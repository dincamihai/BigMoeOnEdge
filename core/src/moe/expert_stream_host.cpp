#include "bmoe/expert_stream_host.h"

#include "expert_stream_source.h"
#include "gguf_offsets.h"
#include "router_hook.h"
#include "bmoe/recipe.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

namespace bmoe {

namespace {

// The names of the expert tensors the streamer owns. Both consumers below start from the same
// question -- what the streamer manages, and by subtraction what it leaves alone -- so they ask it
// once here instead of each carrying its own copy of the triple loop.
std::unordered_set<std::string> expert_tensor_names(const std::vector<LayerExperts> & layers) {
    std::unordered_set<std::string> names;
    for (const LayerExperts & L : layers) {
        if (!L.bound) continue;
        for (int p = 0; p < MoeRecipe::max_exps; ++p)
            if (L.proj[p].tensor) names.insert(L.proj[p].tensor->name);
    }
    return names;
}

// Each layer's bytes that the streamer does NOT manage: everything under blk.<il>. except the
// expert weight tensors it rebinds -- attention, norms, the router, and any per-expert scale left
// mmap-resident. A static property of the file: nothing about decoding changes it, which is why
// the route trace states it once instead of pretending to measure it per step.
std::vector<uint64_t> dense_bytes_per_layer_impl(const GgufOffsets & offs,
                                                 const std::vector<LayerExperts> & layers, int n_layer) {
    const std::unordered_set<std::string> streamed = expert_tensor_names(layers);
    std::vector<uint64_t> out((size_t) (n_layer > 0 ? n_layer : 0), 0);
    for (const auto & kv : offs.size_by_name) {
        int il = -1;
        if (std::sscanf(kv.first.c_str(), "blk.%d.", &il) != 1) continue;
        if (il < 0 || il >= n_layer || streamed.count(kv.first)) continue;
        out[(size_t) il] += kv.second;
    }
    return out;
}

} // namespace

struct ExpertStreamHost::Impl {
    MoeStreamConfig cfg;
    std::string model_path;
    GgufMeta meta;
    const MoeRecipe * recipe = nullptr;
    std::unique_ptr<RouterHook> hook;
    ExpertStreamSource source;
    ExpertStreamHost::CaptureFn capture;
    std::vector<uint64_t> dense_bytes;
    int extra_layers = 0;
    int n_expert = 0;
    int n_layer_streamed = 0;
    bool want_dense_bytes = false;
    bool attached = false;
};

ExpertStreamHost::ExpertStreamHost(MoeStreamConfig cfg, std::string model_path)
    : impl_(new Impl{}) {
    impl_->cfg = std::move(cfg);
    impl_->model_path = std::move(model_path);
}

ExpertStreamHost::~ExpertStreamHost() = default;

void ExpertStreamHost::configure(llama_model_params & mparams) const {
    if (!impl_->cfg.enabled) return;
    // The three properties the rebind depends on. mmap because the streamer hands llama.cpp
    // pointers INTO the mapped file; no extra buffer types because a repacked quantisation is a
    // different layout than the one on disk; no GPU offload because the rebind targets host memory
    // and a tensor living in VRAM cannot be pointed somewhere else.
    mparams.load_mode       = LLAMA_LOAD_MODE_MMAP;
    mparams.use_extra_bufts = false;
    mparams.n_gpu_layers    = 0;
}

void ExpertStreamHost::configure(llama_context_params & cparams) {
    if (!impl_->cfg.enabled) return;
    // The hook has to exist before the context does, because its address is what the context
    // stores. Its recipe is filled in at attach(), once the model can be asked its architecture.
    if (!impl_->hook) impl_->hook = std::make_unique<RouterHook>(MoeRecipe{}, 0);
    cparams.cb_eval           = &RouterHook::c_eval;
    cparams.cb_eval_user_data = impl_->hook.get();
}

bool ExpertStreamHost::attach(llama_model * model, llama_context * ctx, std::string & err) {
    if (!impl_->cfg.enabled) { err = "expert streaming is not enabled in this config"; return false; }
    if (!impl_->hook) { err = "configure(llama_context_params&) was not called before the context"; return false; }

    char arch[128] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    impl_->recipe = find_moe_recipe(arch);
    if (!impl_->recipe) {
        err = std::string("no MoE recipe for architecture '") + arch +
              "' -- add one in core/src/moe/arch_registry.cpp";
        return false;
    }

    impl_->n_layer_streamed = llama_model_n_layer(model) + impl_->extra_layers;
    impl_->hook->set_recipe(*impl_->recipe, impl_->n_layer_streamed);

    // Capture warm-up: one mmap-resident decode, so the eval callback can harvest the expert
    // tensor pointers out of the graph. Any valid token builds the same graph -- the expert tensor
    // structure does not depend on the prompt -- and the KV is wiped afterwards.
    impl_->hook->begin_capture();
    bool captured_ok;
    if (impl_->capture) {
        captured_ok = impl_->capture(ctx);
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_token warm = llama_vocab_bos(vocab);
        if (warm < 0) warm = 0;
        llama_batch b = llama_batch_get_one(&warm, 1);
        captured_ok = llama_decode(ctx, b) == 0;
    }
    impl_->hook->end_capture();
    if (!captured_ok) { err = "capture warm-up decode failed"; return false; }
    llama_memory_clear(llama_get_memory(ctx), true);

    impl_->meta = read_gguf_meta(impl_->model_path.c_str());
    const GgufOffsets & offs = impl_->meta.offsets;
    if (!offs.ok) { err = "cannot read gguf offsets: " + impl_->model_path; return false; }

    std::vector<LayerExperts> layers = impl_->hook->captured();
    int n_expert = 0;
    int n_bound = 0;
    for (LayerExperts & L : layers) {
        if (!L.bound) continue;
        ++n_bound;
        for (int p = 0; p < MoeRecipe::max_exps; ++p) {
            if (!impl_->recipe->exps_suffix[p]) continue; // slot unused by this architecture
            ggml_tensor * t = L.proj[p].tensor;
            if (!t) {
                err = std::string("captured MoE layer is missing expert tensor '") +
                      impl_->recipe->exps_suffix[p] + "'";
                return false;
            }
            auto it = offs.off_by_name.find(t->name);
            if (it == offs.off_by_name.end()) { err = std::string("no gguf offset for tensor ") + t->name; return false; }
            L.proj[p].file_off = it->second;
            L.proj[p].file_idx = offs.file_by_name.at(t->name);
            const int ne2 = (int) t->ne[2];
            if (n_expert == 0) {
                n_expert = ne2;
            } else if (ne2 != n_expert) {
                err = std::string("inconsistent expert count: tensor ") + t->name + " has " +
                      std::to_string(ne2) + ", expected " + std::to_string(n_expert);
                return false;
            }
        }
    }
    if (n_bound == 0) { err = "no MoE expert tensors captured -- is this a MoE model?"; return false; }
    impl_->n_expert = n_expert;

    // Before init consumes `layers`: the dense split needs the captured expert names and the gguf
    // sizes together, so it cannot be recovered afterwards.
    if (impl_->want_dense_bytes)
        impl_->dense_bytes = dense_bytes_per_layer_impl(offs, layers, impl_->n_layer_streamed);

    // The dense (non-expert) weights, for the policies that take them off mmap. The list is every
    // captured weight leaf that IS a gguf tensor -- graph inputs and KV share the leaf shape and
    // are dropped by the offset lookup -- minus the experts the streamer already owns. Built before
    // init, which consumes `layers`.
    if (impl_->cfg.dense_weights == DenseWeightsMode::Anonymous ||
        impl_->cfg.dense_weights == DenseWeightsMode::Pinned) {
        const std::unordered_set<std::string> expert_names = expert_tensor_names(layers);
        std::vector<DenseTensorRef> dense;
        for (const auto & kv : impl_->hook->captured_weights()) {
            const std::string & name = kv.first;
            if (expert_names.count(name)) continue;
            auto off = offs.off_by_name.find(name);
            auto sz  = offs.size_by_name.find(name);
            if (off == offs.off_by_name.end() || sz == offs.size_by_name.end()) continue; // not a file tensor
            DenseTensorRef d;
            d.tensor   = kv.second;
            d.file_off = off->second;
            d.size     = sz->second;
            d.file_idx = offs.file_by_name.at(name);
            dense.push_back(d);
        }
        impl_->source.set_dense_tensors(std::move(dense));
    }

    if (!impl_->source.init(offs.shard_paths, n_expert, std::move(layers), impl_->cfg)) {
        err = "expert stream source init failed";
        return false;
    }
    impl_->hook->set_source(&impl_->source);
    impl_->attached = true;
    return true;
}

bool ExpertStreamHost::attached() const { return impl_->attached; }

void ExpertStreamHost::set_extra_layers(int n) { impl_->extra_layers = n; }
void ExpertStreamHost::set_capture(CaptureFn fn) { impl_->capture = std::move(fn); }
void ExpertStreamHost::want_dense_bytes(bool on) { impl_->want_dense_bytes = on; }

RouterHook & ExpertStreamHost::hook() {
    if (!impl_->hook) impl_->hook = std::make_unique<RouterHook>(MoeRecipe{}, 0);
    return *impl_->hook;
}
ExpertStreamSource & ExpertStreamHost::source() { return impl_->source; }

int ExpertStreamHost::n_expert() const { return impl_->n_expert; }
int ExpertStreamHost::n_layer_streamed() const { return impl_->n_layer_streamed; }
const std::vector<uint64_t> & ExpertStreamHost::dense_bytes_per_layer() const { return impl_->dense_bytes; }
const GgufMeta & ExpertStreamHost::meta() const { return impl_->meta; }

uint64_t ExpertStreamHost::dense_rebound_bytes() const {
    return impl_->attached ? impl_->source.dense_rebound_bytes() : 0;
}

} // namespace bmoe
