#include "bmoe/expert_stream_host.h"

#include "expert_stream_source.h"
#include "gguf_offsets.h"
#include "router_hook.h"
#include "bmoe/recipe.h"

#include "llama.h"

#include <string>
#include <vector>

namespace bmoe {

struct ExpertStreamHost::Impl {
    MoeStreamConfig cfg;
    std::string model_path;
    GgufMeta meta;
    const MoeRecipe * recipe = nullptr;
    std::unique_ptr<RouterHook> hook;
    ExpertStreamSource source;
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

    const int n_layer = llama_model_n_layer(model);
    impl_->hook->set_recipe(*impl_->recipe, n_layer);
    impl_->hook->set_prefetch_layers(impl_->cfg.prefetch_layers);

    // Capture warm-up: one mmap-resident decode, so the eval callback can harvest the expert
    // tensor pointers out of the graph. Any valid token builds the same graph -- the expert tensor
    // structure does not depend on the prompt -- and the KV is wiped afterwards.
    impl_->hook->begin_capture();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token warm = llama_vocab_bos(vocab);
    if (warm < 0) warm = 0;
    llama_batch b = llama_batch_get_one(&warm, 1);
    const int rc = llama_decode(ctx, b);
    impl_->hook->end_capture();
    if (rc != 0) { err = "capture warm-up decode failed"; return false; }
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

    if (!impl_->source.init(offs.shard_paths, n_expert, std::move(layers), impl_->cfg)) {
        err = "expert stream source init failed";
        return false;
    }
    impl_->hook->set_source(&impl_->source);
    impl_->attached = true;
    return true;
}

bool ExpertStreamHost::attached() const { return impl_->attached; }

} // namespace bmoe
