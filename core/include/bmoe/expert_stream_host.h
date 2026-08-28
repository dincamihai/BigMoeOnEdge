#pragma once

// Expert streaming for a host that is not bmoe::Session.
//
// The streamer attaches to llama.cpp through public mechanisms only -- model parameters, the
// eval callback, and (optionally) the expert-ready hook -- so nothing about it requires this
// project's chat layer. This header is that fact made usable: a llama.cpp host constructs one of
// these, lets it fill in the parameters it needs, and hands it the model and context afterwards.
//
// Two phases, and the order is not a style choice. The streamer learns which tensors are the
// expert weights by watching a real graph go past the eval callback, so it cannot be armed before
// a context exists to decode with.
//
//   configure(model_params)    before the model is loaded   -- mmap, no repack, experts on CPU
//   configure(context_params)  before the context is made   -- installs the eval callback
//   attach(model, context)     once, before serving         -- capture decode, then rebind
//
// After attach() the host decodes normally and never calls in again.

#include "bmoe/config.h"

#include "ggml-backend.h"

#include <functional>
#include <vector>

#include <cstdint>
#include <memory>
#include <string>

struct llama_model;
struct llama_context_params;
struct llama_context;
struct llama_model_params;
struct llama_context_params;

namespace bmoe {

class RouterHook;
class ExpertStreamSource;
struct GgufMeta;

class ExpertStreamHost {
public:
    ExpertStreamHost(MoeStreamConfig cfg, std::string model_path);
    ~ExpertStreamHost();

    ExpertStreamHost(const ExpertStreamHost &)             = delete;
    ExpertStreamHost & operator=(const ExpertStreamHost &) = delete;

    // The layout the rebind requires: file-backed mmap, no extra buffer types (a repacked q4_K
    // buffer cannot be rebound), and no GPU offload (the rebind targets host memory).
    void configure(llama_model_params & mparams) const;

    // The eval callback and its user data, for a host that does not fill a llama_context_params
    // itself. llama.cpp's server builds both params structs inside common_init_from_params from a
    // common_params, so it needs the pair rather than the convenience above; they are the same
    // two values.
    ggml_backend_sched_eval_callback eval_callback() const;
    void * eval_user_data();

    // Installs the router hook as the context's eval callback. The host must not overwrite
    // cb_eval afterwards; there is one callback slot and the streamer needs it.
    void configure(llama_context_params & cparams);

    // Stream more layers than the model reports. A speculative MTP head sits at layer index
    // n_layer and routes experts of its own: left out, they stay mmap-resident, which is the one
    // thing streaming exists to avoid on a model that does not fit. Call before attach().
    void set_extra_layers(int n);

    // The router, for a host that tunes it before attach() -- drop policy, prediction, tracing.
    RouterHook & hook();

    // The streamer itself, for the diagnostics and knobs a host may drive after attach().
    ExpertStreamSource & source();

    // What the capture concluded: experts per tensor, and the streamed span it harvested over.
    int n_expert() const;
    int n_layer_streamed() const;

    // Per-layer dense byte totals, for a host that reports them. Empty unless requested before
    // attach(), because computing it means walking the gguf a second time.
    void want_dense_bytes(bool on);
    const std::vector<uint64_t> & dense_bytes_per_layer() const;

    // The gguf metadata the attach read, so a caller does not parse the file a third time.
    const GgufMeta & meta() const;

    // A capture decode the host runs itself, for the graph the helper cannot build: MTP's expert
    // layer is only reached through the draft context, so a speculative host must issue that pass.
    // Return false to fail the attach. Default: one BOS decode on the given context.
    using CaptureFn = std::function<bool(llama_context *)>;
    void set_capture(CaptureFn fn);

    // Runs one capture decode on the given context, harvests the expert tensors it saw, opens the
    // gguf shards and rebinds. The KV is cleared afterwards, so the host starts from a clean cache.
    // Returns false and fills `err` on any failure; the caller may then carry on unstreamed.
    bool attach(llama_model * model, llama_context * ctx, std::string & err);

    // False until attach() has succeeded.
    bool attached() const;

    // Whether reads overlap compute, i.e. whether the fork's expert-ready hook is armed. Asking
    // for overlap on a build without the hook fails the attach rather than running slower in
    // silence, so this is false only before attach().
    bool overlap_enabled() const;

    // Bytes the dense-weights policy read into private buffers and rebound. Zero under the mmap
    // policies, and zero before attach(). A host can check this to tell a policy that ran from one
    // that silently degraded to mmap because nothing handed it the tensors.
    uint64_t dense_rebound_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bmoe
