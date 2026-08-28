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

#include <memory>
#include <string>

struct llama_model;
struct llama_context;
struct llama_model_params;
struct llama_context_params;

namespace bmoe {

class ExpertStreamHost {
public:
    ExpertStreamHost(MoeStreamConfig cfg, std::string model_path);
    ~ExpertStreamHost();

    ExpertStreamHost(const ExpertStreamHost &)             = delete;
    ExpertStreamHost & operator=(const ExpertStreamHost &) = delete;

    // The layout the rebind requires: file-backed mmap, no extra buffer types (a repacked q4_K
    // buffer cannot be rebound), and no GPU offload (the rebind targets host memory).
    void configure(llama_model_params & mparams) const;

    // Installs the router hook as the context's eval callback. The host must not overwrite
    // cb_eval afterwards; there is one callback slot and the streamer needs it.
    void configure(llama_context_params & cparams);

    // Runs one capture decode on the given context, harvests the expert tensors it saw, opens the
    // gguf shards and rebinds. The KV is cleared afterwards, so the host starts from a clean cache.
    // Returns false and fills `err` on any failure; the caller may then carry on unstreamed.
    bool attach(llama_model * model, llama_context * ctx, std::string & err);

    // False until attach() has succeeded.
    bool attached() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bmoe
