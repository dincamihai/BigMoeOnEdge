// Expert streaming, attached by a host that is NOT bmoe::Session.
//
// The engine's value is the streamer; everything else in core/src/engine is a chat layer that
// llama.cpp's own server already does, and does better. This test pins the seam that lets the two
// be separated: a plain llama.cpp host -- model, context, decode loop, nothing else -- turns
// streaming on through a public helper and gets BYTE-IDENTICAL tokens out.
//
// It is the G1 guarantee (streamed == resident) asserted through the HOST API rather than through
// Session, so it fails the moment the helper starts depending on session state again.
//
// Greedy decoding throughout: with argmax there is exactly one correct continuation, so a single
// differing token id is a real divergence and not sampler noise.

#include "bmoe/config.h"
#include "bmoe/expert_stream_host.h"

#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static int g_last_n_expert = 0;

static void expect(const char * name, bool ok, const std::string & detail = {}) {
    if (ok) {
        std::printf("[PASS] %s\n", name);
    } else {
        std::printf("[FAIL] %s%s%s\n", name, detail.empty() ? "" : " -- ", detail.c_str());
        ++failures;
    }
}

// One greedy continuation of a fixed prompt. `host` is null for the resident baseline.
static std::vector<llama_token> run(const char * model_path, bool streamed, int n_gen, std::string & err,
                                   bmoe::DenseWeightsMode dense = bmoe::DenseWeightsMode::Mmap,
                                   uint64_t * dense_bytes = nullptr) {
    bmoe::MoeStreamConfig moe;
    moe.enabled = streamed;
    moe.dense_weights = dense;

    bmoe::ExpertStreamHost host(moe, model_path);

    llama_model_params mparams = llama_model_default_params();
    if (streamed) host.configure(mparams);

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { err = "model load failed"; return {}; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    if (streamed) host.configure(cparams);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { err = "context creation failed"; llama_model_free(model); return {}; }

    if (streamed && !host.attach(model, ctx, err)) {
        llama_free(ctx);
        llama_model_free(model);
        return {};
    }

    if (dense_bytes) *dense_bytes = host.dense_rebound_bytes();
    if (streamed) g_last_n_expert = host.n_expert();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tok = llama_vocab_bos(vocab);
    if (tok < 0) tok = 0;

    std::vector<llama_token> out;
    llama_pos pos = 0;
    for (int i = 0; i < n_gen; ++i) {
        llama_batch b = llama_batch_get_one(&tok, 1);
        b.pos = &pos;
        if (llama_decode(ctx, b) != 0) { err = "decode failed"; break; }
        ++pos;
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        for (int t = 1; t < n_vocab; ++t)
            if (logits[t] > logits[best]) best = t;
        out.push_back(best);
        tok = best;
    }

    llama_free(ctx);
    llama_model_free(model);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tiny-moe.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    std::string err_res, err_str;
    const std::vector<llama_token> resident = run(argv[1], /*streamed*/ false, 8, err_res);
    const std::vector<llama_token> streamed = run(argv[1], /*streamed*/ true, 8, err_str);

    expect("a host with no Session can attach expert streaming", !streamed.empty(), err_str);
    expect("the resident baseline generated", !resident.empty(), err_res);
    expect("streamed == resident, token for token",
           !resident.empty() && streamed == resident,
           "resident " + std::to_string(resident.size()) + " tokens, streamed " +
               std::to_string(streamed.size()));

    // The dense-weights policy is what let a 158 GB model run on a 123 GB box: the experts stream,
    // and the dense side is read into private buffers instead of being left to the page cache. A
    // host that forgets to hand those tensors over gets a streamer that SILENTLY stays on mmap --
    // same answers, none of the memory behaviour -- so what is asserted is the rebind itself, not
    // the tokens.
    uint64_t rebound = 0;
    std::string err_anon;
    const std::vector<llama_token> anon =
        run(argv[1], /*streamed*/ true, 8, err_anon, bmoe::DenseWeightsMode::Anonymous, &rebound);

    // What the harvest concluded, not what the caller assumed. A wrong expert count does not fail
    // loudly -- it makes the streamer slice the tensor on the wrong stride -- so the number the
    // capture arrived at is worth stating out loud and pinning.
    expect("the host reports the expert count it discovered", g_last_n_expert == 8,
           "n_expert " + std::to_string(g_last_n_expert));

    expect("the anonymous dense policy rebinds dense weights off mmap", rebound > 0,
           "rebound " + std::to_string(rebound) + " bytes");
    expect("rebinding the dense weights does not change the answer",
           !resident.empty() && anon == resident, err_anon);

    llama_backend_free();
    std::printf("\nhost attach: %s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
