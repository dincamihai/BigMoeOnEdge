# Upstream is implementing expert streaming, and the open PR is ahead of ours in two ways

column: Todo
created: 2026-08-28

## What is there

`ggml-org/llama.cpp` PR #25294, "llama : stream MoE routed experts from disk". Open since
2026-07-04, +1450/-16 over 17 files, still being updated. Same problem, same shape: a per-layer
cache of expert slabs, a custom op after the router's top-k that remaps expert ids to cache slots,
missing experts fetched by an I/O thread pool while the op waits, O_DIRECT with a buffered
fallback. Flags: `--moe-stream-cache`, `--moe-stream-io-threads`, `--moe-stream-direct`.

Reported on GLM-5.2-UD-Q2_K_XL (a 254 GB file): decode 1.83-2.20 tok/s, prefill up to 5.69, and
5.3x prefill / 2.4x decode against `--cpu-moe` with mmap. Our own Q5 numbers on evo are the same
order (1.2-1.9 tok/s decode), on different hardware.

Status: no maintainer approval. The objection is not technical -- it is that a three-tier
VRAM/RAM/SSD design should be settled first.

## Two things it does that we do not

**Wave-partitional prefill.** Expert GEMMs are split into waves when the cache cannot hold the
working set, so a long prompt is no longer bounded by `n_ubatch`. Their own numbers put prefill at
2.28 -> 5.65 tok/s from this alone. This is the same amortisation argument that came up here from
first principles -- prefill reads each expert once per BATCH rather than once per token -- and they
have implemented it.

**It refuses mmap on purpose.** The PR auto-disables mmap, on the grounds that the page cache is
counterproductive once the model is far larger than RAM. Our streamer is built the other way round:
file-backed mmap plus a rebind, which is also why `n_gpu_layers` must be 0. These are opposite
positions on the same question and theirs is the one with numbers attached.

## Why this card exists

Nothing here is an argument to stop. It is an argument against believing the streamer is the part
that makes this project distinctive: as of today it is being built upstream, by people who will
also get the server, the chat glue and the maintenance for free. Two honest next steps, in order:

1. Build PR #25294 on evo and run it against Q5, same prompt, same measurements. That is a few
   hours and it decides whether our streamer has any remaining reason to exist here.
2. If theirs wins, the useful contribution is the comparison, not the code -- the PR is stalled on
   a design argument and has no third-party numbers in it.

If ours wins on this model, the interesting difference is mmap+rebind versus O_DIRECT, and that is
worth saying in the PR rather than keeping.
