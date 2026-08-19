# KV prefix reuse is lost on recurrent-memory models

column: Todo
created: 2026-08-19

## Symptom

With DeepSeek-V4-Flash-0731 (UD-IQ3_XXS), every turn re-prefills the entire prompt.
`core/src/engine/session.cpp:1152` fires its one-time warning:

    bmoe: this model's memory refuses a partial KV removal; every turn re-prefills the
    whole prompt (prefix reuse is unavailable here)

With `qwen3-coder:30b` (arch `qwen3moe`, classic attention) the same binary, bridge and
flags reuse cleanly: turn1 16 prompt/16 prefilled, turn2 34/17, turn3 53/18 cached 35,
prefill 1.53s -> 0.25s.

## Front ends, and which layer each needs

Four callers sit on the same `core/src/engine/session.cpp`. The engine fix below serves
all four; only the first had a bug of its own.

| Front end | Language | Its own KV bug |
|---|---|---|
| APK — `examples/android/.../ApiServer.kt` | Kotlin | `clearKv` hardcoded true on `/v1/chat/completions`. **ALREADY FIXED**, commit e2f9c5d (2026-08-17). Do not re-fix. |
| `bmoe-server` native binary for Termux (PR 161) | C++ | none of its own |
| `bmoe-cli --session` behind `bmoe-bridge.py` on evo — **a real serving path, used daily** | C++ | none of its own |
| `bmoe-server` (branch `pr-161`, `cli/server_main.cpp`) — native OpenAI HTTP server, supersedes the Python bridge | C++ | none of its own |
| `bmoe-cli` one-shot | C++ | none of its own |

The Python bridge exists only because `bmoe-cli` has no serve mode: there is no HTTP server
anywhere in the C++ tree on `main` (the APK's is Kotlin/NanoHTTPD, Android-only). Branch
`pr-161` adds `bmoe-server` with `/v1/chat/completions`, `/v1/completions`, `/v1/models` and
SSE, keeping the model and expert cache loaded — the same amortisation `--session` gives.
That is the intended Linux serving path; the bridge is the stopgap until it lands.

**UPDATE 2026-08-19: the test vehicle is now `bmoe-server`, not the CLI+bridge.** The native
server is on fork main (d830872 + 94527d8) and gives this card what it needs: it accepts a full
conversation, keeps the KV across turns (`clear_kv = false`), and reports `usage.prompt_tokens`
per request -- which IS the per-turn prefill number the acceptance criteria ask you to log.
Measured with qwen3moe: 841 prefilled on turn 1, 24 on turn 2, prefill 9s -> 1s. Repro:

    ~/BigMoeOnEdge/build-native/cli/bmoe-server -m <model> --port 8790 --host 0.0.0.0 \
      -t 12 -c 4096 --ubatch 512 --moe-stream --cache-mb 12000 --io-threads 8 --overlap \
      --dense-weights anon

Point -m at the dsv4-flash shards for the recurrent-memory case. Watch stderr: the one-time
warning from session.cpp:1152 is the signal, and the Python bridge sent stderr to DEVNULL,
which is why this was invisible for hours (id:490).

**Develop and verify on the evo CLI path.** `bmoe-cli --session` is the same engine with
no Kotlin, no NDK, no APK and no phone in the loop, so the whole cycle is edit → build on
evo → run. The APK and the Termux binary inherit the fix without being rebuilt for it.

Note the phone's actual runtime is unsettled: per engram id:453 the only APK on the box is
the 2026-08-14 dud missing its jniLibs `.so`, while PR 161 produced `bmoe-server` +
`~/bmoe-server-termux.tgz` instead, and whether that was ever copied to the phone was never
verified. Do not assume the APK is what runs there.

## What is already established

- `llama_memory_seq_rm` is called ONLY when the new prompt diverges from the cache.
  The guard at session.cpp:1139 is `if (n_common < im.kv_tokens.size())`. A prompt that
  is a strict extension of the cached tokens never reaches the failing branch.
- So the failure means the re-rendered prompt is NOT a strict extension of what was
  decoded, even when the conversation only grew by one turn.
- Replacing `chat_history` with the caller's `messages_json` (session.cpp:929-935) is
  DELIBERATE and documented: an HTTP API serving several clients cannot append, because
  appending would send a stranger's turns along with this request. **Any fix must keep
  that contract.** "Just append instead of replacing" is not an acceptable answer.
- The *hole* case is already solved: commit 10c5bec parks the surviving block in sequence 1
  and hands it back at shifted positions after a history edit (session.cpp:1046). That
  machinery cannot help the recurrent case, because there is no cuttable cell to move.
- A recurrent state genuinely cannot be cut at a position. Making `seq_rm` succeed on a
  Gated Delta Net is not the goal; not needing it is.

## The open question this card must answer FIRST

llama.cpp's `llama-server`, running the SAME weights (`~/models/dsv4-flash/
DeepSeek-V4-Flash-0731-UD-IQ3_XXS-*.gguf`, ROCm build in `~/llama-rocm`), DOES reuse the
prefix — measured 2026-08-19 on evo: 6015 tokens prefilled on turn 1 (40.8s), 22 tokens
on turn 2 (2.1s); across a 6-request agent conversation, 2125 then 74/78/80/98/68.

Two competing explanations. Decide which is true before writing any fix:

  H1. RENDER INSTABILITY. bmoe's re-render of an unchanged history is not byte-identical
      to what was decoded (generation-prompt suffix; an assistant turn re-templated
      rather than replayed as generated). The divergence is a short tail, `seq_rm` is
      called for those few tokens, and on a recurrent memory it refuses.
      => Fix: make turn N's rendering a strict token-prefix of turn N+1's, e.g. by
         replaying the assistant's generated tokens verbatim instead of re-templating.

  H2. DIFFERENT MEMORY IMPLEMENTATION. llama.cpp implements DeepSeek-V4's attention with
      an ordinary positional KV (Lightning Indexer + HC fused ops), while bmoe's fork
      uses a Gated Delta Net. Then llama-server never needs a rewind because its cache is
      cuttable, and bmoe's is not.
      => Fix: different and larger — bmoe must avoid rewinds structurally, e.g. keep the
         decoded token sequence as ground truth and refuse to accept a caller history
         that is not an extension of it (falling back to full re-prefill only then).

Discriminating experiment: log `n_common`, `im.kv_tokens.size()` and `tokens.size()` on
every turn for both models. If H1, `n_common` is close to but below `kv_tokens.size()`
(a short tail). If H2, the numbers match on qwen3moe and the difference is only that
`seq_rm` returns false on DSv4.

## Acceptance

1. The discriminating experiment is run and its numbers recorded in the commit message.
2. With DeepSeek-V4-Flash through bmoe on evo, a 3-turn conversation prefills the full
   prompt on turn 1 and only the new turn's tokens on turns 2 and 3. Report the same
   triple as the qwen3moe baseline above (prompt/prefilled/cached per turn).
3. The multi-client contract at session.cpp:929-935 still holds: a request carrying a
   DIFFERENT conversation still replaces the running one and does not leak prior turns.
   Needs a test that alternates two distinct histories against one session.
4. `qwen3moe` reuse is unchanged (no regression): re-run the turn1/2/3 numbers above.
5. The byte-identity gates still pass; `clear_kv` still reduces to a full prefill.
6. ctest green on evo.

## Notes for whoever picks this up

- Measure with stderr VISIBLE. The bridge sent stderr to DEVNULL, which is why this was
  invisible for hours (id:490).
- The honest test model is DSv4-Flash through bmoe on evo: ~3 tok/s, 40-80s load. Budget
  for that; do not validate only on qwen3moe, which is the model that already works.
- `bmoe-dsv4.service` on evo runs the BRIDGE with Qwen3-Coder-30B despite its name. To
  test DSv4 you must point `BMOE_MODEL` at the dsv4-flash shards.
- Commits: Conventional Commits, author Helldez only, no AI co-author or session trailers
  (AGENTS.md).

## Related

- engram id:490 (original measurement; its "THE MODEL IS THE VARIABLE" headline is wrong)
- engram id:509 (the correction, with the llama-server counter-measurement)
- engram id:487 (client-side consequence: append-only history, no relevance reranking)
