# bmoe-server answers every turn as if it were the first

## RESOLVED 2026-08-19

Not fixed in place: pr-161's engine had no way to accept a message array at all
(`GenerateRequest` there is `prompt` only; `grep -rn messages_json ~/bmoe-pr161` returns
nothing), so `extract_last_user_message` was the only thing that server could do. The fix was
to PORT `cli/server_main.cpp` onto `main`, whose engine takes `messages_json` and `tools_json`
-- branch `feat/native-server`, merged to fork main as d830872.

Dropped in the port, because main's core has no field for them: `--mmproj` and the image
extraction that fed it (main is text-only), and `--batch-size` (main has n_ubatch, not n_batch).

REVISITED 2026-08-28, and both dropped items were looked at again with the branch still open.
Neither decision reversed the flattening fix: what pr-161 keeps that main must never take back is
`extract_last_user_message` and the ~190 lines around it. Any future pass over that branch takes
named changes onto main's server, never the file.

  - `--batch-size` is now IN (01bb7f2). The reason it was dropped was the missing field, not the
    knob, so the field was added -- and the more valuable half turned out to be the DEFAULT it
    changes. main had `sc.n_batch = cfg.n_ctx` ("one-batch prefill for any prompt that fits the
    context") with n_ubatch following it, so opening a session at a large context reserved compute
    buffers for a full-width prefill graph before decoding a token; that reservation is resident
    and comes out of the expert cache. Both now default to 512 and are set independently, in
    runtime.cpp so bmoe-cli gets them too. Authorship of the original change preserved.
  - `--mmproj` STAYS OUT, and the reason is stronger than "main is text-only": it does nothing on
    pr-161 either. Its engine side loads the projector and then, in generate(), prints
    `bmoe: vision input: %zu image(s) provided (MTMD integration pending)` and drops them -- no
    decode, no embeddings, nothing prepended, while session.h's comment claims the opposite. The
    cost of taking it would be LLAMA_BUILD_MTMD=ON on every build, mtmd linked into bmoe_core, and
    two public API fields that are silent. Revisit only if the MTMD integration is actually written.

A second defect surfaced during acceptance and is fixed in 94527d8: a request carrying `tools`
returned 200 with content "", tool_calls null, finish_reason "stop". The engine had the call in
`RunResult::tool_calls_json` (runtime.h:37) all along and neither response path read it.

Acceptance, all verified live on evo against Qwen3-Coder-30B-A3B:

    turn 1  [user]                    22 prompt tokens -> "noted"
    turn 2  [user, assistant, user]   24 prompt tokens -> "782"
    system message honoured           "Arrr, the capital of France be Paris, matey!"
    fresh conversation                "UNKNOWN" -- no leakage between callers
    /v1/completions raw prompt        "The capital of France is Paris."
    streaming, same history           "782"
    tools, non-streaming              finish_reason tool_calls, get_weather{city:Nuremberg}
    tools, streaming                  same array in one delta
    ctest                             10/10

STILL OPEN, left as follow-ups rather than reopening this card:
  - streaming content deltas still carry the raw <tool_call> text, because pieces stream before
    the final parse reclassifies them. A client should prefer tool_calls when finish_reason says
    so. Same property engram id:492 recorded on the Python bridge.
  - no cached-token count in the response, so a client cannot display prefix reuse, and cannot
    warn when the prefix breaks.

---

column: Done
closed: 2026-08-19
fixed-by: 8dadae6 (merged d830872), 94527d8; follow-up 01bb7f2
created: 2026-08-19
branch: pr-161

## Symptom

`bmoe-server` serves an OpenAI-compatible API but keeps no conversation. Every request is
answered from its last user message alone; all prior turns are discarded silently, with a
200 and a confident wrong answer.

Measured on evo 2026-08-19, native x86-64 build, qwen3moe (Qwen3-Coder-30B-A3B):

    turn 1  messages=[user]                      prompt_tokens=22   -> "noted"
    turn 2  messages=[user, assistant, user]     prompt_tokens=16   -> "I don't know what
                                                                       your favorite number
                                                                       is, since I don't have
                                                                       information about your
                                                                       personal preferences."

Turn 2 carried a THREE-message history and produced FEWER prompt tokens than turn 1. That
inversion is the tell: the history never reached the engine.

## Cause

`cli/server_main.cpp:603`

    prompt = extract_last_user_message(req.body);

and the helper's own comment at line 166: "Extract the last user message content from a chat
messages array." Only the final user turn is rendered; `assistant` turns, `system` turns and
every earlier `user` turn are dropped on the floor.

## Why this matters more than it looks

- It is silent. There is no error, no warning, no truncation notice — just an answer that
  ignores everything said before. A caller cannot tell a lost history from a model that
  forgot.
- It blocks retiring `bmoe-bridge.py`. The bridge is the only working Linux HTTP path today
  precisely because it passes the full `messages_json` to the engine. `bmoe-server` cannot
  replace it until this is fixed.
- It makes the KV prefix-reuse card (.tasks/kv-prefix-reuse-recurrent-memory.md) untestable
  through this server: with a one-message prompt there is no prefix to reuse.

## Fix

Render the whole `messages` array through the chat template and hand it to the engine the
way `bmoe-bridge.py` does — the engine already accepts a full history via `messages_json`
and REPLACES its own with it (session.cpp:929-935), which is the correct multi-client
semantic for an HTTP server serving several callers.

Note `extract_image_urls` (server_main.cpp:305) walks the same array and will need the same
treatment if a non-final message can carry an image.

## Acceptance

1. The measurement above inverts: turn 2's `prompt_tokens` EXCEEDS turn 1's, and the model
   answers "782" to "what is my favourite number times two" after being told it is 391.
2. A `system` message in position 0 is honoured (send one that forbids a word; verify).
3. Two different conversations alternating against one server do not leak turns into each
   other — each request's history is what is served, per the session.cpp:929-935 contract.
4. `/v1/completions` (raw `prompt`, no messages array) is unchanged.
5. Streaming (`stream=true`) carries the same prompt as non-streaming for identical input.

## Repro setup on evo

    cd ~/bmoe-pr161 && cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
    cmake --build build-native --target bmoe-server -j$(nproc)      # ~4 min, 478 KB binary
    ~/bmoe-pr161/build-native/cli/bmoe-server \
      -m /usr/share/ollama/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a \
      --port 8790 --host 0.0.0.0 -t 12 -c 4096 --ubatch 512 \
      --moe-stream --cache-mb 12000 --io-threads 8 --overlap --dense-weights anon

Loads in ~12s. Use qwen3moe for this card — it is the model whose KV reuse already works, so
any prompt-size anomaly is this bug and not the recurrent-memory one.

OPS: do not stop it with `pkill -f bmoe-server` over ssh — the pattern matches the remote
command's own line and kills the ssh session (engram id:486). Use the pid.

## Related

- .tasks/kv-prefix-reuse-recurrent-memory.md — blocked by this card for server-side testing
- engram id:486 (the pkill-over-ssh trap), id:509 (the KV correction)
