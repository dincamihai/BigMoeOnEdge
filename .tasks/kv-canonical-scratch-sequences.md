# Keep reasoning out of the canonical KV: a scratch sequence per turn

column: Todo
created: 2026-08-19

## Why

`kv-prefix-reuse-recurrent-memory.md` established, by measurement and by a control run, that
prefix reuse dies on DeepSeek-V4-Flash for one reason: the model's reasoning tokens are decoded
into the KV and then stripped before they reach `chat_history`, so the next turn's re-render is
not an extension of what was decoded. The engine asks for a rewind of `n_gen` tokens, a Gated
Delta Net refuses it, and `llama_memory_clear` throws the whole prefix away.

With `--no-think` the same weights and the same recurrent memory reuse perfectly (`tail=0`,
`prompt_tokens 13 -> 9 -> 9`). The memory type is not the problem. The reasoning span is.

## The design

Split the two things the KV is being asked to be at once — the conversation the template
describes, and the scratchpad the model needs while answering.

* **Sequence 0 is canonical.** It holds prompts and CLEAN answers only, exactly what re-rendering
  the history produces. Nothing else is ever decoded into it.
* **A scratch sequence holds the turn in progress.** Each turn: `seq_cp(0 -> scratch)`, decode the
  new user turn into scratch, and generate there. The model sees its own reasoning while it works.
* **On completion, commit the clean form.** Decode the new user turn plus the parsed assistant
  answer into sequence 0, then clear scratch.

Every operation on both sequences is an APPEND. `llama_memory_seq_rm` is never called on a partial
range, so nothing can refuse it. That is what makes this work on a recurrent memory, and it is why
the fix is not "make the rewind succeed".

Two properties fall out that the simpler alternatives do not have:

* The model never sees an EARLIER turn's reasoning, which is what the DeepSeek template intends.
  Answer quality is unchanged; there is no flag to maintain and no A/B to run.
* Context grows at the clean rate. Retaining reasoning would have inflated it roughly 5-8x — turn 1
  of the measurement was 13 prompt tokens against 92 generated. See the companion card.

It also fixes the render instability at its root rather than tolerating it: sequence 0's tokens are
BY CONSTRUCTION the tokens a re-render produces, because that is what was decoded into it.

## What the code has to grow

* `cparams.n_seq_max = 2` at `core/src/engine/session.cpp:520` — sequence 1 is already the splice's
  parking bay (`session.cpp:1120`), so scratch needs its own id and this must go to 3. Confirm the
  parking bay and scratch never overlap in a turn rather than assuming it.
* A second decode path: today generation and prefill both write to sequence 0. The commit step
  decodes without sampling, so it is a prefill of a known token list, not a generate.
* `im.kv_tokens` currently mirrors sequence 0 and is the prefix-reuse ground truth. It must keep
  mirroring the CANONICAL sequence, so it records the clean answer's tokens, not the generated ones.
* The multi-client contract at `session.cpp:929-935` is untouched: a caller history that is not an
  extension of the canonical sequence still replaces the conversation and re-prefills in full.

## Verify

Measured with `BMOE_KV_TRACE=1` and stderr visible, the same 3-turn append-only driver as the
companion card. Numbers to beat, all from 2026-08-19 on evo:

1. DSv4-Flash WITH thinking goes from `tail=93 / tail=158`, `rewind=yes`, refusal warning, and
   `prompt_tokens 13 -> 30 -> 53` at 29.9s/49.9s/58.8s, to `tail=0`, no warning, and only the new
   turn prefilled per turn.
2. qwen3moe unchanged: `tail=0` both turns, `prompt_tokens 21 -> 15 -> 15`, 2.16s/0.60s/0.61s.
3. Two clients alternating distinct histories against one session: no leakage, and the non-matching
   history still re-prefills rather than appending onto a stranger's turns.
4. The byte-identity gates still pass and `clear_kv` still reduces to a full prefill.
5. ctest green on evo.

## Notes

* Build with the venv python or 6 of 10 ctest tests fail on a missing gguf/numpy:
  `cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DPYTHON3=$HOME/.venvs/ggufbuild/bin/python3`
* Testing against DSv4 needs `systemctl --user stop ds4` first — it holds ~114 GiB of 123 GiB and
  bmoe-server cannot load the shards alongside it. Restart it when done; it is the user's daily
  server on :8099.
* Commits: Conventional Commits, author Helldez only, no AI co-author or session trailers.

## Related

* `.tasks/kv-prefix-reuse-recurrent-memory.md` — the diagnosis and the measurements
* `.tasks/kv-context-budget-with-reasoning.md` — the context-growth question, deliberately separate
* engram id:509 (llama-server keeps its decoded sequence as ground truth and appends)
