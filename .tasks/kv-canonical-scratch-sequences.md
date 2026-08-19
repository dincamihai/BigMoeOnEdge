# Keep reasoning out of the canonical KV: a scratch sequence per turn

column: Done
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

## Outcome (2026-08-19)

Implemented and measured on evo. DSv4-Flash WITH thinking, 3 append-only turns:

    turn2  prefix n_common=12  kv_tokens=12  tokens=30  tail=0  rewind=no
    turn3  prefix n_common=29  kv_tokens=29  tokens=53  tail=0  rewind=no
    canon commit  keep=0 -> 12 -> 29,  clean=12 -> 29 -> 52,  ok=1
    prompt_tokens 13 -> 18 -> 24   (was 13 -> 30 -> 53)

No `refuses a partial KV removal` warning is printed: the branch is never entered. qwen3moe is
unregressed and slightly better — `prompt_tokens 21 -> 12 -> 12` against 21 -> 15 -> 15, `tail=0`,
warm walls 0.68s/0.62s/0.73s, answers identical. The two-client alternation test passes on BOTH
models: no leak, and the non-matching history still re-prefills instead of appending onto a
stranger's turns. ctest 10/10.

THE PART THAT WAS NOT OBVIOUS, and what two failed attempts cost. The canonical tokens are "this
conversation as a LATER prompt will contain it", and no single render produces that string. First
attempt rendered `im.chat_history` directly: the template kept the LAST assistant's reasoning, so
the canonical sequence was 106 tokens against a 30-token prompt — the reasoning went straight back
in. Clearing `reasoning_content` fixed that and still diverged, because DeepSeek renders a PAST
assistant as `<|Assistant|></think>text` and the LAST one as `<|Assistant|><think></think>text`.
Any render ending in this turn's answer is a string no later prompt ever contains.

The fix is to stop modelling the template: render the same history twice with a throwaway message
appended, differing only in ROLE, and keep the tokens the two renders agree on. They agree through
the end of the answer and diverge at the next role header — exactly the cut wanted, and it names no
marker, role or reasoning syntax. Worth defending in review: it looks like a trick, and the two
obvious direct renders are both wrong.

CARRIED, not done: the split is off when a speculative DRAFT CONTEXT is present (`im.ctx_dft`),
since that context mirrors sequence 0's positions and would have to be forked in step. With a draft
context the engine keeps its previous behaviour. n-gram speculation is unaffected (no draft
context). Worth its own card if MTP is wanted on a thinking model.

One observation, not a defect: at temperature 0 the DSv4 answers now differ slightly from the
re-prefill run (turn 3 reasoned longer and hit the token cap). The prompt tokens are identical; a
reused KV is not bit-identical to a recomputed one, and that is true of prefix reuse anywhere.

## Correction (2026-08-19): it shipped half-broken, and the fix for that is the sentinel pair

The split above landed, but the canonical sequence NEVER HELD THE ANSWER. Every turn committed the
conversation only as far as the user message; the answer arrived a turn late, out of the caller's
history. An 889-token prompt reused 29 tokens.

The cause is the trick this card calls the part that was not obvious. Varying the sentinel's ROLE
assumes a role change moves only where the two renders diverge. It does not. DeepSeek-V4 computes
`last_user_idx` and sets

    keep_reasoning = tp.has or (loop.index0 > last_user_idx.value)

so a USER sentinel puts the answer BEFORE the last user message (reasoning stripped, `</think>x`)
while an ASSISTANT sentinel leaves it after (reasoning kept, `<think></think>x`). The renders
disagree at the answer's own header, every turn — the same last-vs-past asymmetry this card was
written to survive, re-entered through the fix for it.

Fix: hold the ROLE fixed and vary only the CONTENT. Whatever the template keys on is then identical
in both renders, which can only diverge inside the sentinel itself. Commit
"fix(engine): vary the canonical sentinel's content, not its role"; derivation extracted to
`core/src/engine/canonical_prefix.{h,cpp}` and unit-tested against vendored templates.

NOT DeepSeek-specific — rendered offline from the vendored templates, the old cut kept 44 chars on
DeepSeek-V4, stopped short of the answer on Qwen3.5, and kept 22 chars on GLM-4.6 and 10 on
MiniMax-M2, both still inside the preamble. On those two the split was INERT.

Measured before -> after, same 3 turns: canonical 17/26/35 -> 21/30/39, prefilled 18/10/10 -> 18/6/6.

WHY THE VALIDATION IN THIS CARD MISSED IT, which is the lesson worth keeping: every assertion was
about prefix reuse working, and reuse DID work — `tail=0`, no refusal, prompt_tokens 13->18->24.
A canonical that stops before the answer is still a valid prefix; it just makes the next turn
re-prefill the answer, which is invisible when answers are 4 tokens long. The new test asserts the
answer is IN the canonical text, and that assertion is the whole test. It was itself vacuous on
first writing (the oracle's answer string also occurred in the question) and passed against the
broken code until the answer was changed to something the prompt could not contain.

ALSO: ctest is no longer 10/10 as recorded above — `G15 ... the block was not moved` fails on clean
HEAD (verified by stashing; identical numbers 1101/1105 and 1174/1178). Unrelated to this work,
regressed some time after this card was written.
