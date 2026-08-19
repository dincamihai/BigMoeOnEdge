# Context budget when reasoning is retained, and the n_rs_seq rollback window

column: Todo
created: 2026-08-19

Split off deliberately from `kv-canonical-scratch-sequences.md` (user direction 2026-08-19): the
reuse fix lands first and alone, this is the follow-up.

## Why this is a card and not a paragraph

The canonical/scratch design keeps reasoning OUT of sequence 0, so the ~5-8x context inflation it
would otherwise cause does not happen. That is the whole reason it wins over the simpler "keep the
reasoning in the KV" design. So this card is NOT blocking, and it is not a defect in that design.

What it holds is the material that argued the decision, so the next person does not re-derive it,
plus the two questions that survive.

## The inflation number

From the 2026-08-19 measurement on DSv4-Flash: turn 1 was 13 prompt tokens against 92 generated,
whose parsed content is `1, 2, 3.` — roughly 85% of the growth is reasoning. Any design that
retains reasoning in the served context fills `n_ctx` about 5-8x faster on this model. On a short
`-c` that is turns, not conversations.

## The n_rs_seq finding (worth keeping, do not act on it)

This fork's recurrent memory can roll back partially. `llama-memory-recurrent.cpp` in `seq_rm`:

    // partial rollback via per-token snapshot index (bounded by n_rs_seq)
    if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
        const llama_pos rollback = cell.pos - (p0 - 1);
        if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) { ...; return true; }
        return false;

So the refusal we measured was NOT "a recurrent state cannot be cut". It was a rollback of 93 and
then 158 tokens against a window of zero: `n_rs_seq` defaults to 0 and is set only from
`params.speculative.need_n_rs_seq()`, sized to the speculative draft depth.

Widening it is not a general fix and should not be attempted as one. The snapshots are paid for in
tensor width — `const uint32_t n_rows = mem_size * (1 + n_rs_seq)` — so covering a rollback of N
tokens costs (1+N) times the recurrent state, and `n_gen` is unbounded: a long thinking answer is
thousands of tokens. It is the right mechanism for a speculative draft and the wrong one here.

Also recorded, because it was checked and is load-bearing for the scratch design:
`llama_memory_recurrent::seq_cp` does not copy a state, it makes the destination ALIAS the source's
tail cell. That is safe because `find_slot` is copy-on-write — a cell with more than one `seq_id`
is forked into a fresh cell carrying the original's `src`. Decoding into the copy leaves the source
untouched. If that ever changes, the scratch-sequence design breaks silently.

## What the shipped fix changed about this question (2026-08-19)

`kv-canonical-scratch-sequences.md` landed, so the premise above is now half-obsolete and the card
should be read with that in mind. The canonical sequence grows at the CLEAN rate — the 5-8x
inflation does not happen, and the "drop the old reasoning" idea has nothing to drop.

What replaced it is a narrower and more concrete pressure: the WORKING sequence still holds the
whole turn, prompt plus every reasoning token, because that is what the model is generating into.
So the peak context of a session is no longer "the conversation" but

    len(canonical) + this turn's n_gen

and `n_gen` is where the reasoning lives. Measured on DSv4-Flash at `-c 4096`, a 3-turn toy
conversation reached canonical=52 with n_gen up to 220, so the peak was dominated entirely by the
turn in flight. A long thinking answer on a long conversation is what actually approaches n_ctx now,
and it does so within a SINGLE turn rather than by accumulating across them.

That reframes the open items below: a reset that trims history does nothing for a turn whose own
reasoning is what fills the window. Whatever is designed here has to handle the in-turn case, which
is closer to "what happens when a turn cannot finish" than to a budgeting policy.

## What is actually open

1. **Long conversations still end.** Sequence 0 grows at the clean rate, which is slow, but it is
   not bounded. When it approaches `n_ctx` something must give. The shape that fits the append-only
   constraint is a periodic reset: ONE full re-prefill from a trimmed history, paying a single slow
   turn to recover room, rather than a continuous trim which a recurrent memory cannot do at all.
   Decide the trigger (a fraction of `n_ctx`) and what gets dropped.
2. **Nothing measures this yet.** There is no number for how many turns of real DSv4 use fit in a
   given `-c`. Get one before designing the reset, or the threshold is a guess.

## Related

* `.tasks/kv-canonical-scratch-sequences.md` — the fix this defers to
* `.tasks/kv-prefix-reuse-recurrent-memory.md` — the diagnosis and the raw numbers
* engram id:506 (context size is a ceiling, not a cost: you pay for tokens actually sent)
