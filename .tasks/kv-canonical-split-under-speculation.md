# The canonical/working split is off when a draft context is present

column: Todo
created: 2026-08-19

## What is carried

`kv-canonical-scratch-sequences.md` shipped the split that keeps a thinking model's reasoning span
out of the conversation's KV, and with it prefix reuse works on DeepSeek-V4-Flash. It is gated:

    const bool canon_on = chat_on && im.ctx_dft == nullptr;

So a session opened with MTP speculation keeps the PREVIOUS behaviour — on a thinking model with a
recurrent memory that means the full re-prefill every turn the other card measured at
`prompt_tokens 13 -> 30 -> 53`. n-gram speculation is unaffected: it has no draft context, so
`canon_on` stays true and it already gets the fix.

The gate is deliberate, not an oversight. The draft context mirrors sequence 0's positions — the
existing code rewinds it to the same point as the target and warns that missing this makes the
SECOND message of a conversation fail while the first always works — so the split has to fork it in
step with the target: the same `seq_cp` into a working sequence, the same commit of the clean turn
into a canonical one, at the same positions. Half of that is worse than none.

## What to work out

1. Whether the draft context needs a canonical sequence of its own at all, or whether it can simply
   be re-seeded from the target's canonical tokens at the top of each turn. Re-seeding costs a
   draft-context prefill of the whole conversation per turn, which may well be cheaper than it
   sounds — the draft context is small by construction — and it is much less state.
2. `cparams.n_seq_max` for the draft context. The target went to 3 for work, parking bay and
   canonical; the draft context is opened separately and would need the same room.
3. Whether `common_speculative_begin` and the MTP driver tolerate a target sequence that was copied
   rather than decoded. They read the batch's sequence ids, so this is a question about them and
   not about the memory.

## Acceptance

* With MTP on, DSv4-Flash through bmoe reuses its prefix across a 3-turn thinking conversation:
  `tail=0`, no `refuses a partial KV removal` warning, only the new turn prefilled. Same shape as
  the numbers in the canonical-split card.
* The second message of a conversation still works — that is the specific failure the draft-context
  rewind exists to prevent, so it is the specific thing a mistake here brings back.
* No regression with MTP off, or with n-gram speculation, on either model.
* ctest green on evo.

## Notes

* Measure with `BMOE_KV_TRACE=1` and stderr visible.
* Testing against DSv4 needs `systemctl --user stop ds4` first (it holds ~114 GiB of 123 GiB), and
  the model takes 5-10 minutes to load cold. Batch the runs into one window and restart ds4 after —
  it is the user's daily server on :8099.
* Commits: Conventional Commits, author Helldez only, no AI co-author or session trailers.

## Related

* `.tasks/kv-canonical-scratch-sequences.md` — the split this extends
* `.tasks/kv-prefix-reuse-recurrent-memory.md` — the diagnosis and the raw numbers
