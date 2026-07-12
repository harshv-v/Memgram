# Public memory-layer leaderboard

One command ranks every installed memory system on the same footing:

```bash
pip install mem0ai                     # + any of: zep-cloud, letta-client, langmem
export OPENAI_API_KEY=sk-...
export ZEP_API_KEY=...                 # only if ranking zep
# letta: docker run -p 8283:8283 lettaai/letta   (only if ranking letta)

python bench/leaderboard/run_board.py
# BOARD_SYSTEMS=memgram,memgram_v2,mem0,no-memory,zep,letta,langmem BOARD_SEEDS=3 ...
```

Outputs `LEADERBOARD.md` (publishable as-is) and `results_board.json` (raw,
per-question). Systems whose SDK/key/server is missing are skipped with the
reason printed — never a hard failure.

## Publishing checklist (credibility is the product)

1. Run with `BOARD_SEEDS=3` minimum; publish the spread, not just the mean.
2. Commit `LEADERBOARD.md` + `results_board.json` together — raw data or it
   didn't happen.
3. The Zep/Letta/LangMem adapters target the SDK versions noted in their
   docstrings — verify against current vendor docs before publishing, and
   state the pinned versions (auto-included in the output).
4. Keep the disclosure block (we are the Memgram authors) — it's what makes
   a vendor-published board an asset instead of a liability.
5. Publish losses too. If Zep wins the update axis, that line ships.
6. Invite PRs: a competitor fixing their own adapter is the best outcome —
   it makes the board community property.

Optionally add LoCoMo as a second board (`bench/locomo/`) for a number the
field already recognizes.
