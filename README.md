# Discard Hold'em Bot v2

## What to submit

Upload **`bot.py` only**. It is a single self-contained file (about 60 KB, limit 5 MB).
It uses only the Python standard library, so it needs no numpy or scipy and no data files.
The preflop equity table is embedded in the source.

Do not upload `engine.py`, `random_bot.py`, this README, logs or test bots. Only the last
valid upload before **15 Nov 2026, 23:59 IST** counts.

## Before you upload (checklist)

- [ ] `python3.11 engine.py bot.py example_bots/random_bot.py` finishes with zero notices.
- [ ] `python3.11 engine.py bot.py example_bots/random_bot.py --stage 2` finishes with zero notices.
- [ ] Time used is well under the bank (30 s per leg in Stage 1, 75 s per leg in Stage 2).
      The engine prints the sum over both legs, so compare against 60 s and 150 s.
- [ ] Run 20 or more different `--seed` values against your previous bot and any other opponents.
- [ ] Nothing is printed to stdout except protocol JSON (the file contains no `print` calls).

## How the bot decides

**Time governor.** Each decision gets a wall-clock deadline of about 80% of the remaining bank,
divided by hands remaining, times a per-decision share (preflop 14%, flop 24%, discard 22%,
turn 20%, river 20%), capped at 0.5 s. A reserve for I/O overhead is kept back. All sampling loops
stop at their deadline, and below 4 ms the bot skips sampling and uses a fast heuristic. Hands that
end early leave time for later hands.

**Crash guard.** Every message handler is wrapped. Any internal error produces a legal fallback
reply (check, else call, else fold; lowest card for a discard), so a bug cannot forfeit the match.

**Evaluator.** A memoised table-driven 7-card evaluator. It matched `engine.best_hand` on
180,000 comparisons (including ties) with no mismatches.

**Preflop.** An offline equity table for all 1,755 suit-equivalent three-card hands (equity
against a random hand). When the opponent has acted, sampling estimates the shift caused by the
opponent's actions, with the table as a control variate.

**Flop and discard.** The opponent's three-card hand is sampled uniformly and weighted by the
opponent model. The opponent's keep choice is averaged exactly over its three options using the
learned follow rate. All three of your kept pairs are evaluated on the same sampled worlds. For
betting decisions, the best pair is chosen on one half of the samples and scored on the other half,
which removes the upward bias from taking the maximum of three noisy estimates.

**Turn and river.** Opponent pairs (given its revealed discard) are visited in a fixed random
order until the deadline, so a truncated pass is an unbiased subsample. Your own discarded card is
excluded from the opponent's range and from the river. Weights are cached per hand and only newly
observed actions are applied on later requests.

**Betting.** Fold, call and raise values are compared. Raise sizes include several fractions of the
pot, the minimum raise and all-in. A fold-probability model estimates how often the opponent folds
to each size. When we raise, the opponent folds its weakest hands, so equity when called is measured
against only the stronger remaining part of its range, using the same weighted samples as the
equity estimate (with a linear fallback of `0.30 x P(fold)` when no samples are available). A raise
needs both an equity threshold (or a credible bluff) and an expected-value margin over calling.

**Opponent model.** Per-leg, in memory only: preflop raise and call rates, aggression, fold rates by
street, strength seen at showdown, and how often the opponent's discard matches the default
heuristic. It resets on each `init` message, as the rules require.

## Local test results

All runs: 400 hands, local engine, Python 3.12, zero protocol notices. "v2" is the previous
version of this bot (same design, linear adverse-selection shading instead of the range-based one).

| Opponent | Result for this bot (v4) | Time used (of 60 s) |
|---|---|---|
| random_bot, seed 11 | +17,811 | 18.7 s |
| calling station, 2 seeds | +23,042 / +19,900 | about 22-23 s |
| maniac, 2 seeds | +17,316 / +13,619 (v2: +19,284 / +16,965) | about 23-24 s |
| tight bot, 12 paired seeds | +10,128 total vs v2 +3,598; better on all 12 | 2-5 s |
| v2 head-to-head, 10 seeds | +1,117 total (no measurable difference) | 25-31 s |

Notes on the evidence:
- The tight-bot result includes 4 seeds not used to choose the settings (+5,021 vs +1,624).
- Head-to-head vs v2: the 4 seeds used for selection gave +1,829; 6 fresh seeds gave -712.
  Treat it as a tie.
- Individual matches swing by about +/-1,000 chips because the bots use unseeded randomness.
- Two other changes were tried and dropped: judging opponent raises relative to the pot (no
  measurable gain) and a larger required margin before raising (mixed, about -420 over 4 seeds
  vs v2). Switches for them remain in the source (`USE_FRAC`, `RISK_MARGIN`).
- An earlier version (v1) forfeited on the time bank against the calling station.

## Known limits

- Sample sizes are small. Both bots use unseeded randomness, so the same seed can give different
  results on different runs. Treat these numbers as evidence, not proof.
- Against the maniac the bot scored a little below v2 (within observed noise, but not shown equal).
- The raise sizes, the time-share split and the 0.30 fallback constant are untuned.
- Not implemented: a river solver, opponent archetype inference, a balanced bluffing baseline,
  and modelling what your own discard reveals.
- Not yet run on Python 3.11 or with `--stage 2`. Do both before submitting.
- Rules: no state is carried between legs, and no attempt is made to identify duplicate deals
  (Rulebook 5.9, 10.3).
