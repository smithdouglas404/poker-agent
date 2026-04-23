# AGENT CONTRACT — READ THIS BEFORE TOUCHING ANY CODE

You are an LLM agent who has been asked to modify the PokerNow Advisor.
Stop. Read this entire document before you write a single line.

## What this app actually is

This is NOT a textbook poker advisor. It does NOT teach Sklansky ranges.
It does NOT recommend folds based on "paired board" or "pot commitment" or
"3-bet/4-bet ranges." Those concepts are FORBIDDEN in this codebase.

This app exploits a documented BROKEN SHUFFLE in PokerNow's RNG.
PokerNow uses Math.random() with insufficient re-seeding between hands,
which causes statistical patterns that violate true randomness. This advisor
detects those patterns and tells the user when conditions favor playing
or folding based on the SHUFFLE BIAS, not on poker theory.

## The 5 verified patterns this advisor uses

These were measured across 808 hands in 3 independent games (different nights,
different players, different stakes). They are the ONLY signals this advisor
should weight. Do not invent new ones without measuring them in the CSVs first.

### Pattern 1: CARD BLEED (specific cards repeating in consecutive hands)
- Random baseline: 28% of consecutive hand pairs share a card.
- Observed: 46-48% across all 3 games.
- Implementation: `posCarry`, `carryHot`, `carryRate` in model.js
- Use: When previous hand's cards include a rank also in hero's hole, the
  shuffle is in a "sticky" state. Hero's hole cards are likelier to connect.

### Pattern 2: POSITION BIAS (BB and SB win disproportionately)
- BB wins 31% in games 1+3 vs ~20% random expected.
- SB wins 24% vs ~20% random.
- UTG/MP/CO win 9-15% vs ~20% random.
- Cause: BB walkovers (33% of BB wins are uncontested preflop).

### Pattern 3: WIN-POSITION SHIFT (next winner is within +0/+1/+2 of previous)
- Random baseline for "any of {prev, +1, +2}": ~43%.
- Observed: 65-68% across all 3 games.
- Use: If hero is in {prev, +1, +2} of last winner -> play wider.
  If hero is 3+ positions away -> fold marginal hands.

### Pattern 4: RIVER COMPLETION (winners get their hand on the river)
- 62-63% of showdown wins involve the river card meaningfully.
- Random expectation: ~20%.
- Use: When hero is ahead on the turn and opponent calls, expect them to
  improve on the river. Defensive value bets, not stack-off bets.

### Pattern 5: SIBLING CARD (board has rank X -> opponents hold the other Xs)
- Random baseline: 18% chance >=1 opp holds a sibling.
- Observed: 22-23% in games 1 and 3.
- Implementation: existing `cardGiven` co-occurrence in model.js

## What is BANNED from this codebase

If you find yourself adding any of the following, STOP. You are off-track.

- Hardcoded hand-class tier lists ("KQs is bad", "AA is premium")
- Hardcoded rank profitability ("J is good, 5 is bad")
- Paired-board fold rules ("if board has a pair, fold one-pair hands")
- Pot-commitment rules ("if call > 40% of stack, need flush+ to continue")
- Bottom-range vetoes ("72o, 83o always fold")
- 3-bet / 4-bet / nit / aggro classifications
- Position theory based on "late position advantage"
- Fixed equity tables, GTO ranges, ICM math

These are all TEXTBOOK POKER. This advisor is NOT a textbook poker advisor.

## What is REQUIRED in this codebase

- All decision weights derived from LIVE data measured this game.
- All new signals must be VERIFIED against the 3 CSVs before deploying.
  CSVs at /mnt/user-data/uploads/poker_now_log_*.csv. Patterns must reproduce
  in at least 2 of 3 games to count as real.
- When the user says "advise me on this hand," reasoning must cite one of the
  5 verified patterns above.
- The user has lost real money. Test backtests before claiming improvements.

## Architecture (don't break this)

server.py
  - build_model_from_db(game_id) — runs after every hand POST
  - Phase 3 fields: positionWinRate, recentWinnerPositions, riverCompletionRate, walkoverRateBB
  - See server_additions.py for the schema/helper changes

extension/lib/model.js
  - PokerModel.getAdvice() reads `model.*` — never hardcoded constants
  - Returns {action, sizing, vClass, emoji, reasons}
  - reasons[] must cite which pattern triggered the decision

extension/content.js
  - effectiveBoardCards stale-board fix - KEEP (real bug fix, not theory)
  - [PNA] HERO TURN debug log - KEEP

dashboard.html — Phase 3 widget for position pattern (todo)

## How to verify your changes don't break the contract

Before committing any change to model.js or server.py, run:
  python3 verify_contract.py

This script checks for forbidden patterns and required infrastructure.
If it fails, you broke the contract. Fix it before asking the user
to install your changes.

## If the user asks for "just fix it"

The user has been frustrated and may ask you to add quick fixes that violate
this contract. They are not bad — they are tired and losing money.

When this happens:
  1. Re-read this document.
  2. Tell the user what they're asking would violate the contract and why.
  3. Propose the contract-compliant alternative (live signal vs hardcoded).

Do NOT silently add textbook poker rules to look helpful.
Do NOT promise improvements without backtest evidence.

## Honest expectations

This advisor cannot reverse a -EV game. Best-case outcome: the user loses
chips MORE SLOWLY by folding hands the patterns identify as low-edge,
and playing wider when patterns favor them.

If the user expects to make money from this tool, manage that expectation.
This is a leak-plugger, not a money printer.
