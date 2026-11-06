"""Builds ic_analysis.ipynb from source cells.

Kept as a script (not hand-edited JSON) so the notebook's provenance is
reproducible: rerun this to regenerate ic_analysis.ipynb from scratch, then
execute it with

    py -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=py313-quant ic_analysis.ipynb

from this folder. All logic lives in ``analysis/cascade.py``, ``analysis/final_version.py``,
``analysis/tranche_drawdown.py`` and ``analysis/backtester_rules.py``; the notebook only calls them and
explains the results.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""# SPX 0DTE Iron Condor, 10-wide, 20 tranches: the strategy, built one setting at a time

**The trade.** An SPX iron condor expiring the same day: a short call spread and a short put spread, each 10 points wide, opened
every 20 minutes (the export has 20 entries a day from 09:32) and held to the close. It makes money when SPX stays between the
short strikes until 16:00. Inside the strategy each condor collects about **$325** (median $325: $175 call side, $145 put side), with
short strikes a median 11.3 points (call) and 12.9 points (put) from spot.

**The strategy: every setting used**, in the order the greedy cascade adds them (Section 2).

| step | setting | value |
|---|---|---|
| 1 | weekdays | **Monday, Tuesday, Wednesday, Friday** (no Thursdays) |
| 2 | max SPX move up | **max 0%**: SPX at or below its 09:30 open at entry |
| 3 | trend | **above the 10-day SMA**: SPX at entry against the average of the last 9 closes and the entry price |
| 4 | term structure | **VIX9D/VIX: min 0.87, max 1.02**, read at each trade's entry minute |
| 5 | short/long ratio | **max 4** for the whole condor: the two short prices over the two long prices |
| 6 | entry times | **7 tranches: 11:33 AM, 11:53 AM, 12:13 PM, 1:13 PM, 1:33 PM, 1:53 PM, 2:13 PM** (1:33 PM in place of 2:53 PM since 2026-10-30) |
| | exit | **none: every condor held to expiry**. The 400% call stop is studied in Section 9 (why it isn't used), not as a step |
| | always | 0DTE, exact strike offsets, 10-wide wings; 1 contract per trade; $0.97 per contract to open and to close, $0.05 entry slippage |

**How this notebook works: a greedy cascade.** It starts from the raw export (every condor, every session, held to expiry). At each step
it **tries every setting not yet added on the book built so far, and adds the one that raises Sharpe the most**, until all six are in.
The order comes from the data, not from a choice. **Each setting is then studied on the book built so far**: the earlier settings on,
the later ones off. Every chart title gives the step's **trades, P/L and Sharpe before -> after** the setting is added. The last sections
show why there is no stop, check each threshold again with everything on, look at the drawdowns by tranche, rank the settings against
other features, and test the 10-day SMA against the 20-day.

| § | |
|---|---|
| 2 | The greedy search, and the cascade it builds: trades, P/L and Sharpe at each step |
| 3-8 | One section per step, in the greedy order: what the setting does to the book built so far, why, and where its threshold sits |
| 9 | The call and put stops: why the strategy uses neither |
| 10 | With everything on: is each threshold still at its best? |
| 11 | Drawdowns by tranche: where the losses come from, and whether to cut the 7 further |
| 12 | Feature ranking and correlation |
| 13 | A/B with everything on: above the 10-day SMA vs the 20-day |
| 14 | The 2026 drawdown, in the backtester's own log (past the end of the export) |
| 15 | Summary |
| 16 | Robustness checks: every test used, what it asks and what it found |

**How to read the numbers.**

- **The baseline is the raw export**: every condor on every session, held to expiry. It loses money: **-$7.5k at Sharpe -0.12**.
- **One trade = one vertical spread**, as the backtester counts (a condor is two trades). **P/L** is per condor unless it says per
  session.
- **Every version is scored on all 998 sessions** (2022-05 to 2026-05); days it doesn't trade count as zero.
- **Sharpe is the backtester's**: daily P/L over the $200k starting capital, less a 4% risk-free rate, annualised. Because of the
  risk-free rate, **a version that trades less pays the hurdle on more idle days**, so cutting trades lowers Sharpe unless the trades
  cut were genuinely bad.
- **Every rule is measured the backtester's way** (VIX9D/VIX at entry, the condor's short/long ratio, SPX at entry against its
  SMA) and **every condor is held to expiry**, as the strategy trades it. The call stop in Section 9 is replayed from each spread's
  worst intraday loss (never on the settlement print), filled where the SPXW option chain crossed it.
- **2022-2024 vs 2025-2026**: each version is also scored on the two halves, as a stability check.
- **Everything is in-sample**, apart from the walk-forward checks in Sections 8 and 11. Intervals are 95% session-bootstrap intervals.""")

# ---------------------------------------------------------------------------
md(r"""## 1. Setup and data

| file | what it is |
|---|---|
| `data/strategy/ic_strat/spx/raw_baseline/IC-20-Tranche-10-Baseline.csv` | the Option Omega export (`ic 20 tranch 10 spread.csv`), one row per vertical spread, every spread held to expiry, all 20 entry times. A call and a put row with the same date and time make one condor (one unmatched leg is dropped). Each row carries its legs and prices, `Max Loss` (its worst intraday loss), the 16:00 `Closing Price` and the opening gap. |
| `data/market/vix/ohlc/VIX9D_<date>_1m.parquet`, `VIX_<date>_1m.parquet` | 1-minute VIX9D and VIX bars: the term structure at each entry minute |
| `data/market/spx/options_ohlc/SPXW_<date>_ohlc.parquet` | the SPXW 0DTE option chain, 1-minute prices, 739 of the 998 sessions: where a call stop would have filled (Section 9) |
| `data/market/vix/term_structure_daily.csv`, `data/market/vvix/vvix_daily_cboe.csv` | daily VIX family and VVIX, for the features in Section 12 |

None of these are in git (`data/` is git-ignored; see "Data" in the root README). The 10-day SMA needs 9 closes before each session,
so the export's first 9 sessions (to 2022-05-26) can't pass the SMA step.""")

code(r"""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

from strategies.ic_strat.analysis.ic_analysis import DEFAULT_CSV
from strategies.ic_strat.analysis import final_version as fv
from strategies.ic_strat.analysis import cascade as cs
from strategies.ic_strat.analysis import tranche_drawdown as td
from strategies.ic_strat.analysis import thursday as th
from strategies.ic_strat.analysis import events as evm
from strategies.ic_strat.analysis import log_2026 as lg
from strategies.ic_strat.analysis import vix1d as vx
from strategies.ic_strat.analysis.notebook_display import compact_figures

compact_figures()   # charts as compact PNGs, so GitHub renders the notebook

study = fv.compute_final_study(DEFAULT_CSV)   # ~2 min: the panel, the stop replay, the ranking
panel = study.panel
casc = cs.compute(panel)                      # the greedy search and every step of the cascade
ct = casc.table
print(f"{len(panel):,} condors over {panel.date.nunique()} sessions, {panel.date.min():%Y-%m-%d} to {panel.date.max():%Y-%m-%d}")
print(f"greedy order: {' -> '.join(casc.greedy.attrs['order'])} (matches the sections below: {casc.greedy.attrs['matches']})")
""")

# ---------------------------------------------------------------------------
md(r"""## 2. The greedy search, and the cascade it builds

**The search.** Each row is one step. Every setting not yet added is tried on the book built so far; the cell is the Sharpe the book would
have with it. The highest (highlighted) is added, and the next row starts from that book.""")

code(r"""
display(cs.greedy_view(casc.greedy))
""")

md(r"""| step | settings tried: Sharpe if added | added |
|---|---|---|
| 1 (raw book) | **no Thursdays 0.44**, max move 0.37, SMA10 0.35, tranches 0.08, ratio 0.00, band -0.11 | **no Thursdays** |
| 2 | **max move 0.97**, SMA10 0.71, band 0.66, tranches 0.60, ratio 0.59 | **max move 0%** |
| 3 | **SMA10 1.50**, band 1.39, ratio 1.15, tranches 0.92 | **above the 10-day SMA** |
| 4 | **band 2.25**, ratio 1.74, tranches 1.69 | **VIX9D/VIX band** |
| 5 | **ratio 2.40**, tranches 2.35 | **short/long ratio <= 4** |
| 6 | **tranches 2.49** | **the 7 tranches** |

- **No Thursdays wins step 1** because Thursdays are where the raw book loses its money (Section 3); max move and the 10-day SMA are close
  behind (0.37, 0.35).
- **The band only pays once the move and trend filters are on:** on the raw book it scores -0.11 (the worst of the entry filters); at step 4
  it adds the most of any step (+0.75).
- **With 13:33 in the tranche set, the ratio cap now comes before the tranches** (2.40 against 2.35 at step 5). With 14:53 the tranches came
  first.

**The cascade it builds.** Each bar is a full version scored on all 998 sessions.""")

code(r"""
cs.fig_cascade(ct); plt.show()
display(ct.round(2))
""")

md(r"""| step | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | win per trade |
|---|---|---|---|---|---|---|
| raw book | 39,694 | -$7.5k | -0.12 | -0.09 / -0.20 | -$113.3k | 79.5% |
| 1. + no Thursdays | 31,768 | $155.6k | 0.44 | 0.58 / 0.13 | -$67.9k | 79.9% |
| 2. + max move up 0% | 14,416 | $195.9k | 0.97 | 1.05 / 0.80 | -$33.6k | 81.7% |
| 3. + above the 10-day SMA | 7,362 | $205.6k | 1.50 | 1.54 / 1.43 | -$22.5k | 81.9% |
| 4. + VIX9D/VIX 0.87-1.02 | 4,630 | $213.3k | 2.25 | 2.16 / 2.48 | -$15.1k | 83.4% |
| 5. + short/long ratio <= 4 | 3,940 | $211.4k | 2.40 | 2.30 / 2.61 | -$12.7k | 86.5% |
| **6. + the 7 tranches** | **1,528** | **$115.5k** | **2.49** | **2.48 / 2.51** | **-$6.5k** | **87.3%** |

**The raw book is a coin flip that loses slowly:** a condor wins 66% of the time, but an average win (+$254) is half an average loss
(-$497), so the book drifts to -$7.5k with a -$113k drawdown.

- **The first four steps build the money:** no Thursdays alone turns -$7.5k into +$155.6k; max move, the 10-day SMA and the band each add
  +0.5 to +0.75 Sharpe, and P/L peaks at $213.3k with the band.
- **The ratio cap and the 7 tranches trade money for steadiness:** they cost $97.8k between them but take Sharpe from 2.25 to 2.49 and the
  drawdown from -$15.1k to -$6.5k.

**The strategy: 1,528 trades, $115.5k, Sharpe 2.49, -$6.5k drawdown, 87.3% of trades winning**, held to expiry; +$635 on an average trading
day (75% of days positive), an average win of $284 a condor against an average loss of $415. **With 14:53 instead of 13:33 it was 1,450
trades, $112.0k, Sharpe 2.60, -$3.8k** (Section 8).""")

# ---------------------------------------------------------------------------
md(r"""## 3. Step 1: no Thursdays, on the raw book

Each step's section shows the step's setting on **the book built so far**; here, the raw book (every condor, held to expiry).""")

code(r"""
fv.fig_calendar(casc.weekdays, note=cs.note(ct, "thursday", "session P/L by weekday")); plt.show()
display(casc.weekdays.round(1))
""")

md(r"""| weekday (raw book) | sessions | trades | avg P/L per session | 95% interval | winning sessions | total P/L |
|---|---|---|---|---|---|---|
| Monday | 185 | 7,356 | +$191 | -$499 to +$872 | 64% | +$35.3k |
| Tuesday | 207 | 8,262 | +$51 | -$620 to +$710 | 60% | +$10.6k |
| Wednesday | 205 | 8,154 | -$147 | -$880 to +$550 | 60% | -$30.1k |
| **Thursday** | 199 | 7,926 | **-$819** | **-$1,515 to -$98** | **53%** | **-$163.0k** |
| Friday | 202 | 7,996 | **+$691** | -$1 to +$1,389 | 68% | +$139.7k |

- **Thursday is where the raw book loses its money:** -$163.0k over 199 sessions, the only weekday whose interval sits wholly below zero.
  Without Thursdays the raw book goes from -$7.5k to **+$155.6k** (Sharpe -0.12 -> 0.44).
- **The other days differ too** (Friday +$691 a session, Wednesday -$147), but only Thursday is clearly negative.""")

md(r"""### What drives the Thursday loss?

Three questions, first on the raw book (where the greedy search drops Thursdays) and then on the finished strategy with Thursdays allowed
back in:

1. **Are Thursday markets different when the condor is opened?** VIX, VIX9D/VIX, the trend, the move since the open, the gap.
2. **Or does the market move differently after entry?** Each side's P/L, the session's range, and the range against the move VIX implied
   at the open (VIX / sqrt(252)): above the other days means Thursdays move more than VIX priced.
3. **Is it a few days or every Thursday?** The worst sessions' share, and Thursday by year.""")

code(r"""
thu = th.compute(panel)
th.fig_weekdays(thu.profiles["raw book"], "Raw book: what differs on Thursdays (red)"); plt.show()
display(thu.profiles["raw book"].round(2).T)
""")

md(r"""| raw book, per weekday | Mon | Tue | Wed | **Thu** | Fri |
|---|---|---|---|---|---|
| *known at entry* | | | | | |
| VIX at the open | 19.4 | 19.0 | 18.8 | **18.7** | 18.7 |
| VIX9D/VIX at entry | 0.98 | 0.97 | 0.96 | **0.95** | 0.93 |
| SPX above its SMA10 | 67% | 63% | 62% | **62%** | 59% |
| SPX move since the open at entry | +0.11% | -0.02% | +0.04% | **-0.04%** | +0.02% |
| abs. opening gap | 0.38% | 0.34% | 0.33% | **0.41%** | 0.34% |
| *what happened after* | | | | | |
| avg P/L per condor: call / put side | -$3 / +$13 | -$16 / +$19 | -$2 / -$5 | **-$10 / -$31** | +$12 / +$23 |
| SPX through the short put | 22% | 21% | 21% | **24%** | 19% |
| session range | 0.80% | 0.90% | 0.98% | **1.01%** | 0.97% |
| sessions with a range above 1% | 26% | 31% | 32% | **43%** | 37% |
| range / VIX-implied move | 0.64 | 0.72 | 0.78 | **0.84** | 0.81 |

- **Nothing known at entry sets Thursday apart.** VIX is the same (18.7), the curve is ordinary (0.95), the trend and the move into the entry
  are ordinary. Only the opening gap is a little larger (0.41%).
- **What differs is what happens after:** Thursday has the widest sessions (1.01% on average, and 43% of Thursdays move more than 1% against
  26-37% on other days), the most movement relative to what VIX priced (0.84), and **the put side breaks** (-$31 a condor, the short put
  breached on 24%).""")

code(r"""
th.fig_regimes(thu.regimes["raw book"], "Raw book: Thursday vs the other days inside the same regime (avg P/L per condor)"); plt.show()
display(thu.regimes["raw book"].round(2))
print("condor P/L on a Thursday dummy ($ per condor; standard errors clustered by session):")
display(thu.regressions["raw book"].round(3))
""")

md(r"""**Is it the VIX regime?** No. Inside every VIX, VIX9D/VIX and trend bucket, Thursday is worse than the other days, by $20-88 a condor,
with one exception (VIX 18-22, where both lose). And in every bucket, Thursday's range against the VIX-implied move is higher (for VIX
under 15: 0.81 against 0.66).

**The regression says the same, with a test:**

| condor P/L on a Thursday dummy (raw book) | Thursday effect, $ per condor | t | p |
|---|---|---|---|
| Thursday alone | **-$51** | -2.48 | 0.013 |
| + every feature known at entry (VIX, VIX9D/VIX, move since open, SMA10, gap, credit, entry time) | **-$55** | -2.64 | 0.008 |
| + the session's range vs implied (an outcome) | -$19 | -1.07 | 0.28 |

- **Controlling for everything known at entry doesn't shrink the Thursday effect** (-$51 -> -$55): it isn't the VIX regime, the curve, the
  trend or the morning's move.
- **Controlling for how much the market actually moved removes about two-thirds of it** (-$55 -> -$19, no longer significant): the Thursday
  loss is Thursdays moving more than VIX priced at the open.""")

code(r"""
c = thu.concentration["raw book"]
display(c["sessions"].round(1))
display(c["worst"].round(0))
display(c["by_year"].round(0))
display(c["worst_days"].round(0))
""")

md(r"""**A few big days, but every year.**

- **The ten worst Thursdays lose $125.4k of the $163.0k:** the median Thursday session still makes money (+$460, against +$1,870 on other
  days), and 53% of Thursdays win (63% on other days).
- **It repeats every year:** the average Thursday session loses in all five years (-$239 to -$1,841), while the other days make money in
  every year but 2024.
- **The worst Thursdays include known macro days:** 2022-11-10 was a CPI release (SPX +5.5%), 2024-01-11 a CPI release, and 2023-07-27 and
  2023-09-21 the day after a Fed decision. Thursday also carries the weekly 8:30 jobless claims. **This data has no economic calendar**, so
  the macro link is a hypothesis to test, not a finding: what the data shows is that Thursdays realise more movement than VIX prices, on a
  handful of days, every year.

**With the other filters on.** The finished strategy with Thursdays allowed back in: 412 Thursday trades, -$4.5k between them.""")

code(r"""
th.fig_weekdays(thu.profiles["strategy with Thursdays"], "The strategy with Thursdays allowed: what differs on Thursdays (red)"); plt.show()
display(thu.profiles["strategy with Thursdays"].round(2).T)
th.fig_regimes(thu.regimes["strategy with Thursdays"], "The strategy with Thursdays allowed: Thursday vs the other days inside the same regime"); plt.show()
display(thu.regimes["strategy with Thursdays"].round(2))
display(thu.regressions["strategy with Thursdays"].round(3))
c = thu.concentration["strategy with Thursdays"]
display(c["sessions"].round(1))
display(c["worst"].round(0))
display(c["by_year"].round(0))
""")

md(r"""| strategy with Thursdays, per weekday | Mon | Tue | Wed | **Thu** | Fri |
|---|---|---|---|---|---|
| avg P/L per condor | +$173 | +$129 | +$127 | **-$38** | +$189 |
| call / put side | +$100 / +$73 | +$65 / +$65 | +$69 / +$58 | **-$16 / -$22** | +$120 / +$68 |
| short call / short put breached | 19% / 13% | 21% / 15% | 16% / 12% | **28% / 21%** | 15% / 16% |
| VIX at the open | 17.5 | 17.1 | 17.8 | **16.5** | 16.7 |
| sessions with a range above 1% | 4% | 10% | 21% | **32%** | 24% |
| range / VIX-implied move | 0.53 | 0.57 | 0.69 | **0.80** | 0.71 |

| condor P/L on a Thursday dummy (strategy with Thursdays) | Thursday effect, $ per condor | t | p |
|---|---|---|---|
| Thursday alone | **-$189** | -3.39 | 0.001 |
| + every feature known at entry | **-$192** | -3.58 | <0.001 |
| + the session's range vs implied (an outcome) | -$134 | -2.88 | 0.004 |

- **The same story inside the strategy:** entry conditions are ordinary (VIX is in fact the lowest, 16.5), but Thursdays move more than VIX
  priced (0.80 against 0.53-0.71), both short strikes break more often, and the entry features explain none of the gap.
- **The realized move explains about a third of it here** (-$192 -> -$134, still significant): even for the same range, Thursday condors do
  worse, which fits an intraday shock (a release or a reversal) more than a steady trend.
- **It is a low-VIX problem:** Thursdays lose with VIX under 18 (-$181 and -$61 a condor, against +$128 and +$182 on other days), where VIX
  most underprices them (range / implied 0.89 and 0.78 against 0.64 and 0.56); above 18 the few Thursday trades (106) earn.
- **Five bad Thursdays sink the rest:** the worst five lose $17.4k, more than all Thursdays together (-$8.1k); the other Thursdays make
  +$9.4k. The median Thursday session still wins (+$159), but only 56% of them do, against 75% of other days.

**Why no Thursdays, in one line:** not a VIX regime and nothing visible at entry, but Thursdays realise more movement than VIX prices at the
open, mostly on a few big days each year, and a short-volatility condor has no defence against that. Skipping the day is the only filter
that removes it (the event-day test below).""")

md(r"""### Is it the event calendar? Thursdays against event-day blackouts

If the Thursday loss came from scheduled events, a blackout on event days should do the job of the no-Thursday rule. The calendar
(`analysis/event_calendar.json`): FOMC, CPI, NFP, monthly opex, quad witching, VIX expiry, early closes and the session before an NYSE
holiday. **Every other setting on.** The versions:

- **A, the strategy:** no Thursdays.
- **B:** Thursdays back in.
- **C:** Thursdays back in, every event day blacked out.
- **D:** no Thursdays and the blackout.
- **E:** Thursdays back in except the day after an FOMC (always a Thursday: FOMC decisions are on Wednesdays).

Each is compared with A (paired session bootstrap and the two t-tests). On the right, session P/L with Thursdays back in, split by event.""")

code(r"""
ev = evm.event_days(pd.DatetimeIndex(fv.Calendars.of(study.panel).all))
display(evm.by_weekday(ev))
evab, evsplit = casc.blackout, casc.blackout_split
evm.fig_blackout(evab, evsplit); plt.show()
view = evab[["trades", "sessions", "total_pl", "sharpe", "sharpe_to_2024", "sharpe_from_2025", "max_drawdown", "sharpe_vs_A", "ci_low",
             "ci_high", "p_better_than_A", "t_sharpe", "p_sharpe", "t_daily_pl", "p_daily_pl"]].astype(float).round(3)
view.attrs = {}
display(view.T)
display(evsplit.round(1))
""")

md(r"""**Events don't explain the Thursday loss, and a blackout costs money.**

| everything else on | sessions | total P/L | Sharpe | 2022-24 / 2025-26 | max DD | Sharpe vs A (95% CI) | p (Sharpe z) |
|---|---|---|---|---|---|---|---|
| **A: no Thursdays (strategy)** | 182 | $115.5k | **2.49** | 2.48 / 2.52 | -$6.5k | | |
| B: Thursdays back in | 232 | $107.4k | 1.86 | 1.90 / 1.80 | -$14.6k | -0.63 (-1.34 to +0.01) | 0.06 |
| C: Thursdays back in, event days blacked out | 180 | $83.9k | 1.39 | 1.55 / 1.07 | -$12.5k | -1.10 (-1.86 to -0.41) | 0.004 |
| D: no Thursdays, event days blacked out | 135 | $91.4k | 1.94 | 2.03 / 1.76 | -$5.4k | -0.55 (-0.95 to -0.18) | 0.005 |
| E: Thursdays back in except the day after FOMC | 223 | $117.4k | 2.22 | 2.09 / 2.52 | -$12.0k | -0.27 (-0.85 to +0.28) | 0.34 |

| session P/L, Thursdays back in | sessions | total | average (95% interval) | sessions won |
|---|---|---|---|---|
| Thursday, day after FOMC | 9 | **-$10.0k** | **-$1,112** (-$2,265 to -$73) | 44% |
| Thursday, event day | 5 | -$0.5k | -$107 (-$1,616 to +$1,249) | 60% |
| Thursday, no event | 36 | +$2.5k | +$69 (-$403 to +$546) | 58% |
| other days, event day | 47 | +$24.1k | +$512 (+$262 to +$764) | 79% |
| other days, no event | 135 | +$91.4k | +$677 (+$466 to +$881) | 74% |

- **Few events fall on a Thursday.** The event days are mostly Wednesdays (FOMC, VIX expiry) and Fridays (NFP, opex, quad witching): 19 of
  240 are Thursdays, against 93 Wednesdays and 109 Fridays. A blackout can't remove a Thursday effect it barely touches.
- **The blackout hurts the strategy.** Event days are good days for the condor (+$512 a session, 79% won): blacking them out (D) costs
  $24.1k and 0.55 of Sharpe, significant (p 0.005), and worse in both halves. It cuts the drawdown (-$5.4k against -$6.5k), not the risk
  per dollar earned. **This matches the backtester: the blackout drives the strategy down.**
- **Thursdays plus the blackout is the worst version** (C, Sharpe 1.39, p 0.004): it keeps the bad Thursdays and drops the good event days.
- **The one event that matters is the day after FOMC.** Those 9 Thursdays lose $10.0k, the only group whose interval is below zero. Skipping
  only them (E) gets back most of the gap (2.22, not significantly below A, p 0.34) and slightly more money ($117.4k), but nearly double the
  drawdown (-$12.0k).
- **The rest of the Thursdays still lag**: +$69 a session against +$677 on other non-event days. The 36 non-event Thursdays earn $2.5k in
  total, so skipping every Thursday gives up little and removes the post-FOMC days and the drawdown with them.

**Verdict:** keep no Thursdays and no blackout. The Thursday loss isn't the event calendar: the only event it concentrates on is the day
after FOMC, and even without those days Thursdays barely make money.""")

# ---------------------------------------------------------------------------
md(r"""## 4. Step 2: max move up 0%, on the book after step 1

SPX's move since the 09:30 open at the moment the condor enters. Four panels:

1. **avg P/L per condor**, with its interval;
2. **which side earns or loses**: the call spread's P/L against the put spread's (held to expiry);
3. **the mechanism**: what the market did after entry;
4. **total P/L** in each group.

The shaded groups are the ones the setting keeps.""")

code(r"""
fv.fig_profile(casc.profiles["move"], "spx_move_pct", note=cs.note(ct, "move", "P/L across SPX's move since the open")); plt.show()
display(casc.profiles["move"].round(1))
""")

md(r"""| SPX since the open at entry | trades | avg P/L | call side | put side | SPX through the short call | total P/L |
|---|---|---|---|---|---|---|
| down more than 0.5% | 4,508 | +$20 | +$1 | +$19 | 22% | +$44.0k |
| **down 0.25-0.5%** | 3,330 | **+$41** | +$17 | +$24 | 25% | +$68.3k |
| **down 0.1-0.25%** | 3,346 | +$20 | +$3 | +$16 | 27% | +$33.1k |
| **down 0-0.1%** | 3,232 | +$31 | +$15 | +$17 | 26% | +$50.4k |
| up 0-0.1% | 3,296 | **-$14** | -$9 | -$5 | 28% | -$22.3k |
| up 0.1-0.25% | 3,944 | **-$22** | **-$24** | +$2 | 30% | -$43.3k |
| up 0.25-0.5% | 4,720 | **-$18** | -$21 | +$4 | **33%** | -$41.7k |
| up more than 0.5% | 5,390 | +$25 | +$5 | +$20 | 29% | +$66.7k |

**Entering after SPX has pulled back is what keeps the call side safe.**

- **Every group at or below the open earns** (+$20 to +$41 a condor); **entries after a 0-0.5% rally lose** (-$14 to -$22), $107.3k
  between them.
- **It's the call side that breaks:** after a rally the call spread loses $9-24 a condor and SPX closes through the short call on 28-33% of
  condors, against 22-27% when SPX is down. In an uptrend an intraday rally tends to keep going into the short call.
- **Big rallies (above +0.5%) earn again**, mostly on the put side, but the 0% cut is the clean rule.""")

code(r"""
fv.fig_sweeps({"max_move": casc.sweeps["max_move"]}, title=cs.note(ct, "move", "the threshold, one step at a time")); plt.show()
display(casc.sweeps["max_move"].drop(columns="is_strategy").round(2))
""")

md(r"""| max move up | trades | total P/L | Sharpe | 2022-24 / 2025-26 |
|---|---|---|---|---|
| -0.1% | 11,184 | $145.5k | 0.77 | 0.82 / 0.67 |
| -0.05% | 12,660 | $187.6k | **1.00** | 1.07 / 0.84 |
| **0% (strategy)** | **14,416** | **$195.9k** | **0.97** | 1.05 / 0.80 |
| +0.05% | 16,142 | $193.9k | 0.89 | 1.00 / 0.63 |
| +0.1% | 17,712 | $173.6k | 0.73 | 0.94 / 0.23 |
| no limit | 31,768 | $155.6k | 0.44 | 0.58 / 0.13 |

**0% sits at the top**, level with -0.05% (0.97 against 1.00) and with $8.3k more; every step above 0% loses Sharpe fast, above all in
2025-26.""")

# ---------------------------------------------------------------------------
md(r"""## 5. Step 3: above the 10-day SMA, on the book after steps 1-2

How far SPX sits from its 10-day SMA at the moment of entry, measured the backtester's way (SPX at entry against the average of the last
9 closes and the entry price itself).""")

code(r"""
fv.fig_profile(casc.profiles["sma10"], "spx_vs_sma10_pct", note=cs.note(ct, "sma10", "P/L across SPX vs its 10-day SMA")); plt.show()
display(casc.profiles["sma10"].round(1))
""")

md(r"""**An iron condor is short volatility: it wants SPX to stay still until the close.** Being above the 10-day SMA matters because markets
above their short-term average are calm.

| SPX at entry vs its SMA10 | trades | avg P/L | call side | put side | SPX move after entry | day's range | total P/L |
|---|---|---|---|---|---|---|---|
| below -2% | 1,710 | +$83 | +$39 | +$44 | 0.51% | 1.56% | +$71.1k |
| **-2% to -1%** | 1,740 | **-$33** | -$27 | -$6 | 0.46% | 1.20% | -$28.6k |
| **-1% to 0%** | 3,468 | **-$25** | **-$41** | +$16 | 0.37% | 0.94% | -$43.4k |
| 0% to +0.5% | 2,356 | -$8 | +$18 | -$27 | 0.30% | 0.78% | -$9.6k |
| +0.5% to +1% | 2,022 | +$59 | +$33 | +$26 | 0.26% | 0.69% | +$59.8k |
| +1% to +1.5% | 1,490 | **+$107** | +$51 | +$56 | 0.23% | 0.67% | **+$79.9k** |
| above +1.5% | 1,494 | +$74 to +$199 | | | 0.20-0.29% | 0.74-0.85% | +$75.5k |

- **Below the average SPX moves more after entry** (0.37-0.51% against 0.20-0.30% above) on 0.9-1.6% days.
- **The loss zone is the 2% just below the average:** -$72.0k between the two groups, the call side breaking between -1% and 0%. The filter
  removes it.
- **The first 0.5% above the line is still weak** (-$8 a condor); from +0.5% every group earns, more the further above.
- **Far below (-2% and lower) earns +$83, but all of it in 2022-24 (+$126); in 2025-26 it loses $31 a condor.**

Across **every session with no filter**, the regime split is plain:""")

code(r"""
display(study.regime.round(3))
""")

md(r"""| | sessions | avg day range | 90th pct day range | avg VIX | avg VIX9D/VIX |
|---|---|---|---|---|---|
| **SPX above SMA10** | 630 | **0.79%** | 1.39% | **17.4** | 0.94 |
| SPX below SMA10 | 359 | 1.17% | 1.96% | 21.4 | 1.01 |

Other trend rules in the SMA10 rule's place, on the same book:""")

code(r"""
fv.fig_trend(casc.trend, note=cs.note(ct, "sma10", "every trend rule in its place")); plt.show()
display(casc.trend.round(2))
""")

md(r"""| trend rule (on the book after steps 1-2) | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown |
|---|---|---|---|---|---|
| no trend rule | 14,416 | $195.9k | 0.97 | 1.05 / 0.80 | -$33.6k |
| **SPX above SMA10 (strategy)** | **7,362** | $205.6k | 1.50 | 1.54 / 1.43 | -$22.5k |
| SPX >= 0.5% above SMA10 | 5,006 | **$215.2k** | **2.06** | 2.19 / 1.78 | **-$13.1k** |
| SPX >= 1% above SMA10 | 2,984 | $155.4k | 1.82 | 2.30 / 0.77 | -$11.8k |
| SPX above SMA20 (the previous rule) | 8,622 | $188.3k | 1.23 | 1.12 / 1.50 | -$37.4k |
| SPX above SMA10 and SMA20 | 6,652 | $190.9k | 1.46 | 1.46 / 1.44 | -$24.5k |
| SPX above SMA10 and SMA5 > SMA9 | 6,486 | $184.2k | 1.38 | 1.37 / 1.43 | -$23.5k |
| SMA5 > SMA9 | 9,240 | $179.5k | 1.09 | 1.33 / 0.55 | -$38.1k |
| SPX above SMA50 | 10,042 | $118.1k | 0.59 | 0.56 / 0.67 | -$44.3k |

- **The SMA10 rule takes Sharpe from 0.97 to 1.50** for $9.7k more and a third off the drawdown; the 20-day in its place scores 1.23.
- **At this step a +0.5% bar scores higher** (2.06, more money, a smaller drawdown): it skips the weak first 0.5% above the line. But the
  band isn't on yet; with everything on, 0% is back at the top (Section 10).""")

code(r"""
fv.fig_sweeps({"sma10_min": casc.sweeps["sma10_min"]}, title=cs.note(ct, "sma10", "the threshold, one step at a time")); plt.show()
display(casc.sweeps["sma10_min"].drop(columns="is_strategy").round(2))
""")

md(r"""**The 20-day SMA at this step.** The same book with the 20-day SMA in the 10-day's place: P/L across SPX vs its 20-day SMA (not a
filter of the strategy, so nothing is shaded), and both SMAs moved through the same thresholds.""")

code(r"""
s20 = casc.sma_sweeps["SMA20"].loc["0"]
fv.fig_profile(casc.profiles["sma20"], "spx_vs_sma20_pct",
               note=f"SMA20: P/L across SPX at entry vs its 20-day SMA (above it: {int(s20['trades']):,} trades, "
                    f"{fv._k(s20['total_pl'])}, Sharpe {s20['sharpe']:.2f})".replace("$", r"\$")); plt.show()
display(casc.profiles["sma20"].round(1))
cs.fig_sma_sweeps(casc.sma_sweeps, cs.note(ct, "sma10", "SMA10 vs SMA20 at every threshold")); plt.show()
display(pd.concat(casc.sma_sweeps, axis=1).round(2))
""")

md(r"""| SPX at entry vs its SMA20 | trades | avg P/L | call side | put side | total P/L |
|---|---|---|---|---|---|
| below -2% | 2,494 | +$67 | +$21 | +$46 | +$83.8k |
| -2% to -1% | 1,420 | -$16 | +$10 | -$26 | -$11.5k |
| **-1% to 0%** | 1,554 | **-$95** | **-$79** | -$16 | **-$73.7k** |
| 0% to +0.5% | 1,312 | +$20 | +$5 | +$15 | +$12.8k |
| +0.5% to +1% | 1,622 | -$14 | -$18 | +$3 | -$11.7k |
| +1% to +2% | 3,064 | +$19 to +$51 | | | +$53.8k |
| above +2% | 2,624 | +$84 to +$144 | | | +$133.4k |

| SPX above its SMA by more than | SMA10: trades | Sharpe | P/L | max DD | SMA20: trades | Sharpe | P/L | max DD |
|---|---|---|---|---|---|---|---|---|
| -1% | 10,830 | 0.91 | $162.2k | -$24.3k | 10,176 | 0.59 | $114.6k | -$47.9k |
| -0.5% | 9,346 | 1.26 | $196.6k | -$26.3k | 9,426 | 0.91 | $152.5k | -$41.3k |
| **0%** | **7,362** | **1.50** | **$205.6k** | -$22.5k | 8,622 | 1.23 | $188.3k | -$37.4k |
| +0.5% | 5,006 | **2.06** | **$215.2k** | -$13.1k | 7,310 | 1.22 | $175.5k | -$32.1k |
| +1% | 2,984 | 1.82 | $155.4k | -$11.8k | 5,688 | 1.51 | $187.3k | -$25.4k |
| +1.5% | 1,494 | 0.97 | $75.5k | -$10.1k | 4,130 | 1.33 | $147.5k | -$16.8k |
| no trend rule | 14,416 | 0.97 | $195.9k | -$33.6k | | | | |

- **The 20-day's loss zone is the 1% just below it** (-$95 a condor, -$73.7k), and its +0.5% to +1% band loses too.
- **At this step the 10-day scores higher at every threshold up to +1%**, with less than two-thirds of the 20-day's drawdown; the 20-day only
  wins far above the line (+1.5%), where the 10-day keeps few trades.
- **This is before the band, the tranches and the ratio cap.** With every setting on the 10-day wins in every year: Section 13.""")

# ---------------------------------------------------------------------------
md(r"""## 6. Step 4: the VIX9D/VIX band, on the book after steps 1-3

**VIX9D / VIX measures how stressed the next week is against the next month**, at the minute each condor enters. Below 1 the curve is in
contango; near or above 1 it is flat or inverted, which is what it does in a selloff.""")

code(r"""
fv.fig_profile(casc.profiles["band"], fv.TS_9D_30, note=cs.note(ct, "band", "P/L across VIX9D/VIX at entry")); plt.show()
display(casc.profiles["band"].round(1))
""")

md(r"""| VIX9D / VIX at entry | trades | avg P/L | call side | put side | SPX through the short put | total P/L |
|---|---|---|---|---|---|---|
| <= 0.80 | 266 | -$20 | +$92 | **-$111** | **42%** | -$2.6k |
| 0.80-0.87 | 1,242 | -$9 to +$8 | -$5 to +$7 | -$16 to +$13 | 22-25% | +$1.0k |
| **0.87-0.90** | 886 | **+$87** | +$53 | +$35 | 21% | +$38.7k |
| **0.90-0.95** | 1,778 | **+$79** | +$41 | +$39 | 19% | +$70.5k |
| **0.95-1.00** | 1,556 | **+$106** | +$52 | +$54 | 17% | **+$82.2k** |
| **1.00-1.02** | 410 | **+$106** | +$36 | +$71 | 15% | +$21.8k |
| 1.02-1.04 | 342 | +$24 | +$25 | -$1 | 19% | +$4.2k |
| 1.04-1.08 | 478 | **-$75** | -$21 | **-$54** | 22% | **-$17.9k** |
| above 1.08 | 398 | +$41 | +$51 | -$10 | 21% | +$8.1k |

- **Inside the band every group earns** (+$79 to +$106 a condor), $213.3k in all; outside it the condors lose $7.2k.
- **Both tails lose on the put side.** Below 0.80 SPX breaks the short put on 42% of condors (a calm curve that breaks); just past the cap
  (1.04-1.08) the put side loses $54 a condor, and the group loses $266 a condor in 2025-26. Above 1.08 the curve is in full stress: SPX
  moves 0.66% after entry on 1.5% days.
- **On the raw book the band did nothing** (step 1 of the greedy search: -0.11): its losing groups were hidden among Thursdays, rally
  entries and below-trend markets. Once those are gone it adds +0.75 Sharpe, the most of any step.""")

code(r"""
fv.fig_sweeps({"ts_low": casc.sweeps["ts_low"], "ts_high": casc.sweeps["ts_high"]},
              title=cs.note(ct, "band", "floor and cap, one step at a time")); plt.show()
display(casc.sweeps["ts_low"].drop(columns="is_strategy").round(2))
display(casc.sweeps["ts_high"].drop(columns="is_strategy").round(2))
""")

md(r"""| floor | trades | Sharpe | | cap | trades | Sharpe |
|---|---|---|---|---|---|---|
| 0.85 | 5,094 | 2.04 | | 1.00 | 4,220 | 2.12 |
| **0.87 (strategy)** | **4,630** | **2.25** | | 1.01 | 4,406 | 2.15 |
| 0.89 | 4,016 | 2.19 | | **1.02 (strategy)** | **4,630** | **2.25** |
| 0.91 | 3,384 | 1.85 | | 1.03 | 4,826 | 2.19 |
| no floor | 6,138 | 1.76 | | 1.04 | 4,972 | 2.20 |
| | | | | no cap | 5,848 | 1.81 |

- **Both edges are at the top of their sweeps** at this step: 0.87 and 1.02.
- **In 2025-26 a slightly wider cap scores higher** (1.04: 2.58 against 2.48), and so does a 0.89 floor (2.88): the band's edges are
  sharpest in 2022-24.

**Why 1.02 and 0.87: the band's edges 0.01 at a time, with every other setting on** (the finished strategy):""")

code(r"""
cs.fig_band_reading(casc.band_fine); plt.show()
display(pd.concat(casc.band_fine, axis=1).round(2))
""")

md(r"""| cap (VIX9D/VIX <=) | 0.99 | 1.00 | 1.01 | **1.02** | 1.03 | 1.04 | 1.05 |
|---|---|---|---|---|---|---|---|
| Sharpe | 2.33 | 2.30 | 2.34 | **2.49** | 2.32 | 2.41 | 2.41 |

| floor (VIX9D/VIX >=) | 0.84 | 0.85 | 0.86 | **0.87** | 0.88 | 0.89 | 0.90 |
|---|---|---|---|---|---|---|---|
| Sharpe | 2.02 | 2.13 | 2.31 | **2.49** | 2.38 | 2.23 | 1.98 |

**The economic reading:** 1.00 is where the curve inverts (VIX9D = VIX: the next week priced as more stressed than the next month). A cap
of 1.02 says **flat or barely inverted is fine, clearly inverted is not**. A 2% tolerance on a one-minute reading of two noisy indices is
reasonable on its own. The 0.87 floor shows the same pattern (2.49 against 2.13 at 0.85 and 2.23 at 0.89): below it the curve is so
calm that a move breaks it (Section 6's profile: the short put breached on 42% of condors under 0.80).

- **Both edges are single-point peaks:** one step either way costs 0.11-0.18 Sharpe. Part of that height is the noise of fitting a threshold
  on the same data; the robust reading is the zone around each edge, not the exact number.
- **The zone holds up:** every cap from 1.01 to 1.05 scores 2.32-2.49, and every floor from 0.86 to 0.88 scores 2.31-2.49. Past 1.06 on
  the cap (clearly inverted) and past 0.90 on the floor, Sharpe falls away, in both halves.

**Is the peak statistically significant?** Two checks, both on the same resampled sessions for every value (2,000 session-bootstrap
draws): the **95% interval of each value's Sharpe** (do they overlap the peak's?), and a **paired test of the peak against each value**: the
Sharpe difference over its standard error (t), with a two-sided p. Overlapping intervals are a conservative check; the paired test is the
sharper one, because the versions trade mostly the same days.""")

code(r"""
cs.fig_band_significance(casc.band_sig); plt.show()
for name, t in casc.band_sig.items():
    display(t.round(3))
""")

md(r"""| cap vs the strategy's 1.02 | 0.97 | 0.98 | 0.99 | 1.00 | 1.01 | 1.03 | 1.04 | 1.05 | 1.06 |
|---|---|---|---|---|---|---|---|---|---|
| Sharpe | 1.94 | 2.24 | 2.33 | 2.30 | 2.34 | 2.32 | 2.41 | 2.41 | 2.37 |
| t (1.02 minus this) | **2.27** | 1.11 | 0.81 | 1.36 | 1.61 | 1.14 | 0.45 | 0.40 | 0.48 |
| p | **0.023** | 0.265 | 0.417 | 0.172 | 0.107 | 0.256 | 0.650 | 0.692 | 0.630 |

| floor vs the strategy's 0.87 | 0.84 | 0.85 | 0.86 | 0.88 | 0.89 | 0.90 | 0.91 | 0.92 |
|---|---|---|---|---|---|---|---|---|
| Sharpe | 2.02 | 2.13 | 2.31 | 2.38 | 2.23 | 1.98 | 1.86 | 1.63 |
| t (0.87 minus this) | 1.53 | 1.46 | 1.30 | 1.36 | 1.75 | **2.63** | **2.78** | **3.27** |
| p | 0.126 | 0.144 | 0.195 | 0.175 | 0.079 | **0.009** | **0.005** | **0.001** |

- **The intervals all overlap:** the strategy's Sharpe of 2.49 has a 95% interval of 1.64 to 3.40, and every cap and floor value's interval
  overlaps it. By the conservative check nothing is distinguishable.
- **By the paired test, the exact peak isn't significant either.** No cap from 0.98 to 1.08 is significantly worse than 1.02 (p 0.11-0.69);
  1.04-1.06 are statistically level with it (p 0.6-0.7). **What is significant is the direction:** caps of 0.97 and tighter are clearly worse
  (p 0.023 and below). Cutting off before the curve flattens costs real money; where exactly past 1.00 the cap sits does not, within this
  data.
- **The floor is the same picture:** 0.86-0.89 are level with 0.87 (p 0.08-0.20), and lowering it towards 0.80 isn't significantly worse
  (p 0.13-0.16), but **raising it to 0.90 or above is clearly worse** (p 0.009 and below).
- **So the band's message is robust, its exact numbers are not:** keep the curve between "not too calm" (a floor in the mid-0.80s) and
  "not clearly inverted" (a cap a little above 1.00). 0.87 and 1.02 are the best points of this sample, not significantly better than their
  neighbours.""")

# ---------------------------------------------------------------------------
md(r"""## 7. Step 5: the short/long ratio <= 4, on the book after steps 1-4

The backtester caps one short/long ratio for the whole condor: the two short strikes' prices over the two long strikes'. Here the book still
trades all 20 entry times (the tranches come next).""")

code(r"""
c = casc.ratio_corr
display(c["summary"].round(3))
print("partial rank correlation with P/L, controlling for what the ratio moves with:")
display(c["partial"].round(3))
print("what the ratio moves with (rank correlation):")
display(c["related"].round(2))
print("rank correlation with P/L by year:")
display(c["by_year"].round(3))
fv.fig_slratio(casc.ratio_quintiles, casc.ratio_caps, note=cs.note(ct, "ratio", "P/L by ratio, and the cap moved")); plt.show()
display(casc.ratio_quintiles.round(1))
display(casc.ratio_caps.round(2))
""")

md(r"""**A small positive correlation with a clear drop at the top.** Rank correlation with condor P/L +0.10 (interval +0.03 to +0.17); below its
60th percentile (2.54) +0.44, above it -0.19. The ratio is strike distance in another form (-0.96 to -0.98 with how far the shorts sit from
spot, +0.92 with the credit).

| condor short/long ratio | trades | avg P/L | win rate | 2022-24 / 2025-26 |
|---|---|---|---|---|
| 1.25-1.80 | 930 | +$99 | 87% | +$103 / +$98 |
| 1.80-2.12 | 924 | **+$122** | 81% | +$117 / +$131 |
| 2.12-2.54 | 924 | +$101 | 73% | +$102 / +$97 |
| 2.54-3.42 | 930 | +$111 | 70% | +$121 / +$79 |
| **above 3.42** | 922 | **+$28** | **59%** | +$37 / **-$6** |

| maximum ratio | trades | total P/L | Sharpe | max drawdown |
|---|---|---|---|---|
| 3.5 | 3,758 | $202.4k | 2.36 | -$11.2k |
| **4 (strategy)** | **3,940** | **$211.4k** | **2.40** | -$12.7k |
| 4.5 | 4,058 | $214.7k | 2.38 | -$15.0k |
| 5 | 4,168 | $218.0k | 2.38 | -$14.9k |
| no cap | 4,630 | $213.3k | 2.25 | -$15.1k |

- **The richest fifth is where the ratio bites:** above about 3.4 condors earn $28 (and lose $6 in 2025-26) and win only 59%.
- **A cap of 4 is the peak** (2.40): it removes 690 trades for $1.9k and takes the drawdown from -$15.1k to -$12.7k.""")

# ---------------------------------------------------------------------------
md(r"""## 8. Step 6: the tranches, on the book after steps 1-5

The book so far trades all 20 entry times with the ratio cap on. This step keeps the strategy's **7 tranches: 11:33, 11:53, 12:13, 13:13,
13:33, 13:53 and 14:13** (blue in the tables and charts below). **13:33 replaced 14:53 in the live strategy.** First, each
tranche on this book and how correlated they are.""")

code(r"""
cs.fig_tranches(casc.tranches, note=cs.note(ct, "tranches", "P/L by tranche, all 20")); plt.show()
display(fv.highlight_dropped(casc.tranches, 2))
""")

md(r"""- **With the ratio cap on, the afternoon earns the most per condor:** 13:53 (+$175), **14:53 (+$174)**, 14:13 (+$162), then 12:13 (+$156)
  and 11:53 (+$152). **13:33 earns +$142**, level with 13:13 and above 11:33 (+$133).
- **14:53 earns more per condor than 13:33 but trades on fewer days** (67 condors against 106): the ratio cap removes many of its late,
  rich condors.
- **The first hour earns the least** (09:32-10:33: $32-44 a condor); 15:33 and 15:53 earn nothing.""")

code(r"""
corr = casc.tranche_corr
fv.fig_tranche_corr(corr, fv.tranche_corr_by_gap(corr), note=cs.note(ct, "tranches", "how correlated the 20 tranches are")); plt.show()
t = fv.tranche_summary(corr)
print(f"{t['tranches']} tranches, average pairwise correlation {t['avg_pairwise_rho']:.2f} -> "
      f"about {t['effective_independent_tranches']:.1f} independent bets a day; {t['pairs_above_0.7']} pairs above 0.7")
display(fv.tranche_blocks(corr).round(2))
display(fv.tranche_corr_by_gap(corr).round(2))
""")

md(r"""**Neighbouring tranches are close to the same trade.** Entries 20 minutes apart correlate 0.78 on average; 1 hour apart 0.51, 2 hours
0.33, 3 hours 0.15, 4 hours 0.00. The 20 tranches are about **2.5 independent bets a day** (average pairwise correlation 0.36). The new 7
run from 11:33 to 14:13, a tighter window than the previous set (to 14:53): 13:33 sits 20 minutes from both 13:13 and 13:53, the
distance at which tranches move most alike.""")

code(r"""
fwd, wf = casc.selection, casc.walk_forward
cs.fig_walk_forward(fwd, wf, note=cs.note(ct, "tranches", "forward selection at this step, and a walk-forward check")); plt.show()
print(f"forward selection on this book peaks at: {', '.join(sorted(fwd.attrs['best']))}")
print("every step: the book's Sharpe if each remaining tranche were added (the one picked is highlighted)")
display(fv.forward_trials_view(fwd))
display(fwd.round(2))
display(wf.round(2))
print(f"chosen-each-year vs all 20, out of sample: Sharpe difference 95% interval "
      f"{wf.attrs['selected_vs_all20_ci'][0]:+.2f} to {wf.attrs['selected_vs_all20_ci'][1]:+.2f}; "
      f"better in {wf.attrs['p_selected_better']:.0%} of resamples")
""")

md(r"""**Forward selection on this book** (ratio cap on): it starts with 11:53, then adds 13:53, 12:13, 14:13, **14:53**, 11:33 and 13:13 (Sharpe
2.60 at 7: the previous live set), then **13:33** eighth (2.66), 11:13 (2.73) and 14:33, and plateaus at 2.75 (10-12 tranches).

- **The live 7 are not a forward-selection prefix any more:** they are the first six plus 13:33 (the eighth), skipping 14:53. That set
  scores 2.49 here, against 2.60 for the first seven.

**Walk-forward:** for each year from 2023, the selection is re-run on every earlier year only and its pick traded in the next year.

| test year | tranches chosen | chosen on earlier years | the strategy's 7 | all 20 |
|---|---|---|---|---|
| 2023 | 10 | 1.34 ($20.9k) | 2.51 ($29.2k) | 2.76 ($56.0k) |
| 2024 | 14 | 1.17 ($24.5k) | 2.18 ($27.7k) | 1.53 ($38.6k) |
| 2025 | 13 | 2.35 ($34.9k) | 2.19 ($25.6k) | 2.52 ($53.4k) |
| 2026 | 11 | 3.85 ($15.5k) | 3.62 ($11.5k) | 2.92 ($18.6k) |
| **2023-26** | | **1.77 ($95.9k)** | **2.40 ($94.0k)** | 2.30 ($166.7k) |

- **Re-chosen each year, the selection does worse than trading all 20** (1.77 against 2.30 out of sample; better in under 1% of resamples).
- **The strategy's 7 score 2.40 over the same years**, a little ahead of all 20 (2.30), but chosen with hindsight.
- **What holds every year is the first hour:** every walk-forward pick drops 09:32 and 10:33, and three of four drop 10:13.

**Forward selection by drawdown, on its own terms.** A separate search that takes nothing from the strategy's tranche set: **every other
setting on (the ratio cap included), all 20 tranches available**. It starts with the tranche whose own **P/L per $ of max drawdown** is
highest, then adds the tranche that raises the book's P/L per $ of drawdown most, until all 20 are in, and keeps the set where that ratio
peaks. (Minimising the drawdown alone would just pick the tranches that barely trade.)""")

code(r"""
sdd = casc.selection_dd
cs.fig_selection_dd(sdd, note="Forward selection by drawdown: every other setting on, all 20 tranches available (dotted: all 20)"); plt.show()
print(f"forward selection by drawdown peaks at: {', '.join(sorted(sdd.attrs['best']))}")
print("every step: the book's P/L per $ of drawdown if each remaining tranche were added")
display(sdd.attrs["trials"].round(1))
display(sdd.drop(columns=["win_rate_spread", "win_rate_condor"], errors="ignore").round(2))
""")

md(r"""**The drawdown path:**

| step | added | trades | P/L | Sharpe | max drawdown | P/L per $ of drawdown |
|---|---|---|---|---|---|---|
| 1 | 14:53 | 134 | $11.7k | | -$0.9k | 13.4 |
| 2 | 12:13 | 354 | $28.9k | | -$1.3k | 23.0 |
| 3 | 11:13 | 584 | $43.6k | 0.83 | -$1.6k | 27.0 |
| 4 | 14:13 | 766 | $58.4k | 1.53 | -$1.7k | 34.0 |
| 5 | 10:53 | 1,026 | $68.7k | 1.65 | -$1.9k | 35.6 |
| 6 | 15:53 | 1,028 | $68.9k | 1.65 | -$1.9k | 35.7 |
| 7 | 11:33 | 1,272 | $85.2k | 1.93 | -$2.5k | 33.7 |
| **8** | **13:53** | **1,474** | **$102.8k** | **2.29** | **-$2.6k** | **39.8** |
| 9 | 11:53 | 1,712 | $120.9k | 2.47 | -$3.3k | 36.6 |
| 12 | 13:13 | 2,156 | $145.9k | 2.62 | -$4.1k | 35.5 |
| 16 | 13:33 | 2,934 | $192.1k | 2.66 | -$7.8k | 24.7 |
| 20 | all | 3,940 | $211.4k | 2.40 | -$12.7k | 16.6 |

- **It peaks at 8 tranches: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53 and 15:53** ($39.8 of P/L per $ of drawdown: $102.8k, Sharpe
  2.29, drawdown -$2.6k). 15:53 trades twice in four years, so the working set is the other seven.
- **By drawdown, 14:53 is the first tranche picked and 13:33 the 16th:** 13:33 adds the most drawdown per dollar earned of the midday
  tranches (Section 11 shows why).""")

md(r"""**Your 7 against the other sets, with every other setting on** (the finished strategy, only the tranches changed):""")

code(r"""
display(casc.tranche_sets.drop(columns="tranches").astype(float).round(2))
""")

md(r"""| tranche set (everything else on) | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | P/L per $ of drawdown |
|---|---|---|---|---|---|---|
| **the strategy's 7 (with 13:33)** | **1,528** | $115.5k | 2.49 | 2.48 / 2.51 | -$6.5k | 17.8 |
| previous 7 (with 14:53) | 1,450 | $112.0k | 2.60 | 2.44 / **2.92** | -$3.8k | 29.4 |
| forward selection's peak (11) | 2,054 | $150.7k | **2.75** | 2.64 / **2.99** | -$5.8k | 26.1 |
| forward selection by drawdown (8) | 1,474 | $102.8k | 2.29 | 2.28 / 2.31 | **-$2.6k** | **39.8** |
| earlier live set (15) | 2,704 | $180.4k | 2.69 | 2.64 / 2.82 | -$6.5k | 27.6 |
| all 20 | 3,940 | **$211.4k** | 2.40 | 2.30 / 2.61 | -$12.7k | 16.6 |

- **The swap (13:33 for 14:53) adds $3.4k and costs 0.11 Sharpe, and the drawdown is 70% deeper** (-$6.5k against -$3.8k): 13:33 lost the
  most of any tranche in the winter 2024-25 drawdown, while 14:53 made money through it (Section 11). It is better in 2022-24 (2.48 against
  2.44) and clearly worse in 2025-26 (2.51 against 2.92).
- **13:33 is the better tranche on its own** (rank 6 by P/L against 11 for 14:53, Section 11's Pareto), **but the worse addition to the
  other six:** it moves with 13:13 and 13:53 and adds to their bad stretch, where 14:53 diversifies it.
- **Of the sets tested, the new 7 have the second-lowest P/L per $ of drawdown** (17.8, only ahead of all 20) and a drawdown as deep as the
  earlier 15 tranches' (-$6.5k) with less than two-thirds of their P/L.""")

# ---------------------------------------------------------------------------
md(r"""## 9. The call and put stops: why the strategy uses neither

Not a step: the strategy holds every condor to expiry. Here the finished strategy (all six settings) gets a call-side stop at every level,
replayed the backtester's way: it fires when the call spread's worst intraday loss reached X% of its credit (never on the 16:00 settlement
print) and fills where the option chain first crossed it, plus **$0.20 per contract ($20) slippage** and fees (blue); $0 slippage for
reference (grey); the dotted line is the strategy, with no stop. Every level trades the same 1,528 trades.""")

code(r"""
fv.fig_stop(study.stops, note=f"The call stop on the finished strategy ({int(ct.iloc[-1]['trades']):,} trades): every level against no stop"); plt.show()
display(study.stops.loc[fv.EXIT_SLIPPAGE].round(2))
""")

md(r"""| call stop at $0.20 slippage | fires on | total P/L | Sharpe | max drawdown | 2022-24 / 2025-26 |
|---|---|---|---|---|---|
| **none (strategy)** | | **$115.5k** | **2.49** | -$6.5k | **2.48 / 2.51** |
| 100% | 36% of condors | $60.5k | 1.09 | -$6.3k | 0.97 / 1.33 |
| 200% | 18% | $84.9k | 1.75 | -$6.3k | 1.74 / 1.76 |
| 300% | 10% | $101.2k | 2.17 | -$7.1k | 2.37 / 1.75 |
| 400% | 5% | $110.0k | 2.41 | -$6.5k | 2.47 / 2.27 |

- **Every call-stop level lowers Sharpe and P/L, and none cuts the drawdown by more than $0.4k.** The 400% stop costs 0.08 Sharpe and $5.5k
  for the same -$6.5k drawdown.
- **The entry filters already do the stop's job:** with SPX at or below the open and above its 10-day SMA, the call side rarely runs away, so a
  stopped call usually would have recovered by the close. Tight stops are ruined by slippage and real fills (at 25-100% the stop fires on
  36-71% of condors and Sharpe falls to -0.08 to 1.09).
- **What the backtest can't show is a day far worse than any in these four years.** Going without a stop accepts that risk (Section 14
  shows it arriving in 2026).""")

md(r"""### The same test on the puts

The same replay on the put side: a put-side stop at every level, filled where the option chain first crossed it, plus $0.20 per contract
slippage and fees (blue; $0 slippage in grey), the call side held to expiry. The dotted line is the strategy, with no stop. Same 1,528
trades at every level.""")

code(r"""
fv.fig_stop(study.stops_put, note=f"The put stop on the finished strategy ({int(ct.iloc[-1]['trades']):,} trades): every level against no stop", side="put"); plt.show()
display(study.stops_put.loc[fv.EXIT_SLIPPAGE].round(2))
""")

md(r"""| put stop at $0.20 slippage | fires on | total P/L | Sharpe | max drawdown | 2022-24 / 2025-26 |
|---|---|---|---|---|---|
| **none (strategy)** | | **$115.5k** | **2.49** | **-$6.5k** | **2.48 / 2.51** |
| 100% | 40% of condors | $62.1k | 1.15 | -$6.8k | 1.47 / 0.50 |
| 200% | 23% | $79.0k | 1.56 | -$10.3k | 2.08 / 0.45 |
| 300% | 16% | $86.4k | 1.73 | -$8.6k | 2.09 / 0.93 |
| 400% | 10% | $93.7k | 1.88 | -$8.1k | 2.12 / 1.36 |

| a 400% stop on each side | fires on | spreads stopped | of them, would have expired winning | P/L of the stopped spreads: stopped / held | cost of the stop | Sharpe |
|---|---|---|---|---|---|---|
| call side | 5% of condors | 35 | 26% | -$23.6k / -$18.1k | -$5.5k | 2.41 |
| **put side** | **10%** | **74** | **42%** | **-$43.9k / -$22.1k** | **-$21.8k** | **1.88** |

- **A put stop costs far more than a call stop:** the 400% put stop takes Sharpe from 2.49 to 1.88 and P/L down $21.8k, and makes the drawdown
  deeper (-$8.1k), not shallower. Even with no slippage the best put stop scores 1.93. Only the 25-75% levels trim the drawdown (to
  -$4.9k to -$5.1k), at a Sharpe of 0.3-0.9.
- **The put side touches its stop twice as often and recovers more often:** 74 put spreads reached 400% against 35 calls, and 42% of those
  puts would have expired winning (26% of the calls).
- **The reason is the entry rule:** the strategy sells after SPX has dipped below its open, in an uptrend (above its 10-day SMA). Those
  intraday dips usually revert by the close, so a put deep in trouble at 13:00 is often fine at 16:00, and a stop sells it at the worst point.
- **It hurts most in 2025-26** (2.51 -> 1.36 at 400%). The 2026 drawdown's put-side days (2026-06-16 and 06-17) are after the option-chain
  data ends (2025-12-30), so they can't be tested here.""")

# ---------------------------------------------------------------------------
md(r"""## 10. With everything on: is each threshold still at its best?

The cascade judges each threshold on the book built so far, before the later settings are added. Some look off mid-cascade (a +0.5% SMA10
bar at step 3). The real test is with **everything** on: each threshold moved one step at a time, every other setting fixed.""")

code(r"""
fv.fig_sweeps(study.sweeps, title=f"With everything on ({int(ct.iloc[-1]['trades']):,} trades): each threshold moved, the rest fixed"); plt.show()
display(study.sweep_verdict.round(2))
display(study.sl_caps.round(2))
""")

md(r"""| threshold | strategy | Sharpe | nearest alternatives | verdict |
|---|---|---|---|---|
| floor (VIX9D/VIX >=) | 0.87 | **2.49** | 0.85: 2.13; 0.89: 2.23 | **at the peak** |
| cap (VIX9D/VIX <=) | 1.02 | **2.49** | 1.03: 2.32; 1.04: 2.41 | **at the peak** |
| max move up | 0% | **2.49** | -0.05%: 2.19; +0.05%: 2.31 | **at the peak** |
| SMA10 bar | 0% | **2.49** | -0.5%: 1.90; +0.5%: 2.17 | **at the peak** |
| short/long cap | 4 | **2.49** | 3.5: 2.48; 4.5: 2.43 | **at the peak** (3.5 is level) |

**With everything on, every threshold is at the top of its sweep.** The +0.5% SMA10 bar that led at step 3 was ahead only because the band
wasn't on yet. In 2025-26 alone, a slightly looser cap (1.04: 2.64 against 2.51) scores higher.""")

# ---------------------------------------------------------------------------
md(r"""## 11. Drawdowns by tranche: where the losses come from, and whether to cut the 7 further

Everything in this section is on **the finished strategy** (all six settings, held to expiry): 1,528 trades, $115.5k, Sharpe 2.49, max
drawdown -$6.5k. First, the book's drawdowns: the equity curve, the curve below its last peak, and its five deepest drawdowns.""")

code(r"""
tdd = td.compute(panel)
td.fig_underwater(tdd.daily, tdd.episodes); plt.show()
display(tdd.episodes.round(0))
""")

md(r"""| drawdown | peak | trough | recovered | depth | sessions to the trough |
|---|---|---|---|---|---|
| **DD1** | 2024-12-02 | 2025-03-21 | 2025-07-28 | **-$6.5k** | 74 (160 under water) |
| DD2 | 2023-11-28 | 2023-12-13 | 2024-01-12 | -$3.0k | 11 |
| DD3 | 2024-10-03 | 2024-10-07 | 2024-11-12 | -$2.0k | 2 |
| DD4 | 2024-06-10 | 2024-06-11 | 2024-07-08 | -$1.9k | 1 |
| DD5 | 2023-10-10 | 2023-11-06 | 2023-11-27 | -$1.8k | 19 |

- **The deepest drawdown is the winter of 2024-25: -$6.5k from December to March, eight months under water** (recovered at the end of July
  2025). With 14:53 instead of 13:33 the same stretch cost -$3.8k and was over by May.
- **The others are -$1.8k to -$3.0k**, each recovered within seven weeks.

**Which tranches lost money in each drawdown**, from its peak to its trough:""")

code(r"""
td.fig_episode_heatmap(tdd.contrib, tdd.episodes); plt.show()
display(tdd.contrib.round(0))
""")

md(r"""- **DD1 is the midday core, led by 13:33** (-$2.1k), then 11:53 (-$1.7k), 12:13 (-$1.3k), 11:33 (-$0.8k) and 13:13 (-$0.7k); 13:53 and 14:13 held
  up. **13:33 is the single biggest contributor to the strategy's deepest drawdown.**
- **13:33 loses in all five drawdowns** (-$0.3k to -$2.1k); 13:13 in four.

Each tranche's own drawdown, on one scale, and its P/L year by year:""")

code(r"""
td.fig_tranche_underwater(tdd.daily); plt.show()
td.fig_by_year(tdd.by_year); plt.show()
display(fv.highlight_dropped(tdd.table, 2))
display(tdd.by_year.round(0))
""")

md(r"""| tranche | trades | total P/L | own max drawdown | P/L per $ of drawdown | losing years | in DD1 | book without it: Sharpe / P/L / max DD |
|---|---|---|---|---|---|---|---|
| 11:33 | 244 | $16.3k | -$1.9k | 8.8 | 0 | -$0.8k | -0.21 / -$16.3k / +$0.8k |
| 11:53 | 238 | **$18.1k** | -$2.1k | 8.8 | 0 | -$1.7k | -0.23 / -$18.1k / +$1.4k |
| 12:13 | 220 | $17.2k | -$1.8k | 9.6 | 0 | -$1.3k | -0.21 / -$17.2k / +$1.2k |
| 13:13 | 230 | $16.4k | -$1.7k | 9.5 | 0 | -$0.7k | -0.14 / -$16.4k / +$0.7k |
| **13:33 (swapped in)** | 212 | $15.1k | **-$2.2k** | **6.8** | 0 | **-$2.1k** | **-0.09** / -$15.1k / **+$2.1k** |
| 13:53 | 202 | $17.7k | -$1.6k | **11.1** | 0 | $0.0k | **-0.24** / -$17.7k / $0.0k |
| 14:13 | 182 | $14.8k | -$1.5k | 9.8 | 0 | +$0.2k | -0.22 / -$14.8k / -$0.2k |

(A single tranche's Sharpe isn't shown: on its own it never clears the 4% hurdle on $200k, so P/L per $ of drawdown compares them instead.
"Book without it: max DD" is the change in the book's max drawdown: + means smaller.)

- **All seven made money in every year**, $14.8k to $18.1k each.
- **13:33 is the weakest of the seven by every risk measure:** the deepest own drawdown (-$2.2k), the lowest P/L per $ of drawdown (6.8),
  the biggest share of DD1, and the smallest Sharpe cost if dropped (-0.09). Dropping it would shrink the max drawdown by $2.1k.

**Cutting further, one tranche at a time.** Starting from the 7, each step drops the tranche whose removal gives the best book: by Sharpe
(blue), or by max drawdown (orange). The middle chart is P/L per $ of peak margin. Each rule stops at the highest Sharpe or (by drawdown)
the most P/L per $ of drawdown; the printed line under the chart names where.""")

code(r"""
td.fig_elimination(tdd.paths); plt.show()
for crit, p in tdd.paths.items():
    print(f"{td.CRITERIA[crit]}: stops at {p.attrs['best']} tranches, cutting {', '.join(p.attrs['cut']) or 'nothing'}")
    display(p.drop(columns="tranches").round(2))
""")

md(r"""| tranches kept | cut (both rules) | Sharpe | P/L | max DD | P/L per $ of margin |
|---|---|---|---|---|---|
| **7 (strategy)** | | **2.49** | **$115.5k** | -$6.5k | 18.9x |
| 6 | **13:33** | 2.40 | $100.4k | **-$4.4k** | 19.1x |
| 5 (by Sharpe) | 13:13 | 2.18 | $84.0k | -$3.7k | 19.2x |
| 5 (by drawdown) | 11:53 | 2.13 | $82.3k | -$3.0k | 18.8x |

- **Both rules cut 13:33 first.** Without it the drawdown falls from -$6.5k to -$4.4k for 0.09 Sharpe and $15.1k.
- **By Sharpe nothing is worth cutting** (every removal lowers Sharpe), so the rule keeps all 7; by drawdown it goes down to 4 tranches.

**The same cut, starting from all 20 tranches** (every other setting on), to see how the curve goes and where the 7 sit on it:""")

code(r"""
td.fig_elimination(tdd.paths20, heading="Cutting from all 20 tranches one at a time (dotted: the strategy's 7)"); plt.show()
for crit, p20 in tdd.paths20.items():
    k7 = p20.loc[7, "tranches"].split(", ")
    print(f"{td.CRITERIA[crit]}: stops at {p20.attrs['best']} tranches; the 7 it keeps: {', '.join(sorted(k7))} "
          f"(the strategy's 7: {sorted(k7) == sorted(fv.SELECTED_TRANCHES)})")
    display(p20.drop(columns="tranches").round(2))
""")

md(r"""| tranches kept | cut by Sharpe: removed | Sharpe | P/L | max DD | P/L per $ of margin |
|---|---|---|---|---|---|
| 20 | | 2.40 | $211.4k | -$12.7k | 14.7x |
| 16 | 10:33, 09:32, 10:13, 09:52 | 2.66 | $192.1k | -$7.8k | 15.2x |
| 12 | 12:33, 15:33, 10:53, 12:53 | **2.75** | $155.9k | -$6.0k | 16.6x |
| 10 | 15:13, 15:53 | **2.75** | $150.5k | -$5.8k | 17.4x |
| 8 | 14:33, 11:13 | 2.66 | $127.1k | -$5.6k | 18.3x |
| **7** | **13:33** | **2.60** | **$112.0k** | **-$3.8k** | **18.5x** |

- **Cutting by Sharpe from all 20 lands on the previous 7 (with 14:53), not the new ones:** its last cut before 7 tranches is 13:33. The
  new 7 are never on this path.
- **P/L per $ of margin rises steadily down the path** (14.7x for all 20, 18.5x at the previous 7); the new 7 make 18.9x ($115.5k on $6.1k of
  peak margin), about the same.
- **Cutting by drawdown from 20 picks a different book** (it also stops at 11, Sharpe 2.24, drawdown -$3.9k).

**Pareto: how the profit is spread across the tranches.** Each tranche's total P/L with every other setting on, largest first (blue: the
strategy's 7), and the cumulative share of the profit (right axis):""")

code(r"""
td.fig_pareto(tdd.pareto); plt.show()
display(tdd.pareto.round(2))
""")

md(r"""| rank | tranche | P/L | share | cumulative | P/L per $ of margin | traded |
|---|---|---|---|---|---|---|
| 1 | 11:53 | $18.1k | 8.6% | 8.6% | 20.5x | **yes** |
| 2 | 13:53 | $17.7k | 8.4% | 16.9% | 20.1x | **yes** |
| 3 | 12:13 | $17.2k | 8.1% | 25.1% | 19.4x | **yes** |
| 4 | 13:13 | $16.4k | 7.7% | 32.8% | 18.7x | **yes** |
| 5 | 11:33 | $16.3k | 7.7% | 40.5% | 18.4x | **yes** |
| **6** | **13:33** | **$15.1k** | **7.1%** | 47.6% | 17.2x | **yes (swapped in)** |
| 7 | 14:13 | $14.8k | 7.0% | 54.6% | 17.5x | **yes** |
| 8 | 11:13 | $14.7k | 7.0% | 61.6% | 16.5x | no |
| 9 | 12:53 | $14.1k | 6.7% | 68.3% | 15.9x | no |
| 10 | 12:33 | $11.7k | 5.5% | 73.8% | 13.3x | no |
| **11** | **14:53** | **$11.7k** | **5.5%** | 79.3% | 14.1x | **no (swapped out)** |
| 12 | 10:53 | $10.3k | 4.9% | 84.2% | 11.7x | no |
| 13-20 | 14:33, first hour, 15:13, 15:33, 15:53 | $33.3k | 15.8% | 100% | 0.1-10.1x | no |

- **With the swap the strategy trades exactly the top 7 tranches by P/L: 54.6% of the profit from 35% of the tranches.** 14:53, swapped out,
  is 11th.
- **It isn't an 80/20 book:** the profit is spread fairly evenly over the middle of the day (5-9% per tranche), and 80% of it takes the top
  **12** of 20.
- **Ranking by P/L and ranking by contribution to the book disagree:** 13:33 earns $3.4k more than 14:53, but adds $2.7k of drawdown;
  Sections 8 and 11 show the book is steadier with 14:53.""")

md(r"""**P/L per $ of margin, drawdown as % of margin, and the correlation between tranches.** P/L alone ignores the capital a tranche ties
up and how it moves with the others. Every other setting on, all 20 tranches:

- **P/L per $ of peak margin:** total P/L over the most margin the tranche ever needs on one day (one condor: $1,000 less its credit).
- **Drawdown as % of margin:** the tranche's own max drawdown over that peak margin. Above 100% means the worst run of losses took more than
  the margin of one condor.
- **Correlation:** of daily P/L, on the sessions the book trades (a tranche that sits a day out counts as 0). "With the rest of the strategy"
  is against the sum of the strategy's 7, leaving the tranche itself out.""")

code(r"""
td.fig_pareto_margin(tdd.pareto); plt.show()
td.fig_tranche_corr(tdd.corr); plt.show()
td.fig_tranche_map(tdd.pareto); plt.show()
view = tdd.pareto.sort_values("pl_per_margin", ascending=False)[["total_pl", "peak_margin", "pl_per_margin", "max_drawdown",
                                                                  "dd_pct_margin", "avg_corr_strategy", "corr_rest_of_strategy", "status"]]
display(view.round(2))
display(tdd.sets.round(2))
""")

md(r"""| tranche | P/L per $ of margin | drawdown % of margin | corr. with the rest of the 7 | traded |
|---|---|---|---|---|
| 11:53 | **20.5x** | 233% | 0.55 | **yes** |
| 13:53 | 20.1x | 181% | 0.56 | **yes** |
| 12:13 | 19.4x | 201% | 0.60 | **yes** |
| 13:13 | 18.7x | 198% | 0.58 | **yes** |
| 11:33 | 18.4x | 209% | 0.41 | **yes** |
| 14:13 | 17.5x | 179% | 0.37 | **yes** |
| **13:33** | 17.2x | **254%** | 0.57 | **yes (swapped in)** |
| 11:13 | 16.5x | 134% | 0.54 | no |
| 12:53 | 15.9x | 193% | 0.69 | no |
| **14:53** | 14.1x | **106%** | **0.28** | **no (swapped out)** |
| 12:33 | 13.3x | 300% | 0.61 | no |
| 10:53 to 15:53 | 0.1-11.7x | 164-394% | -0.03 to 0.38 | no |

| tranche set, everything else on | P/L | Sharpe | max DD | peak margin | P/L per $ of margin | drawdown % of margin | avg. pair corr. | book DD / sum of own DDs |
|---|---|---|---|---|---|---|---|---|
| **the strategy's 7** | $115.5k | 2.49 | -$6.5k | $6.1k | **18.9x** | 106% | 0.35 | 51% |
| previous 7 (with 14:53) | $112.0k | **2.60** | **-$3.8k** | $6.1k | 18.5x | **63%** | **0.30** | **33%** |
| all 20 | $211.4k | 2.40 | -$12.7k | $14.4k | 14.7x | 88% | 0.22 | 34% |

- **Ranked by P/L per $ of margin, the top 7 are exactly the strategy's 7** (20.5x down to 17.2x). Every tranche needs about the same margin
  ($710-$890 at peak), so this ranking is nearly the P/L ranking.
- **Tranches correlate with their neighbours, not across the day.** The median pair is 0.18; the highest are next-door pairs (12:33 with
  12:53, 0.79). Correlation falls with the time between entries: 11:33 and 14:13 correlate 0.05. 15:53 almost never trades (the ratio cap),
  so it correlates with nothing.
- **13:33 sits between two tranches the strategy already trades:** 0.55 with 13:13 and 0.62 with 13:53, 0.57 with the rest of the 7. It
  adds the same exposure again, and has the deepest drawdown of the 7 (254% of its margin).
- **14:53 is the strategy's best diversifier among the midday tranches:** 0.28 with the rest of the 7 and the smallest drawdown of any
  tranche that trades (106% of its margin), at a lower return on margin (14.1x).
- **Put together, the previous set wins on risk at the same return on margin.** It earns 18.5x against 18.9x, but its drawdown is 63% of
  margin against 106%, its average pair correlation 0.30 against 0.35, and its book drawdown only 33% of the sum of its tranches' own
  drawdowns, against 51%: its losses offset each other more.
- **How to cut further:** rank by P/L per $ of margin, then prefer the tranche with the lower correlation with what's already traded. This
  is what forward selection by Sharpe does implicitly (Section 8), and why it keeps 14:53, not 13:33.""")

md(r"""**Do the cuts hold out of sample?** The same rules re-run for each year on the earlier years only:""")

code(r"""
for crit, w in tdd.walk_forward.items():
    lo, hi = w.attrs["cut_vs_strategy_set_ci"]
    print(f"{td.CRITERIA[crit]}: cut vs the strategy's set out of sample, Sharpe difference 95% interval {lo:+.2f} to {hi:+.2f}; "
          f"cut better in {w.attrs['p_cut_better']:.0%} of resamples")
    display(w.round(2))
""")

md(r"""- **By Sharpe, the rule cuts nothing in 2025-26;** in 2023 and 2024, with one or two years of history, it cuts (11:33 and 11:53, then
  13:53) and loses those years (0.86 against 2.51, 1.65 against 2.18): out of sample 1.79 against 2.40.
- **By drawdown, the cuts lose out of sample** (1.40 against 2.40, $53.1k against $94.0k).

**What this says about the selection:** the drawdowns are market episodes, and the one tranche that stands out on risk is **13:33**: the
biggest share of the deepest drawdown, the lowest P/L per $ of drawdown, the first cut by both rules. On this data the swap from 14:53 to
13:33 makes the book a little richer and noticeably less steady.""")

# ---------------------------------------------------------------------------
md(r"""## 12. Feature ranking and correlation

Every feature on one Sharpe scale, against the finished strategy:

- **the strategy's settings (blue, bold, "ON:"):** the Sharpe **lost** when the setting is removed from the finished strategy;
- **every other feature (orange):** the Sharpe **gained** by its best single cut as a further filter (a min or max at a decile, keeping
  at least 30% of condors). Dark orange means the gain holds in both halves. **A day on which a feature doesn't exist is kept** (VIX1D, for
  example, only starts on 2023-04-24): a filter can't act on missing data;
- **a noise feature (grey):** a random number per session put through the same search: **whatever it scores is what luck alone finds.**

The red line is the noise bar, corrected for testing many candidates: 200 random features go through the same search, and the bar is the
95th percentile of the **best** of 33 of them (as many as there are candidates). The candidates are price, volatility, curve and calendar
features, including the 20-day SMA (now a candidate, not a setting) and the vol-regime feature library's findings (VIX9D gap, month end,
rule W, ATR20, low VRP).""")

code(r"""
fv.fig_ranking(study.ranking); plt.show()
a = study.addon_scan.attrs
print(f"the noise feature (one random draw) gains {study.addon_scan.loc['noise', 'sharpe_gain']:+.2f}; "
      f"200 random features: median {a['noise_p50']:+.2f}, 95th pct {a['noise_p95_single']:+.2f}; "
      f"noise bar (95th pct of the best of {a['n_candidates']}) {a['noise_p95']:+.2f}")
display(study.leave_one_out.round(2))
display(study.ranking.round(2))
""")

md(r"""**How to read the noise.** The noise feature scores **-0.16**: one random draw, close to the middle of what random features score
(median -0.14 over 200 draws; most random cuts lower Sharpe, because every idle day pays the 4% hurdle). One random feature beats +0.05 only
5% of the time, but **the best of 33 random features beats +0.28 5% of the time**, and typically scores +0.09. That is the bar a candidate has
to clear: **+0.28**.

| removed from the finished strategy | trades | Sharpe | change | total P/L change |
|---|---|---|---|---|
| above the 10-day SMA | 2,970 | 1.16 | **-1.33** | -$18.5k |
| max move up 0% | 4,164 | 1.39 | **-1.10** | +$14.0k |
| the VIX9D/VIX band | 2,450 | 1.85 | **-0.64** | +$6.2k |
| no Thursdays | 1,958 | 1.86 | **-0.63** | -$8.1k |
| short/long ratio <= 4 | 1,588 | 2.35 | -0.14 | -$0.5k |
| the 7 tranches (trade all 20) | 3,940 | 2.40 | -0.09 | +$95.9k |
| *added: a 400% call stop* | 1,528 | 2.41 | *-0.08* | *-$5.5k* |

- **The four entry filters pay in both halves.** The 10-day SMA is the most important: without it Sharpe falls to 1.16, the book makes $18.5k
  less and the drawdown nearly doubles (-$6.5k to -$11.9k).
- **The ratio cap and the 7 tranches now pay in 2022-24 only** (+0.19 and +0.18), not in 2025-26 (-0.01 and -0.10): with 13:33 in, the set's
  edge over trading all 20 is gone in the recent half.
- **No candidate clears the noise bar, and none adds in both halves.** The best, SPX's 5-day return (+0.16), is what the luckiest of 33
  random features would typically score, and it only works from 2025 (-0.05 in 2022-24).

**Why the VIX1D features looked so bad, and what they really score.** In an earlier version of this ranking, VIX1D at the open (-0.69) and
VIX1D/VIX30 (-0.76) were the two worst candidates, far below the random-noise control (-0.11); VIX1D/VIX9D was weak and redundant and is left
out of the ranking (`DROPPED_FEATURES` in `final_version.py`). **That was missing data, not a signal:** VIX1D only exists from 2023-04-24, and
20% of the strategy's condors (+$30.8k) come before it. The search treated a blank as failing the rule, so every VIX1D cut threw out all of
2022 to April 2023 and lost ~$31k. **With those days kept (a filter can't act on a day its feature doesn't exist), VIX1D at the open scores
+0.05 and VIX1D/VIX30 -0.04**: mid-table, above the noise control, below the bar, and neither works in both halves. On the days it exists,
VIX1D at the open is only weakly related to the condor's P/L (rank correlation -0.15; P/L per condor $99-160 across its quintiles, lowest
at the highest VIX1D): it measures today's priced risk, which the band and the move filter already capture.

**The three pictures below show it.** First, when VIX1D exists against when the strategy made its money:""")

code(r"""
v1d = vx.compute(panel)
vx.fig_coverage(panel, v1d.coverage); plt.show()
cov = v1d.coverage.copy(); cov.attrs = {}
display(cov)
""")

md(r"""| | from | to | sessions | condors | share | P/L |
|---|---|---|---|---|---|---|
| **before VIX1D exists** | 2022-05-31 | 2023-04-18 | 32 | **152** | **20%** | **+$30.8k** |
| VIX1D exists | 2023-05-08 | 2026-05-04 | 150 | 612 | 80% | +$84.7k |

- **The strategy made 27% of its money before VIX1D was published** (a steep stretch of the equity curve in mid-2022). A rule like "VIX1D at
  the open >= x" can't be true on a day with no VIX1D, so if blanks count as failing, every one of those 152 condors is thrown out.

Second, the same search for the best VIX1D cut, with those blank days dropped (the old treatment) and kept (the fix), against the noise
feature and the noise bar:""")

code(r"""
vx.fig_scores(v1d.scores, study.addon_scan.loc["noise", "sharpe_gain"], study.addon_scan.attrs["noise_p95"]); plt.show()
sc1 = v1d.scores.copy(); sc1.attrs = {}
display(sc1.round(2))
""")

md(r"""| feature | treatment of the days before VIX1D | best cut | condors kept | Sharpe gain | P/L change |
|---|---|---|---|---|---|
| VIX1D at the open | dropped (old) | >= 6.72 | 73% | **-0.66** | -$32.1k |
| VIX1D at the open | **kept (fix)** | >= 6.72 | 93% | **+0.05** | -$1.4k |
| VIX1D / VIX30 | dropped (old) | <= 0.716 | 73% | **-0.76** | -$34.6k |
| VIX1D / VIX30 | **kept (fix)** | <= 0.716 | 92% | **-0.04** | -$3.8k |

- **The same cut, the same threshold: only the treatment of the missing days changes,** and the score jumps by 0.7. Almost all of the old
  -0.66 / -0.76 is the ~$31k of pre-2023 P/L the rule threw away, not anything VIX1D said.
- **Fixed, both land between the noise feature (-0.16) and the noise bar (+0.28):** no better than luck.

Third, what VIX1D says about P/L on the days it exists:""")

code(r"""
vx.fig_quintiles(v1d.quintiles); plt.show()
for name, t in v1d.quintiles.items():
    t2 = t.copy(); t2.attrs = {}
    print(name); display(t2.round(1))
""")

md(r"""| VIX1D at the open (quintile) | 5.6-7.4 | 7.4-8.5 | 8.5-9.3 | 9.3-10.8 | 10.8-25.0 |
|---|---|---|---|---|---|
| avg P/L per condor | +$122 | +$158 | +$160 | +$151 | **+$99** |
| 95% interval | $29 to $200 | $64 to $238 | $81 to $227 | $72 to $221 | -$28 to $180 |

| VIX1D / VIX30 (quintile) | 0.40-0.51 | 0.51-0.55 | 0.55-0.61 | 0.61-0.67 | 0.67-1.07 |
|---|---|---|---|---|---|
| avg P/L per condor | +$131 | +$127 | **+$198** | +$97 | +$137 |

- **No clear direction:** every quintile makes money, and all the intervals overlap. The highest VIX1D days earn the least (+$99, interval
  reaching below zero) but even that isn't separable from the rest. Rank correlations with P/L are -0.15 and -0.07.
- **Why there's little left for VIX1D to say:** it prices today's expected move, and the strategy only trades when the VIX9D/VIX curve is
  calm and SPX has pulled back below its open in an uptrend. On those days VIX1D sits in a narrow, ordinary range (mostly 6-13); the few
  high readings (around the spring 2025 selloff) are too rare to build a filter on.""")

code(r"""
fv.fig_feature_corr(study.feature_corr); plt.show()
display(study.redundant_pairs.round(2))
display(study.feature_corr["P/L"].drop("P/L").sort_values(key=abs, ascending=False).round(2).to_frame("corr_with_pl"))
display(study.addon_test.round(2))
""")

md(r"""**The features fall into a few clusters that say the same thing twice:** vol level (VIX and VIX9D: 0.96), premium (credit and the
condor's ratio: 0.89), trend (SPX vs SMA10 with SPX vs SMA5 and the 5-day return: 0.75-0.77), intraday path (range so far, below the
intraday high, move since the open: 0.75-0.79), and VIX1D with VIX1D/VIX30 (0.77). **VIX9D/VIX at entry stands on its own**: it measures the
curve's shape, not its level. Correlation with P/L is small everywhere (credit +0.31, the ratio +0.26, the rest within ±0.19).

| on top of the strategy | sessions | Sharpe | change | total P/L change | both halves |
|---|---|---|---|---|---|
| SMA5 > SMA9 | 159 | 2.47 | -0.02 | -$5.2k | no (-0.16 in 2022-24, +0.31 in 2025-26) |
| condor credit <= $450 | 181 | 2.49 | 0.00 | -$1.9k | no |
| VVIX rank <= 35 | 162 | 2.43 | -0.06 | -$5.8k | no |
| SPX gap >= 0% | 110 | 1.65 | -0.84 | -$38.8k | no |

**Nothing tested improves the finished strategy.**""")

md(r"""### Shape against level: an A/B on the band

The cluster claim says VIX9D/VIX adds something VIX and VIX9D don't. The test: **every other setting on**, and the band replaced by a
filter on the level.

- **A, the strategy:** VIX9D/VIX between 0.87 and 1.02 (shape).
- **B, VIX level, no band:** only days with VIX at the open inside a range. The range is the best of every decile pair on this data (keeping
  at least half the condors), so the level filter gets every advantage.
- **C, VIX9D level, no band:** the same search on VIX9D at the open.
- **D, shape plus level:** the band, plus the best VIX range searched on top of it.

Each is compared with A: the Sharpe difference with a paired session bootstrap, and the same two t-tests as the SMA A/B in Section 13.""")

code(r"""
sl = casc.shape_level
cs.fig_shape_vs_level(sl); plt.show()
print(f"correlation on the no-band book: VIX vs VIX9D {sl.attrs['corr_vix_vix9d']:.2f}, VIX9D/VIX vs VIX {sl.attrs['corr_shape_vix']:.2f}, "
      f"VIX9D/VIX vs VIX9D {sl.attrs['corr_shape_vix9d']:.2f}")
view = sl[["trades", "total_pl", "sharpe", "sharpe_to_2024", "sharpe_from_2025", "max_drawdown", "sharpe_vs_A", "ci_low", "ci_high",
           "p_better_than_A", "t_sharpe", "p_sharpe", "t_daily_pl", "p_daily_pl"]].astype(float).round(3)
view.attrs = {}
display(view.T)
""")

md(r"""**Shape beats level, and level adds nothing on top of it.**

| everything else on | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max DD | Sharpe vs A (95% CI) | p (Sharpe z) |
|---|---|---|---|---|---|---|---|
| **A: VIX9D/VIX 0.87-1.02 (strategy)** | 1,528 | \$115.5k | **2.49** | 2.48 / 2.52 | -\$6.5k | | |
| B: VIX at the open 13.4-32.2, no band | 2,204 | \$124.5k | 2.08 | 2.17 / 1.94 | -\$5.8k | -0.41 (-1.25 to +0.46) | 0.35 |
| C: VIX9D at the open 12.0-37.2, no band | 2,214 | \$113.7k | 1.82 | 1.69 / 2.10 | -\$6.4k | -0.67 (-1.46 to +0.11) | 0.10 |
| D: the band + VIX at the open 13.4-31.5 | 1,378 | \$107.9k | 2.42 | 2.36 / 2.52 | -\$4.5k | -0.08 (-0.41 to +0.27) | 0.66 |

- **The level filters score lower even with their range picked on the same data.** VIX at its best range gives 2.08 (-0.41) and VIX9D 1.82
  (-0.67); A wins in both halves against both. Neither gap is significant on its own: B is behind in 82% of resamples (p 0.35 on the Sharpe
  z), C in 96% (p 0.10). The level ranges are fitted in-sample and the band isn't re-fitted here, so the true gap is, if anything, larger.
- **The best "level" ranges barely filter.** VIX 13.4-32.2 and VIX9D 12.0-37.2 only cut the quiet tails and the panic days; they keep about
  45% more condors than the band. B earns \$9k more in total by trading those extra condors, at a lower Sharpe. Level can't find what the
  band finds because the band's days aren't a VIX range: 91% of A's condors also pass B's filter, and the band picks among them by shape.
- **Level on top of shape: nothing.** D adds the best VIX range to the band and loses 0.08 of Sharpe (not significant, p 0.66) and \$7.6k.
  Its smaller drawdown comes from 150 fewer condors, not from a better selection.
- **Why:** VIX and VIX9D carry the same information (correlation 0.96), while VIX9D/VIX
  is only loosely tied to them (0.36 with VIX, 0.60 with VIX9D, on the no-band book). The ratio says whether the next week is priced
  above or below the next month, which a level doesn't. That matches the clusters: one vol-level cluster, and the shape on its own.""")

# ---------------------------------------------------------------------------
md(r"""## 13. A/B with everything on: above the 10-day SMA vs the 20-day

No cascade here: **every other setting is on** (no Thursdays, max move 0%, VIX9D/VIX 0.87-1.02, the ratio cap, the 7 tranches, held to
expiry), and only the trend rule changes: SPX at entry above its 10-day SMA (A, the strategy) against above its 20-day SMA (B).
Both SMAs are measured the backtester's way (the prior N - 1 closes and the entry price). The Sharpe difference gets a paired session
bootstrap, and two t-tests say how significant each difference is (below).""")

code(r"""
ab = casc.sma_ab
fv.fig_sma_ab(ab, note=f"A/B with everything on: SMA10 {int(ab.iloc[0]['trades']):,} trades vs SMA20 {int(ab.iloc[1]['trades']):,} trades"); plt.show()
view = ab.astype(float).round(2)
view.attrs = {}
display(view.T)
print(f"trades only the SMA10 takes: {ab.attrs['only_sma10']:,} (P/L {fv._k(ab.attrs['only_sma10_pl'])}); "
      f"only the SMA20 takes: {ab.attrs['only_sma20']:,} (P/L {fv._k(ab.attrs['only_sma20_pl'])})")
display(ab.attrs["by_year"].round(2))
tests = ab.iloc[1:][["sharpe_vs_A", "t_sharpe", "p_sharpe", "t_daily_pl", "p_daily_pl"]].astype(float)
tests.attrs = {}
print("significance against A (the strategy's 10-day SMA); a negative t means worse than A:")
display(tests.round(3))
""")

md(r"""| everything on, trend rule | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | win per trade |
|---|---|---|---|---|---|---|
| **A: above the 10-day SMA (strategy)** | 1,528 | **$115.5k** | **2.49** | **2.48 / 2.51** | **-$6.5k** | **87.3%** |
| B: above the 20-day SMA | 1,886 | $108.4k | 1.84 | 1.70 / 2.23 | -$12.1k | 85.2% |

| year | Sharpe A (SMA10) / B (SMA20) | P/L A / B |
|---|---|---|
| 2022 (part) | **2.92** / 1.92 | **$21.5k** / $16.1k |
| 2023 | **2.51** / 2.28 | $29.2k / $29.0k |
| 2024 | **2.18** / 1.29 | **$27.7k** / $25.9k |
| 2025 | **2.19** / 1.90 | $25.6k / $26.0k |
| 2026 (part) | 3.62 / 3.58 | $11.5k / $11.5k |

- **The 10-day wins on every measure:** higher Sharpe (2.49 against 1.84) in every year, $7.1k more P/L, and about half the drawdown (-$6.5k
  against -$12.1k).
- **The 20-day's extra trades don't pay:** the 480 trades only the 20-day takes earn $1.1k; the 122 only the 10-day takes earn $8.2k.
- **The 10-day is better in 96% of resamples** (interval -1.34 to +0.07 for SMA20 minus SMA10), just short of a 95% bar.

**Statistical significance.** Two tests, each against the strategy's 10-day SMA on the same 998 sessions (a negative t means worse than the
10-day):

- **t on the Sharpe difference:** the difference over its standard error from the paired session bootstrap, two-sided. It tests the
  question the A/B asks: is one rule's risk-adjusted return higher?
- **t on the daily P/L difference:** a paired t-test on the session-by-session P/L difference, every session (no-trade days as zero),
  two-sided. It tests whether one rule makes more money on an average day, ignoring risk.

| against the 10-day SMA | Sharpe difference | t (Sharpe) | p (Sharpe) | t (daily P/L) | p (daily P/L) |
|---|---|---|---|---|---|
| B: above the 20-day SMA | -0.65 | **-1.77** | 0.077 | -0.55 | 0.58 |

- **SMA10 against SMA20: t = 1.77 on the Sharpe difference (p = 0.08 two-sided, about 0.04 one-sided).** Significant at 10% but not at
  the usual 5% two-sided bar.
- **The daily P/L barely differs** (t = 0.55, p = 0.58): **the 10-day's edge is risk, not return.**

**The same A/B on the tranches forward selection by drawdown picks** (Section 8: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53, 15:53),
every other setting on, to check the result doesn't depend on the strategy's own tranche set.""")

code(r"""
ab2 = casc.sma_ab_dd
fv.fig_sma_ab(ab2, note=f"A/B on the drawdown-picked tranches: SMA10 {int(ab2.iloc[0]['trades']):,} trades vs SMA20 {int(ab2.iloc[1]['trades']):,} trades"); plt.show()
view2 = ab2.astype(float).round(2)
view2.attrs = {}
display(view2.T)
by2 = ab2.attrs["by_year"].round(2)
by2.attrs = {}
display(by2)
""")

md(r"""| drawdown-picked tranches, everything else on | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | t (Sharpe) | p | t (daily P/L) | p |
|---|---|---|---|---|---|---|---|---|---|
| **A: above the 10-day SMA** | 1,474 | $102.8k | **2.29** | **2.28 / 2.31** | **-$2.6k** | | | | |
| B: above the 20-day SMA | 1,800 | $103.1k | 1.97 | 1.92 / 2.06 | -$7.4k | -0.89 | 0.38 | +0.02 | 0.98 |

- **The 10-day still leads**, on Sharpe (2.29 against 1.97) and above all on drawdown (-$2.6k against -$7.4k), for the same money.
- **But on this tranche set the gap isn't significant** (t = 0.89, p = 0.38): the 10-day's edge is real in direction and in drawdown, but how
  big it looks depends on which tranches it's paired with.""")

# ---------------------------------------------------------------------------
md(r"""## 14. The 2026 drawdown, in the backtester's own log

Everything above is rebuilt from the 20-tranche export, which ends 2026-05-07. **The backtester's log of the strategy** runs to
2026-09-28. **It was run before the tranche change, with 14:53 instead of 13:33** (11:33, 11:53, 12:13, 13:13, 13:53, 14:13, 14:53; no
Thursdays, held to expiry), so this section is about that set. It shows a drawdown the rest of this notebook can't see: from a peak on
2026-05-11, **-$13.4k by 2026-06-17**, three and a half times anything in the four years before, and **not recovered** by the end of the
log. How the new set (with 13:33) would have fared in 2026 needs a new backtest.

First, a check that the log and this notebook describe the same strategy (the rebuild run on the log's own tranches), on the sessions both
cover:""")

code(r"""
log = lg.load_log()
lc = lg.condors(log)
rec = lg.reconcile(log, panel)
print(f"sessions both cover: {rec.attrs['window'][0]:%Y-%m-%d} to {rec.attrs['window'][1]:%Y-%m-%d}")
display(rec.round(2))
""")

md(r"""| 2022-05-31 to 2026-05-07 | trades | total P/L | Sharpe | max drawdown |
|---|---|---|---|---|
| the backtester's log | 1,446 | $111.3k | 2.51 | -$3.8k |
| this notebook's rebuild | 1,450 | $112.0k | 2.53 | -$3.8k |

**The rebuild matches the backtester's log almost exactly** (4 trades and $0.7k apart over four years), so the rules in this notebook are
the backtester's. The difference is what came after 2026-05-07.""")

code(r"""
ldaily = lg.daily_matrix(log)
leps = td.episodes(ldaily)
td.fig_underwater(ldaily, leps); plt.show()
display(leps.round(0))
dd_start, dd_end = leps.iloc[0]["peak"] + pd.Timedelta(days=1), leps.iloc[0]["trough"]
display(td.episode_contrib(ldaily, leps).round(0))
""")

md(r"""| drawdown (backtester's log) | peak | trough | recovered | depth |
|---|---|---|---|---|
| **2026** | **2026-05-11** | **2026-06-17** | **not by 2026-09-28** | **-$13.4k** |
| next deepest | 2024-12-02 | 2025-03-21 | 2025-05-21 | -$3.8k |
| third | 2023-11-28 | 2023-12-20 | 2023-12-29 | -$2.5k |

- **Every tranche lost in it** (-$1.2k to -$2.8k each, 14:53 the most): it isn't one weak tranche, it is whole days going wrong.
- **After the trough the strategy traded 8 sessions to the end of September and made +$2.1k**; the book was still $11.3k below its peak.

**What happened: three days.**""")

code(r"""
display(lg.before_vs_drawdown(lc, log, dd_start, dd_end).round(2))
display(lg.worst_days(lc).round(2))
""")

md(r"""| the log's worst days | P/L | condors | lost a full wing | side that broke | VIX at entry | SPX from entry to close |
|---|---|---|---|---|---|---|
| **2026-05-12** (Tue) | **-$5.2k** | 7 | **7** | call (-$6.0k) | 19.0 | **+0.55%** |
| **2026-06-17** (Wed) | **-$4.8k** | 7 | **6** | put (-$5.5k) | 16.9 | **-1.15%** |
| **2026-06-16** (Tue) | **-$3.6k** | 7 | **5** | put (-$4.7k) | 15.9 | -0.34% |
| 2024-12-10 (Tue) | -$2.4k | 4 | 2 | put | 14.1 | -0.29% |
| 2023-05-23 (Tue) | -$1.7k | 3 | 1 | put | 17.9 | -0.46% |

- **The whole drawdown is three sessions:** on 2026-05-12, 06-16 and 06-17 all seven tranches were open and SPX ran through one side of
  nearly all of them. Together they lose $13.6k; the other three sessions of the drawdown net +$0.2k.
- **Each of the three is worse than any day in the four years before** (the previous worst, 2024-12-10, lost $2.4k with 4 condors open). A
  day with all seven tranches open and one side broken costs about $5k: that is the strategy's real worst day, and in 2022-2025 it never
  happened.
- **VIX gave no warning:** it was 16-19 at entry, normal for the strategy (17.7 on average before). The moves after entry were 0.3-1.2%,
  against 0.20% on an average day.

| | before the drawdown | the drawdown |
|---|---|---|
| condors / sessions | 723 / 187 | 31 / 6 |
| win rate | 81% | **26%** |
| average loss per losing condor | -$399 | **-$657** |
| condors that lost a full wing | 37 (5%) | **19 (61%)** |
| average credit per condor | $325 | **$245** |
| SPX move from entry to close (absolute) | 0.20% | **0.53%** |
| short put breached | 15% | **48%** |

**Why 2026 hurts more: less credit for each breach.**""")

code(r"""
ly = lg.by_year(log, lc)
lg.fig_by_year(ly); plt.show()
display(ly.round(2))
""")

md(r"""| year | SPX (median) | credit per spread | short strike from spot | condors | full-wing losses | avg P/L per condor | worst day |
|---|---|---|---|---|---|---|---|
| 2022 | 3,951 | $145 | 0.52% | 115 | 9 | +$183 | -$1.6k |
| 2023 | 4,341 | $175 | 0.21% | 164 | 8 | +$161 | -$1.7k |
| 2024 | 5,310 | $175 | 0.16% | 179 | 8 | +$137 | -$2.4k |
| 2025 | 6,444 | $150 | 0.22% | 212 | 12 | +$130 | -$1.6k |
| **2026 (to Sep)** | **7,369** | **$130** | 0.29% | 115 | **22** | **+$4** | **-$5.2k** |

- **The credit has been shrinking since 2024** ($175 -> $150 -> $130 a spread) while the wing stays 10 points: a breached spread now loses
  up to $870 against $825 before, and a winning one pays less. Each full-wing loss takes more winning trades to earn back (about 6.7
  winners in 2026 against 4.7 in 2023-24).
- **The strikes are not closer:** they sit further from spot in 2026 (0.29%) than in 2023-25. What changed is the size of the moves (0.25% from
  entry to close on average in 2026, against 0.18% in 2024-25) and how many hit all seven tranches the same day.
- **22 of the 59 full-wing losses in the whole log came in 2026**, 18 of them in those three days.

**What it says about the strategy:**

- **The risk was always there, it just hadn't shown up.** Seven tranches spread over three hours on one day, with no stop, are close to one
  big position: a day that runs through one side costs ~$5k. The four years before simply had no such day on a day the filters let through.
- **Two of the three days were scheduled news:** 2026-05-12 was a CPI release (-$5.2k) and 2026-06-17 an FOMC decision (-$4.8k);
  2026-06-16, the first day of that FOMC meeting, isn't on the calendar. In the log, event days lost $6.3k in 2026 but **made money in every
  year before** (+$4.9k, +$3.8k, +$1.5k, +$6.5k in 2022-25). On the rebuild an event blackout costs 0.55 of Sharpe and $24.1k (Section 3).
  **A blackout would have avoided this drawdown, and cost more than it saves in every earlier year:** one bad year on two events isn't
  enough to add it.
- **On the rebuild this set's drawdowns were shallow** (-$3.8k at most in 2022-2026). The log shows the tail those four years didn't
  contain, and it is the number to size the strategy on.""")

# ---------------------------------------------------------------------------
md(r"""## 15. Summary

**The greedy cascade, step by step:**

| step | trades | P/L | Sharpe | what it does |
|---|---|---|---|---|
| raw book | 39,694 | -$7.5k | -0.12 | wins 66% of condors, but the losses are twice the wins |
| + no Thursdays | 31,768 | $155.6k | 0.44 | removes the weekday where the raw book loses (-$163.0k) |
| + max move 0% | 14,416 | $195.9k | 0.97 | enters on pullbacks, so an intraday rally doesn't run into the short call |
| + above the 10-day SMA | 7,362 | $205.6k | 1.50 | keeps calm markets, where SPX moves less after entry |
| + VIX9D/VIX 0.87-1.02 | 4,630 | $213.3k | 2.25 | keeps a curve that isn't bracing for a move; both tails lose on the put side |
| + short/long <= 4 | 3,940 | $211.4k | 2.40 | skips the richest, nearest-the-money condors |
| **+ the 7 tranches** | **1,528** | **$115.5k** | **2.49** | keeps 11:33-14:13 (with 13:33); drawdown -$12.7k -> -$6.5k |

**The strategy: 1,528 trades, $115.5k, Sharpe 2.49, -$6.5k drawdown, 87.3% of trades winning**, held to expiry; +$635 on an average trading
day (75% of days positive).

- **The swap of 14:53 for 13:33 is a step back on this data:** +$3.4k, but Sharpe 2.60 -> 2.49, the drawdown 70% deeper (-$3.8k -> -$6.5k)
  and 2025-26 much weaker (2.92 -> 2.51). 13:33 earns more on its own (the 6th tranche by P/L, against 11th) but is the biggest single
  loser in the deepest drawdown and the first tranche both cutting rules remove (Sections 8 and 11).
- **No Thursdays isn't a VIX regime:** nothing known at entry differs on Thursdays, but they move more than VIX prices (the realized range
  explains two-thirds of the loss on the raw book), on a few big days every year. **It isn't the event calendar:** few events fall on Thursdays, an event-day blackout costs 0.55 of
  Sharpe (event days are good days), and the only event the loss concentrates on is the day after FOMC (9 Thursdays, -$10.0k).
- **The settings work together.** The band is the worst entry filter on the raw book and adds the most at step 4; with everything on, every
  threshold is at the top of its sweep, but **the band's exact edges aren't significant** (no cap from 0.98 to 1.08 is significantly worse
  than 1.02); the direction is (caps of 0.97 and tighter and floors of 0.90 and higher are clearly worse).
- **The 10-day SMA is the most important setting:** removing it costs 1.33 Sharpe and nearly doubles the drawdown. With everything on it
  beats the 20-day in every year (t = 1.77, p = 0.08: significant at 10%, not at 5%); the edge is lower risk, not more money.
- **No stop:** every call-stop level lowers Sharpe and P/L, and a put stop is worse still (400%: Sharpe 1.88, -$21.8k; 42% of the stopped
  puts would have expired winning) (Section 9).
- **No candidate feature beats the noise bar** (+0.28, the best of 33 random features); the VIX1D features only looked bad because VIX1D
  starts in April 2023 (Section 12).
- **The backtester's log shows a 2026 drawdown the export doesn't reach** (on the previous 7, with 14:53): -$13.4k from 2026-05-11 to 06-17,
  not recovered by 2026-09-28, from three days on which all seven tranches lost a full wing (Section 14). Two of the three were CPI and FOMC days,
  but event days made money in every earlier year: size the strategy on that tail rather than black out events, and backtest the new
  tranche set.

**What this notebook doesn't establish:** out-of-sample performance (apart from the walk-forward checks); the backtester's totals on the
new tranche set (the rebuild matches the backtester trade by trade on the previous 7; the set with 13:33 hasn't been backtested); the first
sessions, which need SPX closes before the export.""")

# ---------------------------------------------------------------------------
md(r"""## 16. Robustness checks

Every check this notebook uses, grouped by the question it answers. "Section" says where to find it.

### 1. Is the difference real, or luck?

| check | how it works | where it's used | what it found | section |
|---|---|---|---|---|
| **Session bootstrap** | Resample whole trading days (all tranches of a day together) 2,000 times; the spread of the results gives a 95% interval | P/L per condor in every profile, the ratio quintiles, the VIX1D quintiles | Most profile buckets overlap; the clear ones are named in the text | 3-12 |
| **Paired bootstrap** | Resample the same days for two versions at once and look at the Sharpe *difference*; reports a 95% interval and how often one beats the other | Band caps and floors, SMA10 vs SMA20, shape vs level | A difference counts only when its interval excludes zero | 6, 12, 13 |
| **t-test on daily P/L** | Paired t-test: do the two versions make different money per day? | SMA A/B, shape vs level | SMA10 vs SMA20: the daily P/L difference is small; the edge is lower risk | 12, 13 |
| **Sharpe z-score** | Sharpe difference divided by its bootstrap standard error | SMA A/B, shape vs level | SMA10 over SMA20: t = 1.77, p = 0.08 (10% level, not 5%); shape over VIX level p = 0.35, over VIX9D level p = 0.10 | 12, 13 |
| **Peak against its neighbours** | 95% interval at the chosen value against each nearby value, plus a z-test for each pair | Term-structure cap (1.02) and floor (0.87) | No cap from 0.98 to 1.08 is significantly worse than 1.02; caps of 0.97 and lower, and floors of 0.90 and higher, are | 6 |
| **Regression, errors clustered by session** | Condor P/L on a Thursday dummy, controlling for everything known at entry | The no-Thursday rule | The Thursday loss stays (-$51 -> -$55, p = 0.008) with every entry feature in; it fades only once the day's realized range is added | 3 |
| **Event-day blackout A/B** | FOMC, CPI, NFP, opex, quad witching, VIX expiry, early closes and pre-holiday sessions blacked out, against and alongside the no-Thursday rule | The blackout costs 0.55 Sharpe (p 0.005): event days are good days; the Thursday loss concentrates only on the day after FOMC | 3 |

### 2. Did we find a signal, or fit noise?

| check | how it works | what it found | section |
|---|---|---|---|
| **Noise feature** | A random feature goes through the same threshold search as the real ones | It "gains" -0.16: the yardstick for a meaningless filter | 12 |
| **Family-wise noise bar** | The best of 33 random features, 200 times over; the bar is the 95th percentile (+0.28). A real feature must beat the best of 33 random ones, not just one | **No candidate feature clears it**: nothing tested improves the finished strategy | 12 |
| **Feature correlation clusters** | Groups features that say the same thing so one signal isn't counted twice | Five clusters (vol level, premium, trend, intraday path, VIX1D); VIX9D/VIX stands alone | 12 |
| **Shape vs level A/B** | The band against VIX and VIX9D level filters given their *best in-sample* range | The band still scores higher (2.49 against 2.08 and 1.82); level on top of it adds nothing | 12 |
| **Missing-data rule** | A filter keeps the days its feature is blank, instead of silently dropping them | Fixed a false VIX1D result: VIX1D starts 2023-04-24, so dropping blank days threw out a fifth of the strategy | 12 |

### 3. Does it hold over time?

| check | how it works | what it found | section |
|---|---|---|---|
| **Split halves** | Every table reports 2022-24 and 2025-26 separately | Each filter is flagged when it pays in one half only | all |
| **By year** | Results for each calendar year | SMA10 beats SMA20 in every year with everything on; the Thursday loss shows up in all five years; the 2026 drawdown is broken down by year | 3, 13, 14 |
| **Walk-forward tranche selection** | Re-run the selection on earlier years only, trade its pick in the next year | Re-chosen each year it does worse than all 20 (1.77 against 2.30); what holds every year is dropping the first hour | 8, 11 |
| **Live log reconciliation** | The rebuild against the backtester's own 2026 trade log | Matches on the previous 7: 1,446 against 1,450 trades, $111.3k against $112.0k | 14 |

### 4. Is the setting on a plateau or a spike?

| check | how it works | what it found | section |
|---|---|---|---|
| **Greedy cascade** | Add the setting with the biggest Sharpe gain at each step | Order: Thursday, move, SMA10, band, ratio, tranches | 2-8 |
| **Leave-one-out** | Turn each setting off with all the others on | Every entry filter pays in both halves; the 10-day SMA matters most (-1.33 Sharpe without it) | 12 |
| **Everything-on sweeps** | Sweep each threshold with every other setting fixed | Every threshold sits at the top of its sweep | 10 |
| **Fine band sweep** | Cap and floor in steps of 0.01 | 1.02 is the highest cap, but its neighbours from 0.99 to 1.05 score 2.30-2.41 and none is significantly worse: the edge is the region, not the exact value | 6 |
| **SMA10 vs SMA20** | Compared at step 3 and again with everything on | SMA10 wins with everything on; the edge is lower risk | 5, 13 |
| **Stop scans** | Call and put stops across levels, $0.20 exit slippage | Every stop lowers Sharpe; hence no stop | 9 |

### 5. Which tranches, and where do the losses come from?

| check | how it works | what it found | section |
|---|---|---|---|
| **Tranche-set comparison** | The live 7, the previous 7 (14:53), the earlier 15, all 20, and the sets picked by Sharpe and by drawdown | The previous 7 (with 14:53) scores higher than the live set: 2.60 against 2.49, drawdown -$3.8k against -$6.5k | 8 |
| **Forward selection** by Sharpe and by P/L per $ of drawdown | Add the tranche that helps most, one at a time; the drawdown version starts from all 20 and ignores the live set | Both stop at 11 tranches | 8 |
| **Drawdown episodes** | Each tranche's share of every drawdown | 13:33 loses in all 5 of the deepest drawdowns | 11 |
| **Backward elimination** from 7 and from 20 | Remove the tranche that hurts most, one at a time, tracking P/L per $ of peak margin | 13:33 is the first tranche both rules cut | 11 |
| **Pareto** | Tranches sorted by P/L, with a cumulative-share line | Shows how much of the P/L the top tranches carry | 11 |
| **P/L per $ of margin, drawdown % of margin, correlation** | Each tranche's return on the margin it ties up, its drawdown against that margin, and its daily P/L correlation with the other tranches | The live 7 lead on return on margin; the previous 7 (with 14:53) match it (18.5x against 18.9x) with a drawdown of 63% of margin against 106%, because 14:53 correlates least with the rest | 11 |

### What isn't tested

- **A true out-of-sample period.** Every threshold and the tranche set were chosen on the same 998 sessions they are scored on; only the
  tranche selection has a walk-forward check.
- **Entry costs.** Entries are taken at the backtester's fills; slippage is varied only on stop exits.
- **Structural breaks.** Stability is judged by halves and by year, not with a formal break test.
- **Event days beyond Thursday.** The blackout is tested against the no-Thursday rule (Section 3: it costs Sharpe); whether events drove
  the 2026 drawdown isn't tested.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "py313-quant", "language": "python", "name": "py313-quant"}
with open("ic_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote ic_analysis.ipynb with {len(cells)} cells.")
