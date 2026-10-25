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
short strikes until 16:00. Inside the strategy each condor collects about **$326** (median $325: $175 call side, $145 put side), with
short strikes a median 11.4 points (call) and 12.9 points (put) from spot.

**The strategy: every setting used**, in the order the greedy cascade adds them (Section 2).

| step | setting | value |
|---|---|---|
| 1 | weekdays | **Monday, Tuesday, Wednesday, Friday** (no Thursdays) |
| 2 | max SPX move up | **max 0%**: SPX at or below its 09:30 open at entry |
| 3 | trend | **above the 10-day SMA**: SPX at entry against the average of the last 9 closes and the entry price |
| 4 | term structure | **VIX9D/VIX: min 0.87, max 1.02**, read at each trade's entry minute |
| 5 | entry times | **7 tranches: 11:33 AM, 11:53 AM, 12:13 PM, 1:13 PM, 1:53 PM, 2:13 PM, 2:53 PM** (the first seven forward selection adds) |
| 6 | short/long ratio | **max 4** for the whole condor: the two short prices over the two long prices |
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
| 9 | The call stop: why the strategy doesn't use one |
| 10 | With everything on: is each threshold still at its best? |
| 11 | Drawdowns by tranche: where the losses come from, and whether to cut the 7 further |
| 12 | Feature ranking and correlation |
| 13 | A/B with everything on: above the 10-day SMA vs the 20-day |
| 14 | The 2026 drawdown, in the backtester's own log (past the end of the export) |
| 15 | Summary |

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
- **Everything is in-sample**, apart from the walk-forward checks in Sections 7 and 11. Intervals are 95% session-bootstrap intervals.""")

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
from strategies.ic_strat.analysis import log_2026 as lg
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
| 1 (raw book) | **no Thursdays 0.44**, max move 0.37, SMA10 0.35, tranches 0.12, ratio 0.00, band -0.11 | **no Thursdays** |
| 2 | **max move 0.97**, SMA10 0.71, band 0.66, ratio 0.59, tranches 0.56 | **max move 0%** |
| 3 | **SMA10 1.50**, band 1.39, ratio 1.15, tranches 1.03 | **above the 10-day SMA** |
| 4 | **band 2.25**, ratio 1.74, tranches 1.66 | **VIX9D/VIX band** |
| 5 | **tranches 2.48**, ratio 2.40 | **the 7 tranches** |
| 6 | **ratio 2.60** | **short/long ratio <= 4** |

- **No Thursdays wins step 1** because Thursdays are where the raw book loses its money (Section 3); max move and the 10-day SMA are close
  behind (0.37, 0.35).
- **The band only pays once the move and trend filters are on:** on the raw book it scores -0.11 (the worst of the entry filters); at step 4
  it adds the most of any step (+0.75).
- **The tranche step is the fixed set of 7**; Section 7 shows they are the first seven forward selection picks at that point, and what a
  selection by drawdown picks on its own.

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
| 4. + VIX9D/VIX 0.87-1.02 | 4,630 | **$213.3k** | 2.25 | 2.16 / 2.48 | -$15.1k | 83.4% |
| 5. + 7 forward-selected tranches | 1,582 | $113.4k | 2.48 | 2.33 / 2.83 | -$4.9k | 85.7% |
| **6. + short/long ratio <= 4** | **1,450** | **$112.0k** | **2.60** | **2.44 / 2.92** | **-$3.8k** | **87.4%** |

**The raw book is a coin flip that loses slowly:** a condor wins 66% of the time, but an average win (+$254) is half an average loss
(-$497), so the book drifts to -$7.5k with a -$113k drawdown.

- **The first four steps build the money:** no Thursdays alone turns -$7.5k into +$155.6k; max move, the 10-day SMA and the band each add
  +0.5 to +0.75 Sharpe, and P/L peaks at $213.3k with the band.
- **The 7 tranches and the ratio cap trade money for steadiness:** they cost $101.2k between them but take Sharpe from 2.25 to 2.60 and the
  drawdown from -$15.1k to -$3.8k.

**The strategy: 1,450 trades, $112.0k, Sharpe 2.60, -$3.8k drawdown, 87.4% of trades winning**, held to expiry; +$599 on an average trading
day (76% of days positive), an average win of $285 a condor against an average loss of $399.""")

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
| avg P/L per condor | +$174 | +$122 | +$143 | **-$22** | +$195 |
| call / put side | +$100 / +$74 | +$61 / +$61 | +$85 / +$58 | **-$1 / -$21** | +$111 / +$85 |
| short call / short put breached | 19% / 14% | 22% / 16% | 16% / 13% | **27% / 22%** | 15% / 15% |
| VIX at the open | 17.4 | 17.5 | 17.8 | **16.5** | 16.7 |
| sessions with a range above 1% | 4% | 11% | 23% | **32%** | 26% |
| range / VIX-implied move | 0.54 | 0.58 | 0.72 | **0.80** | 0.72 |

| condor P/L on a Thursday dummy (strategy with Thursdays) | Thursday effect, $ per condor | t | p |
|---|---|---|---|
| Thursday alone | **-$176** | -3.32 | 0.001 |
| + every feature known at entry | **-$181** | -3.54 | <0.001 |
| + the session's range vs implied (an outcome) | -$122 | -2.77 | 0.006 |

- **The same story inside the strategy:** entry conditions are ordinary (VIX is in fact the lowest, 16.5), but Thursdays move more than VIX
  priced (0.80 against 0.54-0.72), both short strikes break more often, and the entry features explain none of the gap.
- **The realized move explains about a third of it here** (-$181 -> -$122, still significant): even for the same range, Thursday condors do
  worse, which fits an intraday shock (a release or a reversal) more than a steady trend.
- **It is a low-VIX problem:** Thursdays lose with VIX under 18 (-$150 and -$64 a condor, against +$133 and +$189 on other days), where VIX
  most underprices them (range / implied 0.91 and 0.78 against 0.66 and 0.56); above 18 the few Thursday trades (102) earn.
- **Five bad Thursdays sink the rest:** the worst five lose $14.9k, more than all Thursdays together (-$4.5k); the other Thursdays make
  +$10.4k. The median Thursday session still wins (+$202), but only 56% of them do, against 77% of other days.

**Why no Thursdays, in one line:** not a VIX regime and nothing visible at entry, but Thursdays realise more movement than VIX prices at the
open, mostly on a few big days each year (macro releases are the likely cause, untested here), and a short-volatility condor has no defence
against that. Skipping the day is the only filter that removes it.""")

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
  sharpest in 2022-24.""")

# ---------------------------------------------------------------------------
md(r"""## 7. Step 5: the tranches, on the book after steps 1-4

The book so far trades all 20 entry times. This step keeps the strategy's **7 tranches: 11:33, 11:53, 12:13, 13:13, 13:53, 14:13 and
14:53**. They are the first seven tranches forward selection adds on this book (below). First, each tranche on this book and how
correlated they are.""")

code(r"""
cs.fig_tranches(casc.tranches, note=cs.note(ct, "tranches", "P/L by tranche, all 20")); plt.show()
display(fv.highlight_dropped(casc.tranches, 2))
""")

md(r"""- **The strategy keeps six of the seven best earners:** 13:53 (+$166 a condor), 12:13 (+$156), 11:53 (+$152), 13:13 (+$142), 14:13 (+$139)
  and 11:33 (+$133), plus 14:53 (+$114). 11:13, 13:33 and 12:53 earn a little more than 14:53 (+$122-128), but forward selection adds 14:53
  first: it adds more to the book's Sharpe.
- **The first hour earns the least:** 09:32-10:33 make $32-44 a condor. The last two are weak (15:33 +$23, 15:53 -$21).""")

code(r"""
corr = casc.tranche_corr
fv.fig_tranche_corr(corr, fv.tranche_corr_by_gap(corr), note=cs.note(ct, "tranches", "how correlated the 20 tranches are")); plt.show()
t = fv.tranche_summary(corr)
print(f"{t['tranches']} tranches, average pairwise correlation {t['avg_pairwise_rho']:.2f} -> "
      f"about {t['effective_independent_tranches']:.1f} independent bets a day; {t['pairs_above_0.7']} pairs above 0.7")
display(fv.tranche_blocks(corr).round(2))
display(fv.tranche_corr_by_gap(corr).round(2))
""")

md(r"""**Neighbouring tranches are close to the same trade.** Entries 20 minutes apart correlate 0.74 on average; 1 hour apart 0.51, 2 hours
0.29, 3 hours 0.17, 4 hours 0.07. The 20 tranches are about **2.8 independent bets a day** (average pairwise correlation 0.33). The 7 are
spread from 11:33 to 14:53, never more than an hour apart, so they share most of each day's risk: what they give up against all 20 is
mostly the first and the last hour, the weakest tranches.""")

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

md(r"""**Forward selection on this book:** it starts with 11:53 (the best tranche alone), then adds 13:53, 12:13, 14:13, 11:33, 13:13 and 14:53:
**the strategy's 7, in that order** (Sharpe 2.48 at step 7). It goes on to 11:13 (2.56) and 13:33, where it peaks (2.58, 9 tranches), is
flat to 12 tranches and falls after.

**Walk-forward:** for each year from 2023, the selection is re-run on every earlier year only and its pick traded in the next year.

| test year | tranches chosen | chosen on earlier years | the strategy's 7 | all 20 |
|---|---|---|---|---|
| 2023 | 11 | 1.25 ($24.2k) | 2.24 ($27.3k) | 2.43 ($56.3k) |
| 2024 | 13 | 1.31 ($27.8k) | 2.05 ($26.0k) | 1.44 ($40.0k) |
| 2025 | 12 | 2.03 ($31.5k) | 2.57 ($27.3k) | 2.31 ($49.8k) |
| 2026 | 12 | 3.78 ($17.6k) | 3.59 ($11.7k) | 3.00 ($20.0k) |
| **2023-26** | | **1.69 ($101.1k)** | **2.39 ($92.3k)** | 2.11 ($166.1k) |

- **Re-chosen each year, the selection does worse than trading all 20** (1.69 against 2.11 out of sample; better in only 2% of resamples):
  with little history it keeps too few tranches, and not always the right ones.
- **The strategy's 7 score 2.39 over the same years**, ahead of all 20 (2.11), but that set was chosen on all the data, so part of its edge
  is hindsight. It makes $92.3k against all 20's $166.1k.
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

- **It peaks at 8 tranches: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53 and 15:53**, at $39.8 of P/L per $ of drawdown: 1,474 trades,
  $102.8k, Sharpe 2.29 (2.28 / 2.31 in the two halves), drawdown -$2.6k.
- **15:53 is a passenger:** with the ratio cap on it trades twice in four years, so it neither helps nor hurts; the working set is the
  other seven.
- **Past the peak, each tranche adds more drawdown than profit:** 11:53 and 13:13 still raise Sharpe (to 2.62 at 12) but deepen the
  drawdown to -$4.1k; from 13 tranches on the drawdown grows fast (-$5.2k to -$12.7k).
- **Drawdown and Sharpe point to different sizes:** the ratio peaks at 8 tranches with Sharpe 2.29; Sharpe keeps rising to 2.66 at 16
  tranches, where the drawdown is three times as deep.""")

md(r"""**Your 7 against the other sets, with every other setting on** (the finished strategy, only the tranches changed):""")

code(r"""
display(casc.tranche_sets.drop(columns="tranches").astype(float).round(2))
""")

md(r"""| tranche set (everything else on) | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | P/L per $ of drawdown |
|---|---|---|---|---|---|---|
| **the strategy's 7** | **1,450** | $112.0k | 2.60 | 2.44 / 2.92 | -$3.8k | 29.4 |
| forward selection's peak (the 7 + 11:13, 13:33) | 1,892 | $141.9k | **2.73** | 2.64 / 2.93 | -$6.0k | 23.7 |
| forward selection by drawdown (8) | 1,474 | $102.8k | 2.29 | 2.28 / 2.31 | **-$2.6k** | **39.8** |
| previous live set (15) | 2,704 | $180.4k | 2.69 | 2.64 / 2.82 | -$6.5k | 27.6 |
| all 20 | 3,940 | **$211.4k** | 2.40 | 2.30 / 2.61 | -$12.7k | 16.6 |

- **The drawdown pick has the smallest drawdown (-$2.6k) and the most P/L per $ of drawdown (39.8)**, but the lowest Sharpe here (2.29)
  and the weakest 2025-26 (2.31).
- **The 7 come next on drawdown (-$3.8k, 29.4)**, with the best 2025-26 Sharpe alongside the Sharpe pick.
- **They don't have the highest Sharpe:** adding 11:13 and 13:33 (the 9) scores 2.73 for $29.8k more, and the previous 15 score 2.69 for
  $68.4k more, each with a drawdown $2-3k deeper. Fewer tranches means fewer trading days, and every idle day pays the 4% hurdle.
- **The trade-off is size against depth:** the 7 are the smallest, steadiest book; the 9 and the 15 score a higher Sharpe but draw down
  further.""")

# ---------------------------------------------------------------------------
md(r"""## 8. Step 6: the short/long ratio <= 4, on the book after steps 1-5

The backtester caps one short/long ratio for the whole condor: the two short strikes' prices over the two long strikes'.""")

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

md(r"""**On the 7 tranches the ratio's relationship with P/L is a hump.** Rank correlation with condor P/L +0.16 (interval +0.05 to +0.26);
below its 60th percentile (2.65) +0.51, above it -0.08. The ratio is strike distance in another form (-0.95 to -0.96 with how far the shorts
sit from spot, +0.89 with the credit).

| condor short/long ratio | trades | avg P/L | win rate | 2022-24 / 2025-26 |
|---|---|---|---|---|
| 1.25-1.87 | 318 | +$134 | 90% | +$89 / +$156 |
| 1.87-2.23 | 316 | **+$197** | 87% | +$195 / +$201 |
| 2.23-2.65 | 316 | +$188 | 82% | +$205 / +$152 |
| 2.65-3.23 | 316 | +$117 | 72% | +$140 / +$37 |
| above 3.23 | 316 | +$81 | **65%** | +$79 / +$93 |

| maximum ratio | trades | total P/L | Sharpe | max drawdown |
|---|---|---|---|---|
| 3.5 | 1,350 | $104.4k | 2.49 | -$3.3k |
| **4 (strategy)** | **1,450** | **$112.0k** | **2.60** | -$3.8k |
| 4.5 | 1,502 | $113.8k | 2.57 | -$4.7k |
| 5 | 1,552 | $113.0k | 2.50 | -$5.2k |
| no cap | 1,582 | $113.4k | 2.48 | -$4.9k |

- **On the 7 tranches the cap removes only 132 trades.** Those earn little ($1.3k between them) and add risk: a cap of 4 is the peak (2.60
  against 2.48 with no cap), and the drawdown is $1.1k smaller.""")

# ---------------------------------------------------------------------------
md(r"""## 9. The call stop: why the strategy doesn't use one

Not a step: the strategy holds every condor to expiry. Here the finished strategy (all six settings) gets a call-side stop at every level,
replayed the backtester's way: it fires when the call spread's worst intraday loss reached X% of its credit (never on the 16:00 settlement
print) and fills where the option chain first crossed it, plus **$0.20 per contract ($20) slippage** and fees (blue); $0 slippage for
reference (grey); the dotted line is the strategy, with no stop. Every level trades the same 1,450 trades.""")

code(r"""
fv.fig_stop(study.stops, note=f"The call stop on the finished strategy ({int(ct.iloc[-1]['trades']):,} trades): every level against no stop"); plt.show()
display(study.stops.loc[fv.EXIT_SLIPPAGE].round(2))
""")

md(r"""| call stop at $0.20 slippage | fires on | total P/L | Sharpe | max drawdown | 2022-24 / 2025-26 |
|---|---|---|---|---|---|
| **none (strategy)** | | **$112.0k** | **2.60** | **-$3.8k** | **2.44 / 2.92** |
| 100% | 36% of condors | $56.8k | 1.02 | -$4.3k | 0.76 / 1.52 |
| 200% | 16% | $87.1k | 1.91 | -$3.9k | 1.77 / 2.20 |
| 300% | 9% | $98.1k | 2.23 | -$4.0k | 2.21 / 2.29 |
| 400% | 4% | $105.8k | 2.46 | -$3.8k | 2.36 / 2.67 |

- **Every stop level lowers Sharpe and P/L, in both halves, and none makes the drawdown smaller.** The 400% stop the strategy used until now
  costs 0.14 Sharpe and $6.2k for the same -$3.8k drawdown.
- **The entry filters already do the stop's job:** with SPX at or below the open and above its 10-day SMA, the call side rarely runs away, so a
  stopped call usually would have recovered by the close. Tight stops are ruined by slippage and real fills (at 25-100% the stop fires on
  36-70% of condors and Sharpe falls to -0.18 to 1.02).
- **What the backtest can't show is a day far worse than any in these four years.** Going without a stop accepts that risk; the -$3.8k
  drawdown is the worst the data shows, not the worst possible.""")

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
| floor (VIX9D/VIX >=) | 0.87 | **2.60** | 0.85: 2.27; 0.89: 2.31 | **at the peak** |
| cap (VIX9D/VIX <=) | 1.02 | **2.60** | 1.03: 2.38; 1.04: 2.49 | **at the peak** |
| max move up | 0% | **2.60** | -0.05%: 2.26; +0.05%: 2.44 | **at the peak** |
| SMA10 bar | 0% | **2.60** | -0.5%: 2.06; +0.5%: 2.22 | **at the peak** |
| short/long cap | 4 | **2.60** | 3.5: 2.49; 4.5: 2.57 | **at the peak** |

**With everything on, every threshold is at the top of its sweep, by a clear margin.** The +0.5% SMA10 bar that led at step 3 was ahead only
because the band wasn't on yet.""")

# ---------------------------------------------------------------------------
md(r"""## 11. Drawdowns by tranche: where the losses come from, and whether to cut the 7 further

Everything in this section is on **the finished strategy** (all six settings, held to expiry): 1,450 trades, $112.0k, Sharpe 2.60, max
drawdown -$3.8k. First, the book's drawdowns: the equity curve, the curve below its last peak, and its five deepest drawdowns.""")

code(r"""
tdd = td.compute(panel)
td.fig_underwater(tdd.daily, tdd.episodes); plt.show()
display(tdd.episodes.round(0))
""")

md(r"""| drawdown | peak | trough | recovered | depth | sessions to the trough |
|---|---|---|---|---|---|
| **DD1** | 2024-12-02 | 2024-12-10 | 2025-05-21 | **-$3.8k** | 6 (115 under water) |
| DD2 | 2023-11-28 | 2023-12-20 | 2023-12-29 | -$2.5k | 16 |
| DD3 | 2024-10-03 | 2024-10-07 | 2024-11-12 | -$2.3k | 2 |
| DD4 | 2025-05-27 | 2025-06-02 | 2025-06-17 | -$1.8k | 4 |
| DD5 | 2023-05-22 | 2023-05-23 | 2023-05-30 | -$1.7k | 1 |

- **The deepest drawdown is -$3.8k, reached in six sessions in December 2024**; recovering took until May 2025 (115 sessions).
- **The others are -$1.7k to -$2.5k**, each recovered within six weeks.

**Which tranches lost money in each drawdown**, from its peak to its trough:""")

code(r"""
td.fig_episode_heatmap(tdd.contrib, tdd.episodes); plt.show()
display(tdd.contrib.round(0))
""")

md(r"""- **DD1 is the late-morning tranches:** 11:53 (-$1.15k), 12:13 (-$1.14k), 13:13 (-$0.8k) and 11:33 (-$0.7k); the three afternoon tranches
  didn't lose in it.
- **The others are spread out:** 13:13 lost in all five (-$0.3k to -$0.9k), and with 13:53-14:53 it carries most of DD2, DD4 and DD5.

Each tranche's own drawdown, on one scale, and its P/L year by year:""")

code(r"""
td.fig_tranche_underwater(tdd.daily); plt.show()
td.fig_by_year(tdd.by_year); plt.show()
display(tdd.table.round(2))
display(tdd.by_year.round(0))
""")

md(r"""| tranche | trades | total P/L | own max drawdown | P/L per $ of drawdown | losing years | in DD1 | book without it: Sharpe / P/L / max DD |
|---|---|---|---|---|---|---|---|
| 11:33 | 244 | $16.3k | -$1.9k | 8.8 | 0 | -$0.7k | -0.20 / -$16.3k / +$0.7k |
| 11:53 | 238 | **$18.1k** | **-$2.1k** | 8.8 | 0 | **-$1.1k** | -0.22 / -$18.1k / +$0.8k |
| 12:13 | 220 | $17.2k | -$1.8k | 9.6 | 0 | **-$1.1k** | -0.21 / -$17.2k / +$1.0k |
| 13:13 | 230 | $16.4k | -$1.7k | 9.5 | 0 | -$0.8k | -0.16 / -$16.4k / +$0.8k |
| 13:53 | 202 | $17.7k | -$1.6k | 11.1 | 0 | $0.0k | **-0.28** / -$17.7k / $0.0k |
| 14:13 | 182 | $14.8k | -$1.5k | 9.8 | 0 | $0.0k | -0.22 / -$14.8k / $0.0k |
| 14:53 | 134 | $11.7k | **-$0.9k** | **13.4** | 0 | $0.0k | -0.20 / -$11.7k / -$0.6k |

(A single tranche's Sharpe isn't shown: on its own it never clears the 4% hurdle on $200k, so P/L per $ of drawdown compares them instead.
"Book without it: max DD" is the change in the book's max drawdown: + means smaller.)

- **All seven made money in every year**, $11.7k to $18.1k each.
- **Dropping any one costs 0.16 to 0.28 Sharpe and $12k-18k.** The four late-morning tranches each carry part of DD1, so dropping one would
  shrink it by $0.7k-1.0k, but each is among the best earners.

**Cutting further, one tranche at a time.** Starting from the 7, each step drops the tranche whose removal gives the best book: by Sharpe
(blue), or by max drawdown (orange). The circle marks where each rule stops: the highest Sharpe, or (by drawdown) the most P/L per $ of
drawdown.""")

code(r"""
td.fig_elimination(tdd.paths); plt.show()
for crit, p in tdd.paths.items():
    print(f"{td.CRITERIA[crit]}: stops at {p.attrs['best']} tranches, cutting {', '.join(p.attrs['cut']) or 'nothing'}")
    display(p.drop(columns="tranches").round(2))
""")

md(r"""| tranches kept | cut by Sharpe: removed | Sharpe | P/L | max DD | cut by drawdown: removed | Sharpe | P/L | max DD |
|---|---|---|---|---|---|---|---|---|
| **7 (strategy)** | | **2.60** | **$112.0k** | **-$3.8k** | | 2.60 | $112.0k | -$3.8k |
| 6 | 13:13 | 2.43 | $95.7k | -$3.0k | 12:13 | 2.39 | $94.8k | -$2.8k |
| 5 | 11:33 | 2.18 | $79.4k | -$2.3k | 13:13 | 2.14 | $78.5k | -$2.0k |
| 4 | 14:53 | 1.86 | $67.7k | -$2.9k | 11:53 | 1.63 | $60.4k | -$1.4k |

- **By Sharpe there is nothing to cut:** every removal lowers it, so the rule keeps all 7.
- **By drawdown every cut shrinks the drawdown, and the P/L with it:** at 4 tranches the drawdown is -$1.4k but the book makes $60.4k at
  Sharpe 1.63.

**Do the cuts hold out of sample?** The same rules re-run for each year on the earlier years only:""")

code(r"""
for crit, w in tdd.walk_forward.items():
    lo, hi = w.attrs["cut_vs_strategy_set_ci"]
    print(f"{td.CRITERIA[crit]}: cut vs the strategy's set out of sample, Sharpe difference 95% interval {lo:+.2f} to {hi:+.2f}; "
          f"cut better in {w.attrs['p_cut_better']:.0%} of resamples")
    display(w.round(2))
""")

md(r"""- **By Sharpe, the rule cuts nothing in 2024-26**; only in 2023, with a single year of history, does it cut (11:33 and 11:53), and that
  year it loses (0.58 against 2.48).
- **By drawdown, the cuts lose out of sample** (1.11 against 2.53, $46.0k against $91.0k).

**What this says about cutting the selection:** nothing. Every one of the 7 made money every year, each one's removal lowers Sharpe, and the
drawdowns are shallow market episodes that a smaller book would only make smaller in proportion.""")

# ---------------------------------------------------------------------------
md(r"""## 12. Feature ranking and correlation

Every feature on one Sharpe scale, against the finished strategy:

- **the strategy's settings (blue, bold, "ON:"):** the Sharpe **lost** when the setting is removed from the finished strategy;
- **every other feature (orange):** the Sharpe **gained** by its best single cut as a further filter (a min or max at a decile, keeping
  at least 30% of condors). Dark orange means the gain holds in both halves;
- **a noise feature (grey):** a random number per session put through the same search: **whatever it scores is what luck alone finds.**

The red line is the strict noise bar: the 95th percentile of the same search on 200 random features. The candidates are price,
volatility, curve and calendar features, including the 20-day SMA (now a candidate, not a setting) and the vol-regime feature library's
findings (VIX9D gap, month end, rule W, ATR20, low VRP).""")

code(r"""
fv.fig_ranking(study.ranking); plt.show()
print(f"the noise feature gains {study.addon_scan.loc['noise', 'sharpe_gain']:+.2f}; strict noise bar "
      f"(95th pct of 200 noise features) {study.addon_scan.attrs['noise_p95']:+.2f}")
display(study.leave_one_out.round(2))
display(study.ranking.round(2))
""")

md(r"""**The noise feature scores -0.11**, and the strict bar is **+0.01**.

| removed from the finished strategy | trades | Sharpe | change | total P/L change |
|---|---|---|---|---|
| above the 10-day SMA | 2,864 | 1.32 | **-1.28** | -$11.2k |
| max move up 0% | 3,946 | 1.42 | **-1.18** | +$10.4k |
| the VIX9D/VIX band | 2,324 | 1.89 | **-0.71** | +$4.2k |
| no Thursdays | 1,862 | 2.03 | **-0.56** | -$4.5k |
| the 7 tranches (trade all 20) | 3,940 | 2.40 | -0.20 | +$99.3k |
| short/long ratio <= 4 | 1,582 | 2.48 | -0.12 | +$1.3k |
| *added: a 400% call stop* | 1,450 | 2.46 | *-0.14* | *-$6.2k* |

- **Every setting pays, in both halves.** The 10-day SMA is now the most important: without it Sharpe falls to 1.32, the book loses $11.2k
  and the drawdown triples (-$3.8k to -$11.8k).
- **No candidate adds in both halves.** SPX's 5-day return (+0.10) clears the noise bar overall but only from 2025 (-0.03 in 2022-24).
  SMA5 > SMA9 loses (-0.04).""")

code(r"""
fv.fig_feature_corr(study.feature_corr); plt.show()
display(study.redundant_pairs.round(2))
display(study.feature_corr["P/L"].drop("P/L").sort_values(key=abs, ascending=False).round(2).to_frame("corr_with_pl"))
display(study.addon_test.round(2))
""")

md(r"""**The features fall into a few clusters that say the same thing twice:** vol level (VIX and VIX9D: 0.96), premium (credit and the
condor's ratio: 0.89), trend (SPX vs SMA10 with SPX vs SMA5 and the 5-day return: 0.74-0.77), intraday path (range so far, below the
intraday high, move since the open: 0.75-0.80). **VIX9D/VIX at entry stands on its own**: it measures the curve's shape, not its level.
Correlation with P/L is small everywhere (credit +0.30, the ratio +0.26, SPX vs SMA20 +0.17, the rest within ±0.16).

| on top of the strategy | sessions | Sharpe | change | total P/L change | both halves |
|---|---|---|---|---|---|
| SMA5 > SMA9 | 164 | 2.55 | -0.04 | -$5.6k | no (-0.12 in 2022-24, +0.14 in 2025-26) |
| condor credit <= $450 | 186 | 2.59 | -0.01 | -$2.8k | no |
| VVIX rank <= 35 | 166 | 2.53 | -0.07 | -$5.8k | no |
| SPX gap >= 0% | 111 | 1.78 | -0.81 | -$36.5k | no |

**Nothing tested improves the finished strategy.**""")

# ---------------------------------------------------------------------------
md(r"""## 13. A/B with everything on: above the 10-day SMA vs the 20-day

No cascade here: **every other setting is on** (no Thursdays, max move 0%, VIX9D/VIX 0.87-1.02, the 7 tranches, ratio <= 4, held to
expiry), and only the trend rule changes: SPX at entry above its 10-day SMA (A, the strategy) against above its 20-day SMA (B), and above
both. Both SMAs are measured the backtester's way (the prior N - 1 closes and the entry price). The Sharpe difference gets a paired session
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
| **A: above the 10-day SMA (strategy)** | 1,450 | **$112.0k** | **2.60** | **2.44 / 2.92** | **-$3.8k** | **87.4%** |
| B: above the 20-day SMA | 1,778 | $105.7k | 1.97 | 1.66 / 2.70 | -$9.1k | 85.4% |
| above both | 1,320 | $102.4k | 2.38 | 2.16 / 2.83 | -$3.8k | 87.1% |

| year | Sharpe A (SMA10) / B (SMA20) | P/L A / B |
|---|---|---|
| 2022 (part) | **2.91** / 1.81 | **$21.1k** / $15.2k |
| 2023 | **2.48** / 2.36 | $26.7k / $26.9k |
| 2024 | **2.08** / 1.18 | **$24.6k** / $22.4k |
| 2025 | **2.69** / 2.46 | $27.9k / **$29.5k** |
| 2026 (part) | 3.59 / 3.57 | $11.7k / $11.7k |

- **On the 7 tranches the 10-day wins on every measure:** higher Sharpe (2.60 against 1.97) in every year, $6.4k more P/L, and well under half
  the drawdown (-$3.8k against -$9.1k).
- **The 20-day's extra trades don't pay:** the 458 trades only the 20-day takes earn $3.3k; the 130 only the 10-day takes earn $9.7k.
- **The 10-day is better in 96% of resamples** (interval -1.33 to +0.07 for SMA20 minus SMA10), just short of a 95% bar.

**Statistical significance.** Two tests, each against the strategy's 10-day SMA on the same 998 sessions (a negative t means worse than the
10-day):

- **t on the Sharpe difference:** the difference over its standard error from the paired session bootstrap, two-sided. It tests the
  question the A/B asks: is one rule's risk-adjusted return higher?
- **t on the daily P/L difference:** a paired t-test on the session-by-session P/L difference, every session (no-trade days as zero),
  two-sided. It tests whether one rule makes more money on an average day, ignoring risk.

| against the 10-day SMA | Sharpe difference | t (Sharpe) | p (Sharpe) | t (daily P/L) | p (daily P/L) |
|---|---|---|---|---|---|
| B: above the 20-day SMA | -0.63 | **-1.73** | 0.08 | -0.54 | 0.59 |
| above both | -0.21 | -1.65 | 0.10 | **-2.16** | **0.03** |

- **SMA10 against SMA20: t = 1.73 on the Sharpe difference (p = 0.08 two-sided, about 0.04 one-sided).** Significant at 10% but not at
  the usual 5% two-sided bar; it says the same as the bootstrap's 96%.
- **The daily P/L barely differs** (t = 0.54, p = 0.59): the two rules make about the same money ($112.0k against $105.7k). **The 10-day's
  edge is risk, not return:** the same money with fewer bad days, hence half the drawdown and the higher Sharpe.
- **Adding the 20-day on top of the 10-day ("above both") makes significantly less money** (t = -2.16, p = 0.03): it drops the 130 trades
  only the 10-day takes, which earn $9.7k, and adds nothing.

**The same A/B on the tranches forward selection by drawdown picks** (Section 7: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53, 15:53),
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
| above both | 1,350 | $95.8k | 2.18 | 2.20 / 2.15 | -$2.6k | -0.65 | 0.52 | -1.40 | 0.16 |

- **The 10-day still leads**, on Sharpe (2.29 against 1.97, ahead in four of five years) and above all on drawdown (-$2.6k against -$7.4k),
  for exactly the same money ($102.8k against $103.1k).
- **But on this tranche set the gap isn't significant** (t = 0.89, p = 0.38; the 10-day is better in 82% of resamples, against 96% on the
  strategy's 7). The 10-day's edge is real in direction and in drawdown, but how big it looks depends on which tranches it's paired with.""")

# ---------------------------------------------------------------------------
md(r"""## 14. The 2026 drawdown, in the backtester's own log

Everything above is rebuilt from the 20-tranche export, which ends 2026-05-07. **The backtester's log of the live strategy** (the 7
tranches, no Thursdays, held to expiry) runs to 2026-09-28, and it shows a drawdown the rest of this notebook can't see: from a peak on
2026-05-11, **-$13.4k by 2026-06-17**, three and a half times anything in the four years before, and **not recovered** by the end of the
log.

First, a check that the log and this notebook describe the same strategy, on the sessions both cover:""")

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

**The rebuild matches the backtester's log almost exactly** (4 trades and $0.7k apart over four years), so the rest of the notebook is the
live strategy. The difference is what came after 2026-05-07.""")

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
- **The likely trigger is scheduled news.** 2026-06-16 and 06-17 were the days of a Fed (FOMC) meeting, with the decision on the 17th; the
  Thursday analysis (Section 3) found the same pattern: the strategy's worst days are days the market moves more than VIX priced, often
  macro days. This data has no economic calendar, so this is a hypothesis: **an event-day filter (no trading on FOMC decision days, and
  possibly CPI days) is the change most worth backtesting.**
- **The tranche study (Section 11) said the drawdowns were shallow:** on the rebuild they were (-$3.8k). The log shows the tail those four
  years didn't contain, and it is the number to size the strategy on.""")

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
| + the 7 tranches | 1,582 | $113.4k | 2.48 | keeps the late morning and early afternoon; drawdown -$15.1k -> -$4.9k |
| **+ short/long <= 4** | **1,450** | **$112.0k** | **2.60** | skips the richest condors the 7 still take |

**The strategy: 1,450 trades, $112.0k, Sharpe 2.60, -$3.8k drawdown, 87.4% of trades winning**, held to expiry; +$599 on an average trading
day (76% of days positive), an average win of $285 a condor against an average loss of $399.

- **No Thursdays isn't a VIX regime:** nothing known at entry differs on Thursdays, but they move more than VIX prices (the realized range
  explains two-thirds of the loss), on a few big days every year; macro releases are the likely cause, untested here.
- **The settings work together.** The band is the worst entry filter on the raw book and adds the most at step 4; with everything on, every
  threshold is at the top of its sweep.
- **The 10-day SMA is the most important setting:** removing it costs 1.28 Sharpe and triples the drawdown. With everything on it beats the
  20-day in every year (t = 1.73 on the Sharpe difference, p = 0.08: significant at 10%, not at 5%); the edge is lower risk, not more
  money. On the drawdown-picked tranches it still leads (2.29 against 1.97, a third of the drawdown) but not significantly (p = 0.38).
- **No stop:** on this strategy every call-stop level lowers Sharpe and P/L and none reduces the drawdown (Section 9). The cost is exposure to
  a day worse than any in this sample.
- **The 7 tranches are the smallest, steadiest set:** the smallest drawdown and the most P/L per $ of drawdown of the sets tested. Adding
  11:13 and 13:33 (forward selection's peak) scores a higher Sharpe (2.73) for $29.8k more and a deeper drawdown; the previous 15, 2.69 for
  $68.4k more. None of the 7 should be cut: each made money every year and each one's removal lowers Sharpe.

- **The backtester's log shows a 2026 drawdown the export doesn't reach:** -$13.4k from 2026-05-11 to 06-17, not recovered by 2026-09-28,
  from three days on which all seven tranches lost a full wing (Section 14). The rebuild matches the log to 2026-05-07; size the strategy on
  that tail, not on the -$3.8k of 2022-2026, and backtest an event-day (FOMC/CPI) filter.

**What this notebook doesn't establish:** out-of-sample performance (apart from the walk-forward checks); the backtester's totals on these
settings (the rules were matched trade by trade against a backtest on earlier settings, with the 20-day SMA; the 10-day and the 7 tranches
haven't been checked against a backtest log); the first sessions, which need SPX closes before the export; and a put stop, which is untested.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "py313-quant", "language": "python", "name": "py313-quant"}
with open("ic_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote ic_analysis.ipynb with {len(cells)} cells.")
