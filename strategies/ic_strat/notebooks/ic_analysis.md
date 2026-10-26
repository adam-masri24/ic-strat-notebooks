# SPX 0DTE Iron Condor, 10-wide, 20 tranches: the strategy, built one setting at a time

**The trade.** An SPX iron condor expiring the same day: a short call spread and a short put spread, each 10 points wide, opened
every 20 minutes (the export has 20 entries a day from 09:32) and held to the close. It makes money when SPX stays between the
short strikes until 16:00. Inside the strategy each condor collects about **\$326** (median \$325: \$175 call side, \$145 put side), with
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
| | always | 0DTE, exact strike offsets, 10-wide wings; 1 contract per trade; \$0.97 per contract to open and to close, \$0.05 entry slippage |

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

**How to read the numbers.**

- **The baseline is the raw export**: every condor on every session, held to expiry. It loses money: **-\$7.5k at Sharpe -0.12**.
- **One trade = one vertical spread**, as the backtester counts (a condor is two trades). **P/L** is per condor unless it says per
  session.
- **Every version is scored on all 998 sessions** (2022-05 to 2026-05); days it doesn't trade count as zero.
- **Sharpe is the backtester's**: daily P/L over the \$200k starting capital, less a 4% risk-free rate, annualised. Because of the
  risk-free rate, **a version that trades less pays the hurdle on more idle days**, so cutting trades lowers Sharpe unless the trades
  cut were genuinely bad.
- **Every rule is measured the backtester's way** (VIX9D/VIX at entry, the condor's short/long ratio, SPX at entry against its
  SMA) and **every condor is held to expiry**, as the strategy trades it. The call stop in Section 9 is replayed from each spread's
  worst intraday loss (never on the settlement print), filled where the SPXW option chain crossed it.
- **2022-2024 vs 2025-2026**: each version is also scored on the two halves, as a stability check.
- **Everything is in-sample**, apart from the walk-forward checks in Sections 7 and 11. Intervals are 95% session-bootstrap intervals.

## 1. Setup and data

| file | what it is |
|---|---|
| `data/strategy/ic_strat/spx/raw_baseline/IC-20-Tranche-10-Baseline.csv` | the Option Omega export (`ic 20 tranch 10 spread.csv`), one row per vertical spread, every spread held to expiry, all 20 entry times. A call and a put row with the same date and time make one condor (one unmatched leg is dropped). Each row carries its legs and prices, `Max Loss` (its worst intraday loss), the 16:00 `Closing Price` and the opening gap. |
| `data/market/vix/ohlc/VIX9D_<date>_1m.parquet`, `VIX_<date>_1m.parquet` | 1-minute VIX9D and VIX bars: the term structure at each entry minute |
| `data/market/spx/options_ohlc/SPXW_<date>_ohlc.parquet` | the SPXW 0DTE option chain, 1-minute prices, 739 of the 998 sessions: where a call stop would have filled (Section 9) |
| `data/market/vix/term_structure_daily.csv`, `data/market/vvix/vvix_daily_cboe.csv` | daily VIX family and VVIX, for the features in Section 12 |

None of these are in git (`data/` is git-ignored; see "Data" in the root README). The 10-day SMA needs 9 closes before each session,
so the export's first 9 sessions (to 2022-05-26) can't pass the SMA step.

    19,847 condors over 998 sessions, 2022-05-16 to 2026-05-07
    greedy order: thursday -> move -> sma10 -> band -> tranches -> ratio (matches the sections below: True)
    

## 2. The greedy search, and the cascade it builds

**The search.** Each row is one step. Every setting not yet added is tried on the book built so far; the cell is the Sharpe the book would
have with it. The highest (highlighted) is added, and the next row starts from that book.


<table id="T_01dda">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_01dda_level0_col0" class="col_heading level0 col0" >picked</th>
      <th id="T_01dda_level0_col1" class="col_heading level0 col1" >VIX9D/VIX 0.87-1.02 at entry</th>
      <th id="T_01dda_level0_col2" class="col_heading level0 col2" >max move up 0%</th>
      <th id="T_01dda_level0_col3" class="col_heading level0 col3" >above the 10-day SMA</th>
      <th id="T_01dda_level0_col4" class="col_heading level0 col4" >short/long ratio <= 4</th>
      <th id="T_01dda_level0_col5" class="col_heading level0 col5" >no Thursdays</th>
      <th id="T_01dda_level0_col6" class="col_heading level0 col6" >7 forward-selected tranches</th>
    </tr>
    <tr>
      <th class="index_name level0" >step</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_01dda_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_01dda_row0_col0" class="data row0 col0" >no Thursdays</td>
      <td id="T_01dda_row0_col1" class="data row0 col1" >-0.11</td>
      <td id="T_01dda_row0_col2" class="data row0 col2" >0.37</td>
      <td id="T_01dda_row0_col3" class="data row0 col3" >0.35</td>
      <td id="T_01dda_row0_col4" class="data row0 col4" >-0.00</td>
      <td id="T_01dda_row0_col5" class="data row0 col5" >0.44</td>
      <td id="T_01dda_row0_col6" class="data row0 col6" >0.12</td>
    </tr>
    <tr>
      <th id="T_01dda_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_01dda_row1_col0" class="data row1 col0" >max move up 0%</td>
      <td id="T_01dda_row1_col1" class="data row1 col1" >0.66</td>
      <td id="T_01dda_row1_col2" class="data row1 col2" >0.97</td>
      <td id="T_01dda_row1_col3" class="data row1 col3" >0.71</td>
      <td id="T_01dda_row1_col4" class="data row1 col4" >0.59</td>
      <td id="T_01dda_row1_col5" class="data row1 col5" ></td>
      <td id="T_01dda_row1_col6" class="data row1 col6" >0.56</td>
    </tr>
    <tr>
      <th id="T_01dda_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_01dda_row2_col0" class="data row2 col0" >above the 10-day SMA</td>
      <td id="T_01dda_row2_col1" class="data row2 col1" >1.39</td>
      <td id="T_01dda_row2_col2" class="data row2 col2" ></td>
      <td id="T_01dda_row2_col3" class="data row2 col3" >1.50</td>
      <td id="T_01dda_row2_col4" class="data row2 col4" >1.15</td>
      <td id="T_01dda_row2_col5" class="data row2 col5" ></td>
      <td id="T_01dda_row2_col6" class="data row2 col6" >1.03</td>
    </tr>
    <tr>
      <th id="T_01dda_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_01dda_row3_col0" class="data row3 col0" >VIX9D/VIX 0.87-1.02 at entry</td>
      <td id="T_01dda_row3_col1" class="data row3 col1" >2.25</td>
      <td id="T_01dda_row3_col2" class="data row3 col2" ></td>
      <td id="T_01dda_row3_col3" class="data row3 col3" ></td>
      <td id="T_01dda_row3_col4" class="data row3 col4" >1.74</td>
      <td id="T_01dda_row3_col5" class="data row3 col5" ></td>
      <td id="T_01dda_row3_col6" class="data row3 col6" >1.66</td>
    </tr>
    <tr>
      <th id="T_01dda_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_01dda_row4_col0" class="data row4 col0" >7 forward-selected tranches</td>
      <td id="T_01dda_row4_col1" class="data row4 col1" ></td>
      <td id="T_01dda_row4_col2" class="data row4 col2" ></td>
      <td id="T_01dda_row4_col3" class="data row4 col3" ></td>
      <td id="T_01dda_row4_col4" class="data row4 col4" >2.40</td>
      <td id="T_01dda_row4_col5" class="data row4 col5" ></td>
      <td id="T_01dda_row4_col6" class="data row4 col6" >2.48</td>
    </tr>
    <tr>
      <th id="T_01dda_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_01dda_row5_col0" class="data row5 col0" >short/long ratio <= 4</td>
      <td id="T_01dda_row5_col1" class="data row5 col1" ></td>
      <td id="T_01dda_row5_col2" class="data row5 col2" ></td>
      <td id="T_01dda_row5_col3" class="data row5 col3" ></td>
      <td id="T_01dda_row5_col4" class="data row5 col4" >2.60</td>
      <td id="T_01dda_row5_col5" class="data row5 col5" ></td>
      <td id="T_01dda_row5_col6" class="data row5 col6" ></td>
    </tr>
  </tbody>
</table>



| step | settings tried: Sharpe if added | added |
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

**The cascade it builds.** Each bar is a full version scored on all 998 sessions.


    
![png](ic_analysis_files/output_6_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trades</th>
      <th>sessions</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>win_rate_trade</th>
      <th>win_rate_condor</th>
      <th>avg_pl_condor</th>
      <th>trades_change</th>
      <th>total_pl_change</th>
      <th>sharpe_change</th>
    </tr>
    <tr>
      <th>step</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>raw book</th>
      <td>39694.0</td>
      <td>998.0</td>
      <td>-7464.36</td>
      <td>-0.12</td>
      <td>-0.09</td>
      <td>-0.20</td>
      <td>-113337.60</td>
      <td>79.47</td>
      <td>66.09</td>
      <td>-0.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1. + no Thursdays</th>
      <td>31768.0</td>
      <td>799.0</td>
      <td>155558.08</td>
      <td>0.44</td>
      <td>0.58</td>
      <td>0.13</td>
      <td>-67902.08</td>
      <td>79.86</td>
      <td>67.04</td>
      <td>9.79</td>
      <td>-7926.0</td>
      <td>163022.44</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>2. + max move up 0%</th>
      <td>14416.0</td>
      <td>656.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>1.05</td>
      <td>0.80</td>
      <td>-33625.72</td>
      <td>81.65</td>
      <td>69.71</td>
      <td>27.18</td>
      <td>-17352.0</td>
      <td>40348.88</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>3. + above the 10-day SMA</th>
      <td>7362.0</td>
      <td>404.0</td>
      <td>205624.72</td>
      <td>1.50</td>
      <td>1.54</td>
      <td>1.43</td>
      <td>-22504.00</td>
      <td>81.87</td>
      <td>70.80</td>
      <td>55.86</td>
      <td>-7054.0</td>
      <td>9717.76</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>4. + VIX9D/VIX 0.87-1.02 at entry</th>
      <td>4630.0</td>
      <td>294.0</td>
      <td>213278.80</td>
      <td>2.25</td>
      <td>2.16</td>
      <td>2.48</td>
      <td>-15130.04</td>
      <td>83.39</td>
      <td>73.95</td>
      <td>92.13</td>
      <td>-2732.0</td>
      <td>7654.08</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>5. + 7 forward-selected tranches</th>
      <td>1582.0</td>
      <td>188.0</td>
      <td>113370.92</td>
      <td>2.48</td>
      <td>2.33</td>
      <td>2.83</td>
      <td>-4888.92</td>
      <td>85.65</td>
      <td>79.14</td>
      <td>143.33</td>
      <td>-3048.0</td>
      <td>-99907.88</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>6. + short/long ratio &lt;= 4</th>
      <td>1450.0</td>
      <td>187.0</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>-3808.68</td>
      <td>87.38</td>
      <td>80.97</td>
      <td>154.54</td>
      <td>-132.0</td>
      <td>-1326.92</td>
      <td>0.12</td>
    </tr>
  </tbody>
</table>
</div>


| step | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | win per trade |
|---|---|---|---|---|---|---|
| raw book | 39,694 | -\$7.5k | -0.12 | -0.09 / -0.20 | -\$113.3k | 79.5% |
| 1. + no Thursdays | 31,768 | \$155.6k | 0.44 | 0.58 / 0.13 | -\$67.9k | 79.9% |
| 2. + max move up 0% | 14,416 | \$195.9k | 0.97 | 1.05 / 0.80 | -\$33.6k | 81.7% |
| 3. + above the 10-day SMA | 7,362 | \$205.6k | 1.50 | 1.54 / 1.43 | -\$22.5k | 81.9% |
| 4. + VIX9D/VIX 0.87-1.02 | 4,630 | **\$213.3k** | 2.25 | 2.16 / 2.48 | -\$15.1k | 83.4% |
| 5. + 7 forward-selected tranches | 1,582 | \$113.4k | 2.48 | 2.33 / 2.83 | -\$4.9k | 85.7% |
| **6. + short/long ratio <= 4** | **1,450** | **\$112.0k** | **2.60** | **2.44 / 2.92** | **-\$3.8k** | **87.4%** |

**The raw book is a coin flip that loses slowly:** a condor wins 66% of the time, but an average win (+\$254) is half an average loss
(-\$497), so the book drifts to -\$7.5k with a -\$113k drawdown.

- **The first four steps build the money:** no Thursdays alone turns -\$7.5k into +\$155.6k; max move, the 10-day SMA and the band each add
  +0.5 to +0.75 Sharpe, and P/L peaks at \$213.3k with the band.
- **The 7 tranches and the ratio cap trade money for steadiness:** they cost \$101.2k between them but take Sharpe from 2.25 to 2.60 and the
  drawdown from -\$15.1k to -\$3.8k.

**The strategy: 1,450 trades, \$112.0k, Sharpe 2.60, -\$3.8k drawdown, 87.4% of trades winning**, held to expiry; +\$599 on an average trading
day (76% of days positive), an average win of \$285 a condor against an average loss of \$399.

## 3. Step 1: no Thursdays, on the raw book

Each step's section shows the step's setting on **the book built so far**; here, the raw book (every condor, held to expiry).


    
![png](ic_analysis_files/output_9_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>avg_session_pl</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>winning_sessions_pct</th>
      <th>total_pl</th>
    </tr>
    <tr>
      <th>weekday</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monday</th>
      <td>185</td>
      <td>7356</td>
      <td>191.0</td>
      <td>-498.8</td>
      <td>871.9</td>
      <td>63.8</td>
      <td>35336.4</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>207</td>
      <td>8262</td>
      <td>51.3</td>
      <td>-619.7</td>
      <td>710.4</td>
      <td>60.4</td>
      <td>10627.7</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>205</td>
      <td>8154</td>
      <td>-146.7</td>
      <td>-880.1</td>
      <td>549.8</td>
      <td>59.5</td>
      <td>-30074.8</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>199</td>
      <td>7926</td>
      <td>-819.2</td>
      <td>-1514.5</td>
      <td>-98.0</td>
      <td>52.8</td>
      <td>-163022.4</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>202</td>
      <td>7996</td>
      <td>691.4</td>
      <td>-0.6</td>
      <td>1389.1</td>
      <td>68.3</td>
      <td>139668.8</td>
    </tr>
  </tbody>
</table>
</div>


| weekday (raw book) | sessions | trades | avg P/L per session | 95% interval | winning sessions | total P/L |
|---|---|---|---|---|---|---|
| Monday | 185 | 7,356 | +\$191 | -\$499 to +\$872 | 64% | +\$35.3k |
| Tuesday | 207 | 8,262 | +\$51 | -\$620 to +\$710 | 60% | +\$10.6k |
| Wednesday | 205 | 8,154 | -\$147 | -\$880 to +\$550 | 60% | -\$30.1k |
| **Thursday** | 199 | 7,926 | **-\$819** | **-\$1,515 to -\$98** | **53%** | **-\$163.0k** |
| Friday | 202 | 7,996 | **+\$691** | -\$1 to +\$1,389 | 68% | +\$139.7k |

- **Thursday is where the raw book loses its money:** -\$163.0k over 199 sessions, the only weekday whose interval sits wholly below zero.
  Without Thursdays the raw book goes from -\$7.5k to **+\$155.6k** (Sharpe -0.12 -> 0.44).
- **The other days differ too** (Friday +\$691 a session, Wednesday -\$147), but only Thursday is clearly negative.

### What drives the Thursday loss?

Three questions, first on the raw book (where the greedy search drops Thursdays) and then on the finished strategy with Thursdays allowed
back in:

1. **Are Thursday markets different when the condor is opened?** VIX, VIX9D/VIX, the trend, the move since the open, the gap.
2. **Or does the market move differently after entry?** Each side's P/L, the session's range, and the range against the move VIX implied
   at the open (VIX / sqrt(252)): above the other days means Thursdays move more than VIX priced.
3. **Is it a few days or every Thursday?** The worst sessions' share, and Thursday by year.


    
![png](ic_analysis_files/output_12_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>weekday</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sessions</th>
      <td>185.00</td>
      <td>207.00</td>
      <td>205.00</td>
      <td>199.00</td>
      <td>202.00</td>
    </tr>
    <tr>
      <th>trades</th>
      <td>7356.00</td>
      <td>8262.00</td>
      <td>8154.00</td>
      <td>7926.00</td>
      <td>7996.00</td>
    </tr>
    <tr>
      <th>avg_pl</th>
      <td>9.61</td>
      <td>2.57</td>
      <td>-7.38</td>
      <td>-41.14</td>
      <td>34.93</td>
    </tr>
    <tr>
      <th>avg_pl_call</th>
      <td>-2.95</td>
      <td>-16.33</td>
      <td>-2.39</td>
      <td>-10.16</td>
      <td>11.94</td>
    </tr>
    <tr>
      <th>avg_pl_put</th>
      <td>12.56</td>
      <td>18.90</td>
      <td>-4.98</td>
      <td>-30.98</td>
      <td>22.99</td>
    </tr>
    <tr>
      <th>call_breached_pct</th>
      <td>28.96</td>
      <td>30.48</td>
      <td>25.97</td>
      <td>28.44</td>
      <td>24.94</td>
    </tr>
    <tr>
      <th>put_breached_pct</th>
      <td>22.10</td>
      <td>20.89</td>
      <td>21.17</td>
      <td>23.74</td>
      <td>19.06</td>
    </tr>
    <tr>
      <th>vix_open</th>
      <td>19.39</td>
      <td>18.96</td>
      <td>18.84</td>
      <td>18.74</td>
      <td>18.73</td>
    </tr>
    <tr>
      <th>ts_9d_30_entry</th>
      <td>0.98</td>
      <td>0.97</td>
      <td>0.96</td>
      <td>0.95</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>above_sma10_pct</th>
      <td>66.56</td>
      <td>62.91</td>
      <td>61.98</td>
      <td>62.05</td>
      <td>59.20</td>
    </tr>
    <tr>
      <th>move_since_open</th>
      <td>0.11</td>
      <td>-0.02</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>abs_gap</th>
      <td>0.38</td>
      <td>0.34</td>
      <td>0.33</td>
      <td>0.41</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>day_range_pct</th>
      <td>0.80</td>
      <td>0.90</td>
      <td>0.98</td>
      <td>1.01</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>big_range_days_pct</th>
      <td>26.49</td>
      <td>31.40</td>
      <td>31.71</td>
      <td>43.22</td>
      <td>37.13</td>
    </tr>
    <tr>
      <th>abs_move_after_entry</th>
      <td>0.29</td>
      <td>0.31</td>
      <td>0.42</td>
      <td>0.39</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>range_vs_implied</th>
      <td>0.64</td>
      <td>0.72</td>
      <td>0.78</td>
      <td>0.84</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>35336.36</td>
      <td>10627.72</td>
      <td>-30074.76</td>
      <td>-163022.44</td>
      <td>139668.76</td>
    </tr>
  </tbody>
</table>
</div>


| raw book, per weekday | Mon | Tue | Wed | **Thu** | Fri |
|---|---|---|---|---|---|
| *known at entry* | | | | | |
| VIX at the open | 19.4 | 19.0 | 18.8 | **18.7** | 18.7 |
| VIX9D/VIX at entry | 0.98 | 0.97 | 0.96 | **0.95** | 0.93 |
| SPX above its SMA10 | 67% | 63% | 62% | **62%** | 59% |
| SPX move since the open at entry | +0.11% | -0.02% | +0.04% | **-0.04%** | +0.02% |
| abs. opening gap | 0.38% | 0.34% | 0.33% | **0.41%** | 0.34% |
| *what happened after* | | | | | |
| avg P/L per condor: call / put side | -\$3 / +\$13 | -\$16 / +\$19 | -\$2 / -\$5 | **-\$10 / -\$31** | +\$12 / +\$23 |
| SPX through the short put | 22% | 21% | 21% | **24%** | 19% |
| session range | 0.80% | 0.90% | 0.98% | **1.01%** | 0.97% |
| sessions with a range above 1% | 26% | 31% | 32% | **43%** | 37% |
| range / VIX-implied move | 0.64 | 0.72 | 0.78 | **0.84** | 0.81 |

- **Nothing known at entry sets Thursday apart.** VIX is the same (18.7), the curve is ordinary (0.95), the trend and the move into the entry
  are ordinary. Only the opening gap is a little larger (0.41%).
- **What differs is what happens after:** Thursday has the widest sessions (1.01% on average, and 43% of Thursdays move more than 1% against
  26-37% on other days), the most movement relative to what VIX priced (0.84), and **the put side breaks** (-\$31 a condor, the short put
  breached on 24%).


    
![png](ic_analysis_files/output_14_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>thursday_trades</th>
      <th>thursday_avg_pl</th>
      <th>other_days_avg_pl</th>
      <th>difference</th>
      <th>thursday_range_vs_implied</th>
      <th>other_range_vs_implied</th>
    </tr>
    <tr>
      <th>regime</th>
      <th>bucket</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">VIX at the open</th>
      <th>&lt;= 15</th>
      <td>1840</td>
      <td>-73.08</td>
      <td>0.26</td>
      <td>-73.35</td>
      <td>0.81</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>15 to 18</th>
      <td>2462</td>
      <td>-51.56</td>
      <td>21.05</td>
      <td>-72.61</td>
      <td>0.81</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>18 to 22</th>
      <td>1952</td>
      <td>-8.11</td>
      <td>-17.40</td>
      <td>9.29</td>
      <td>0.87</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>&gt; 22</th>
      <td>1672</td>
      <td>-29.19</td>
      <td>33.52</td>
      <td>-62.71</td>
      <td>0.87</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">VIX9D/VIX at entry</th>
      <th>&lt;= 0.9</th>
      <td>2038</td>
      <td>-35.80</td>
      <td>-15.90</td>
      <td>-19.91</td>
      <td>0.77</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>0.9 to 0.95</th>
      <td>2076</td>
      <td>-66.90</td>
      <td>21.21</td>
      <td>-88.11</td>
      <td>0.87</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>0.95 to 1</th>
      <td>1718</td>
      <td>-34.90</td>
      <td>28.81</td>
      <td>-63.71</td>
      <td>0.99</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>&gt; 1</th>
      <td>2094</td>
      <td>-25.91</td>
      <td>8.68</td>
      <td>-34.59</td>
      <td>0.98</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">SPX vs its SMA10, %</th>
      <th>&lt;= -1</th>
      <td>1482</td>
      <td>-73.56</td>
      <td>8.19</td>
      <td>-81.76</td>
      <td>1.07</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>-1 to 0</th>
      <td>1450</td>
      <td>-60.42</td>
      <td>-19.45</td>
      <td>-40.97</td>
      <td>1.00</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>0 to 1</th>
      <td>2480</td>
      <td>-37.85</td>
      <td>5.40</td>
      <td>-43.26</td>
      <td>0.90</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>&gt; 1</th>
      <td>2438</td>
      <td>-20.59</td>
      <td>32.54</td>
      <td>-53.13</td>
      <td>0.78</td>
      <td>0.69</td>
    </tr>
  </tbody>
</table>
</div>


    condor P/L on a Thursday dummy ($ per condor; standard errors clustered by session):
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>thursday_effect</th>
      <th>se</th>
      <th>t</th>
      <th>p</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thursday alone</th>
      <td>19847</td>
      <td>-50.930</td>
      <td>20.566</td>
      <td>-2.476</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>+ every entry feature</th>
      <td>19650</td>
      <td>-54.568</td>
      <td>20.639</td>
      <td>-2.644</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>+ the session's range vs implied (an outcome)</th>
      <td>19650</td>
      <td>-19.234</td>
      <td>17.937</td>
      <td>-1.072</td>
      <td>0.284</td>
    </tr>
  </tbody>
</table>
</div>


**Is it the VIX regime?** No. Inside every VIX, VIX9D/VIX and trend bucket, Thursday is worse than the other days, by \$20-88 a condor,
with one exception (VIX 18-22, where both lose). And in every bucket, Thursday's range against the VIX-implied move is higher (for VIX
under 15: 0.81 against 0.66).

**The regression says the same, with a test:**

| condor P/L on a Thursday dummy (raw book) | Thursday effect, \$ per condor | t | p |
|---|---|---|---|
| Thursday alone | **-\$51** | -2.48 | 0.013 |
| + every feature known at entry (VIX, VIX9D/VIX, move since open, SMA10, gap, credit, entry time) | **-\$55** | -2.64 | 0.008 |
| + the session's range vs implied (an outcome) | -\$19 | -1.07 | 0.28 |

- **Controlling for everything known at entry doesn't shrink the Thursday effect** (-\$51 -> -\$55): it isn't the VIX regime, the curve, the
  trend or the morning's move.
- **Controlling for how much the market actually moved removes about two-thirds of it** (-\$55 -> -\$19, no longer significant): the Thursday
  loss is Thursdays moving more than VIX priced at the open.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>median_session_pl</th>
      <th>mean_session_pl</th>
      <th>winning_sessions_pct</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thursday</th>
      <td>460.4</td>
      <td>-819.2</td>
      <td>52.8</td>
    </tr>
    <tr>
      <th>other days</th>
      <td>1870.4</td>
      <td>194.7</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl</th>
      <th>thursday_total</th>
      <th>rest_of_thursdays</th>
    </tr>
    <tr>
      <th>worst_n</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>-67438.0</td>
      <td>-163022.0</td>
      <td>-95584.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-125433.0</td>
      <td>-163022.0</td>
      <td>-37589.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-227342.0</td>
      <td>-163022.0</td>
      <td>64320.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>day</th>
      <th>Thursday</th>
      <th>other days</th>
      <th>thursday_sessions</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>-490.0</td>
      <td>561.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>-1841.0</td>
      <td>429.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>-453.0</td>
      <td>-115.0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>-239.0</td>
      <td>107.0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>-1075.0</td>
      <td>9.0</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-02-27</th>
      <td>-14979.0</td>
    </tr>
    <tr>
      <th>2022-06-09</th>
      <td>-14283.0</td>
    </tr>
    <tr>
      <th>2022-11-10</th>
      <td>-13399.0</td>
    </tr>
    <tr>
      <th>2023-08-24</th>
      <td>-12486.0</td>
    </tr>
    <tr>
      <th>2025-01-23</th>
      <td>-12293.0</td>
    </tr>
    <tr>
      <th>2025-09-04</th>
      <td>-11886.0</td>
    </tr>
    <tr>
      <th>2023-09-21</th>
      <td>-11803.0</td>
    </tr>
    <tr>
      <th>2023-03-09</th>
      <td>-11549.0</td>
    </tr>
    <tr>
      <th>2026-01-29</th>
      <td>-11418.0</td>
    </tr>
    <tr>
      <th>2022-08-25</th>
      <td>-11341.0</td>
    </tr>
  </tbody>
</table>
</div>


**A few big days, but every year.**

- **The ten worst Thursdays lose \$125.4k of the \$163.0k:** the median Thursday session still makes money (+\$460, against +\$1,870 on other
  days), and 53% of Thursdays win (63% on other days).
- **It repeats every year:** the average Thursday session loses in all five years (-\$239 to -\$1,841), while the other days make money in
  every year but 2024.
- **The worst Thursdays include known macro days:** 2022-11-10 was a CPI release (SPX +5.5%), 2024-01-11 a CPI release, and 2023-07-27 and
  2023-09-21 the day after a Fed decision. Thursday also carries the weekly 8:30 jobless claims. **This data has no economic calendar**, so
  the macro link is a hypothesis to test, not a finding: what the data shows is that Thursdays realise more movement than VIX prices, on a
  handful of days, every year.

**With the other filters on.** The finished strategy with Thursdays allowed back in: 412 Thursday trades, -\$4.5k between them.


    
![png](ic_analysis_files/output_18_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>weekday</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sessions</th>
      <td>47.00</td>
      <td>53.00</td>
      <td>48.00</td>
      <td>50.00</td>
      <td>39.00</td>
    </tr>
    <tr>
      <th>trades</th>
      <td>352.00</td>
      <td>436.00</td>
      <td>374.00</td>
      <td>412.00</td>
      <td>288.00</td>
    </tr>
    <tr>
      <th>avg_pl</th>
      <td>173.93</td>
      <td>121.85</td>
      <td>143.22</td>
      <td>-21.91</td>
      <td>195.04</td>
    </tr>
    <tr>
      <th>avg_pl_call</th>
      <td>100.06</td>
      <td>61.18</td>
      <td>85.31</td>
      <td>-1.09</td>
      <td>110.52</td>
    </tr>
    <tr>
      <th>avg_pl_put</th>
      <td>73.87</td>
      <td>60.67</td>
      <td>57.90</td>
      <td>-20.82</td>
      <td>84.53</td>
    </tr>
    <tr>
      <th>call_breached_pct</th>
      <td>19.32</td>
      <td>22.48</td>
      <td>15.51</td>
      <td>27.18</td>
      <td>15.28</td>
    </tr>
    <tr>
      <th>put_breached_pct</th>
      <td>14.20</td>
      <td>15.60</td>
      <td>13.37</td>
      <td>22.33</td>
      <td>14.58</td>
    </tr>
    <tr>
      <th>vix_open</th>
      <td>17.35</td>
      <td>17.45</td>
      <td>17.79</td>
      <td>16.47</td>
      <td>16.72</td>
    </tr>
    <tr>
      <th>ts_9d_30_entry</th>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.95</td>
      <td>0.93</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>above_sma10_pct</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>move_since_open</th>
      <td>-0.23</td>
      <td>-0.33</td>
      <td>-0.29</td>
      <td>-0.38</td>
      <td>-0.33</td>
    </tr>
    <tr>
      <th>abs_gap</th>
      <td>0.24</td>
      <td>0.28</td>
      <td>0.25</td>
      <td>0.31</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>day_range_pct</th>
      <td>0.59</td>
      <td>0.67</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>big_range_days_pct</th>
      <td>4.26</td>
      <td>11.32</td>
      <td>22.92</td>
      <td>32.00</td>
      <td>25.64</td>
    </tr>
    <tr>
      <th>abs_move_after_entry</th>
      <td>0.17</td>
      <td>0.19</td>
      <td>0.25</td>
      <td>0.30</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>range_vs_implied</th>
      <td>0.54</td>
      <td>0.58</td>
      <td>0.72</td>
      <td>0.80</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>30612.12</td>
      <td>26564.16</td>
      <td>26781.44</td>
      <td>-4513.28</td>
      <td>28086.28</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](ic_analysis_files/output_18_2.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>thursday_trades</th>
      <th>thursday_avg_pl</th>
      <th>other_days_avg_pl</th>
      <th>difference</th>
      <th>thursday_range_vs_implied</th>
      <th>other_range_vs_implied</th>
    </tr>
    <tr>
      <th>regime</th>
      <th>bucket</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">VIX at the open</th>
      <th>&lt;= 15</th>
      <td>148</td>
      <td>-150.24</td>
      <td>132.92</td>
      <td>-283.17</td>
      <td>0.91</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>15 to 18</th>
      <td>162</td>
      <td>-64.35</td>
      <td>188.63</td>
      <td>-252.98</td>
      <td>0.78</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>18 to 22</th>
      <td>72</td>
      <td>212.51</td>
      <td>95.54</td>
      <td>116.97</td>
      <td>0.72</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>&gt; 22</th>
      <td>30</td>
      <td>277.79</td>
      <td>188.43</td>
      <td>89.35</td>
      <td>0.56</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">VIX9D/VIX at entry</th>
      <th>&lt;= 0.9</th>
      <td>92</td>
      <td>-86.86</td>
      <td>175.13</td>
      <td>-261.99</td>
      <td>0.79</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>0.9 to 0.95</th>
      <td>204</td>
      <td>-7.13</td>
      <td>154.37</td>
      <td>-161.50</td>
      <td>0.84</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>0.95 to 1</th>
      <td>106</td>
      <td>24.35</td>
      <td>139.80</td>
      <td>-115.45</td>
      <td>0.84</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>&gt; 1</th>
      <td>10</td>
      <td>-216.08</td>
      <td>169.18</td>
      <td>-385.26</td>
      <td>0.97</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">SPX vs its SMA10, %</th>
      <th>0 to 1</th>
      <td>220</td>
      <td>-111.16</td>
      <td>138.60</td>
      <td>-249.76</td>
      <td>0.89</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>&gt; 1</th>
      <td>192</td>
      <td>80.36</td>
      <td>177.01</td>
      <td>-96.65</td>
      <td>0.71</td>
      <td>0.64</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>thursday_effect</th>
      <th>se</th>
      <th>t</th>
      <th>p</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thursday alone</th>
      <td>931</td>
      <td>-176.453</td>
      <td>53.185</td>
      <td>-3.318</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>+ every entry feature</th>
      <td>931</td>
      <td>-181.069</td>
      <td>51.125</td>
      <td>-3.542</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>+ the session's range vs implied (an outcome)</th>
      <td>931</td>
      <td>-122.292</td>
      <td>44.091</td>
      <td>-2.774</td>
      <td>0.006</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>median_session_pl</th>
      <th>mean_session_pl</th>
      <th>winning_sessions_pct</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thursday</th>
      <td>201.7</td>
      <td>-90.3</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>other days</th>
      <td>587.2</td>
      <td>599.2</td>
      <td>76.5</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl</th>
      <th>thursday_total</th>
      <th>rest_of_thursdays</th>
    </tr>
    <tr>
      <th>worst_n</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>-14918.0</td>
      <td>-4513.0</td>
      <td>10404.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-24387.0</td>
      <td>-4513.0</td>
      <td>19874.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-31196.0</td>
      <td>-4513.0</td>
      <td>26683.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>day</th>
      <th>Thursday</th>
      <th>other days</th>
      <th>thursday_sessions</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>1707.0</td>
      <td>843.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>-772.0</td>
      <td>594.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>157.0</td>
      <td>464.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>-197.0</td>
      <td>517.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>-229.0</td>
      <td>1173.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


| strategy with Thursdays, per weekday | Mon | Tue | Wed | **Thu** | Fri |
|---|---|---|---|---|---|
| avg P/L per condor | +\$174 | +\$122 | +\$143 | **-\$22** | +\$195 |
| call / put side | +\$100 / +\$74 | +\$61 / +\$61 | +\$85 / +\$58 | **-\$1 / -\$21** | +\$111 / +\$85 |
| short call / short put breached | 19% / 14% | 22% / 16% | 16% / 13% | **27% / 22%** | 15% / 15% |
| VIX at the open | 17.4 | 17.5 | 17.8 | **16.5** | 16.7 |
| sessions with a range above 1% | 4% | 11% | 23% | **32%** | 26% |
| range / VIX-implied move | 0.54 | 0.58 | 0.72 | **0.80** | 0.72 |

| condor P/L on a Thursday dummy (strategy with Thursdays) | Thursday effect, \$ per condor | t | p |
|---|---|---|---|
| Thursday alone | **-\$176** | -3.32 | 0.001 |
| + every feature known at entry | **-\$181** | -3.54 | <0.001 |
| + the session's range vs implied (an outcome) | -\$122 | -2.77 | 0.006 |

- **The same story inside the strategy:** entry conditions are ordinary (VIX is in fact the lowest, 16.5), but Thursdays move more than VIX
  priced (0.80 against 0.54-0.72), both short strikes break more often, and the entry features explain none of the gap.
- **The realized move explains about a third of it here** (-\$181 -> -\$122, still significant): even for the same range, Thursday condors do
  worse, which fits an intraday shock (a release or a reversal) more than a steady trend.
- **It is a low-VIX problem:** Thursdays lose with VIX under 18 (-\$150 and -\$64 a condor, against +\$133 and +\$189 on other days), where VIX
  most underprices them (range / implied 0.91 and 0.78 against 0.66 and 0.56); above 18 the few Thursday trades (102) earn.
- **Five bad Thursdays sink the rest:** the worst five lose \$14.9k, more than all Thursdays together (-\$4.5k); the other Thursdays make
  +\$10.4k. The median Thursday session still wins (+\$202), but only 56% of them do, against 77% of other days.

**Why no Thursdays, in one line:** not a VIX regime and nothing visible at entry, but Thursdays realise more movement than VIX prices at the
open, mostly on a few big days each year (macro releases are the likely cause, untested here), and a short-volatility condor has no defence
against that. Skipping the day is the only filter that removes it.

## 4. Step 2: max move up 0%, on the book after step 1

SPX's move since the 09:30 open at the moment the condor enters. Four panels:

1. **avg P/L per condor**, with its interval;
2. **which side earns or loses**: the call spread's P/L against the put spread's (held to expiry);
3. **the mechanism**: what the market did after entry;
4. **total P/L** in each group.

The shaded groups are the ones the setting keeps.


    
![png](ic_analysis_files/output_21_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>trades</th>
      <th>sessions</th>
      <th>avg_pl</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>win_rate</th>
      <th>avg_pl_to_2024</th>
      <th>avg_pl_from_2025</th>
      <th>total_pl</th>
      <th>avg_pl_call</th>
      <th>avg_pl_put</th>
      <th>call_breached_pct</th>
      <th>put_breached_pct</th>
      <th>abs_move_after_entry_pct</th>
      <th>day_range_pct</th>
      <th>in_strategy</th>
    </tr>
    <tr>
      <th>bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;= -0.5</th>
      <td>2254</td>
      <td>4508</td>
      <td>271</td>
      <td>19.5</td>
      <td>-23.9</td>
      <td>55.5</td>
      <td>70.5</td>
      <td>27.3</td>
      <td>2.3</td>
      <td>44049.5</td>
      <td>1.0</td>
      <td>18.6</td>
      <td>22.3</td>
      <td>19.0</td>
      <td>0.4</td>
      <td>1.4</td>
      <td>True</td>
    </tr>
    <tr>
      <th>-0.5 to -0.25</th>
      <td>1665</td>
      <td>3330</td>
      <td>378</td>
      <td>41.0</td>
      <td>1.8</td>
      <td>73.3</td>
      <td>71.4</td>
      <td>32.5</td>
      <td>56.4</td>
      <td>68343.8</td>
      <td>16.8</td>
      <td>24.2</td>
      <td>25.0</td>
      <td>18.6</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>True</td>
    </tr>
    <tr>
      <th>-0.25 to -0.1</th>
      <td>1673</td>
      <td>3346</td>
      <td>482</td>
      <td>19.8</td>
      <td>-14.5</td>
      <td>54.5</td>
      <td>67.5</td>
      <td>25.8</td>
      <td>7.3</td>
      <td>33070.8</td>
      <td>3.5</td>
      <td>16.3</td>
      <td>26.9</td>
      <td>19.8</td>
      <td>0.3</td>
      <td>0.7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>-0.1 to 0</th>
      <td>1616</td>
      <td>3232</td>
      <td>523</td>
      <td>31.2</td>
      <td>-0.4</td>
      <td>64.9</td>
      <td>69.2</td>
      <td>35.5</td>
      <td>22.2</td>
      <td>50442.9</td>
      <td>14.6</td>
      <td>16.6</td>
      <td>25.7</td>
      <td>19.4</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0 to 0.1</th>
      <td>1648</td>
      <td>3296</td>
      <td>570</td>
      <td>-13.6</td>
      <td>-49.3</td>
      <td>20.6</td>
      <td>63.9</td>
      <td>3.8</td>
      <td>-51.8</td>
      <td>-22343.2</td>
      <td>-9.0</td>
      <td>-4.6</td>
      <td>28.3</td>
      <td>22.8</td>
      <td>0.3</td>
      <td>0.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.1 to 0.25</th>
      <td>1972</td>
      <td>3944</td>
      <td>536</td>
      <td>-21.9</td>
      <td>-57.4</td>
      <td>14.6</td>
      <td>63.1</td>
      <td>-8.5</td>
      <td>-47.7</td>
      <td>-43259.4</td>
      <td>-24.0</td>
      <td>2.0</td>
      <td>29.8</td>
      <td>22.6</td>
      <td>0.4</td>
      <td>0.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.25 to 0.5</th>
      <td>2360</td>
      <td>4720</td>
      <td>456</td>
      <td>-17.7</td>
      <td>-53.5</td>
      <td>16.0</td>
      <td>62.7</td>
      <td>-20.9</td>
      <td>-11.9</td>
      <td>-41727.8</td>
      <td>-21.3</td>
      <td>3.6</td>
      <td>32.5</td>
      <td>22.7</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>False</td>
    </tr>
    <tr>
      <th>&gt; 0.5</th>
      <td>2695</td>
      <td>5390</td>
      <td>307</td>
      <td>24.8</td>
      <td>-8.7</td>
      <td>57.2</td>
      <td>68.5</td>
      <td>15.4</td>
      <td>41.3</td>
      <td>66715.4</td>
      <td>4.5</td>
      <td>20.2</td>
      <td>28.6</td>
      <td>20.8</td>
      <td>0.3</td>
      <td>1.2</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


| SPX since the open at entry | trades | avg P/L | call side | put side | SPX through the short call | total P/L |
|---|---|---|---|---|---|---|
| down more than 0.5% | 4,508 | +\$20 | +\$1 | +\$19 | 22% | +\$44.0k |
| **down 0.25-0.5%** | 3,330 | **+\$41** | +\$17 | +\$24 | 25% | +\$68.3k |
| **down 0.1-0.25%** | 3,346 | +\$20 | +\$3 | +\$16 | 27% | +\$33.1k |
| **down 0-0.1%** | 3,232 | +\$31 | +\$15 | +\$17 | 26% | +\$50.4k |
| up 0-0.1% | 3,296 | **-\$14** | -\$9 | -\$5 | 28% | -\$22.3k |
| up 0.1-0.25% | 3,944 | **-\$22** | **-\$24** | +\$2 | 30% | -\$43.3k |
| up 0.25-0.5% | 4,720 | **-\$18** | -\$21 | +\$4 | **33%** | -\$41.7k |
| up more than 0.5% | 5,390 | +\$25 | +\$5 | +\$20 | 29% | +\$66.7k |

**Entering after SPX has pulled back is what keeps the call side safe.**

- **Every group at or below the open earns** (+\$20 to +\$41 a condor); **entries after a 0-0.5% rally lose** (-\$14 to -\$22), \$107.3k
  between them.
- **It's the call side that breaks:** after a rally the call spread loses \$9-24 a condor and SPX closes through the short call on 28-33% of
  condors, against 22-27% when SPX is down. In an uptrend an intraday rally tends to keep going into the short call.
- **Big rallies (above +0.5%) earn again**, mostly on the put side, but the 0% cut is the clean rule.


    
![png](ic_analysis_files/output_23_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>max_move</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-0.3</th>
      <td>375.0</td>
      <td>7034.0</td>
      <td>91312.04</td>
      <td>0.51</td>
      <td>-45124.20</td>
      <td>0.49</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>-0.2</th>
      <td>440.0</td>
      <td>8740.0</td>
      <td>123366.40</td>
      <td>0.71</td>
      <td>-38913.72</td>
      <td>0.72</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>-0.1</th>
      <td>543.0</td>
      <td>11184.0</td>
      <td>145464.04</td>
      <td>0.77</td>
      <td>-34233.88</td>
      <td>0.82</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>-0.05</th>
      <td>597.0</td>
      <td>12660.0</td>
      <td>187613.60</td>
      <td>1.00</td>
      <td>-34167.96</td>
      <td>1.07</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>0</th>
      <td>656.0</td>
      <td>14416.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>-33625.72</td>
      <td>1.05</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>702.0</td>
      <td>16142.0</td>
      <td>193891.52</td>
      <td>0.89</td>
      <td>-35646.00</td>
      <td>1.00</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>738.0</td>
      <td>17712.0</td>
      <td>173563.72</td>
      <td>0.73</td>
      <td>-35464.80</td>
      <td>0.94</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>770.0</td>
      <td>20452.0</td>
      <td>159178.12</td>
      <td>0.59</td>
      <td>-46788.04</td>
      <td>0.88</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>788.0</td>
      <td>22736.0</td>
      <td>110211.16</td>
      <td>0.33</td>
      <td>-79311.88</td>
      <td>0.64</td>
      <td>-0.34</td>
    </tr>
    <tr>
      <th>off</th>
      <td>799.0</td>
      <td>31768.0</td>
      <td>155558.08</td>
      <td>0.44</td>
      <td>-67902.08</td>
      <td>0.58</td>
      <td>0.13</td>
    </tr>
  </tbody>
</table>
</div>


| max move up | trades | total P/L | Sharpe | 2022-24 / 2025-26 |
|---|---|---|---|---|
| -0.1% | 11,184 | \$145.5k | 0.77 | 0.82 / 0.67 |
| -0.05% | 12,660 | \$187.6k | **1.00** | 1.07 / 0.84 |
| **0% (strategy)** | **14,416** | **\$195.9k** | **0.97** | 1.05 / 0.80 |
| +0.05% | 16,142 | \$193.9k | 0.89 | 1.00 / 0.63 |
| +0.1% | 17,712 | \$173.6k | 0.73 | 0.94 / 0.23 |
| no limit | 31,768 | \$155.6k | 0.44 | 0.58 / 0.13 |

**0% sits at the top**, level with -0.05% (0.97 against 1.00) and with \$8.3k more; every step above 0% loses Sharpe fast, above all in
2025-26.

## 5. Step 3: above the 10-day SMA, on the book after steps 1-2

How far SPX sits from its 10-day SMA at the moment of entry, measured the backtester's way (SPX at entry against the average of the last
9 closes and the entry price itself).


    
![png](ic_analysis_files/output_26_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>trades</th>
      <th>sessions</th>
      <th>avg_pl</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>win_rate</th>
      <th>avg_pl_to_2024</th>
      <th>avg_pl_from_2025</th>
      <th>total_pl</th>
      <th>avg_pl_call</th>
      <th>avg_pl_put</th>
      <th>call_breached_pct</th>
      <th>put_breached_pct</th>
      <th>abs_move_after_entry_pct</th>
      <th>day_range_pct</th>
      <th>in_strategy</th>
    </tr>
    <tr>
      <th>bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;= -2</th>
      <td>855</td>
      <td>1710</td>
      <td>83</td>
      <td>83.1</td>
      <td>30.2</td>
      <td>127.7</td>
      <td>81.1</td>
      <td>126.2</td>
      <td>-31.2</td>
      <td>71052.6</td>
      <td>39.0</td>
      <td>44.1</td>
      <td>13.3</td>
      <td>13.2</td>
      <td>0.5</td>
      <td>1.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>-2 to -1</th>
      <td>870</td>
      <td>1740</td>
      <td>125</td>
      <td>-32.9</td>
      <td>-104.6</td>
      <td>32.4</td>
      <td>67.0</td>
      <td>-42.0</td>
      <td>-15.4</td>
      <td>-28579.6</td>
      <td>-26.5</td>
      <td>-6.3</td>
      <td>23.9</td>
      <td>20.7</td>
      <td>0.5</td>
      <td>1.2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>-1 to 0</th>
      <td>1734</td>
      <td>3468</td>
      <td>219</td>
      <td>-25.0</td>
      <td>-70.7</td>
      <td>17.4</td>
      <td>63.4</td>
      <td>-35.5</td>
      <td>-1.1</td>
      <td>-43400.9</td>
      <td>-41.2</td>
      <td>16.1</td>
      <td>31.6</td>
      <td>19.1</td>
      <td>0.4</td>
      <td>0.9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0 to 0.5</th>
      <td>1178</td>
      <td>2356</td>
      <td>180</td>
      <td>-8.2</td>
      <td>-61.8</td>
      <td>49.1</td>
      <td>62.3</td>
      <td>-16.6</td>
      <td>5.4</td>
      <td>-9623.6</td>
      <td>18.5</td>
      <td>-26.7</td>
      <td>24.7</td>
      <td>25.3</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0.5 to 1</th>
      <td>1011</td>
      <td>2022</td>
      <td>155</td>
      <td>59.2</td>
      <td>2.6</td>
      <td>105.3</td>
      <td>71.9</td>
      <td>44.0</td>
      <td>80.8</td>
      <td>59827.3</td>
      <td>32.7</td>
      <td>26.5</td>
      <td>24.7</td>
      <td>19.4</td>
      <td>0.3</td>
      <td>0.7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1 to 1.5</th>
      <td>745</td>
      <td>1490</td>
      <td>122</td>
      <td>107.2</td>
      <td>53.2</td>
      <td>158.0</td>
      <td>75.8</td>
      <td>108.7</td>
      <td>102.4</td>
      <td>79892.4</td>
      <td>51.0</td>
      <td>56.2</td>
      <td>25.8</td>
      <td>19.6</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1.5 to 2</th>
      <td>348</td>
      <td>696</td>
      <td>62</td>
      <td>99.1</td>
      <td>35.6</td>
      <td>159.0</td>
      <td>77.0</td>
      <td>150.0</td>
      <td>-9.7</td>
      <td>34481.8</td>
      <td>42.4</td>
      <td>56.7</td>
      <td>25.0</td>
      <td>15.5</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2 to 3</th>
      <td>307</td>
      <td>614</td>
      <td>41</td>
      <td>74.2</td>
      <td>-31.9</td>
      <td>160.6</td>
      <td>74.9</td>
      <td>102.7</td>
      <td>25.4</td>
      <td>22784.8</td>
      <td>40.3</td>
      <td>34.0</td>
      <td>21.8</td>
      <td>16.3</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>True</td>
    </tr>
    <tr>
      <th>&gt; 3</th>
      <td>92</td>
      <td>184</td>
      <td>12</td>
      <td>198.5</td>
      <td>100.9</td>
      <td>241.6</td>
      <td>89.1</td>
      <td>198.9</td>
      <td>197.9</td>
      <td>18262.0</td>
      <td>85.6</td>
      <td>112.9</td>
      <td>10.9</td>
      <td>4.3</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


**An iron condor is short volatility: it wants SPX to stay still until the close.** Being above the 10-day SMA matters because markets
above their short-term average are calm.

| SPX at entry vs its SMA10 | trades | avg P/L | call side | put side | SPX move after entry | day's range | total P/L |
|---|---|---|---|---|---|---|---|
| below -2% | 1,710 | +\$83 | +\$39 | +\$44 | 0.51% | 1.56% | +\$71.1k |
| **-2% to -1%** | 1,740 | **-\$33** | -\$27 | -\$6 | 0.46% | 1.20% | -\$28.6k |
| **-1% to 0%** | 3,468 | **-\$25** | **-\$41** | +\$16 | 0.37% | 0.94% | -\$43.4k |
| 0% to +0.5% | 2,356 | -\$8 | +\$18 | -\$27 | 0.30% | 0.78% | -\$9.6k |
| +0.5% to +1% | 2,022 | +\$59 | +\$33 | +\$26 | 0.26% | 0.69% | +\$59.8k |
| +1% to +1.5% | 1,490 | **+\$107** | +\$51 | +\$56 | 0.23% | 0.67% | **+\$79.9k** |
| above +1.5% | 1,494 | +\$74 to +\$199 | | | 0.20-0.29% | 0.74-0.85% | +\$75.5k |

- **Below the average SPX moves more after entry** (0.37-0.51% against 0.20-0.30% above) on 0.9-1.6% days.
- **The loss zone is the 2% just below the average:** -\$72.0k between the two groups, the call side breaking between -1% and 0%. The filter
  removes it.
- **The first 0.5% above the line is still weak** (-\$8 a condor); from +0.5% every group earns, more the further above.
- **Far below (-2% and lower) earns +\$83, but all of it in 2022-24 (+\$126); in 2025-26 it loses \$31 a condor.**

Across **every session with no filter**, the regime split is plain:


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>avg_day_range_pct</th>
      <th>day_range_p90_pct</th>
      <th>avg_vix</th>
      <th>avg_ts_9d_30</th>
      <th>avg_session_pl_all_tranches</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SPX above SMA10</th>
      <td>630</td>
      <td>0.786</td>
      <td>1.393</td>
      <td>17.356</td>
      <td>0.944</td>
      <td>40.151</td>
    </tr>
    <tr>
      <th>SPX below SMA10</th>
      <td>359</td>
      <td>1.168</td>
      <td>1.960</td>
      <td>21.426</td>
      <td>1.010</td>
      <td>-122.328</td>
    </tr>
  </tbody>
</table>
</div>


| | sessions | avg day range | 90th pct day range | avg VIX | avg VIX9D/VIX |
|---|---|---|---|---|---|
| **SPX above SMA10** | 630 | **0.79%** | 1.39% | **17.4** | 0.94 |
| SPX below SMA10 | 359 | 1.17% | 1.96% | 21.4 | 1.01 |

Other trend rules in the SMA10 rule's place, on the same book:


    
![png](ic_analysis_files/output_30_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>sharpe_vs_strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no trend rule</th>
      <td>656.0</td>
      <td>14416.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>-33625.72</td>
      <td>1.05</td>
      <td>0.80</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>SPX &gt; SMA10 (strategy)</th>
      <td>404.0</td>
      <td>7362.0</td>
      <td>205624.72</td>
      <td>1.50</td>
      <td>-22504.00</td>
      <td>1.54</td>
      <td>1.43</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SPX &gt;= 0.5% above SMA10</th>
      <td>291.0</td>
      <td>5006.0</td>
      <td>215248.36</td>
      <td>2.06</td>
      <td>-13105.32</td>
      <td>2.19</td>
      <td>1.78</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>SPX &gt;= 1% above SMA10</th>
      <td>182.0</td>
      <td>2984.0</td>
      <td>155421.04</td>
      <td>1.82</td>
      <td>-11751.40</td>
      <td>2.30</td>
      <td>0.77</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>SPX &gt; SMA20 (previous rule)</th>
      <td>437.0</td>
      <td>8622.0</td>
      <td>188322.32</td>
      <td>1.23</td>
      <td>-37390.80</td>
      <td>1.12</td>
      <td>1.50</td>
      <td>-0.27</td>
    </tr>
    <tr>
      <th>SPX &gt; SMA10 and SMA20</th>
      <td>367.0</td>
      <td>6652.0</td>
      <td>190861.12</td>
      <td>1.46</td>
      <td>-24526.96</td>
      <td>1.46</td>
      <td>1.44</td>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>SPX &gt; SMA50</th>
      <td>457.0</td>
      <td>10042.0</td>
      <td>118131.52</td>
      <td>0.59</td>
      <td>-44316.56</td>
      <td>0.56</td>
      <td>0.67</td>
      <td>-0.91</td>
    </tr>
    <tr>
      <th>SMA5 &gt; SMA9</th>
      <td>416.0</td>
      <td>9240.0</td>
      <td>179532.40</td>
      <td>1.09</td>
      <td>-38134.16</td>
      <td>1.33</td>
      <td>0.55</td>
      <td>-0.41</td>
    </tr>
    <tr>
      <th>SPX &gt; SMA10 and SMA5 &gt; SMA9</th>
      <td>337.0</td>
      <td>6486.0</td>
      <td>184210.16</td>
      <td>1.38</td>
      <td>-23526.32</td>
      <td>1.37</td>
      <td>1.43</td>
      <td>-0.12</td>
    </tr>
  </tbody>
</table>
</div>


| trend rule (on the book after steps 1-2) | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown |
|---|---|---|---|---|---|
| no trend rule | 14,416 | \$195.9k | 0.97 | 1.05 / 0.80 | -\$33.6k |
| **SPX above SMA10 (strategy)** | **7,362** | \$205.6k | 1.50 | 1.54 / 1.43 | -\$22.5k |
| SPX >= 0.5% above SMA10 | 5,006 | **\$215.2k** | **2.06** | 2.19 / 1.78 | **-\$13.1k** |
| SPX >= 1% above SMA10 | 2,984 | \$155.4k | 1.82 | 2.30 / 0.77 | -\$11.8k |
| SPX above SMA20 (the previous rule) | 8,622 | \$188.3k | 1.23 | 1.12 / 1.50 | -\$37.4k |
| SPX above SMA10 and SMA20 | 6,652 | \$190.9k | 1.46 | 1.46 / 1.44 | -\$24.5k |
| SPX above SMA10 and SMA5 > SMA9 | 6,486 | \$184.2k | 1.38 | 1.37 / 1.43 | -\$23.5k |
| SMA5 > SMA9 | 9,240 | \$179.5k | 1.09 | 1.33 / 0.55 | -\$38.1k |
| SPX above SMA50 | 10,042 | \$118.1k | 0.59 | 0.56 / 0.67 | -\$44.3k |

- **The SMA10 rule takes Sharpe from 0.97 to 1.50** for \$9.7k more and a third off the drawdown; the 20-day in its place scores 1.23.
- **At this step a +0.5% bar scores higher** (2.06, more money, a smaller drawdown): it skips the weak first 0.5% above the line. But the
  band isn't on yet; with everything on, 0% is back at the top (Section 10).


    
![png](ic_analysis_files/output_32_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>sma10_min</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>535.0</td>
      <td>10830.0</td>
      <td>162223.80</td>
      <td>0.91</td>
      <td>-24304.16</td>
      <td>0.78</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>-0.5</th>
      <td>479.0</td>
      <td>9346.0</td>
      <td>196631.76</td>
      <td>1.26</td>
      <td>-26311.64</td>
      <td>1.28</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>0</th>
      <td>404.0</td>
      <td>7362.0</td>
      <td>205624.72</td>
      <td>1.50</td>
      <td>-22504.00</td>
      <td>1.54</td>
      <td>1.43</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>291.0</td>
      <td>5006.0</td>
      <td>215248.36</td>
      <td>2.06</td>
      <td>-13105.32</td>
      <td>2.19</td>
      <td>1.78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>182.0</td>
      <td>2984.0</td>
      <td>155421.04</td>
      <td>1.82</td>
      <td>-11751.40</td>
      <td>2.30</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>93.0</td>
      <td>1494.0</td>
      <td>75528.64</td>
      <td>0.97</td>
      <td>-10057.40</td>
      <td>1.52</td>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>798.0</td>
      <td>41046.88</td>
      <td>0.26</td>
      <td>-6322.08</td>
      <td>0.45</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>off</th>
      <td>656.0</td>
      <td>14416.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>-33625.72</td>
      <td>1.05</td>
      <td>0.80</td>
    </tr>
  </tbody>
</table>
</div>


**The 20-day SMA at this step.** The same book with the 20-day SMA in the 10-day's place: P/L across SPX vs its 20-day SMA (not a
filter of the strategy, so nothing is shaded), and both SMAs moved through the same thresholds.


    
![png](ic_analysis_files/output_34_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>trades</th>
      <th>sessions</th>
      <th>avg_pl</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>win_rate</th>
      <th>avg_pl_to_2024</th>
      <th>avg_pl_from_2025</th>
      <th>total_pl</th>
      <th>avg_pl_call</th>
      <th>avg_pl_put</th>
      <th>call_breached_pct</th>
      <th>put_breached_pct</th>
      <th>abs_move_after_entry_pct</th>
      <th>day_range_pct</th>
      <th>in_strategy</th>
    </tr>
    <tr>
      <th>bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;= -2</th>
      <td>1247</td>
      <td>2494</td>
      <td>120</td>
      <td>67.2</td>
      <td>20.0</td>
      <td>109.1</td>
      <td>78.8</td>
      <td>82.3</td>
      <td>28.0</td>
      <td>83759.6</td>
      <td>21.3</td>
      <td>45.9</td>
      <td>16.2</td>
      <td>13.6</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>-2 to -1</th>
      <td>710</td>
      <td>1420</td>
      <td>93</td>
      <td>-16.2</td>
      <td>-101.6</td>
      <td>52.8</td>
      <td>68.6</td>
      <td>-54.8</td>
      <td>45.1</td>
      <td>-11524.8</td>
      <td>9.8</td>
      <td>-26.1</td>
      <td>21.4</td>
      <td>21.5</td>
      <td>0.4</td>
      <td>1.2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>-1 to 0</th>
      <td>777</td>
      <td>1554</td>
      <td>102</td>
      <td>-94.8</td>
      <td>-175.1</td>
      <td>-16.5</td>
      <td>57.5</td>
      <td>-65.8</td>
      <td>-144.7</td>
      <td>-73683.8</td>
      <td>-78.8</td>
      <td>-16.0</td>
      <td>31.8</td>
      <td>20.3</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0 to 0.5</th>
      <td>656</td>
      <td>1312</td>
      <td>103</td>
      <td>19.5</td>
      <td>-53.6</td>
      <td>84.1</td>
      <td>65.5</td>
      <td>28.4</td>
      <td>3.3</td>
      <td>12795.7</td>
      <td>4.9</td>
      <td>14.6</td>
      <td>30.3</td>
      <td>21.0</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.5 to 1</th>
      <td>811</td>
      <td>1622</td>
      <td>141</td>
      <td>-14.5</td>
      <td>-79.5</td>
      <td>48.1</td>
      <td>63.3</td>
      <td>-27.8</td>
      <td>4.5</td>
      <td>-11732.7</td>
      <td>-17.8</td>
      <td>3.3</td>
      <td>29.7</td>
      <td>22.2</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1 to 1.5</th>
      <td>779</td>
      <td>1558</td>
      <td>109</td>
      <td>51.0</td>
      <td>-17.7</td>
      <td>112.5</td>
      <td>70.2</td>
      <td>-2.9</td>
      <td>137.6</td>
      <td>39729.5</td>
      <td>24.1</td>
      <td>26.9</td>
      <td>26.1</td>
      <td>20.2</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1.5 to 2</th>
      <td>753</td>
      <td>1506</td>
      <td>122</td>
      <td>18.7</td>
      <td>-52.3</td>
      <td>81.1</td>
      <td>64.8</td>
      <td>1.5</td>
      <td>59.3</td>
      <td>14084.4</td>
      <td>-5.7</td>
      <td>24.5</td>
      <td>29.3</td>
      <td>21.0</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2 to 3</th>
      <td>914</td>
      <td>1828</td>
      <td>114</td>
      <td>83.5</td>
      <td>26.2</td>
      <td>131.4</td>
      <td>73.9</td>
      <td>103.0</td>
      <td>20.5</td>
      <td>76333.7</td>
      <td>51.8</td>
      <td>31.7</td>
      <td>25.7</td>
      <td>20.4</td>
      <td>0.3</td>
      <td>0.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>&gt; 3</th>
      <td>398</td>
      <td>796</td>
      <td>49</td>
      <td>143.5</td>
      <td>59.3</td>
      <td>207.2</td>
      <td>81.9</td>
      <td>195.0</td>
      <td>59.2</td>
      <td>57111.8</td>
      <td>81.1</td>
      <td>62.3</td>
      <td>16.1</td>
      <td>15.8</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](ic_analysis_files/output_34_2.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="7" halign="left">SMA10</th>
      <th colspan="7" halign="left">SMA20</th>
    </tr>
    <tr>
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>535.0</td>
      <td>10830.0</td>
      <td>162223.80</td>
      <td>0.91</td>
      <td>-24304.16</td>
      <td>0.78</td>
      <td>1.20</td>
      <td>493.0</td>
      <td>10176.0</td>
      <td>114638.56</td>
      <td>0.59</td>
      <td>-47938.12</td>
      <td>0.67</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>-0.5</th>
      <td>479.0</td>
      <td>9346.0</td>
      <td>196631.76</td>
      <td>1.26</td>
      <td>-26311.64</td>
      <td>1.28</td>
      <td>1.21</td>
      <td>468.0</td>
      <td>9426.0</td>
      <td>152496.56</td>
      <td>0.91</td>
      <td>-41343.12</td>
      <td>0.90</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>0</th>
      <td>404.0</td>
      <td>7362.0</td>
      <td>205624.72</td>
      <td>1.50</td>
      <td>-22504.00</td>
      <td>1.54</td>
      <td>1.43</td>
      <td>437.0</td>
      <td>8622.0</td>
      <td>188322.32</td>
      <td>1.23</td>
      <td>-37390.80</td>
      <td>1.12</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>291.0</td>
      <td>5006.0</td>
      <td>215248.36</td>
      <td>2.06</td>
      <td>-13105.32</td>
      <td>2.19</td>
      <td>1.78</td>
      <td>382.0</td>
      <td>7310.0</td>
      <td>175526.60</td>
      <td>1.22</td>
      <td>-32126.84</td>
      <td>1.03</td>
      <td>1.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>182.0</td>
      <td>2984.0</td>
      <td>155421.04</td>
      <td>1.82</td>
      <td>-11751.40</td>
      <td>2.30</td>
      <td>0.77</td>
      <td>296.0</td>
      <td>5688.0</td>
      <td>187259.28</td>
      <td>1.51</td>
      <td>-25365.64</td>
      <td>1.34</td>
      <td>1.96</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>93.0</td>
      <td>1494.0</td>
      <td>75528.64</td>
      <td>0.97</td>
      <td>-10057.40</td>
      <td>1.52</td>
      <td>-0.12</td>
      <td>229.0</td>
      <td>4130.0</td>
      <td>147529.80</td>
      <td>1.33</td>
      <td>-16824.08</td>
      <td>1.61</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>798.0</td>
      <td>41046.88</td>
      <td>0.26</td>
      <td>-6322.08</td>
      <td>0.45</td>
      <td>-0.05</td>
      <td>153.0</td>
      <td>2624.0</td>
      <td>133445.44</td>
      <td>1.53</td>
      <td>-11317.16</td>
      <td>2.14</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>off</th>
      <td>656.0</td>
      <td>14416.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>-33625.72</td>
      <td>1.05</td>
      <td>0.80</td>
      <td>656.0</td>
      <td>14416.0</td>
      <td>195906.96</td>
      <td>0.97</td>
      <td>-33625.72</td>
      <td>1.05</td>
      <td>0.80</td>
    </tr>
  </tbody>
</table>
</div>


| SPX at entry vs its SMA20 | trades | avg P/L | call side | put side | total P/L |
|---|---|---|---|---|---|
| below -2% | 2,494 | +\$67 | +\$21 | +\$46 | +\$83.8k |
| -2% to -1% | 1,420 | -\$16 | +\$10 | -\$26 | -\$11.5k |
| **-1% to 0%** | 1,554 | **-\$95** | **-\$79** | -\$16 | **-\$73.7k** |
| 0% to +0.5% | 1,312 | +\$20 | +\$5 | +\$15 | +\$12.8k |
| +0.5% to +1% | 1,622 | -\$14 | -\$18 | +\$3 | -\$11.7k |
| +1% to +2% | 3,064 | +\$19 to +\$51 | | | +\$53.8k |
| above +2% | 2,624 | +\$84 to +\$144 | | | +\$133.4k |

| SPX above its SMA by more than | SMA10: trades | Sharpe | P/L | max DD | SMA20: trades | Sharpe | P/L | max DD |
|---|---|---|---|---|---|---|---|---|
| -1% | 10,830 | 0.91 | \$162.2k | -\$24.3k | 10,176 | 0.59 | \$114.6k | -\$47.9k |
| -0.5% | 9,346 | 1.26 | \$196.6k | -\$26.3k | 9,426 | 0.91 | \$152.5k | -\$41.3k |
| **0%** | **7,362** | **1.50** | **\$205.6k** | -\$22.5k | 8,622 | 1.23 | \$188.3k | -\$37.4k |
| +0.5% | 5,006 | **2.06** | **\$215.2k** | -\$13.1k | 7,310 | 1.22 | \$175.5k | -\$32.1k |
| +1% | 2,984 | 1.82 | \$155.4k | -\$11.8k | 5,688 | 1.51 | \$187.3k | -\$25.4k |
| +1.5% | 1,494 | 0.97 | \$75.5k | -\$10.1k | 4,130 | 1.33 | \$147.5k | -\$16.8k |
| no trend rule | 14,416 | 0.97 | \$195.9k | -\$33.6k | | | | |

- **The 20-day's loss zone is the 1% just below it** (-\$95 a condor, -\$73.7k), and its +0.5% to +1% band loses too.
- **At this step the 10-day scores higher at every threshold up to +1%**, with less than two-thirds of the 20-day's drawdown; the 20-day only
  wins far above the line (+1.5%), where the 10-day keeps few trades.
- **This is before the band, the tranches and the ratio cap.** With every setting on the 10-day wins in every year: Section 13.

## 6. Step 4: the VIX9D/VIX band, on the book after steps 1-3

**VIX9D / VIX measures how stressed the next week is against the next month**, at the minute each condor enters. Below 1 the curve is in
contango; near or above 1 it is flat or inverted, which is what it does in a selloff.


    
![png](ic_analysis_files/output_37_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors</th>
      <th>trades</th>
      <th>sessions</th>
      <th>avg_pl</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>win_rate</th>
      <th>avg_pl_to_2024</th>
      <th>avg_pl_from_2025</th>
      <th>total_pl</th>
      <th>avg_pl_call</th>
      <th>avg_pl_put</th>
      <th>call_breached_pct</th>
      <th>put_breached_pct</th>
      <th>abs_move_after_entry_pct</th>
      <th>day_range_pct</th>
      <th>in_strategy</th>
    </tr>
    <tr>
      <th>bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;= 0.8</th>
      <td>133</td>
      <td>266</td>
      <td>22</td>
      <td>-19.5</td>
      <td>-201.6</td>
      <td>156.1</td>
      <td>59.4</td>
      <td>-90.9</td>
      <td>23.5</td>
      <td>-2597.0</td>
      <td>91.9</td>
      <td>-111.4</td>
      <td>18.8</td>
      <td>42.1</td>
      <td>0.2</td>
      <td>0.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.8 to 0.85</th>
      <td>389</td>
      <td>778</td>
      <td>58</td>
      <td>8.0</td>
      <td>-85.2</td>
      <td>89.6</td>
      <td>64.0</td>
      <td>26.3</td>
      <td>-9.6</td>
      <td>3117.7</td>
      <td>-5.0</td>
      <td>13.0</td>
      <td>35.5</td>
      <td>22.1</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.85 to 0.87</th>
      <td>232</td>
      <td>464</td>
      <td>59</td>
      <td>-9.0</td>
      <td>-126.6</td>
      <td>97.9</td>
      <td>62.1</td>
      <td>-56.5</td>
      <td>66.0</td>
      <td>-2080.2</td>
      <td>7.1</td>
      <td>-16.1</td>
      <td>28.4</td>
      <td>25.0</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0.87 to 0.9</th>
      <td>443</td>
      <td>886</td>
      <td>85</td>
      <td>87.3</td>
      <td>12.9</td>
      <td>145.7</td>
      <td>71.8</td>
      <td>140.9</td>
      <td>35.0</td>
      <td>38682.2</td>
      <td>52.8</td>
      <td>34.5</td>
      <td>24.6</td>
      <td>20.8</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0.9 to 0.95</th>
      <td>889</td>
      <td>1778</td>
      <td>156</td>
      <td>79.4</td>
      <td>33.1</td>
      <td>124.1</td>
      <td>71.4</td>
      <td>72.5</td>
      <td>91.6</td>
      <td>70546.7</td>
      <td>40.5</td>
      <td>38.8</td>
      <td>26.7</td>
      <td>18.9</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0.95 to 1</th>
      <td>778</td>
      <td>1556</td>
      <td>123</td>
      <td>105.7</td>
      <td>61.7</td>
      <td>144.7</td>
      <td>76.5</td>
      <td>95.6</td>
      <td>132.7</td>
      <td>82245.4</td>
      <td>52.0</td>
      <td>53.8</td>
      <td>21.1</td>
      <td>16.8</td>
      <td>0.2</td>
      <td>0.8</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1 to 1.02</th>
      <td>205</td>
      <td>410</td>
      <td>54</td>
      <td>106.4</td>
      <td>-2.3</td>
      <td>193.6</td>
      <td>80.0</td>
      <td>98.3</td>
      <td>153.2</td>
      <td>21804.6</td>
      <td>35.9</td>
      <td>70.5</td>
      <td>22.0</td>
      <td>14.6</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1.02 to 1.04</th>
      <td>171</td>
      <td>342</td>
      <td>38</td>
      <td>24.4</td>
      <td>-115.6</td>
      <td>143.5</td>
      <td>69.0</td>
      <td>11.6</td>
      <td>78.1</td>
      <td>4173.5</td>
      <td>25.1</td>
      <td>-0.7</td>
      <td>20.5</td>
      <td>19.3</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1.04 to 1.08</th>
      <td>239</td>
      <td>478</td>
      <td>48</td>
      <td>-74.9</td>
      <td>-218.1</td>
      <td>69.2</td>
      <td>61.1</td>
      <td>-0.6</td>
      <td>-265.6</td>
      <td>-17895.3</td>
      <td>-20.8</td>
      <td>-54.1</td>
      <td>24.3</td>
      <td>22.2</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>&gt; 1.08</th>
      <td>199</td>
      <td>398</td>
      <td>31</td>
      <td>40.5</td>
      <td>-68.6</td>
      <td>123.6</td>
      <td>78.4</td>
      <td>31.8</td>
      <td>68.0</td>
      <td>8063.9</td>
      <td>50.5</td>
      <td>-10.0</td>
      <td>9.5</td>
      <td>20.6</td>
      <td>0.7</td>
      <td>1.5</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


| VIX9D / VIX at entry | trades | avg P/L | call side | put side | SPX through the short put | total P/L |
|---|---|---|---|---|---|---|
| <= 0.80 | 266 | -\$20 | +\$92 | **-\$111** | **42%** | -\$2.6k |
| 0.80-0.87 | 1,242 | -\$9 to +\$8 | -\$5 to +\$7 | -\$16 to +\$13 | 22-25% | +\$1.0k |
| **0.87-0.90** | 886 | **+\$87** | +\$53 | +\$35 | 21% | +\$38.7k |
| **0.90-0.95** | 1,778 | **+\$79** | +\$41 | +\$39 | 19% | +\$70.5k |
| **0.95-1.00** | 1,556 | **+\$106** | +\$52 | +\$54 | 17% | **+\$82.2k** |
| **1.00-1.02** | 410 | **+\$106** | +\$36 | +\$71 | 15% | +\$21.8k |
| 1.02-1.04 | 342 | +\$24 | +\$25 | -\$1 | 19% | +\$4.2k |
| 1.04-1.08 | 478 | **-\$75** | -\$21 | **-\$54** | 22% | **-\$17.9k** |
| above 1.08 | 398 | +\$41 | +\$51 | -\$10 | 21% | +\$8.1k |

- **Inside the band every group earns** (+\$79 to +\$106 a condor), \$213.3k in all; outside it the condors lose \$7.2k.
- **Both tails lose on the put side.** Below 0.80 SPX breaks the short put on 42% of condors (a calm curve that breaks); just past the cap
  (1.04-1.08) the put side loses \$54 a condor, and the group loses \$266 a condor in 2025-26. Above 1.08 the curve is in full stress: SPX
  moves 0.66% after entry on 1.5% days.
- **On the raw book the band did nothing** (step 1 of the greedy search: -0.11): its losing groups were hidden among Thursdays, rally
  entries and below-trend markets. Once those are gone it adds +0.75 Sharpe, the most of any step.


    
![png](ic_analysis_files/output_39_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>ts_low</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.8</th>
      <td>343.0</td>
      <td>5872.0</td>
      <td>214316.32</td>
      <td>1.86</td>
      <td>-14130.12</td>
      <td>1.79</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>0.83</th>
      <td>331.0</td>
      <td>5482.0</td>
      <td>207863.92</td>
      <td>1.87</td>
      <td>-16273.96</td>
      <td>1.69</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>315.0</td>
      <td>5094.0</td>
      <td>211198.64</td>
      <td>2.04</td>
      <td>-17007.32</td>
      <td>1.86</td>
      <td>2.49</td>
    </tr>
    <tr>
      <th>0.87</th>
      <td>294.0</td>
      <td>4630.0</td>
      <td>213278.80</td>
      <td>2.25</td>
      <td>-15130.04</td>
      <td>2.16</td>
      <td>2.48</td>
    </tr>
    <tr>
      <th>0.89</th>
      <td>267.0</td>
      <td>4016.0</td>
      <td>194393.96</td>
      <td>2.19</td>
      <td>-12802.16</td>
      <td>1.96</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>0.91</th>
      <td>241.0</td>
      <td>3384.0</td>
      <td>157541.04</td>
      <td>1.85</td>
      <td>-14611.96</td>
      <td>1.70</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>0.93</th>
      <td>192.0</td>
      <td>2724.0</td>
      <td>149398.44</td>
      <td>1.94</td>
      <td>-12457.60</td>
      <td>1.87</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>149.0</td>
      <td>1966.0</td>
      <td>104049.96</td>
      <td>1.44</td>
      <td>-12738.72</td>
      <td>1.33</td>
      <td>2.03</td>
    </tr>
    <tr>
      <th>off</th>
      <td>350.0</td>
      <td>6138.0</td>
      <td>211719.28</td>
      <td>1.76</td>
      <td>-14619.20</td>
      <td>1.70</td>
      <td>1.87</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>ts_high</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.98</th>
      <td>238.0</td>
      <td>3740.0</td>
      <td>176307.40</td>
      <td>2.01</td>
      <td>-15291.76</td>
      <td>1.91</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>268.0</td>
      <td>4220.0</td>
      <td>191474.20</td>
      <td>2.12</td>
      <td>-16004.24</td>
      <td>2.04</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>1.01</th>
      <td>278.0</td>
      <td>4406.0</td>
      <td>199453.36</td>
      <td>2.15</td>
      <td>-15203.52</td>
      <td>2.04</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>1.02</th>
      <td>294.0</td>
      <td>4630.0</td>
      <td>213278.80</td>
      <td>2.25</td>
      <td>-15130.04</td>
      <td>2.16</td>
      <td>2.48</td>
    </tr>
    <tr>
      <th>1.03</th>
      <td>300.0</td>
      <td>4826.0</td>
      <td>212412.56</td>
      <td>2.19</td>
      <td>-13764.08</td>
      <td>2.05</td>
      <td>2.55</td>
    </tr>
    <tr>
      <th>1.04</th>
      <td>304.0</td>
      <td>4972.0</td>
      <td>217452.32</td>
      <td>2.20</td>
      <td>-13391.84</td>
      <td>2.06</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>1.05</th>
      <td>311.0</td>
      <td>5086.0</td>
      <td>216688.16</td>
      <td>2.15</td>
      <td>-16772.84</td>
      <td>2.04</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>1.06</th>
      <td>319.0</td>
      <td>5192.0</td>
      <td>214305.52</td>
      <td>2.09</td>
      <td>-17570.00</td>
      <td>2.03</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>1.08</th>
      <td>332.0</td>
      <td>5450.0</td>
      <td>199557.00</td>
      <td>1.80</td>
      <td>-23348.80</td>
      <td>1.88</td>
      <td>1.64</td>
    </tr>
    <tr>
      <th>1.1</th>
      <td>337.0</td>
      <td>5576.0</td>
      <td>199706.56</td>
      <td>1.75</td>
      <td>-25235.44</td>
      <td>1.79</td>
      <td>1.68</td>
    </tr>
    <tr>
      <th>off</th>
      <td>348.0</td>
      <td>5848.0</td>
      <td>207620.88</td>
      <td>1.81</td>
      <td>-25235.44</td>
      <td>1.85</td>
      <td>1.76</td>
    </tr>
  </tbody>
</table>
</div>


| floor | trades | Sharpe | | cap | trades | Sharpe |
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

## 7. Step 5: the tranches, on the book after steps 1-4

The book so far trades all 20 entry times. This step keeps the strategy's **7 tranches: 11:33, 11:53, 12:13, 13:13, 13:53, 14:13 and
14:53**. They are the first seven tranches forward selection adds on this book (below). First, each tranche on this book and how
correlated they are.


    
![png](ic_analysis_files/output_42_0.png)
    



<table id="T_bcc19">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bcc19_level0_col0" class="col_heading level0 col0" >condors</th>
      <th id="T_bcc19_level0_col1" class="col_heading level0 col1" >trades</th>
      <th id="T_bcc19_level0_col2" class="col_heading level0 col2" >avg_pl</th>
      <th id="T_bcc19_level0_col3" class="col_heading level0 col3" >ci_low</th>
      <th id="T_bcc19_level0_col4" class="col_heading level0 col4" >ci_high</th>
      <th id="T_bcc19_level0_col5" class="col_heading level0 col5" >total_pl</th>
      <th id="T_bcc19_level0_col6" class="col_heading level0 col6" >sharpe_if_dropped</th>
      <th id="T_bcc19_level0_col7" class="col_heading level0 col7" >status</th>
    </tr>
    <tr>
      <th class="index_name level0" >entry_time</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bcc19_level0_row0" class="row_heading level0 row0" >09:32</th>
      <td id="T_bcc19_row0_col0" class="data row0 col0" >139</td>
      <td id="T_bcc19_row0_col1" class="data row0 col1" >278</td>
      <td id="T_bcc19_row0_col2" class="data row0 col2" >31.85</td>
      <td id="T_bcc19_row0_col3" class="data row0 col3" >-31.26</td>
      <td id="T_bcc19_row0_col4" class="data row0 col4" >96.27</td>
      <td id="T_bcc19_row0_col5" class="data row0 col5" >4427.68</td>
      <td id="T_bcc19_row0_col6" class="data row0 col6" >0.04</td>
      <td id="T_bcc19_row0_col7" class="data row0 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row1" class="row_heading level0 row1" >09:52</th>
      <td id="T_bcc19_row1_col0" class="data row1 col0" >131</td>
      <td id="T_bcc19_row1_col1" class="data row1 col1" >262</td>
      <td id="T_bcc19_row1_col2" class="data row1 col2" >34.85</td>
      <td id="T_bcc19_row1_col3" class="data row1 col3" >-31.65</td>
      <td id="T_bcc19_row1_col4" class="data row1 col4" >95.74</td>
      <td id="T_bcc19_row1_col5" class="data row1 col5" >4565.72</td>
      <td id="T_bcc19_row1_col6" class="data row1 col6" >0.04</td>
      <td id="T_bcc19_row1_col7" class="data row1 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row2" class="row_heading level0 row2" >10:13</th>
      <td id="T_bcc19_row2_col0" class="data row2 col0" >112</td>
      <td id="T_bcc19_row2_col1" class="data row2 col1" >224</td>
      <td id="T_bcc19_row2_col2" class="data row2 col2" >43.86</td>
      <td id="T_bcc19_row2_col3" class="data row2 col3" >-26.94</td>
      <td id="T_bcc19_row2_col4" class="data row2 col4" >108.33</td>
      <td id="T_bcc19_row2_col5" class="data row2 col5" >4912.44</td>
      <td id="T_bcc19_row2_col6" class="data row2 col6" >0.05</td>
      <td id="T_bcc19_row2_col7" class="data row2 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row3" class="row_heading level0 row3" >10:33</th>
      <td id="T_bcc19_row3_col0" class="data row3 col0" >122</td>
      <td id="T_bcc19_row3_col1" class="data row3 col1" >244</td>
      <td id="T_bcc19_row3_col2" class="data row3 col2" >43.01</td>
      <td id="T_bcc19_row3_col3" class="data row3 col3" >-29.20</td>
      <td id="T_bcc19_row3_col4" class="data row3 col4" >109.34</td>
      <td id="T_bcc19_row3_col5" class="data row3 col5" >5246.64</td>
      <td id="T_bcc19_row3_col6" class="data row3 col6" >0.07</td>
      <td id="T_bcc19_row3_col7" class="data row3 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row4" class="row_heading level0 row4" >10:53</th>
      <td id="T_bcc19_row4_col0" class="data row4 col0" >131</td>
      <td id="T_bcc19_row4_col1" class="data row4 col1" >262</td>
      <td id="T_bcc19_row4_col2" class="data row4 col2" >78.01</td>
      <td id="T_bcc19_row4_col3" class="data row4 col3" >16.48</td>
      <td id="T_bcc19_row4_col4" class="data row4 col4" >136.45</td>
      <td id="T_bcc19_row4_col5" class="data row4 col5" >10218.72</td>
      <td id="T_bcc19_row4_col6" class="data row4 col6" >0.00</td>
      <td id="T_bcc19_row4_col7" class="data row4 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row5" class="row_heading level0 row5" >11:13</th>
      <td id="T_bcc19_row5_col0" class="data row5 col0" >115</td>
      <td id="T_bcc19_row5_col1" class="data row5 col1" >230</td>
      <td id="T_bcc19_row5_col2" class="data row5 col2" >128.06</td>
      <td id="T_bcc19_row5_col3" class="data row5 col3" >65.88</td>
      <td id="T_bcc19_row5_col4" class="data row5 col4" >183.16</td>
      <td id="T_bcc19_row5_col5" class="data row5 col5" >14726.80</td>
      <td id="T_bcc19_row5_col6" class="data row5 col6" >-0.06</td>
      <td id="T_bcc19_row5_col7" class="data row5 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row6" class="row_heading level0 row6" >11:33</th>
      <td id="T_bcc19_row6_col0" class="data row6 col0" >122</td>
      <td id="T_bcc19_row6_col1" class="data row6 col1" >244</td>
      <td id="T_bcc19_row6_col2" class="data row6 col2" >133.44</td>
      <td id="T_bcc19_row6_col3" class="data row6 col3" >77.09</td>
      <td id="T_bcc19_row6_col4" class="data row6 col4" >188.55</td>
      <td id="T_bcc19_row6_col5" class="data row6 col5" >16279.64</td>
      <td id="T_bcc19_row6_col6" class="data row6 col6" >-0.09</td>
      <td id="T_bcc19_row6_col7" class="data row6 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row7" class="row_heading level0 row7" >11:53</th>
      <td id="T_bcc19_row7_col0" class="data row7 col0" >119</td>
      <td id="T_bcc19_row7_col1" class="data row7 col1" >238</td>
      <td id="T_bcc19_row7_col2" class="data row7 col2" >151.93</td>
      <td id="T_bcc19_row7_col3" class="data row7 col3" >99.79</td>
      <td id="T_bcc19_row7_col4" class="data row7 col4" >202.39</td>
      <td id="T_bcc19_row7_col5" class="data row7 col5" >18079.28</td>
      <td id="T_bcc19_row7_col6" class="data row7 col6" >-0.10</td>
      <td id="T_bcc19_row7_col7" class="data row7 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row8" class="row_heading level0 row8" >12:13</th>
      <td id="T_bcc19_row8_col0" class="data row8 col0" >110</td>
      <td id="T_bcc19_row8_col1" class="data row8 col1" >220</td>
      <td id="T_bcc19_row8_col2" class="data row8 col2" >156.40</td>
      <td id="T_bcc19_row8_col3" class="data row8 col3" >103.37</td>
      <td id="T_bcc19_row8_col4" class="data row8 col4" >203.10</td>
      <td id="T_bcc19_row8_col5" class="data row8 col5" >17204.20</td>
      <td id="T_bcc19_row8_col6" class="data row8 col6" >-0.09</td>
      <td id="T_bcc19_row8_col7" class="data row8 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row9" class="row_heading level0 row9" >12:33</th>
      <td id="T_bcc19_row9_col0" class="data row9 col0" >115</td>
      <td id="T_bcc19_row9_col1" class="data row9 col1" >230</td>
      <td id="T_bcc19_row9_col2" class="data row9 col2" >101.97</td>
      <td id="T_bcc19_row9_col3" class="data row9 col3" >37.67</td>
      <td id="T_bcc19_row9_col4" class="data row9 col4" >161.83</td>
      <td id="T_bcc19_row9_col5" class="data row9 col5" >11726.80</td>
      <td id="T_bcc19_row9_col6" class="data row9 col6" >-0.01</td>
      <td id="T_bcc19_row9_col7" class="data row9 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row10" class="row_heading level0 row10" >12:53</th>
      <td id="T_bcc19_row10_col0" class="data row10 col0" >119</td>
      <td id="T_bcc19_row10_col1" class="data row10 col1" >238</td>
      <td id="T_bcc19_row10_col2" class="data row10 col2" >122.09</td>
      <td id="T_bcc19_row10_col3" class="data row10 col3" >59.76</td>
      <td id="T_bcc19_row10_col4" class="data row10 col4" >180.05</td>
      <td id="T_bcc19_row10_col5" class="data row10 col5" >14529.28</td>
      <td id="T_bcc19_row10_col6" class="data row10 col6" >-0.03</td>
      <td id="T_bcc19_row10_col7" class="data row10 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row11" class="row_heading level0 row11" >13:13</th>
      <td id="T_bcc19_row11_col0" class="data row11 col0" >115</td>
      <td id="T_bcc19_row11_col1" class="data row11 col1" >230</td>
      <td id="T_bcc19_row11_col2" class="data row11 col2" >142.34</td>
      <td id="T_bcc19_row11_col3" class="data row11 col3" >84.21</td>
      <td id="T_bcc19_row11_col4" class="data row11 col4" >199.19</td>
      <td id="T_bcc19_row11_col5" class="data row11 col5" >16368.80</td>
      <td id="T_bcc19_row11_col6" class="data row11 col6" >-0.08</td>
      <td id="T_bcc19_row11_col7" class="data row11 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row12" class="row_heading level0 row12" >13:33</th>
      <td id="T_bcc19_row12_col0" class="data row12 col0" >110</td>
      <td id="T_bcc19_row12_col1" class="data row12 col1" >220</td>
      <td id="T_bcc19_row12_col2" class="data row12 col2" >125.69</td>
      <td id="T_bcc19_row12_col3" class="data row12 col3" >56.78</td>
      <td id="T_bcc19_row12_col4" class="data row12 col4" >188.97</td>
      <td id="T_bcc19_row12_col5" class="data row12 col5" >13826.20</td>
      <td id="T_bcc19_row12_col6" class="data row12 col6" >-0.05</td>
      <td id="T_bcc19_row12_col7" class="data row12 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row13" class="row_heading level0 row13" >13:53</th>
      <td id="T_bcc19_row13_col0" class="data row13 col0" >107</td>
      <td id="T_bcc19_row13_col1" class="data row13 col1" >214</td>
      <td id="T_bcc19_row13_col2" class="data row13 col2" >165.76</td>
      <td id="T_bcc19_row13_col3" class="data row13 col3" >108.62</td>
      <td id="T_bcc19_row13_col4" class="data row13 col4" >224.09</td>
      <td id="T_bcc19_row13_col5" class="data row13 col5" >17735.84</td>
      <td id="T_bcc19_row13_col6" class="data row13 col6" >-0.12</td>
      <td id="T_bcc19_row13_col7" class="data row13 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row14" class="row_heading level0 row14" >14:13</th>
      <td id="T_bcc19_row14_col0" class="data row14 col0" >111</td>
      <td id="T_bcc19_row14_col1" class="data row14 col1" >222</td>
      <td id="T_bcc19_row14_col2" class="data row14 col2" >139.41</td>
      <td id="T_bcc19_row14_col3" class="data row14 col3" >78.27</td>
      <td id="T_bcc19_row14_col4" class="data row14 col4" >193.31</td>
      <td id="T_bcc19_row14_col5" class="data row14 col5" >15474.32</td>
      <td id="T_bcc19_row14_col6" class="data row14 col6" >-0.10</td>
      <td id="T_bcc19_row14_col7" class="data row14 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row15" class="row_heading level0 row15" >14:33</th>
      <td id="T_bcc19_row15_col0" class="data row15 col0" >111</td>
      <td id="T_bcc19_row15_col1" class="data row15 col1" >222</td>
      <td id="T_bcc19_row15_col2" class="data row15 col2" >74.84</td>
      <td id="T_bcc19_row15_col3" class="data row15 col3" >9.37</td>
      <td id="T_bcc19_row15_col4" class="data row15 col4" >140.85</td>
      <td id="T_bcc19_row15_col5" class="data row15 col5" >8307.32</td>
      <td id="T_bcc19_row15_col6" class="data row15 col6" >-0.01</td>
      <td id="T_bcc19_row15_col7" class="data row15 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row16" class="row_heading level0 row16" >14:53</th>
      <td id="T_bcc19_row16_col0" class="data row16 col0" >107</td>
      <td id="T_bcc19_row16_col1" class="data row16 col1" >214</td>
      <td id="T_bcc19_row16_col2" class="data row16 col2" >114.29</td>
      <td id="T_bcc19_row16_col3" class="data row16 col3" >52.78</td>
      <td id="T_bcc19_row16_col4" class="data row16 col4" >169.79</td>
      <td id="T_bcc19_row16_col5" class="data row16 col5" >12228.84</td>
      <td id="T_bcc19_row16_col6" class="data row16 col6" >-0.07</td>
      <td id="T_bcc19_row16_col7" class="data row16 col7" >traded</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row17" class="row_heading level0 row17" >15:13</th>
      <td id="T_bcc19_row17_col0" class="data row17 col0" >105</td>
      <td id="T_bcc19_row17_col1" class="data row17 col1" >210</td>
      <td id="T_bcc19_row17_col2" class="data row17 col2" >70.82</td>
      <td id="T_bcc19_row17_col3" class="data row17 col3" >7.06</td>
      <td id="T_bcc19_row17_col4" class="data row17 col4" >137.75</td>
      <td id="T_bcc19_row17_col5" class="data row17 col5" >7436.60</td>
      <td id="T_bcc19_row17_col6" class="data row17 col6" >-0.01</td>
      <td id="T_bcc19_row17_col7" class="data row17 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row18" class="row_heading level0 row18" >15:33</th>
      <td id="T_bcc19_row18_col0" class="data row18 col0" >101</td>
      <td id="T_bcc19_row18_col1" class="data row18 col1" >202</td>
      <td id="T_bcc19_row18_col2" class="data row18 col2" >23.16</td>
      <td id="T_bcc19_row18_col3" class="data row18 col3" >-42.84</td>
      <td id="T_bcc19_row18_col4" class="data row18 col4" >84.47</td>
      <td id="T_bcc19_row18_col5" class="data row18 col5" >2339.12</td>
      <td id="T_bcc19_row18_col6" class="data row18 col6" >0.02</td>
      <td id="T_bcc19_row18_col7" class="data row18 col7" >dropped</td>
    </tr>
    <tr>
      <th id="T_bcc19_level0_row19" class="row_heading level0 row19" >15:53</th>
      <td id="T_bcc19_row19_col0" class="data row19 col0" >113</td>
      <td id="T_bcc19_row19_col1" class="data row19 col1" >226</td>
      <td id="T_bcc19_row19_col2" class="data row19 col2" >-20.84</td>
      <td id="T_bcc19_row19_col3" class="data row19 col3" >-71.71</td>
      <td id="T_bcc19_row19_col4" class="data row19 col4" >28.55</td>
      <td id="T_bcc19_row19_col5" class="data row19 col5" >-2355.44</td>
      <td id="T_bcc19_row19_col6" class="data row19 col6" >0.05</td>
      <td id="T_bcc19_row19_col7" class="data row19 col7" >dropped</td>
    </tr>
  </tbody>
</table>



- **The strategy keeps six of the seven best earners:** 13:53 (+\$166 a condor), 12:13 (+\$156), 11:53 (+\$152), 13:13 (+\$142), 14:13 (+\$139)
  and 11:33 (+\$133), plus 14:53 (+\$114). 11:13, 13:33 and 12:53 earn a little more than 14:53 (+\$122-128), but forward selection adds 14:53
  first: it adds more to the book's Sharpe.
- **The first hour earns the least:** 09:32-10:33 make \$32-44 a condor. The last two are weak (15:33 +\$23, 15:53 -\$21).


    
![png](ic_analysis_files/output_44_0.png)
    


    20 tranches, average pairwise correlation 0.33 -> about 2.8 independent bets a day; 24 pairs above 0.7
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>from</th>
      <th>to</th>
      <th>tranches</th>
      <th>members</th>
      <th>avg_rho_inside</th>
    </tr>
    <tr>
      <th>block</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>09:32</td>
      <td>10:33</td>
      <td>4</td>
      <td>09:32, 09:52, 10:13, 10:33</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10:53</td>
      <td>12:53</td>
      <td>7</td>
      <td>10:53, 11:13, 11:33, 11:53, 12:13, 12:33, 12:53</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13:13</td>
      <td>13:33</td>
      <td>2</td>
      <td>13:13, 13:33</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13:53</td>
      <td>14:13</td>
      <td>2</td>
      <td>13:53, 14:13</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14:33</td>
      <td>15:13</td>
      <td>3</td>
      <td>14:33, 14:53, 15:13</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15:33</td>
      <td>15:33</td>
      <td>1</td>
      <td>15:33</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15:53</td>
      <td>15:53</td>
      <td>1</td>
      <td>15:53</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>min</th>
      <th>max</th>
      <th>pairs</th>
    </tr>
    <tr>
      <th>gap_minutes</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.44</td>
      <td>0.89</td>
      <td>19</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.61</td>
      <td>0.64</td>
      <td>0.29</td>
      <td>0.87</td>
      <td>18</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.51</td>
      <td>0.55</td>
      <td>0.18</td>
      <td>0.74</td>
      <td>17</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.47</td>
      <td>0.44</td>
      <td>0.22</td>
      <td>0.73</td>
      <td>16</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.38</td>
      <td>0.32</td>
      <td>0.17</td>
      <td>0.70</td>
      <td>15</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.29</td>
      <td>0.31</td>
      <td>-0.07</td>
      <td>0.61</td>
      <td>14</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.27</td>
      <td>0.27</td>
      <td>0.06</td>
      <td>0.51</td>
      <td>13</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0.21</td>
      <td>0.26</td>
      <td>-0.12</td>
      <td>0.53</td>
      <td>12</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.17</td>
      <td>0.12</td>
      <td>-0.05</td>
      <td>0.52</td>
      <td>11</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.12</td>
      <td>0.03</td>
      <td>-0.06</td>
      <td>0.56</td>
      <td>10</td>
    </tr>
    <tr>
      <th>220</th>
      <td>0.11</td>
      <td>0.04</td>
      <td>-0.05</td>
      <td>0.59</td>
      <td>9</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0.07</td>
      <td>0.07</td>
      <td>-0.10</td>
      <td>0.38</td>
      <td>8</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0.03</td>
      <td>0.01</td>
      <td>-0.07</td>
      <td>0.21</td>
      <td>7</td>
    </tr>
    <tr>
      <th>280</th>
      <td>-0.00</td>
      <td>-0.02</td>
      <td>-0.09</td>
      <td>0.16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>300</th>
      <td>-0.02</td>
      <td>-0.03</td>
      <td>-0.11</td>
      <td>0.08</td>
      <td>5</td>
    </tr>
    <tr>
      <th>320</th>
      <td>-0.00</td>
      <td>0.02</td>
      <td>-0.15</td>
      <td>0.08</td>
      <td>4</td>
    </tr>
    <tr>
      <th>340</th>
      <td>0.01</td>
      <td>-0.06</td>
      <td>-0.12</td>
      <td>0.22</td>
      <td>3</td>
    </tr>
    <tr>
      <th>360</th>
      <td>0.07</td>
      <td>0.07</td>
      <td>-0.03</td>
      <td>0.17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>380</th>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


**Neighbouring tranches are close to the same trade.** Entries 20 minutes apart correlate 0.74 on average; 1 hour apart 0.51, 2 hours
0.29, 3 hours 0.17, 4 hours 0.07. The 20 tranches are about **2.8 independent bets a day** (average pairwise correlation 0.33). The 7 are
spread from 11:33 to 14:53, never more than an hour apart, so they share most of each day's risk: what they give up against all 20 is
mostly the first and the last hour, the weakest tranches.


    
![png](ic_analysis_files/output_46_0.png)
    


    forward selection on this book peaks at: 11:13, 11:33, 11:53, 12:13, 13:13, 13:33, 13:53, 14:13, 14:53
    every step: the book's Sharpe if each remaining tranche were added (the one picked is highlighted)
    


<table id="T_8c690">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8c690_level0_col0" class="col_heading level0 col0" >picked</th>
      <th id="T_8c690_level0_col1" class="col_heading level0 col1" >09:32</th>
      <th id="T_8c690_level0_col2" class="col_heading level0 col2" >09:52</th>
      <th id="T_8c690_level0_col3" class="col_heading level0 col3" >10:13</th>
      <th id="T_8c690_level0_col4" class="col_heading level0 col4" >10:33</th>
      <th id="T_8c690_level0_col5" class="col_heading level0 col5" >10:53</th>
      <th id="T_8c690_level0_col6" class="col_heading level0 col6" >11:13</th>
      <th id="T_8c690_level0_col7" class="col_heading level0 col7" >11:33</th>
      <th id="T_8c690_level0_col8" class="col_heading level0 col8" >11:53</th>
      <th id="T_8c690_level0_col9" class="col_heading level0 col9" >12:13</th>
      <th id="T_8c690_level0_col10" class="col_heading level0 col10" >12:33</th>
      <th id="T_8c690_level0_col11" class="col_heading level0 col11" >12:53</th>
      <th id="T_8c690_level0_col12" class="col_heading level0 col12" >13:13</th>
      <th id="T_8c690_level0_col13" class="col_heading level0 col13" >13:33</th>
      <th id="T_8c690_level0_col14" class="col_heading level0 col14" >13:53</th>
      <th id="T_8c690_level0_col15" class="col_heading level0 col15" >14:13</th>
      <th id="T_8c690_level0_col16" class="col_heading level0 col16" >14:33</th>
      <th id="T_8c690_level0_col17" class="col_heading level0 col17" >14:53</th>
      <th id="T_8c690_level0_col18" class="col_heading level0 col18" >15:13</th>
      <th id="T_8c690_level0_col19" class="col_heading level0 col19" >15:33</th>
      <th id="T_8c690_level0_col20" class="col_heading level0 col20" >15:53</th>
    </tr>
    <tr>
      <th class="index_name level0" >step</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
      <th class="blank col12" >&nbsp;</th>
      <th class="blank col13" >&nbsp;</th>
      <th class="blank col14" >&nbsp;</th>
      <th class="blank col15" >&nbsp;</th>
      <th class="blank col16" >&nbsp;</th>
      <th class="blank col17" >&nbsp;</th>
      <th class="blank col18" >&nbsp;</th>
      <th class="blank col19" >&nbsp;</th>
      <th class="blank col20" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8c690_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_8c690_row0_col0" class="data row0 col0" >11:53</td>
      <td id="T_8c690_row0_col1" class="data row0 col1" >-2.98</td>
      <td id="T_8c690_row0_col2" class="data row0 col2" >-3.15</td>
      <td id="T_8c690_row0_col3" class="data row0 col3" >-3.44</td>
      <td id="T_8c690_row0_col4" class="data row0 col4" >-3.15</td>
      <td id="T_8c690_row0_col5" class="data row0 col5" >-2.60</td>
      <td id="T_8c690_row0_col6" class="data row0 col6" >-2.32</td>
      <td id="T_8c690_row0_col7" class="data row0 col7" >-2.03</td>
      <td id="T_8c690_row0_col8" class="data row0 col8" >-1.93</td>
      <td id="T_8c690_row0_col9" class="data row0 col9" >-2.24</td>
      <td id="T_8c690_row0_col10" class="data row0 col10" >-2.71</td>
      <td id="T_8c690_row0_col11" class="data row0 col11" >-2.19</td>
      <td id="T_8c690_row0_col12" class="data row0 col12" >-2.10</td>
      <td id="T_8c690_row0_col13" class="data row0 col13" >-2.30</td>
      <td id="T_8c690_row0_col14" class="data row0 col14" >-2.02</td>
      <td id="T_8c690_row0_col15" class="data row0 col15" >-2.27</td>
      <td id="T_8c690_row0_col16" class="data row0 col16" >-3.05</td>
      <td id="T_8c690_row0_col17" class="data row0 col17" >-2.82</td>
      <td id="T_8c690_row0_col18" class="data row0 col18" >-3.35</td>
      <td id="T_8c690_row0_col19" class="data row0 col19" >-4.39</td>
      <td id="T_8c690_row0_col20" class="data row0 col20" >-5.91</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_8c690_row1_col0" class="data row1 col0" >13:53</td>
      <td id="T_8c690_row1_col1" class="data row1 col1" >-0.73</td>
      <td id="T_8c690_row1_col2" class="data row1 col2" >-0.74</td>
      <td id="T_8c690_row1_col3" class="data row1 col3" >-0.72</td>
      <td id="T_8c690_row1_col4" class="data row1 col4" >-0.64</td>
      <td id="T_8c690_row1_col5" class="data row1 col5" >-0.26</td>
      <td id="T_8c690_row1_col6" class="data row1 col6" >0.09</td>
      <td id="T_8c690_row1_col7" class="data row1 col7" >0.20</td>
      <td id="T_8c690_row1_col8" class="data row1 col8" ></td>
      <td id="T_8c690_row1_col9" class="data row1 col9" >0.29</td>
      <td id="T_8c690_row1_col10" class="data row1 col10" >-0.15</td>
      <td id="T_8c690_row1_col11" class="data row1 col11" >0.07</td>
      <td id="T_8c690_row1_col12" class="data row1 col12" >0.23</td>
      <td id="T_8c690_row1_col13" class="data row1 col13" >0.02</td>
      <td id="T_8c690_row1_col14" class="data row1 col14" >0.37</td>
      <td id="T_8c690_row1_col15" class="data row1 col15" >0.17</td>
      <td id="T_8c690_row1_col16" class="data row1 col16" >-0.48</td>
      <td id="T_8c690_row1_col17" class="data row1 col17" >-0.13</td>
      <td id="T_8c690_row1_col18" class="data row1 col18" >-0.58</td>
      <td id="T_8c690_row1_col19" class="data row1 col19" >-1.13</td>
      <td id="T_8c690_row1_col20" class="data row1 col20" >-1.76</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_8c690_row2_col0" class="data row2 col0" >12:13</td>
      <td id="T_8c690_row2_col1" class="data row2 col1" >0.54</td>
      <td id="T_8c690_row2_col2" class="data row2 col2" >0.56</td>
      <td id="T_8c690_row2_col3" class="data row2 col3" >0.59</td>
      <td id="T_8c690_row2_col4" class="data row2 col4" >0.58</td>
      <td id="T_8c690_row2_col5" class="data row2 col5" >0.88</td>
      <td id="T_8c690_row2_col6" class="data row2 col6" >1.15</td>
      <td id="T_8c690_row2_col7" class="data row2 col7" >1.24</td>
      <td id="T_8c690_row2_col8" class="data row2 col8" ></td>
      <td id="T_8c690_row2_col9" class="data row2 col9" >1.32</td>
      <td id="T_8c690_row2_col10" class="data row2 col10" >0.97</td>
      <td id="T_8c690_row2_col11" class="data row2 col11" >1.09</td>
      <td id="T_8c690_row2_col12" class="data row2 col12" >1.25</td>
      <td id="T_8c690_row2_col13" class="data row2 col13" >1.05</td>
      <td id="T_8c690_row2_col14" class="data row2 col14" ></td>
      <td id="T_8c690_row2_col15" class="data row2 col15" >1.20</td>
      <td id="T_8c690_row2_col16" class="data row2 col16" >0.81</td>
      <td id="T_8c690_row2_col17" class="data row2 col17" >1.12</td>
      <td id="T_8c690_row2_col18" class="data row2 col18" >0.79</td>
      <td id="T_8c690_row2_col19" class="data row2 col19" >0.47</td>
      <td id="T_8c690_row2_col20" class="data row2 col20" >0.14</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_8c690_row3_col0" class="data row3 col0" >14:13</td>
      <td id="T_8c690_row3_col1" class="data row3 col1" >1.28</td>
      <td id="T_8c690_row3_col2" class="data row3 col2" >1.30</td>
      <td id="T_8c690_row3_col3" class="data row3 col3" >1.31</td>
      <td id="T_8c690_row3_col4" class="data row3 col4" >1.26</td>
      <td id="T_8c690_row3_col5" class="data row3 col5" >1.50</td>
      <td id="T_8c690_row3_col6" class="data row3 col6" >1.70</td>
      <td id="T_8c690_row3_col7" class="data row3 col7" >1.76</td>
      <td id="T_8c690_row3_col8" class="data row3 col8" ></td>
      <td id="T_8c690_row3_col9" class="data row3 col9" ></td>
      <td id="T_8c690_row3_col10" class="data row3 col10" >1.53</td>
      <td id="T_8c690_row3_col11" class="data row3 col11" >1.62</td>
      <td id="T_8c690_row3_col12" class="data row3 col12" >1.78</td>
      <td id="T_8c690_row3_col13" class="data row3 col13" >1.61</td>
      <td id="T_8c690_row3_col14" class="data row3 col14" ></td>
      <td id="T_8c690_row3_col15" class="data row3 col15" >1.78</td>
      <td id="T_8c690_row3_col16" class="data row3 col16" >1.49</td>
      <td id="T_8c690_row3_col17" class="data row3 col17" >1.74</td>
      <td id="T_8c690_row3_col18" class="data row3 col18" >1.51</td>
      <td id="T_8c690_row3_col19" class="data row3 col19" >1.30</td>
      <td id="T_8c690_row3_col20" class="data row3 col20" >1.10</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_8c690_row4_col0" class="data row4 col0" >11:33</td>
      <td id="T_8c690_row4_col1" class="data row4 col1" >1.72</td>
      <td id="T_8c690_row4_col2" class="data row4 col2" >1.73</td>
      <td id="T_8c690_row4_col3" class="data row4 col3" >1.75</td>
      <td id="T_8c690_row4_col4" class="data row4 col4" >1.70</td>
      <td id="T_8c690_row4_col5" class="data row4 col5" >1.90</td>
      <td id="T_8c690_row4_col6" class="data row4 col6" >2.05</td>
      <td id="T_8c690_row4_col7" class="data row4 col7" >2.11</td>
      <td id="T_8c690_row4_col8" class="data row4 col8" ></td>
      <td id="T_8c690_row4_col9" class="data row4 col9" ></td>
      <td id="T_8c690_row4_col10" class="data row4 col10" >1.89</td>
      <td id="T_8c690_row4_col11" class="data row4 col11" >1.95</td>
      <td id="T_8c690_row4_col12" class="data row4 col12" >2.08</td>
      <td id="T_8c690_row4_col13" class="data row4 col13" >1.93</td>
      <td id="T_8c690_row4_col14" class="data row4 col14" ></td>
      <td id="T_8c690_row4_col15" class="data row4 col15" ></td>
      <td id="T_8c690_row4_col16" class="data row4 col16" >1.81</td>
      <td id="T_8c690_row4_col17" class="data row4 col17" >2.03</td>
      <td id="T_8c690_row4_col18" class="data row4 col18" >1.86</td>
      <td id="T_8c690_row4_col19" class="data row4 col19" >1.71</td>
      <td id="T_8c690_row4_col20" class="data row4 col20" >1.59</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_8c690_row5_col0" class="data row5 col0" >13:13</td>
      <td id="T_8c690_row5_col1" class="data row5 col1" >2.03</td>
      <td id="T_8c690_row5_col2" class="data row5 col2" >2.03</td>
      <td id="T_8c690_row5_col3" class="data row5 col3" >2.04</td>
      <td id="T_8c690_row5_col4" class="data row5 col4" >1.97</td>
      <td id="T_8c690_row5_col5" class="data row5 col5" >2.13</td>
      <td id="T_8c690_row5_col6" class="data row5 col6" >2.25</td>
      <td id="T_8c690_row5_col7" class="data row5 col7" ></td>
      <td id="T_8c690_row5_col8" class="data row5 col8" ></td>
      <td id="T_8c690_row5_col9" class="data row5 col9" ></td>
      <td id="T_8c690_row5_col10" class="data row5 col10" >2.15</td>
      <td id="T_8c690_row5_col11" class="data row5 col11" >2.19</td>
      <td id="T_8c690_row5_col12" class="data row5 col12" >2.33</td>
      <td id="T_8c690_row5_col13" class="data row5 col13" >2.20</td>
      <td id="T_8c690_row5_col14" class="data row5 col14" ></td>
      <td id="T_8c690_row5_col15" class="data row5 col15" ></td>
      <td id="T_8c690_row5_col16" class="data row5 col16" >2.12</td>
      <td id="T_8c690_row5_col17" class="data row5 col17" >2.31</td>
      <td id="T_8c690_row5_col18" class="data row5 col18" >2.17</td>
      <td id="T_8c690_row5_col19" class="data row5 col19" >2.05</td>
      <td id="T_8c690_row5_col20" class="data row5 col20" >1.95</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_8c690_row6_col0" class="data row6 col0" >14:53</td>
      <td id="T_8c690_row6_col1" class="data row6 col1" >2.24</td>
      <td id="T_8c690_row6_col2" class="data row6 col2" >2.25</td>
      <td id="T_8c690_row6_col3" class="data row6 col3" >2.25</td>
      <td id="T_8c690_row6_col4" class="data row6 col4" >2.19</td>
      <td id="T_8c690_row6_col5" class="data row6 col5" >2.34</td>
      <td id="T_8c690_row6_col6" class="data row6 col6" >2.43</td>
      <td id="T_8c690_row6_col7" class="data row6 col7" ></td>
      <td id="T_8c690_row6_col8" class="data row6 col8" ></td>
      <td id="T_8c690_row6_col9" class="data row6 col9" ></td>
      <td id="T_8c690_row6_col10" class="data row6 col10" >2.32</td>
      <td id="T_8c690_row6_col11" class="data row6 col11" >2.35</td>
      <td id="T_8c690_row6_col12" class="data row6 col12" ></td>
      <td id="T_8c690_row6_col13" class="data row6 col13" >2.35</td>
      <td id="T_8c690_row6_col14" class="data row6 col14" ></td>
      <td id="T_8c690_row6_col15" class="data row6 col15" ></td>
      <td id="T_8c690_row6_col16" class="data row6 col16" >2.32</td>
      <td id="T_8c690_row6_col17" class="data row6 col17" >2.48</td>
      <td id="T_8c690_row6_col18" class="data row6 col18" >2.36</td>
      <td id="T_8c690_row6_col19" class="data row6 col19" >2.27</td>
      <td id="T_8c690_row6_col20" class="data row6 col20" >2.20</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_8c690_row7_col0" class="data row7 col0" >11:13</td>
      <td id="T_8c690_row7_col1" class="data row7 col1" >2.39</td>
      <td id="T_8c690_row7_col2" class="data row7 col2" >2.40</td>
      <td id="T_8c690_row7_col3" class="data row7 col3" >2.40</td>
      <td id="T_8c690_row7_col4" class="data row7 col4" >2.36</td>
      <td id="T_8c690_row7_col5" class="data row7 col5" >2.48</td>
      <td id="T_8c690_row7_col6" class="data row7 col6" >2.56</td>
      <td id="T_8c690_row7_col7" class="data row7 col7" ></td>
      <td id="T_8c690_row7_col8" class="data row7 col8" ></td>
      <td id="T_8c690_row7_col9" class="data row7 col9" ></td>
      <td id="T_8c690_row7_col10" class="data row7 col10" >2.46</td>
      <td id="T_8c690_row7_col11" class="data row7 col11" >2.49</td>
      <td id="T_8c690_row7_col12" class="data row7 col12" ></td>
      <td id="T_8c690_row7_col13" class="data row7 col13" >2.49</td>
      <td id="T_8c690_row7_col14" class="data row7 col14" ></td>
      <td id="T_8c690_row7_col15" class="data row7 col15" ></td>
      <td id="T_8c690_row7_col16" class="data row7 col16" >2.41</td>
      <td id="T_8c690_row7_col17" class="data row7 col17" ></td>
      <td id="T_8c690_row7_col18" class="data row7 col18" >2.45</td>
      <td id="T_8c690_row7_col19" class="data row7 col19" >2.40</td>
      <td id="T_8c690_row7_col20" class="data row7 col20" >2.35</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_8c690_row8_col0" class="data row8 col0" >13:33</td>
      <td id="T_8c690_row8_col1" class="data row8 col1" >2.48</td>
      <td id="T_8c690_row8_col2" class="data row8 col2" >2.50</td>
      <td id="T_8c690_row8_col3" class="data row8 col3" >2.48</td>
      <td id="T_8c690_row8_col4" class="data row8 col4" >2.43</td>
      <td id="T_8c690_row8_col5" class="data row8 col5" >2.53</td>
      <td id="T_8c690_row8_col6" class="data row8 col6" ></td>
      <td id="T_8c690_row8_col7" class="data row8 col7" ></td>
      <td id="T_8c690_row8_col8" class="data row8 col8" ></td>
      <td id="T_8c690_row8_col9" class="data row8 col9" ></td>
      <td id="T_8c690_row8_col10" class="data row8 col10" >2.53</td>
      <td id="T_8c690_row8_col11" class="data row8 col11" >2.56</td>
      <td id="T_8c690_row8_col12" class="data row8 col12" ></td>
      <td id="T_8c690_row8_col13" class="data row8 col13" >2.58</td>
      <td id="T_8c690_row8_col14" class="data row8 col14" ></td>
      <td id="T_8c690_row8_col15" class="data row8 col15" ></td>
      <td id="T_8c690_row8_col16" class="data row8 col16" >2.51</td>
      <td id="T_8c690_row8_col17" class="data row8 col17" ></td>
      <td id="T_8c690_row8_col18" class="data row8 col18" >2.55</td>
      <td id="T_8c690_row8_col19" class="data row8 col19" >2.51</td>
      <td id="T_8c690_row8_col20" class="data row8 col20" >2.46</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_8c690_row9_col0" class="data row9 col0" >12:53</td>
      <td id="T_8c690_row9_col1" class="data row9 col1" >2.51</td>
      <td id="T_8c690_row9_col2" class="data row9 col2" >2.52</td>
      <td id="T_8c690_row9_col3" class="data row9 col3" >2.51</td>
      <td id="T_8c690_row9_col4" class="data row9 col4" >2.47</td>
      <td id="T_8c690_row9_col5" class="data row9 col5" >2.56</td>
      <td id="T_8c690_row9_col6" class="data row9 col6" ></td>
      <td id="T_8c690_row9_col7" class="data row9 col7" ></td>
      <td id="T_8c690_row9_col8" class="data row9 col8" ></td>
      <td id="T_8c690_row9_col9" class="data row9 col9" ></td>
      <td id="T_8c690_row9_col10" class="data row9 col10" >2.55</td>
      <td id="T_8c690_row9_col11" class="data row9 col11" >2.57</td>
      <td id="T_8c690_row9_col12" class="data row9 col12" ></td>
      <td id="T_8c690_row9_col13" class="data row9 col13" ></td>
      <td id="T_8c690_row9_col14" class="data row9 col14" ></td>
      <td id="T_8c690_row9_col15" class="data row9 col15" ></td>
      <td id="T_8c690_row9_col16" class="data row9 col16" >2.54</td>
      <td id="T_8c690_row9_col17" class="data row9 col17" ></td>
      <td id="T_8c690_row9_col18" class="data row9 col18" >2.57</td>
      <td id="T_8c690_row9_col19" class="data row9 col19" >2.54</td>
      <td id="T_8c690_row9_col20" class="data row9 col20" >2.49</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row10" class="row_heading level0 row10" >11</th>
      <td id="T_8c690_row10_col0" class="data row10 col0" >15:13</td>
      <td id="T_8c690_row10_col1" class="data row10 col1" >2.52</td>
      <td id="T_8c690_row10_col2" class="data row10 col2" >2.53</td>
      <td id="T_8c690_row10_col3" class="data row10 col3" >2.52</td>
      <td id="T_8c690_row10_col4" class="data row10 col4" >2.48</td>
      <td id="T_8c690_row10_col5" class="data row10 col5" >2.56</td>
      <td id="T_8c690_row10_col6" class="data row10 col6" ></td>
      <td id="T_8c690_row10_col7" class="data row10 col7" ></td>
      <td id="T_8c690_row10_col8" class="data row10 col8" ></td>
      <td id="T_8c690_row10_col9" class="data row10 col9" ></td>
      <td id="T_8c690_row10_col10" class="data row10 col10" >2.54</td>
      <td id="T_8c690_row10_col11" class="data row10 col11" ></td>
      <td id="T_8c690_row10_col12" class="data row10 col12" ></td>
      <td id="T_8c690_row10_col13" class="data row10 col13" ></td>
      <td id="T_8c690_row10_col14" class="data row10 col14" ></td>
      <td id="T_8c690_row10_col15" class="data row10 col15" ></td>
      <td id="T_8c690_row10_col16" class="data row10 col16" >2.55</td>
      <td id="T_8c690_row10_col17" class="data row10 col17" ></td>
      <td id="T_8c690_row10_col18" class="data row10 col18" >2.57</td>
      <td id="T_8c690_row10_col19" class="data row10 col19" >2.54</td>
      <td id="T_8c690_row10_col20" class="data row10 col20" >2.49</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row11" class="row_heading level0 row11" >12</th>
      <td id="T_8c690_row11_col0" class="data row11 col0" >10:53</td>
      <td id="T_8c690_row11_col1" class="data row11 col1" >2.52</td>
      <td id="T_8c690_row11_col2" class="data row11 col2" >2.53</td>
      <td id="T_8c690_row11_col3" class="data row11 col3" >2.52</td>
      <td id="T_8c690_row11_col4" class="data row11 col4" >2.49</td>
      <td id="T_8c690_row11_col5" class="data row11 col5" >2.57</td>
      <td id="T_8c690_row11_col6" class="data row11 col6" ></td>
      <td id="T_8c690_row11_col7" class="data row11 col7" ></td>
      <td id="T_8c690_row11_col8" class="data row11 col8" ></td>
      <td id="T_8c690_row11_col9" class="data row11 col9" ></td>
      <td id="T_8c690_row11_col10" class="data row11 col10" >2.54</td>
      <td id="T_8c690_row11_col11" class="data row11 col11" ></td>
      <td id="T_8c690_row11_col12" class="data row11 col12" ></td>
      <td id="T_8c690_row11_col13" class="data row11 col13" ></td>
      <td id="T_8c690_row11_col14" class="data row11 col14" ></td>
      <td id="T_8c690_row11_col15" class="data row11 col15" ></td>
      <td id="T_8c690_row11_col16" class="data row11 col16" >2.53</td>
      <td id="T_8c690_row11_col17" class="data row11 col17" ></td>
      <td id="T_8c690_row11_col18" class="data row11 col18" ></td>
      <td id="T_8c690_row11_col19" class="data row11 col19" >2.53</td>
      <td id="T_8c690_row11_col20" class="data row11 col20" >2.49</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row12" class="row_heading level0 row12" >13</th>
      <td id="T_8c690_row12_col0" class="data row12 col0" >12:33</td>
      <td id="T_8c690_row12_col1" class="data row12 col1" >2.51</td>
      <td id="T_8c690_row12_col2" class="data row12 col2" >2.52</td>
      <td id="T_8c690_row12_col3" class="data row12 col3" >2.51</td>
      <td id="T_8c690_row12_col4" class="data row12 col4" >2.47</td>
      <td id="T_8c690_row12_col5" class="data row12 col5" ></td>
      <td id="T_8c690_row12_col6" class="data row12 col6" ></td>
      <td id="T_8c690_row12_col7" class="data row12 col7" ></td>
      <td id="T_8c690_row12_col8" class="data row12 col8" ></td>
      <td id="T_8c690_row12_col9" class="data row12 col9" ></td>
      <td id="T_8c690_row12_col10" class="data row12 col10" >2.54</td>
      <td id="T_8c690_row12_col11" class="data row12 col11" ></td>
      <td id="T_8c690_row12_col12" class="data row12 col12" ></td>
      <td id="T_8c690_row12_col13" class="data row12 col13" ></td>
      <td id="T_8c690_row12_col14" class="data row12 col14" ></td>
      <td id="T_8c690_row12_col15" class="data row12 col15" ></td>
      <td id="T_8c690_row12_col16" class="data row12 col16" >2.54</td>
      <td id="T_8c690_row12_col17" class="data row12 col17" ></td>
      <td id="T_8c690_row12_col18" class="data row12 col18" ></td>
      <td id="T_8c690_row12_col19" class="data row12 col19" >2.53</td>
      <td id="T_8c690_row12_col20" class="data row12 col20" >2.49</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row13" class="row_heading level0 row13" >14</th>
      <td id="T_8c690_row13_col0" class="data row13 col0" >14:33</td>
      <td id="T_8c690_row13_col1" class="data row13 col1" >2.49</td>
      <td id="T_8c690_row13_col2" class="data row13 col2" >2.50</td>
      <td id="T_8c690_row13_col3" class="data row13 col3" >2.49</td>
      <td id="T_8c690_row13_col4" class="data row13 col4" >2.45</td>
      <td id="T_8c690_row13_col5" class="data row13 col5" ></td>
      <td id="T_8c690_row13_col6" class="data row13 col6" ></td>
      <td id="T_8c690_row13_col7" class="data row13 col7" ></td>
      <td id="T_8c690_row13_col8" class="data row13 col8" ></td>
      <td id="T_8c690_row13_col9" class="data row13 col9" ></td>
      <td id="T_8c690_row13_col10" class="data row13 col10" ></td>
      <td id="T_8c690_row13_col11" class="data row13 col11" ></td>
      <td id="T_8c690_row13_col12" class="data row13 col12" ></td>
      <td id="T_8c690_row13_col13" class="data row13 col13" ></td>
      <td id="T_8c690_row13_col14" class="data row13 col14" ></td>
      <td id="T_8c690_row13_col15" class="data row13 col15" ></td>
      <td id="T_8c690_row13_col16" class="data row13 col16" >2.52</td>
      <td id="T_8c690_row13_col17" class="data row13 col17" ></td>
      <td id="T_8c690_row13_col18" class="data row13 col18" ></td>
      <td id="T_8c690_row13_col19" class="data row13 col19" >2.51</td>
      <td id="T_8c690_row13_col20" class="data row13 col20" >2.47</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row14" class="row_heading level0 row14" >15</th>
      <td id="T_8c690_row14_col0" class="data row14 col0" >09:52</td>
      <td id="T_8c690_row14_col1" class="data row14 col1" >2.48</td>
      <td id="T_8c690_row14_col2" class="data row14 col2" >2.49</td>
      <td id="T_8c690_row14_col3" class="data row14 col3" >2.48</td>
      <td id="T_8c690_row14_col4" class="data row14 col4" >2.45</td>
      <td id="T_8c690_row14_col5" class="data row14 col5" ></td>
      <td id="T_8c690_row14_col6" class="data row14 col6" ></td>
      <td id="T_8c690_row14_col7" class="data row14 col7" ></td>
      <td id="T_8c690_row14_col8" class="data row14 col8" ></td>
      <td id="T_8c690_row14_col9" class="data row14 col9" ></td>
      <td id="T_8c690_row14_col10" class="data row14 col10" ></td>
      <td id="T_8c690_row14_col11" class="data row14 col11" ></td>
      <td id="T_8c690_row14_col12" class="data row14 col12" ></td>
      <td id="T_8c690_row14_col13" class="data row14 col13" ></td>
      <td id="T_8c690_row14_col14" class="data row14 col14" ></td>
      <td id="T_8c690_row14_col15" class="data row14 col15" ></td>
      <td id="T_8c690_row14_col16" class="data row14 col16" ></td>
      <td id="T_8c690_row14_col17" class="data row14 col17" ></td>
      <td id="T_8c690_row14_col18" class="data row14 col18" ></td>
      <td id="T_8c690_row14_col19" class="data row14 col19" >2.48</td>
      <td id="T_8c690_row14_col20" class="data row14 col20" >2.45</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row15" class="row_heading level0 row15" >16</th>
      <td id="T_8c690_row15_col0" class="data row15 col0" >15:33</td>
      <td id="T_8c690_row15_col1" class="data row15 col1" >2.44</td>
      <td id="T_8c690_row15_col2" class="data row15 col2" ></td>
      <td id="T_8c690_row15_col3" class="data row15 col3" >2.44</td>
      <td id="T_8c690_row15_col4" class="data row15 col4" >2.41</td>
      <td id="T_8c690_row15_col5" class="data row15 col5" ></td>
      <td id="T_8c690_row15_col6" class="data row15 col6" ></td>
      <td id="T_8c690_row15_col7" class="data row15 col7" ></td>
      <td id="T_8c690_row15_col8" class="data row15 col8" ></td>
      <td id="T_8c690_row15_col9" class="data row15 col9" ></td>
      <td id="T_8c690_row15_col10" class="data row15 col10" ></td>
      <td id="T_8c690_row15_col11" class="data row15 col11" ></td>
      <td id="T_8c690_row15_col12" class="data row15 col12" ></td>
      <td id="T_8c690_row15_col13" class="data row15 col13" ></td>
      <td id="T_8c690_row15_col14" class="data row15 col14" ></td>
      <td id="T_8c690_row15_col15" class="data row15 col15" ></td>
      <td id="T_8c690_row15_col16" class="data row15 col16" ></td>
      <td id="T_8c690_row15_col17" class="data row15 col17" ></td>
      <td id="T_8c690_row15_col18" class="data row15 col18" ></td>
      <td id="T_8c690_row15_col19" class="data row15 col19" >2.46</td>
      <td id="T_8c690_row15_col20" class="data row15 col20" >2.43</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row16" class="row_heading level0 row16" >17</th>
      <td id="T_8c690_row16_col0" class="data row16 col0" >09:32</td>
      <td id="T_8c690_row16_col1" class="data row16 col1" >2.42</td>
      <td id="T_8c690_row16_col2" class="data row16 col2" ></td>
      <td id="T_8c690_row16_col3" class="data row16 col3" >2.42</td>
      <td id="T_8c690_row16_col4" class="data row16 col4" >2.39</td>
      <td id="T_8c690_row16_col5" class="data row16 col5" ></td>
      <td id="T_8c690_row16_col6" class="data row16 col6" ></td>
      <td id="T_8c690_row16_col7" class="data row16 col7" ></td>
      <td id="T_8c690_row16_col8" class="data row16 col8" ></td>
      <td id="T_8c690_row16_col9" class="data row16 col9" ></td>
      <td id="T_8c690_row16_col10" class="data row16 col10" ></td>
      <td id="T_8c690_row16_col11" class="data row16 col11" ></td>
      <td id="T_8c690_row16_col12" class="data row16 col12" ></td>
      <td id="T_8c690_row16_col13" class="data row16 col13" ></td>
      <td id="T_8c690_row16_col14" class="data row16 col14" ></td>
      <td id="T_8c690_row16_col15" class="data row16 col15" ></td>
      <td id="T_8c690_row16_col16" class="data row16 col16" ></td>
      <td id="T_8c690_row16_col17" class="data row16 col17" ></td>
      <td id="T_8c690_row16_col18" class="data row16 col18" ></td>
      <td id="T_8c690_row16_col19" class="data row16 col19" ></td>
      <td id="T_8c690_row16_col20" class="data row16 col20" >2.40</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row17" class="row_heading level0 row17" >18</th>
      <td id="T_8c690_row17_col0" class="data row17 col0" >10:13</td>
      <td id="T_8c690_row17_col1" class="data row17 col1" ></td>
      <td id="T_8c690_row17_col2" class="data row17 col2" ></td>
      <td id="T_8c690_row17_col3" class="data row17 col3" >2.37</td>
      <td id="T_8c690_row17_col4" class="data row17 col4" >2.35</td>
      <td id="T_8c690_row17_col5" class="data row17 col5" ></td>
      <td id="T_8c690_row17_col6" class="data row17 col6" ></td>
      <td id="T_8c690_row17_col7" class="data row17 col7" ></td>
      <td id="T_8c690_row17_col8" class="data row17 col8" ></td>
      <td id="T_8c690_row17_col9" class="data row17 col9" ></td>
      <td id="T_8c690_row17_col10" class="data row17 col10" ></td>
      <td id="T_8c690_row17_col11" class="data row17 col11" ></td>
      <td id="T_8c690_row17_col12" class="data row17 col12" ></td>
      <td id="T_8c690_row17_col13" class="data row17 col13" ></td>
      <td id="T_8c690_row17_col14" class="data row17 col14" ></td>
      <td id="T_8c690_row17_col15" class="data row17 col15" ></td>
      <td id="T_8c690_row17_col16" class="data row17 col16" ></td>
      <td id="T_8c690_row17_col17" class="data row17 col17" ></td>
      <td id="T_8c690_row17_col18" class="data row17 col18" ></td>
      <td id="T_8c690_row17_col19" class="data row17 col19" ></td>
      <td id="T_8c690_row17_col20" class="data row17 col20" >2.36</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row18" class="row_heading level0 row18" >19</th>
      <td id="T_8c690_row18_col0" class="data row18 col0" >15:53</td>
      <td id="T_8c690_row18_col1" class="data row18 col1" ></td>
      <td id="T_8c690_row18_col2" class="data row18 col2" ></td>
      <td id="T_8c690_row18_col3" class="data row18 col3" ></td>
      <td id="T_8c690_row18_col4" class="data row18 col4" >2.30</td>
      <td id="T_8c690_row18_col5" class="data row18 col5" ></td>
      <td id="T_8c690_row18_col6" class="data row18 col6" ></td>
      <td id="T_8c690_row18_col7" class="data row18 col7" ></td>
      <td id="T_8c690_row18_col8" class="data row18 col8" ></td>
      <td id="T_8c690_row18_col9" class="data row18 col9" ></td>
      <td id="T_8c690_row18_col10" class="data row18 col10" ></td>
      <td id="T_8c690_row18_col11" class="data row18 col11" ></td>
      <td id="T_8c690_row18_col12" class="data row18 col12" ></td>
      <td id="T_8c690_row18_col13" class="data row18 col13" ></td>
      <td id="T_8c690_row18_col14" class="data row18 col14" ></td>
      <td id="T_8c690_row18_col15" class="data row18 col15" ></td>
      <td id="T_8c690_row18_col16" class="data row18 col16" ></td>
      <td id="T_8c690_row18_col17" class="data row18 col17" ></td>
      <td id="T_8c690_row18_col18" class="data row18 col18" ></td>
      <td id="T_8c690_row18_col19" class="data row18 col19" ></td>
      <td id="T_8c690_row18_col20" class="data row18 col20" >2.31</td>
    </tr>
    <tr>
      <th id="T_8c690_level0_row19" class="row_heading level0 row19" >20</th>
      <td id="T_8c690_row19_col0" class="data row19 col0" >10:33</td>
      <td id="T_8c690_row19_col1" class="data row19 col1" ></td>
      <td id="T_8c690_row19_col2" class="data row19 col2" ></td>
      <td id="T_8c690_row19_col3" class="data row19 col3" ></td>
      <td id="T_8c690_row19_col4" class="data row19 col4" >2.25</td>
      <td id="T_8c690_row19_col5" class="data row19 col5" ></td>
      <td id="T_8c690_row19_col6" class="data row19 col6" ></td>
      <td id="T_8c690_row19_col7" class="data row19 col7" ></td>
      <td id="T_8c690_row19_col8" class="data row19 col8" ></td>
      <td id="T_8c690_row19_col9" class="data row19 col9" ></td>
      <td id="T_8c690_row19_col10" class="data row19 col10" ></td>
      <td id="T_8c690_row19_col11" class="data row19 col11" ></td>
      <td id="T_8c690_row19_col12" class="data row19 col12" ></td>
      <td id="T_8c690_row19_col13" class="data row19 col13" ></td>
      <td id="T_8c690_row19_col14" class="data row19 col14" ></td>
      <td id="T_8c690_row19_col15" class="data row19 col15" ></td>
      <td id="T_8c690_row19_col16" class="data row19 col16" ></td>
      <td id="T_8c690_row19_col17" class="data row19 col17" ></td>
      <td id="T_8c690_row19_col18" class="data row19 col18" ></td>
      <td id="T_8c690_row19_col19" class="data row19 col19" ></td>
      <td id="T_8c690_row19_col20" class="data row19 col20" ></td>
    </tr>
  </tbody>
</table>




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added</th>
      <th>tranches</th>
      <th>condors</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>win_rate_spread</th>
      <th>win_rate_condor</th>
      <th>avg_pl_condor</th>
    </tr>
    <tr>
      <th>step</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>11:53</td>
      <td>1</td>
      <td>119</td>
      <td>238</td>
      <td>18079.28</td>
      <td>-1.93</td>
      <td>-1.79</td>
      <td>-2.21</td>
      <td>-2052.68</td>
      <td>89.08</td>
      <td>84.03</td>
      <td>151.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13:53</td>
      <td>2</td>
      <td>226</td>
      <td>452</td>
      <td>35815.12</td>
      <td>0.37</td>
      <td>0.33</td>
      <td>0.45</td>
      <td>-2378.04</td>
      <td>87.39</td>
      <td>82.30</td>
      <td>158.47</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12:13</td>
      <td>3</td>
      <td>336</td>
      <td>672</td>
      <td>53019.32</td>
      <td>1.32</td>
      <td>1.24</td>
      <td>1.49</td>
      <td>-3965.36</td>
      <td>87.20</td>
      <td>81.85</td>
      <td>157.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14:13</td>
      <td>4</td>
      <td>447</td>
      <td>894</td>
      <td>68493.64</td>
      <td>1.78</td>
      <td>1.65</td>
      <td>2.05</td>
      <td>-4037.84</td>
      <td>86.24</td>
      <td>80.76</td>
      <td>153.23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11:33</td>
      <td>5</td>
      <td>569</td>
      <td>1138</td>
      <td>84773.28</td>
      <td>2.11</td>
      <td>2.05</td>
      <td>2.24</td>
      <td>-4817.12</td>
      <td>86.82</td>
      <td>81.20</td>
      <td>148.99</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13:13</td>
      <td>6</td>
      <td>684</td>
      <td>1368</td>
      <td>101142.08</td>
      <td>2.33</td>
      <td>2.25</td>
      <td>2.50</td>
      <td>-5515.52</td>
      <td>86.70</td>
      <td>80.70</td>
      <td>147.87</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14:53</td>
      <td>7</td>
      <td>791</td>
      <td>1582</td>
      <td>113370.92</td>
      <td>2.48</td>
      <td>2.33</td>
      <td>2.83</td>
      <td>-4888.92</td>
      <td>85.65</td>
      <td>79.14</td>
      <td>143.33</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11:13</td>
      <td>8</td>
      <td>906</td>
      <td>1812</td>
      <td>128097.72</td>
      <td>2.56</td>
      <td>2.45</td>
      <td>2.84</td>
      <td>-5252.32</td>
      <td>86.20</td>
      <td>79.47</td>
      <td>141.39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13:33</td>
      <td>9</td>
      <td>1016</td>
      <td>2032</td>
      <td>141923.92</td>
      <td>2.58</td>
      <td>2.47</td>
      <td>2.85</td>
      <td>-8413.48</td>
      <td>86.17</td>
      <td>79.23</td>
      <td>139.69</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12:53</td>
      <td>10</td>
      <td>1135</td>
      <td>2270</td>
      <td>156453.20</td>
      <td>2.57</td>
      <td>2.41</td>
      <td>2.99</td>
      <td>-9768.00</td>
      <td>86.26</td>
      <td>79.12</td>
      <td>137.84</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15:13</td>
      <td>11</td>
      <td>1240</td>
      <td>2480</td>
      <td>163889.80</td>
      <td>2.57</td>
      <td>2.47</td>
      <td>2.86</td>
      <td>-9566.52</td>
      <td>85.36</td>
      <td>77.98</td>
      <td>132.17</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10:53</td>
      <td>12</td>
      <td>1371</td>
      <td>2742</td>
      <td>174108.52</td>
      <td>2.57</td>
      <td>2.52</td>
      <td>2.70</td>
      <td>-9393.80</td>
      <td>85.52</td>
      <td>77.75</td>
      <td>126.99</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12:33</td>
      <td>13</td>
      <td>1486</td>
      <td>2972</td>
      <td>185835.32</td>
      <td>2.54</td>
      <td>2.43</td>
      <td>2.83</td>
      <td>-10625.20</td>
      <td>85.46</td>
      <td>77.39</td>
      <td>125.06</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14:33</td>
      <td>14</td>
      <td>1597</td>
      <td>3194</td>
      <td>194142.64</td>
      <td>2.52</td>
      <td>2.38</td>
      <td>2.88</td>
      <td>-10638.60</td>
      <td>85.07</td>
      <td>76.58</td>
      <td>121.57</td>
    </tr>
    <tr>
      <th>15</th>
      <td>09:52</td>
      <td>15</td>
      <td>1728</td>
      <td>3456</td>
      <td>198708.36</td>
      <td>2.49</td>
      <td>2.38</td>
      <td>2.78</td>
      <td>-12483.64</td>
      <td>85.13</td>
      <td>76.22</td>
      <td>114.99</td>
    </tr>
    <tr>
      <th>16</th>
      <td>15:33</td>
      <td>16</td>
      <td>1829</td>
      <td>3658</td>
      <td>201047.48</td>
      <td>2.46</td>
      <td>2.39</td>
      <td>2.65</td>
      <td>-12015.16</td>
      <td>84.17</td>
      <td>75.12</td>
      <td>109.92</td>
    </tr>
    <tr>
      <th>17</th>
      <td>09:32</td>
      <td>17</td>
      <td>1968</td>
      <td>3936</td>
      <td>205475.16</td>
      <td>2.42</td>
      <td>2.31</td>
      <td>2.69</td>
      <td>-13004.20</td>
      <td>84.38</td>
      <td>75.15</td>
      <td>104.41</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10:13</td>
      <td>18</td>
      <td>2080</td>
      <td>4160</td>
      <td>210387.60</td>
      <td>2.37</td>
      <td>2.27</td>
      <td>2.62</td>
      <td>-15060.12</td>
      <td>84.45</td>
      <td>75.05</td>
      <td>101.15</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15:53</td>
      <td>19</td>
      <td>2193</td>
      <td>4386</td>
      <td>208032.16</td>
      <td>2.31</td>
      <td>2.21</td>
      <td>2.60</td>
      <td>-15095.88</td>
      <td>83.24</td>
      <td>74.01</td>
      <td>94.86</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10:33</td>
      <td>20</td>
      <td>2315</td>
      <td>4630</td>
      <td>213278.80</td>
      <td>2.25</td>
      <td>2.16</td>
      <td>2.48</td>
      <td>-15130.04</td>
      <td>83.39</td>
      <td>73.95</td>
      <td>92.13</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>tranches_selected</th>
      <th>dropped_by_selection</th>
      <th>selected_sharpe</th>
      <th>live_sharpe</th>
      <th>all20_sharpe</th>
      <th>selected_pl</th>
      <th>live_pl</th>
      <th>all20_pl</th>
    </tr>
    <tr>
      <th>test_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023</th>
      <td>2022-05 to 2022</td>
      <td>11.0</td>
      <td>09:32, 10:13, 10:33, 10:53, 11:13, 11:33, 11:5...</td>
      <td>1.25</td>
      <td>2.24</td>
      <td>2.43</td>
      <td>24240.60</td>
      <td>27291.64</td>
      <td>56324.20</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2022-05 to 2023</td>
      <td>13.0</td>
      <td>09:32, 10:33, 10:53, 12:53, 13:53, 14:33, 15:53</td>
      <td>1.31</td>
      <td>2.05</td>
      <td>1.44</td>
      <td>27810.76</td>
      <td>25997.60</td>
      <td>39950.48</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>2022-05 to 2024</td>
      <td>12.0</td>
      <td>09:32, 09:52, 10:13, 10:33, 12:33, 12:53, 14:3...</td>
      <td>2.03</td>
      <td>2.57</td>
      <td>2.31</td>
      <td>31527.00</td>
      <td>27280.52</td>
      <td>49777.28</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>2022-05 to 2025</td>
      <td>12.0</td>
      <td>09:32, 09:52, 10:13, 10:33, 12:33, 14:33, 15:3...</td>
      <td>3.78</td>
      <td>3.59</td>
      <td>3.00</td>
      <td>17551.44</td>
      <td>11734.36</td>
      <td>20006.92</td>
    </tr>
    <tr>
      <th>out of sample</th>
      <td>expanding</td>
      <td>NaN</td>
      <td>-</td>
      <td>1.69</td>
      <td>2.39</td>
      <td>2.11</td>
      <td>101129.80</td>
      <td>92304.12</td>
      <td>166058.88</td>
    </tr>
  </tbody>
</table>
</div>


    chosen-each-year vs all 20, out of sample: Sharpe difference 95% interval -0.79 to -0.04; better in 2% of resamples
    

**Forward selection on this book:** it starts with 11:53 (the best tranche alone), then adds 13:53, 12:13, 14:13, 11:33, 13:13 and 14:53:
**the strategy's 7, in that order** (Sharpe 2.48 at step 7). It goes on to 11:13 (2.56) and 13:33, where it peaks (2.58, 9 tranches), is
flat to 12 tranches and falls after.

**Walk-forward:** for each year from 2023, the selection is re-run on every earlier year only and its pick traded in the next year.

| test year | tranches chosen | chosen on earlier years | the strategy's 7 | all 20 |
|---|---|---|---|---|
| 2023 | 11 | 1.25 (\$24.2k) | 2.24 (\$27.3k) | 2.43 (\$56.3k) |
| 2024 | 13 | 1.31 (\$27.8k) | 2.05 (\$26.0k) | 1.44 (\$40.0k) |
| 2025 | 12 | 2.03 (\$31.5k) | 2.57 (\$27.3k) | 2.31 (\$49.8k) |
| 2026 | 12 | 3.78 (\$17.6k) | 3.59 (\$11.7k) | 3.00 (\$20.0k) |
| **2023-26** | | **1.69 (\$101.1k)** | **2.39 (\$92.3k)** | 2.11 (\$166.1k) |

- **Re-chosen each year, the selection does worse than trading all 20** (1.69 against 2.11 out of sample; better in only 2% of resamples):
  with little history it keeps too few tranches, and not always the right ones.
- **The strategy's 7 score 2.39 over the same years**, ahead of all 20 (2.11), but that set was chosen on all the data, so part of its edge
  is hindsight. It makes \$92.3k against all 20's \$166.1k.
- **What holds every year is the first hour:** every walk-forward pick drops 09:32 and 10:33, and three of four drop 10:13.

**Forward selection by drawdown, on its own terms.** A separate search that takes nothing from the strategy's tranche set: **every other
setting on (the ratio cap included), all 20 tranches available**. It starts with the tranche whose own **P/L per \$ of max drawdown** is
highest, then adds the tranche that raises the book's P/L per \$ of drawdown most, until all 20 are in, and keeps the set where that ratio
peaks. (Minimising the drawdown alone would just pick the tranches that barely trade.)


    
![png](ic_analysis_files/output_48_0.png)
    


    forward selection by drawdown peaks at: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53, 15:53
    every step: the book's P/L per $ of drawdown if each remaining tranche were added
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>09:32</th>
      <th>09:52</th>
      <th>10:13</th>
      <th>10:33</th>
      <th>10:53</th>
      <th>11:13</th>
      <th>11:33</th>
      <th>11:53</th>
      <th>12:13</th>
      <th>12:33</th>
      <th>12:53</th>
      <th>13:13</th>
      <th>13:33</th>
      <th>13:53</th>
      <th>14:13</th>
      <th>14:33</th>
      <th>14:53</th>
      <th>15:13</th>
      <th>15:33</th>
      <th>15:53</th>
    </tr>
    <tr>
      <th>step</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>1.9</td>
      <td>1.4</td>
      <td>2.0</td>
      <td>6.8</td>
      <td>12.3</td>
      <td>8.8</td>
      <td>8.8</td>
      <td>9.6</td>
      <td>4.4</td>
      <td>8.3</td>
      <td>9.5</td>
      <td>6.8</td>
      <td>11.1</td>
      <td>9.8</td>
      <td>6.1</td>
      <td>13.4</td>
      <td>3.7</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>5.8</td>
      <td>7.1</td>
      <td>7.0</td>
      <td>11.9</td>
      <td>15.8</td>
      <td>20.1</td>
      <td>16.8</td>
      <td>23.0</td>
      <td>8.9</td>
      <td>12.5</td>
      <td>14.0</td>
      <td>12.7</td>
      <td>13.4</td>
      <td>14.6</td>
      <td>9.6</td>
      <td>NaN</td>
      <td>8.8</td>
      <td>12.7</td>
      <td>13.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.4</td>
      <td>13.9</td>
      <td>13.5</td>
      <td>15.5</td>
      <td>21.7</td>
      <td>27.0</td>
      <td>22.8</td>
      <td>20.5</td>
      <td>NaN</td>
      <td>14.3</td>
      <td>16.9</td>
      <td>18.5</td>
      <td>17.3</td>
      <td>17.4</td>
      <td>23.6</td>
      <td>17.8</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>19.2</td>
      <td>23.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.5</td>
      <td>18.1</td>
      <td>16.9</td>
      <td>18.3</td>
      <td>22.8</td>
      <td>NaN</td>
      <td>24.5</td>
      <td>22.3</td>
      <td>NaN</td>
      <td>20.8</td>
      <td>20.3</td>
      <td>24.7</td>
      <td>20.2</td>
      <td>31.9</td>
      <td>34.0</td>
      <td>17.4</td>
      <td>NaN</td>
      <td>16.8</td>
      <td>27.0</td>
      <td>27.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17.5</td>
      <td>25.2</td>
      <td>23.3</td>
      <td>23.8</td>
      <td>35.6</td>
      <td>NaN</td>
      <td>31.3</td>
      <td>26.6</td>
      <td>NaN</td>
      <td>26.4</td>
      <td>25.5</td>
      <td>30.5</td>
      <td>26.7</td>
      <td>32.3</td>
      <td>NaN</td>
      <td>21.5</td>
      <td>NaN</td>
      <td>21.1</td>
      <td>34.0</td>
      <td>34.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>17.6</td>
      <td>24.1</td>
      <td>23.4</td>
      <td>20.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.7</td>
      <td>29.9</td>
      <td>NaN</td>
      <td>28.3</td>
      <td>30.9</td>
      <td>31.6</td>
      <td>30.3</td>
      <td>31.5</td>
      <td>NaN</td>
      <td>28.7</td>
      <td>NaN</td>
      <td>27.9</td>
      <td>31.6</td>
      <td>35.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17.6</td>
      <td>24.1</td>
      <td>23.5</td>
      <td>20.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.7</td>
      <td>29.9</td>
      <td>NaN</td>
      <td>28.3</td>
      <td>30.9</td>
      <td>31.6</td>
      <td>30.3</td>
      <td>31.6</td>
      <td>NaN</td>
      <td>28.8</td>
      <td>NaN</td>
      <td>27.9</td>
      <td>31.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22.6</td>
      <td>24.8</td>
      <td>24.3</td>
      <td>20.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.3</td>
      <td>NaN</td>
      <td>30.4</td>
      <td>29.3</td>
      <td>31.6</td>
      <td>26.8</td>
      <td>39.8</td>
      <td>NaN</td>
      <td>30.5</td>
      <td>NaN</td>
      <td>24.3</td>
      <td>26.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23.7</td>
      <td>29.3</td>
      <td>28.7</td>
      <td>24.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36.6</td>
      <td>NaN</td>
      <td>35.9</td>
      <td>34.6</td>
      <td>35.8</td>
      <td>31.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36.6</td>
      <td>NaN</td>
      <td>28.2</td>
      <td>30.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>27.3</td>
      <td>25.9</td>
      <td>24.9</td>
      <td>28.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.6</td>
      <td>29.8</td>
      <td>33.4</td>
      <td>26.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>31.1</td>
      <td>36.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>28.9</td>
      <td>28.9</td>
      <td>27.7</td>
      <td>30.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.5</td>
      <td>31.7</td>
      <td>35.5</td>
      <td>29.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.6</td>
      <td>38.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>27.4</td>
      <td>27.8</td>
      <td>26.7</td>
      <td>26.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.6</td>
      <td>31.7</td>
      <td>35.5</td>
      <td>28.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>25.9</td>
      <td>27.3</td>
      <td>26.3</td>
      <td>25.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.6</td>
      <td>29.9</td>
      <td>NaN</td>
      <td>27.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.8</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>25.6</td>
      <td>24.0</td>
      <td>23.3</td>
      <td>26.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26.9</td>
      <td>NaN</td>
      <td>24.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24.0</td>
      <td>27.2</td>
      <td>26.3</td>
      <td>25.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.7</td>
      <td>NaN</td>
      <td>27.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>24.4</td>
      <td>24.1</td>
      <td>23.5</td>
      <td>24.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>22.4</td>
      <td>20.4</td>
      <td>20.0</td>
      <td>24.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>22.9</td>
      <td>20.9</td>
      <td>20.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NaN</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added</th>
      <th>tranches</th>
      <th>condors</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>avg_pl_condor</th>
      <th>pl_per_dd</th>
    </tr>
    <tr>
      <th>step</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14:53</td>
      <td>1</td>
      <td>67</td>
      <td>134</td>
      <td>11656.04</td>
      <td>-3.58</td>
      <td>-4.21</td>
      <td>-2.36</td>
      <td>-870.76</td>
      <td>173.97</td>
      <td>13.39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12:13</td>
      <td>2</td>
      <td>177</td>
      <td>354</td>
      <td>28860.24</td>
      <td>-0.30</td>
      <td>-0.69</td>
      <td>0.55</td>
      <td>-1255.52</td>
      <td>163.05</td>
      <td>22.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11:13</td>
      <td>3</td>
      <td>292</td>
      <td>584</td>
      <td>43587.04</td>
      <td>0.83</td>
      <td>0.60</td>
      <td>1.34</td>
      <td>-1614.28</td>
      <td>149.27</td>
      <td>27.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14:13</td>
      <td>4</td>
      <td>383</td>
      <td>766</td>
      <td>58368.96</td>
      <td>1.53</td>
      <td>1.26</td>
      <td>2.09</td>
      <td>-1717.44</td>
      <td>152.40</td>
      <td>33.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10:53</td>
      <td>5</td>
      <td>513</td>
      <td>1026</td>
      <td>68705.56</td>
      <td>1.65</td>
      <td>1.58</td>
      <td>1.78</td>
      <td>-1931.64</td>
      <td>133.93</td>
      <td>35.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15:53</td>
      <td>6</td>
      <td>514</td>
      <td>1028</td>
      <td>68872.68</td>
      <td>1.65</td>
      <td>1.59</td>
      <td>1.78</td>
      <td>-1931.64</td>
      <td>133.99</td>
      <td>35.66</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11:33</td>
      <td>7</td>
      <td>636</td>
      <td>1272</td>
      <td>85152.32</td>
      <td>1.93</td>
      <td>1.95</td>
      <td>1.89</td>
      <td>-2524.92</td>
      <td>133.89</td>
      <td>33.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13:53</td>
      <td>8</td>
      <td>737</td>
      <td>1474</td>
      <td>102826.44</td>
      <td>2.29</td>
      <td>2.28</td>
      <td>2.31</td>
      <td>-2582.60</td>
      <td>139.52</td>
      <td>39.82</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11:53</td>
      <td>9</td>
      <td>856</td>
      <td>1712</td>
      <td>120905.72</td>
      <td>2.47</td>
      <td>2.46</td>
      <td>2.49</td>
      <td>-3301.32</td>
      <td>141.24</td>
      <td>36.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14:33</td>
      <td>10</td>
      <td>936</td>
      <td>1872</td>
      <td>129529.32</td>
      <td>2.52</td>
      <td>2.46</td>
      <td>2.65</td>
      <td>-3404.40</td>
      <td>138.39</td>
      <td>38.05</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15:33</td>
      <td>11</td>
      <td>963</td>
      <td>1926</td>
      <td>129578.56</td>
      <td>2.49</td>
      <td>2.49</td>
      <td>2.48</td>
      <td>-3404.40</td>
      <td>134.56</td>
      <td>38.06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13:13</td>
      <td>12</td>
      <td>1078</td>
      <td>2156</td>
      <td>145947.36</td>
      <td>2.62</td>
      <td>2.62</td>
      <td>2.63</td>
      <td>-4114.08</td>
      <td>135.39</td>
      <td>35.48</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12:33</td>
      <td>13</td>
      <td>1193</td>
      <td>2386</td>
      <td>157674.16</td>
      <td>2.60</td>
      <td>2.50</td>
      <td>2.79</td>
      <td>-5153.72</td>
      <td>132.17</td>
      <td>30.59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15:13</td>
      <td>14</td>
      <td>1243</td>
      <td>2486</td>
      <td>162943.16</td>
      <td>2.60</td>
      <td>2.51</td>
      <td>2.80</td>
      <td>-5153.72</td>
      <td>131.09</td>
      <td>31.62</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12:53</td>
      <td>15</td>
      <td>1361</td>
      <td>2722</td>
      <td>177051.32</td>
      <td>2.60</td>
      <td>2.45</td>
      <td>2.93</td>
      <td>-6386.48</td>
      <td>130.09</td>
      <td>27.72</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13:33</td>
      <td>16</td>
      <td>1467</td>
      <td>2934</td>
      <td>192145.04</td>
      <td>2.66</td>
      <td>2.53</td>
      <td>2.94</td>
      <td>-7776.56</td>
      <td>130.98</td>
      <td>24.71</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10:33</td>
      <td>17</td>
      <td>1588</td>
      <td>3176</td>
      <td>197464.56</td>
      <td>2.57</td>
      <td>2.48</td>
      <td>2.78</td>
      <td>-8059.00</td>
      <td>124.35</td>
      <td>24.50</td>
    </tr>
    <tr>
      <th>18</th>
      <td>09:32</td>
      <td>18</td>
      <td>1727</td>
      <td>3454</td>
      <td>201892.24</td>
      <td>2.52</td>
      <td>2.39</td>
      <td>2.82</td>
      <td>-8799.76</td>
      <td>116.90</td>
      <td>22.94</td>
    </tr>
    <tr>
      <th>19</th>
      <td>09:52</td>
      <td>19</td>
      <td>1858</td>
      <td>3716</td>
      <td>206457.96</td>
      <td>2.47</td>
      <td>2.36</td>
      <td>2.70</td>
      <td>-10644.80</td>
      <td>111.12</td>
      <td>19.40</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10:13</td>
      <td>20</td>
      <td>1970</td>
      <td>3940</td>
      <td>211370.40</td>
      <td>2.40</td>
      <td>2.30</td>
      <td>2.61</td>
      <td>-12700.72</td>
      <td>107.29</td>
      <td>16.64</td>
    </tr>
  </tbody>
</table>
</div>


**The drawdown path:**

| step | added | trades | P/L | Sharpe | max drawdown | P/L per \$ of drawdown |
|---|---|---|---|---|---|---|
| 1 | 14:53 | 134 | \$11.7k | | -\$0.9k | 13.4 |
| 2 | 12:13 | 354 | \$28.9k | | -\$1.3k | 23.0 |
| 3 | 11:13 | 584 | \$43.6k | 0.83 | -\$1.6k | 27.0 |
| 4 | 14:13 | 766 | \$58.4k | 1.53 | -\$1.7k | 34.0 |
| 5 | 10:53 | 1,026 | \$68.7k | 1.65 | -\$1.9k | 35.6 |
| 6 | 15:53 | 1,028 | \$68.9k | 1.65 | -\$1.9k | 35.7 |
| 7 | 11:33 | 1,272 | \$85.2k | 1.93 | -\$2.5k | 33.7 |
| **8** | **13:53** | **1,474** | **\$102.8k** | **2.29** | **-\$2.6k** | **39.8** |
| 9 | 11:53 | 1,712 | \$120.9k | 2.47 | -\$3.3k | 36.6 |
| 12 | 13:13 | 2,156 | \$145.9k | 2.62 | -\$4.1k | 35.5 |
| 16 | 13:33 | 2,934 | \$192.1k | 2.66 | -\$7.8k | 24.7 |
| 20 | all | 3,940 | \$211.4k | 2.40 | -\$12.7k | 16.6 |

- **It peaks at 8 tranches: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53 and 15:53**, at \$39.8 of P/L per \$ of drawdown: 1,474 trades,
  \$102.8k, Sharpe 2.29 (2.28 / 2.31 in the two halves), drawdown -\$2.6k.
- **15:53 is a passenger:** with the ratio cap on it trades twice in four years, so it neither helps nor hurts; the working set is the
  other seven.
- **Past the peak, each tranche adds more drawdown than profit:** 11:53 and 13:13 still raise Sharpe (to 2.62 at 12) but deepen the
  drawdown to -\$4.1k; from 13 tranches on the drawdown grows fast (-\$5.2k to -\$12.7k).
- **Drawdown and Sharpe point to different sizes:** the ratio peaks at 8 tranches with Sharpe 2.29; Sharpe keeps rising to 2.66 at 16
  tranches, where the drawdown is three times as deep.

**Your 7 against the other sets, with every other setting on** (the finished strategy, only the tranches changed):


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>pl_per_dd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the strategy's 7</th>
      <td>1450.0</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>-3808.68</td>
      <td>29.42</td>
    </tr>
    <tr>
      <th>forward selection's peak (9)</th>
      <td>1892.0</td>
      <td>141864.52</td>
      <td>2.73</td>
      <td>2.64</td>
      <td>2.93</td>
      <td>-5975.20</td>
      <td>23.74</td>
    </tr>
    <tr>
      <th>forward selection by drawdown (8)</th>
      <td>1474.0</td>
      <td>102826.44</td>
      <td>2.29</td>
      <td>2.28</td>
      <td>2.31</td>
      <td>-2582.60</td>
      <td>39.82</td>
    </tr>
    <tr>
      <th>previous live set (15)</th>
      <td>2704.0</td>
      <td>180418.24</td>
      <td>2.69</td>
      <td>2.64</td>
      <td>2.82</td>
      <td>-6545.16</td>
      <td>27.57</td>
    </tr>
    <tr>
      <th>all 20</th>
      <td>3940.0</td>
      <td>211370.40</td>
      <td>2.40</td>
      <td>2.30</td>
      <td>2.61</td>
      <td>-12700.72</td>
      <td>16.64</td>
    </tr>
  </tbody>
</table>
</div>


| tranche set (everything else on) | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | P/L per \$ of drawdown |
|---|---|---|---|---|---|---|
| **the strategy's 7** | **1,450** | \$112.0k | 2.60 | 2.44 / 2.92 | -\$3.8k | 29.4 |
| forward selection's peak (the 7 + 11:13, 13:33) | 1,892 | \$141.9k | **2.73** | 2.64 / 2.93 | -\$6.0k | 23.7 |
| forward selection by drawdown (8) | 1,474 | \$102.8k | 2.29 | 2.28 / 2.31 | **-\$2.6k** | **39.8** |
| previous live set (15) | 2,704 | \$180.4k | 2.69 | 2.64 / 2.82 | -\$6.5k | 27.6 |
| all 20 | 3,940 | **\$211.4k** | 2.40 | 2.30 / 2.61 | -\$12.7k | 16.6 |

- **The drawdown pick has the smallest drawdown (-\$2.6k) and the most P/L per \$ of drawdown (39.8)**, but the lowest Sharpe here (2.29)
  and the weakest 2025-26 (2.31).
- **The 7 come next on drawdown (-\$3.8k, 29.4)**, with the best 2025-26 Sharpe alongside the Sharpe pick.
- **They don't have the highest Sharpe:** adding 11:13 and 13:33 (the 9) scores 2.73 for \$29.8k more, and the previous 15 score 2.69 for
  \$68.4k more, each with a drawdown \$2-3k deeper. Fewer tranches means fewer trading days, and every idle day pays the 4% hurdle.
- **The trade-off is size against depth:** the 7 are the smallest, steadiest book; the 9 and the 15 score a higher Sharpe but draw down
  further.

## 8. Step 6: the short/long ratio <= 4, on the book after steps 1-5

The backtester caps one short/long ratio for the whole condor: the two short strikes' prices over the two long strikes'.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spearman_condor</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>spearman_session</th>
      <th>pearson_condor</th>
      <th>p60_ratio</th>
      <th>spearman_below_p60</th>
      <th>spearman_above_p60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>whole condor</th>
      <td>0.156</td>
      <td>0.049</td>
      <td>0.262</td>
      <td>0.09</td>
      <td>-0.11</td>
      <td>2.651</td>
      <td>0.507</td>
      <td>-0.078</td>
    </tr>
  </tbody>
</table>
</div>


    partial rank correlation with P/L, controlling for what the ratio moves with:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>whole condor</th>
    </tr>
    <tr>
      <th>controlling for</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>none</th>
      <td>0.156</td>
    </tr>
    <tr>
      <th>credit</th>
      <td>-0.055</td>
    </tr>
    <tr>
      <th>strike distance (both sides)</th>
      <td>0.044</td>
    </tr>
    <tr>
      <th>entry time</th>
      <td>0.149</td>
    </tr>
    <tr>
      <th>VIX9D</th>
      <td>0.167</td>
    </tr>
    <tr>
      <th>credit, VIX9D, entry time</th>
      <td>-0.019</td>
    </tr>
  </tbody>
</table>
</div>


    what the ratio moves with (rank correlation):
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>whole condor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>credit</th>
      <td>0.89</td>
    </tr>
    <tr>
      <th>otm_points_call</th>
      <td>-0.95</td>
    </tr>
    <tr>
      <th>otm_points_put</th>
      <td>-0.96</td>
    </tr>
    <tr>
      <th>mins_from_open</th>
      <td>0.50</td>
    </tr>
    <tr>
      <th>vix9d_open</th>
      <td>-0.48</td>
    </tr>
    <tr>
      <th>vvix_prev</th>
      <td>-0.42</td>
    </tr>
  </tbody>
</table>
</div>


    rank correlation with P/L by year:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>whole condor</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>0.576</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>-0.006</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>-0.012</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>0.207</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>0.586</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](ic_analysis_files/output_54_7.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>side</th>
      <th>quintile</th>
      <th>range</th>
      <th>condors</th>
      <th>trades</th>
      <th>sessions</th>
      <th>win_rate</th>
      <th>avg_pl</th>
      <th>total_pl</th>
      <th>avg_pl_to_2024</th>
      <th>avg_pl_from_2025</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>whole condor</td>
      <td>1</td>
      <td>1.25 to 1.87</td>
      <td>159</td>
      <td>318</td>
      <td>67</td>
      <td>89.9</td>
      <td>134.2</td>
      <td>21345.1</td>
      <td>89.3</td>
      <td>156.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>whole condor</td>
      <td>2</td>
      <td>1.87 to 2.23</td>
      <td>158</td>
      <td>316</td>
      <td>86</td>
      <td>86.7</td>
      <td>197.0</td>
      <td>31123.0</td>
      <td>195.0</td>
      <td>200.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>whole condor</td>
      <td>3</td>
      <td>2.23 to 2.65</td>
      <td>158</td>
      <td>316</td>
      <td>95</td>
      <td>82.3</td>
      <td>187.7</td>
      <td>29660.0</td>
      <td>205.0</td>
      <td>151.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>whole condor</td>
      <td>4</td>
      <td>2.65 to 3.23</td>
      <td>158</td>
      <td>316</td>
      <td>90</td>
      <td>71.5</td>
      <td>116.7</td>
      <td>18437.0</td>
      <td>140.2</td>
      <td>37.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>whole condor</td>
      <td>5</td>
      <td>3.23 to 8.38</td>
      <td>158</td>
      <td>316</td>
      <td>79</td>
      <td>65.2</td>
      <td>81.1</td>
      <td>12806.0</td>
      <td>78.8</td>
      <td>92.5</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>kept_pct</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>side</th>
      <th>max_ratio</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">whole condor</th>
      <th>2.5</th>
      <td>54.36</td>
      <td>860</td>
      <td>73415.60</td>
      <td>2.03</td>
      <td>-3012.16</td>
      <td>1.62</td>
      <td>2.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73.96</td>
      <td>1170</td>
      <td>94966.20</td>
      <td>2.45</td>
      <td>-2357.04</td>
      <td>2.30</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>3.5</th>
      <td>85.34</td>
      <td>1350</td>
      <td>104427.00</td>
      <td>2.49</td>
      <td>-3271.64</td>
      <td>2.33</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91.66</td>
      <td>1450</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>94.94</td>
      <td>1502</td>
      <td>113803.12</td>
      <td>2.57</td>
      <td>-4716.16</td>
      <td>2.45</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>98.10</td>
      <td>1552</td>
      <td>112994.12</td>
      <td>2.50</td>
      <td>-5195.04</td>
      <td>2.36</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>5.5</th>
      <td>98.74</td>
      <td>1562</td>
      <td>112785.72</td>
      <td>2.48</td>
      <td>-4888.92</td>
      <td>2.33</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>99.37</td>
      <td>1572</td>
      <td>113926.32</td>
      <td>2.50</td>
      <td>-4888.92</td>
      <td>2.37</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>7</th>
      <td>99.87</td>
      <td>1580</td>
      <td>112947.80</td>
      <td>2.46</td>
      <td>-4888.92</td>
      <td>2.31</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>off</th>
      <td>100.00</td>
      <td>1582</td>
      <td>113370.92</td>
      <td>2.48</td>
      <td>-4888.92</td>
      <td>2.33</td>
      <td>2.83</td>
    </tr>
  </tbody>
</table>
</div>


**On the 7 tranches the ratio's relationship with P/L is a hump.** Rank correlation with condor P/L +0.16 (interval +0.05 to +0.26);
below its 60th percentile (2.65) +0.51, above it -0.08. The ratio is strike distance in another form (-0.95 to -0.96 with how far the shorts
sit from spot, +0.89 with the credit).

| condor short/long ratio | trades | avg P/L | win rate | 2022-24 / 2025-26 |
|---|---|---|---|---|
| 1.25-1.87 | 318 | +\$134 | 90% | +\$89 / +\$156 |
| 1.87-2.23 | 316 | **+\$197** | 87% | +\$195 / +\$201 |
| 2.23-2.65 | 316 | +\$188 | 82% | +\$205 / +\$152 |
| 2.65-3.23 | 316 | +\$117 | 72% | +\$140 / +\$37 |
| above 3.23 | 316 | +\$81 | **65%** | +\$79 / +\$93 |

| maximum ratio | trades | total P/L | Sharpe | max drawdown |
|---|---|---|---|---|
| 3.5 | 1,350 | \$104.4k | 2.49 | -\$3.3k |
| **4 (strategy)** | **1,450** | **\$112.0k** | **2.60** | -\$3.8k |
| 4.5 | 1,502 | \$113.8k | 2.57 | -\$4.7k |
| 5 | 1,552 | \$113.0k | 2.50 | -\$5.2k |
| no cap | 1,582 | \$113.4k | 2.48 | -\$4.9k |

- **On the 7 tranches the cap removes only 132 trades.** Those earn little (\$1.3k between them) and add risk: a cap of 4 is the peak (2.60
  against 2.48 with no cap), and the drawdown is \$1.1k smaller.

## 9. The call and put stops: why the strategy uses neither

Not a step: the strategy holds every condor to expiry. Here the finished strategy (all six settings) gets a call-side stop at every level,
replayed the backtester's way: it fires when the call spread's worst intraday loss reached X% of its credit (never on the 16:00 settlement
print) and fills where the option chain first crossed it, plus **\$0.20 per contract (\$20) slippage** and fees (blue); \$0 slippage for
reference (grey); the dotted line is the strategy, with no stop. Every level trades the same 1,450 trades.


    
![png](ic_analysis_files/output_57_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>call_stop_hit_rate</th>
      <th>sessions</th>
      <th>trades</th>
      <th>win_rate</th>
      <th>avg_pl</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>stop_pct</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>off</th>
      <td>0.00</td>
      <td>187</td>
      <td>1450</td>
      <td>80.97</td>
      <td>154.54</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
    </tr>
    <tr>
      <th>25</th>
      <td>69.79</td>
      <td>187</td>
      <td>1450</td>
      <td>82.07</td>
      <td>38.58</td>
      <td>27973.08</td>
      <td>-0.18</td>
      <td>-6408.37</td>
      <td>-0.27</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>50</th>
      <td>52.41</td>
      <td>187</td>
      <td>1450</td>
      <td>65.24</td>
      <td>57.99</td>
      <td>42039.20</td>
      <td>0.47</td>
      <td>-4868.86</td>
      <td>0.34</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>75</th>
      <td>42.62</td>
      <td>187</td>
      <td>1450</td>
      <td>53.79</td>
      <td>67.60</td>
      <td>49010.97</td>
      <td>0.75</td>
      <td>-4292.98</td>
      <td>0.55</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>100</th>
      <td>35.59</td>
      <td>187</td>
      <td>1450</td>
      <td>54.48</td>
      <td>78.41</td>
      <td>56848.43</td>
      <td>1.02</td>
      <td>-4292.98</td>
      <td>0.76</td>
      <td>1.52</td>
    </tr>
    <tr>
      <th>125</th>
      <td>28.55</td>
      <td>187</td>
      <td>1450</td>
      <td>60.41</td>
      <td>91.47</td>
      <td>66316.24</td>
      <td>1.35</td>
      <td>-4392.98</td>
      <td>1.09</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>150</th>
      <td>23.59</td>
      <td>187</td>
      <td>1450</td>
      <td>64.69</td>
      <td>99.98</td>
      <td>72482.71</td>
      <td>1.52</td>
      <td>-4392.98</td>
      <td>1.30</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>175</th>
      <td>19.03</td>
      <td>187</td>
      <td>1450</td>
      <td>68.97</td>
      <td>112.06</td>
      <td>81243.63</td>
      <td>1.76</td>
      <td>-4450.50</td>
      <td>1.58</td>
      <td>2.12</td>
    </tr>
    <tr>
      <th>200</th>
      <td>16.14</td>
      <td>187</td>
      <td>1450</td>
      <td>71.59</td>
      <td>120.19</td>
      <td>87141.21</td>
      <td>1.91</td>
      <td>-3882.56</td>
      <td>1.77</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>250</th>
      <td>11.45</td>
      <td>187</td>
      <td>1450</td>
      <td>75.72</td>
      <td>131.93</td>
      <td>95650.81</td>
      <td>2.16</td>
      <td>-4056.56</td>
      <td>2.11</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>300</th>
      <td>9.10</td>
      <td>187</td>
      <td>1450</td>
      <td>77.10</td>
      <td>135.28</td>
      <td>98074.45</td>
      <td>2.23</td>
      <td>-3982.62</td>
      <td>2.21</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>350</th>
      <td>6.48</td>
      <td>187</td>
      <td>1450</td>
      <td>78.62</td>
      <td>141.47</td>
      <td>102569.15</td>
      <td>2.37</td>
      <td>-3808.68</td>
      <td>2.31</td>
      <td>2.49</td>
    </tr>
    <tr>
      <th>400</th>
      <td>4.41</td>
      <td>187</td>
      <td>1450</td>
      <td>79.59</td>
      <td>145.98</td>
      <td>105832.53</td>
      <td>2.46</td>
      <td>-3808.68</td>
      <td>2.36</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>


| call stop at \$0.20 slippage | fires on | total P/L | Sharpe | max drawdown | 2022-24 / 2025-26 |
|---|---|---|---|---|---|
| **none (strategy)** | | **\$112.0k** | **2.60** | **-\$3.8k** | **2.44 / 2.92** |
| 100% | 36% of condors | \$56.8k | 1.02 | -\$4.3k | 0.76 / 1.52 |
| 200% | 16% | \$87.1k | 1.91 | -\$3.9k | 1.77 / 2.20 |
| 300% | 9% | \$98.1k | 2.23 | -\$4.0k | 2.21 / 2.29 |
| 400% | 4% | \$105.8k | 2.46 | -\$3.8k | 2.36 / 2.67 |

- **Every stop level lowers Sharpe and P/L, in both halves, and none makes the drawdown smaller.** The 400% stop the strategy used until now
  costs 0.14 Sharpe and \$6.2k for the same -\$3.8k drawdown.
- **The entry filters already do the stop's job:** with SPX at or below the open and above its 10-day SMA, the call side rarely runs away, so a
  stopped call usually would have recovered by the close. Tight stops are ruined by slippage and real fills (at 25-100% the stop fires on
  36-70% of condors and Sharpe falls to -0.18 to 1.02).
- **What the backtest can't show is a day far worse than any in these four years.** Going without a stop accepts that risk; the -\$3.8k
  drawdown is the worst the data shows, not the worst possible.

### The same test on the puts

The same replay on the put side: a put-side stop at every level, filled where the option chain first crossed it, plus \$0.20 per contract
slippage and fees (blue; \$0 slippage in grey), the call side held to expiry. The dotted line is the strategy, with no stop. Same 1,450
trades at every level.


    
![png](ic_analysis_files/output_60_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>put_stop_hit_rate</th>
      <th>sessions</th>
      <th>trades</th>
      <th>win_rate</th>
      <th>avg_pl</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>stop_pct</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>off</th>
      <td>0.00</td>
      <td>187</td>
      <td>1450</td>
      <td>80.97</td>
      <td>154.54</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
    </tr>
    <tr>
      <th>25</th>
      <td>74.76</td>
      <td>187</td>
      <td>1450</td>
      <td>84.00</td>
      <td>52.85</td>
      <td>38313.67</td>
      <td>0.34</td>
      <td>-4098.26</td>
      <td>0.40</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>50</th>
      <td>57.52</td>
      <td>187</td>
      <td>1450</td>
      <td>82.21</td>
      <td>66.15</td>
      <td>47958.01</td>
      <td>0.77</td>
      <td>-3926.68</td>
      <td>0.92</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>75</th>
      <td>47.03</td>
      <td>187</td>
      <td>1450</td>
      <td>71.17</td>
      <td>75.81</td>
      <td>54964.50</td>
      <td>1.03</td>
      <td>-3191.60</td>
      <td>1.29</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>100</th>
      <td>38.76</td>
      <td>187</td>
      <td>1450</td>
      <td>63.45</td>
      <td>86.71</td>
      <td>62867.32</td>
      <td>1.29</td>
      <td>-4687.50</td>
      <td>1.57</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>125</th>
      <td>32.83</td>
      <td>187</td>
      <td>1450</td>
      <td>59.03</td>
      <td>93.36</td>
      <td>67687.65</td>
      <td>1.40</td>
      <td>-5564.73</td>
      <td>1.73</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>150</th>
      <td>30.48</td>
      <td>187</td>
      <td>1450</td>
      <td>59.31</td>
      <td>91.11</td>
      <td>66053.96</td>
      <td>1.31</td>
      <td>-7358.33</td>
      <td>1.63</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>175</th>
      <td>26.07</td>
      <td>187</td>
      <td>1450</td>
      <td>62.62</td>
      <td>98.54</td>
      <td>71441.15</td>
      <td>1.44</td>
      <td>-8558.52</td>
      <td>1.90</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>200</th>
      <td>22.34</td>
      <td>187</td>
      <td>1450</td>
      <td>65.79</td>
      <td>107.43</td>
      <td>77883.22</td>
      <td>1.64</td>
      <td>-7830.13</td>
      <td>2.11</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>250</th>
      <td>18.76</td>
      <td>187</td>
      <td>1450</td>
      <td>69.10</td>
      <td>112.57</td>
      <td>81612.85</td>
      <td>1.73</td>
      <td>-5315.73</td>
      <td>2.06</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>300</th>
      <td>15.72</td>
      <td>187</td>
      <td>1450</td>
      <td>71.59</td>
      <td>115.66</td>
      <td>83851.58</td>
      <td>1.77</td>
      <td>-4637.92</td>
      <td>2.07</td>
      <td>1.13</td>
    </tr>
    <tr>
      <th>350</th>
      <td>13.10</td>
      <td>187</td>
      <td>1450</td>
      <td>73.52</td>
      <td>116.88</td>
      <td>84741.26</td>
      <td>1.75</td>
      <td>-4601.38</td>
      <td>1.99</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>400</th>
      <td>9.79</td>
      <td>187</td>
      <td>1450</td>
      <td>75.72</td>
      <td>124.47</td>
      <td>90244.15</td>
      <td>1.91</td>
      <td>-4544.33</td>
      <td>2.10</td>
      <td>1.51</td>
    </tr>
  </tbody>
</table>
</div>


| put stop at \$0.20 slippage | fires on | total P/L | Sharpe | max drawdown | 2022-24 / 2025-26 |
|---|---|---|---|---|---|
| **none (strategy)** | | **\$112.0k** | **2.60** | **-\$3.8k** | **2.44 / 2.92** |
| 100% | 39% of condors | \$62.9k | 1.29 | -\$4.7k | 1.57 / 0.72 |
| 200% | 22% | \$77.9k | 1.64 | -\$7.8k | 2.11 / 0.72 |
| 300% | 16% | \$83.9k | 1.77 | -\$4.6k | 2.07 / 1.13 |
| 400% | 10% | \$90.2k | 1.91 | -\$4.5k | 2.10 / 1.51 |

| a 400% stop on each side | fires on | spreads stopped | of them, would have expired winning | P/L of the stopped spreads: stopped / held | cost of the stop | Sharpe |
|---|---|---|---|---|---|---|
| call side | 4% of condors | 32 | 31% | -\$21.1k / -\$14.9k | -\$6.2k | 2.46 |
| **put side** | **10%** | **71** | **42%** | **-\$41.7k / -\$19.9k** | **-\$21.8k** | **1.91** |

- **A put stop costs far more than a call stop:** the 400% put stop takes Sharpe from 2.60 to 1.91 and P/L down \$21.8k (the call stop:
  2.46, -\$6.2k), and it makes the drawdown deeper (-\$4.5k), not shallower. Even with no slippage the best put stop scores 1.96. Only the 75%
  level trims the drawdown (to -\$3.2k), at a Sharpe of 1.03.
- **The put side touches its stop twice as often and recovers more often:** 71 put spreads reached 400% against 32 calls, and 42% of
  those puts would have expired winning (31% of the calls). Stopping them turns -\$19.9k of held losses into -\$41.7k.
- **The reason is the entry rule:** the strategy sells its condor after SPX has dipped below its open, in an uptrend (above its 10-day
  SMA). Those intraday dips usually revert by the close, so a put that is deep in trouble at 13:00 is often fine at 16:00, and a stop
  sells it at the worst point.
- **It hurts most in 2025-26** (2.92 -> 1.51 at 400%).
- **The 2026 drawdown (Section 14) can't be tested here:** its two put-side days (2026-06-16 and 06-17) are after the option-chain data
  ends. Whether a put stop would have cut them is a question for the backtester, but over 2022-2026 a put stop costs much more than it
  saves.

## 10. With everything on: is each threshold still at its best?

The cascade judges each threshold on the book built so far, before the later settings are added. Some look off mid-cascade (a +0.5% SMA10
bar at step 3). The real test is with **everything** on: each threshold moved one step at a time, every other setting fixed.


    
![png](ic_analysis_files/output_63_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>strategy_value</th>
      <th>strategy_sharpe</th>
      <th>best_value</th>
      <th>best_sharpe</th>
      <th>best_sharpe_to_2024</th>
      <th>best_sharpe_from_2025</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ts_high</th>
      <td>1.02</td>
      <td>2.6</td>
      <td>1.02</td>
      <td>2.6</td>
      <td>1.02</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>ts_low</th>
      <td>0.87</td>
      <td>2.6</td>
      <td>0.87</td>
      <td>2.6</td>
      <td>0.87</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>max_move</th>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sma10_min</th>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>2.6</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>kept_pct</th>
      <th>trades</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
    </tr>
    <tr>
      <th>side</th>
      <th>max_ratio</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">whole condor</th>
      <th>2.5</th>
      <td>54.36</td>
      <td>860</td>
      <td>73415.60</td>
      <td>2.03</td>
      <td>-3012.16</td>
      <td>1.62</td>
      <td>2.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73.96</td>
      <td>1170</td>
      <td>94966.20</td>
      <td>2.45</td>
      <td>-2357.04</td>
      <td>2.30</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>3.5</th>
      <td>85.34</td>
      <td>1350</td>
      <td>104427.00</td>
      <td>2.49</td>
      <td>-3271.64</td>
      <td>2.33</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91.66</td>
      <td>1450</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>94.94</td>
      <td>1502</td>
      <td>113803.12</td>
      <td>2.57</td>
      <td>-4716.16</td>
      <td>2.45</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>98.10</td>
      <td>1552</td>
      <td>112994.12</td>
      <td>2.50</td>
      <td>-5195.04</td>
      <td>2.36</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>5.5</th>
      <td>98.74</td>
      <td>1562</td>
      <td>112785.72</td>
      <td>2.48</td>
      <td>-4888.92</td>
      <td>2.33</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>99.37</td>
      <td>1572</td>
      <td>113926.32</td>
      <td>2.50</td>
      <td>-4888.92</td>
      <td>2.37</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>7</th>
      <td>99.87</td>
      <td>1580</td>
      <td>112947.80</td>
      <td>2.46</td>
      <td>-4888.92</td>
      <td>2.31</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>off</th>
      <td>100.00</td>
      <td>1582</td>
      <td>113370.92</td>
      <td>2.48</td>
      <td>-4888.92</td>
      <td>2.33</td>
      <td>2.83</td>
    </tr>
  </tbody>
</table>
</div>


| threshold | strategy | Sharpe | nearest alternatives | verdict |
|---|---|---|---|---|
| floor (VIX9D/VIX >=) | 0.87 | **2.60** | 0.85: 2.27; 0.89: 2.31 | **at the peak** |
| cap (VIX9D/VIX <=) | 1.02 | **2.60** | 1.03: 2.38; 1.04: 2.49 | **at the peak** |
| max move up | 0% | **2.60** | -0.05%: 2.26; +0.05%: 2.44 | **at the peak** |
| SMA10 bar | 0% | **2.60** | -0.5%: 2.06; +0.5%: 2.22 | **at the peak** |
| short/long cap | 4 | **2.60** | 3.5: 2.49; 4.5: 2.57 | **at the peak** |

**With everything on, every threshold is at the top of its sweep, by a clear margin.** The +0.5% SMA10 bar that led at step 3 was ahead only
because the band wasn't on yet.

## 11. Drawdowns by tranche: where the losses come from, and whether to cut the 7 further

Everything in this section is on **the finished strategy** (all six settings, held to expiry): 1,450 trades, \$112.0k, Sharpe 2.60, max
drawdown -\$3.8k. First, the book's drawdowns: the equity curve, the curve below its last peak, and its five deepest drawdowns.


    
![png](ic_analysis_files/output_66_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peak</th>
      <th>trough</th>
      <th>recovered</th>
      <th>depth</th>
      <th>sessions_to_trough</th>
      <th>sessions_under</th>
    </tr>
    <tr>
      <th>drawdown</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DD1</th>
      <td>2024-12-02</td>
      <td>2024-12-10</td>
      <td>2025-05-21</td>
      <td>-3809.0</td>
      <td>6</td>
      <td>115</td>
    </tr>
    <tr>
      <th>DD2</th>
      <td>2023-11-28</td>
      <td>2023-12-20</td>
      <td>2023-12-29</td>
      <td>-2543.0</td>
      <td>16</td>
      <td>21</td>
    </tr>
    <tr>
      <th>DD3</th>
      <td>2024-10-03</td>
      <td>2024-10-07</td>
      <td>2024-11-12</td>
      <td>-2338.0</td>
      <td>2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>DD4</th>
      <td>2025-05-27</td>
      <td>2025-06-02</td>
      <td>2025-06-17</td>
      <td>-1796.0</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>DD5</th>
      <td>2023-05-22</td>
      <td>2023-05-23</td>
      <td>2023-05-30</td>
      <td>-1721.0</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


| drawdown | peak | trough | recovered | depth | sessions to the trough |
|---|---|---|---|---|---|
| **DD1** | 2024-12-02 | 2024-12-10 | 2025-05-21 | **-\$3.8k** | 6 (115 under water) |
| DD2 | 2023-11-28 | 2023-12-20 | 2023-12-29 | -\$2.5k | 16 |
| DD3 | 2024-10-03 | 2024-10-07 | 2024-11-12 | -\$2.3k | 2 |
| DD4 | 2025-05-27 | 2025-06-02 | 2025-06-17 | -\$1.8k | 4 |
| DD5 | 2023-05-22 | 2023-05-23 | 2023-05-30 | -\$1.7k | 1 |

- **The deepest drawdown is -\$3.8k, reached in six sessions in December 2024**; recovering took until May 2025 (115 sessions).
- **The others are -\$1.7k to -\$2.5k**, each recovered within six weeks.

**Which tranches lost money in each drawdown**, from its peak to its trough:


    
![png](ic_analysis_files/output_68_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DD1</th>
      <th>DD2</th>
      <th>DD3</th>
      <th>DD4</th>
      <th>DD5</th>
    </tr>
    <tr>
      <th>tranche</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11:33</th>
      <td>-702.0</td>
      <td>336.0</td>
      <td>-729.0</td>
      <td>-513.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11:53</th>
      <td>-1150.0</td>
      <td>-61.0</td>
      <td>-699.0</td>
      <td>271.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12:13</th>
      <td>-1145.0</td>
      <td>-486.0</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13:13</th>
      <td>-813.0</td>
      <td>-919.0</td>
      <td>-296.0</td>
      <td>-744.0</td>
      <td>-634.0</td>
    </tr>
    <tr>
      <th>13:53</th>
      <td>0.0</td>
      <td>-707.0</td>
      <td>0.0</td>
      <td>-699.0</td>
      <td>-581.0</td>
    </tr>
    <tr>
      <th>14:13</th>
      <td>0.0</td>
      <td>-38.0</td>
      <td>0.0</td>
      <td>-559.0</td>
      <td>-506.0</td>
    </tr>
    <tr>
      <th>14:53</th>
      <td>0.0</td>
      <td>-669.0</td>
      <td>-614.0</td>
      <td>346.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>book</th>
      <td>-3809.0</td>
      <td>-2543.0</td>
      <td>-2338.0</td>
      <td>-1796.0</td>
      <td>-1721.0</td>
    </tr>
  </tbody>
</table>
</div>


- **DD1 is the late-morning tranches:** 11:53 (-\$1.15k), 12:13 (-\$1.14k), 13:13 (-\$0.8k) and 11:33 (-\$0.7k); the three afternoon tranches
  didn't lose in it.
- **The others are spread out:** 13:13 lost in all five (-\$0.3k to -\$0.9k), and with 13:53-14:53 it carries most of DD2, DD4 and DD5.

Each tranche's own drawdown, on one scale, and its P/L year by year:


    
![png](ic_analysis_files/output_70_0.png)
    



    
![png](ic_analysis_files/output_70_1.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trades</th>
      <th>total_pl</th>
      <th>max_drawdown</th>
      <th>pl_per_dd</th>
      <th>worst_day</th>
      <th>pl_in_worst_dd</th>
      <th>share_of_worst_dd_pct</th>
      <th>pl_on_20_worst_days</th>
      <th>pl_in_top_dds</th>
      <th>sharpe_if_dropped</th>
      <th>pl_if_dropped</th>
      <th>max_dd_if_dropped</th>
    </tr>
    <tr>
      <th>tranche</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11:33</th>
      <td>244</td>
      <td>16279.64</td>
      <td>-1853.68</td>
      <td>8.78</td>
      <td>-758.88</td>
      <td>-701.64</td>
      <td>18.42</td>
      <td>-5317.32</td>
      <td>-1607.16</td>
      <td>-0.20</td>
      <td>-16279.64</td>
      <td>701.64</td>
    </tr>
    <tr>
      <th>11:53</th>
      <td>238</td>
      <td>18079.28</td>
      <td>-2052.68</td>
      <td>8.81</td>
      <td>-798.88</td>
      <td>-1149.64</td>
      <td>30.18</td>
      <td>-6134.20</td>
      <td>-1638.28</td>
      <td>-0.22</td>
      <td>-18079.28</td>
      <td>807.24</td>
    </tr>
    <tr>
      <th>12:13</th>
      <td>220</td>
      <td>17204.20</td>
      <td>-1783.04</td>
      <td>9.65</td>
      <td>-758.88</td>
      <td>-1144.64</td>
      <td>30.05</td>
      <td>-3985.32</td>
      <td>-1529.40</td>
      <td>-0.21</td>
      <td>-17204.20</td>
      <td>975.40</td>
    </tr>
    <tr>
      <th>13:13</th>
      <td>230</td>
      <td>16368.80</td>
      <td>-1729.04</td>
      <td>9.47</td>
      <td>-743.88</td>
      <td>-812.76</td>
      <td>21.34</td>
      <td>-5624.44</td>
      <td>-3405.04</td>
      <td>-0.16</td>
      <td>-16368.80</td>
      <td>812.76</td>
    </tr>
    <tr>
      <th>13:53</th>
      <td>202</td>
      <td>17674.12</td>
      <td>-1595.52</td>
      <td>11.08</td>
      <td>-723.88</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-2638.04</td>
      <td>-1986.52</td>
      <td>-0.28</td>
      <td>-17674.12</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>14:13</th>
      <td>182</td>
      <td>14781.92</td>
      <td>-1508.64</td>
      <td>9.80</td>
      <td>-648.88</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-1766.04</td>
      <td>-1102.64</td>
      <td>-0.22</td>
      <td>-14781.92</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>14:53</th>
      <td>134</td>
      <td>11656.04</td>
      <td>-870.76</td>
      <td>13.39</td>
      <td>-668.88</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-51.40</td>
      <td>-936.64</td>
      <td>-0.20</td>
      <td>-11656.04</td>
      <td>-594.08</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
      <th>2025</th>
      <th>2026</th>
      <th>losing_years</th>
    </tr>
    <tr>
      <th>tranche</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11:33</th>
      <td>721.0</td>
      <td>7093.0</td>
      <td>4362.0</td>
      <td>2741.0</td>
      <td>1363.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11:53</th>
      <td>1393.0</td>
      <td>8215.0</td>
      <td>2865.0</td>
      <td>3862.0</td>
      <td>1744.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12:13</th>
      <td>2751.0</td>
      <td>4635.0</td>
      <td>3762.0</td>
      <td>4487.0</td>
      <td>1569.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13:13</th>
      <td>5168.0</td>
      <td>3873.0</td>
      <td>2242.0</td>
      <td>3590.0</td>
      <td>1496.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13:53</th>
      <td>3700.0</td>
      <td>900.0</td>
      <td>6336.0</td>
      <td>4987.0</td>
      <td>1750.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14:13</th>
      <td>4042.0</td>
      <td>440.0</td>
      <td>4473.0</td>
      <td>3768.0</td>
      <td>2059.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14:53</th>
      <td>3292.0</td>
      <td>1575.0</td>
      <td>544.0</td>
      <td>4491.0</td>
      <td>1754.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


| tranche | trades | total P/L | own max drawdown | P/L per \$ of drawdown | losing years | in DD1 | book without it: Sharpe / P/L / max DD |
|---|---|---|---|---|---|---|---|
| 11:33 | 244 | \$16.3k | -\$1.9k | 8.8 | 0 | -\$0.7k | -0.20 / -\$16.3k / +\$0.7k |
| 11:53 | 238 | **\$18.1k** | **-\$2.1k** | 8.8 | 0 | **-\$1.1k** | -0.22 / -\$18.1k / +\$0.8k |
| 12:13 | 220 | \$17.2k | -\$1.8k | 9.6 | 0 | **-\$1.1k** | -0.21 / -\$17.2k / +\$1.0k |
| 13:13 | 230 | \$16.4k | -\$1.7k | 9.5 | 0 | -\$0.8k | -0.16 / -\$16.4k / +\$0.8k |
| 13:53 | 202 | \$17.7k | -\$1.6k | 11.1 | 0 | \$0.0k | **-0.28** / -\$17.7k / \$0.0k |
| 14:13 | 182 | \$14.8k | -\$1.5k | 9.8 | 0 | \$0.0k | -0.22 / -\$14.8k / \$0.0k |
| 14:53 | 134 | \$11.7k | **-\$0.9k** | **13.4** | 0 | \$0.0k | -0.20 / -\$11.7k / -\$0.6k |

(A single tranche's Sharpe isn't shown: on its own it never clears the 4% hurdle on \$200k, so P/L per \$ of drawdown compares them instead.
"Book without it: max DD" is the change in the book's max drawdown: + means smaller.)

- **All seven made money in every year**, \$11.7k to \$18.1k each.
- **Dropping any one costs 0.16 to 0.28 Sharpe and \$12k-18k.** The four late-morning tranches each carry part of DD1, so dropping one would
  shrink it by \$0.7k-1.0k, but each is among the best earners.

**Cutting further, one tranche at a time.** Starting from the 7, each step drops the tranche whose removal gives the best book: by Sharpe
(blue), or by max drawdown (orange). The circle marks where each rule stops: the highest Sharpe, or (by drawdown) the most P/L per \$ of
drawdown.


    
![png](ic_analysis_files/output_72_0.png)
    


    cut by Sharpe: stops at 7 tranches, cutting nothing
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>removed</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>pl_per_dd</th>
    </tr>
    <tr>
      <th>kept</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>-</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>-3808.68</td>
      <td>29.42</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13:13</td>
      <td>95675.20</td>
      <td>2.43</td>
      <td>2.28</td>
      <td>2.73</td>
      <td>-2995.92</td>
      <td>31.94</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11:33</td>
      <td>79395.56</td>
      <td>2.18</td>
      <td>1.94</td>
      <td>2.64</td>
      <td>-2294.28</td>
      <td>34.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14:53</td>
      <td>67739.52</td>
      <td>1.86</td>
      <td>1.76</td>
      <td>2.03</td>
      <td>-2925.08</td>
      <td>23.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14:13</td>
      <td>52957.60</td>
      <td>1.35</td>
      <td>1.29</td>
      <td>1.49</td>
      <td>-3331.48</td>
      <td>15.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12:13</td>
      <td>35753.40</td>
      <td>0.38</td>
      <td>0.34</td>
      <td>0.45</td>
      <td>-1744.16</td>
      <td>20.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13:53</td>
      <td>18079.28</td>
      <td>-1.93</td>
      <td>-1.79</td>
      <td>-2.21</td>
      <td>-2052.68</td>
      <td>8.81</td>
    </tr>
  </tbody>
</table>
</div>


    cut by drawdown: stops at 4 tranches, cutting 11:53, 12:13, 13:13
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>removed</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>max_drawdown</th>
      <th>pl_per_dd</th>
    </tr>
    <tr>
      <th>kept</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>-</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>-3808.68</td>
      <td>29.42</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12:13</td>
      <td>94839.80</td>
      <td>2.39</td>
      <td>2.26</td>
      <td>2.65</td>
      <td>-2833.28</td>
      <td>33.47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13:13</td>
      <td>78471.00</td>
      <td>2.14</td>
      <td>2.01</td>
      <td>2.39</td>
      <td>-2041.64</td>
      <td>38.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11:53</td>
      <td>60391.72</td>
      <td>1.63</td>
      <td>1.45</td>
      <td>1.97</td>
      <td>-1424.40</td>
      <td>42.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14:13</td>
      <td>45609.80</td>
      <td>1.04</td>
      <td>0.85</td>
      <td>1.39</td>
      <td>-1375.64</td>
      <td>33.16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14:53</td>
      <td>33953.76</td>
      <td>0.21</td>
      <td>0.29</td>
      <td>0.04</td>
      <td>-1378.28</td>
      <td>24.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11:33</td>
      <td>17674.12</td>
      <td>-2.14</td>
      <td>-2.32</td>
      <td>-1.78</td>
      <td>-1595.52</td>
      <td>11.08</td>
    </tr>
  </tbody>
</table>
</div>


| tranches kept | cut by Sharpe: removed | Sharpe | P/L | max DD | cut by drawdown: removed | Sharpe | P/L | max DD |
|---|---|---|---|---|---|---|---|---|
| **7 (strategy)** | | **2.60** | **\$112.0k** | **-\$3.8k** | | 2.60 | \$112.0k | -\$3.8k |
| 6 | 13:13 | 2.43 | \$95.7k | -\$3.0k | 12:13 | 2.39 | \$94.8k | -\$2.8k |
| 5 | 11:33 | 2.18 | \$79.4k | -\$2.3k | 13:13 | 2.14 | \$78.5k | -\$2.0k |
| 4 | 14:53 | 1.86 | \$67.7k | -\$2.9k | 11:53 | 1.63 | \$60.4k | -\$1.4k |

- **By Sharpe there is nothing to cut:** every removal lowers it, so the rule keeps all 7.
- **By drawdown every cut shrinks the drawdown, and the P/L with it:** at 4 tranches the drawdown is -\$1.4k but the book makes \$60.4k at
  Sharpe 1.63.

**Do the cuts hold out of sample?** The same rules re-run for each year on the earlier years only:

    cut by Sharpe: cut vs the strategy's set out of sample, Sharpe difference 95% interval -0.69 to -0.31; cut better in 0% of resamples
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kept</th>
      <th>cut</th>
      <th>cut_sharpe</th>
      <th>strategy_set_sharpe</th>
      <th>cut_pl</th>
      <th>strategy_set_pl</th>
      <th>cut_max_dd</th>
      <th>strategy_set_max_dd</th>
    </tr>
    <tr>
      <th>test_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023</th>
      <td>5.0</td>
      <td>11:33, 11:53</td>
      <td>0.58</td>
      <td>2.48</td>
      <td>11424.08</td>
      <td>26732.80</td>
      <td>-4632.28</td>
      <td>-2542.80</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>7.0</td>
      <td>-</td>
      <td>2.08</td>
      <td>2.08</td>
      <td>24584.48</td>
      <td>24584.48</td>
      <td>-3808.68</td>
      <td>-3808.68</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>7.0</td>
      <td>-</td>
      <td>2.69</td>
      <td>2.69</td>
      <td>27925.56</td>
      <td>27925.56</td>
      <td>-1796.04</td>
      <td>-1796.04</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>7.0</td>
      <td>-</td>
      <td>3.59</td>
      <td>3.59</td>
      <td>11734.36</td>
      <td>11734.36</td>
      <td>-154.88</td>
      <td>-154.88</td>
    </tr>
    <tr>
      <th>out of sample</th>
      <td>NaN</td>
      <td>-</td>
      <td>2.04</td>
      <td>2.53</td>
      <td>75668.48</td>
      <td>90977.20</td>
      <td>-4632.28</td>
      <td>-3808.68</td>
    </tr>
  </tbody>
</table>
</div>


    cut by drawdown: cut vs the strategy's set out of sample, Sharpe difference 95% interval -1.88 to -0.97; cut better in 0% of resamples
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kept</th>
      <th>cut</th>
      <th>cut_sharpe</th>
      <th>strategy_set_sharpe</th>
      <th>cut_pl</th>
      <th>strategy_set_pl</th>
      <th>cut_max_dd</th>
      <th>strategy_set_max_dd</th>
    </tr>
    <tr>
      <th>test_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023</th>
      <td>3.0</td>
      <td>11:33, 11:53, 12:13, 14:53</td>
      <td>-0.61</td>
      <td>2.48</td>
      <td>5213.16</td>
      <td>26732.80</td>
      <td>-3918.08</td>
      <td>-2542.80</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>6.0</td>
      <td>13:13</td>
      <td>2.11</td>
      <td>2.08</td>
      <td>22342.00</td>
      <td>24584.48</td>
      <td>-2995.92</td>
      <td>-3808.68</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>3.0</td>
      <td>11:53, 12:13, 13:13, 14:53</td>
      <td>0.87</td>
      <td>2.69</td>
      <td>11496.64</td>
      <td>27925.56</td>
      <td>-1770.52</td>
      <td>-1796.04</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>4.0</td>
      <td>11:53, 12:13, 13:13</td>
      <td>2.63</td>
      <td>3.59</td>
      <td>6925.60</td>
      <td>11734.36</td>
      <td>-237.40</td>
      <td>-154.88</td>
    </tr>
    <tr>
      <th>out of sample</th>
      <td>NaN</td>
      <td>-</td>
      <td>1.11</td>
      <td>2.53</td>
      <td>45977.40</td>
      <td>90977.20</td>
      <td>-3918.08</td>
      <td>-3808.68</td>
    </tr>
  </tbody>
</table>
</div>


- **By Sharpe, the rule cuts nothing in 2024-26**; only in 2023, with a single year of history, does it cut (11:33 and 11:53), and that
  year it loses (0.58 against 2.48).
- **By drawdown, the cuts lose out of sample** (1.11 against 2.53, \$46.0k against \$91.0k).

**What this says about cutting the selection:** nothing. Every one of the 7 made money every year, each one's removal lowers Sharpe, and the
drawdowns are shallow market episodes that a smaller book would only make smaller in proportion.

## 12. Feature ranking and correlation

Every feature on one Sharpe scale, against the finished strategy:

- **the strategy's settings (blue, bold, "ON:"):** the Sharpe **lost** when the setting is removed from the finished strategy;
- **every other feature (orange):** the Sharpe **gained** by its best single cut as a further filter (a min or max at a decile, keeping
  at least 30% of condors). Dark orange means the gain holds in both halves;
- **a noise feature (grey):** a random number per session put through the same search: **whatever it scores is what luck alone finds.**

The red line is the strict noise bar: the 95th percentile of the same search on 200 random features. The candidates are price,
volatility, curve and calendar features, including the 20-day SMA (now a candidate, not a setting) and the vol-regime feature library's
findings (VIX9D gap, month end, rule W, ATR20, low VRP).


    
![png](ic_analysis_files/output_77_0.png)
    


    the noise feature gains -0.11; strict noise bar (95th pct of 200 noise features) +0.01
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>trades</th>
      <th>win_rate</th>
      <th>avg_pl</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>sharpe_vs_strategy</th>
      <th>total_pl_vs_strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strategy</th>
      <td>187.0</td>
      <td>1450.0</td>
      <td>80.97</td>
      <td>154.54</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>with a 400% call stop (not traded)</th>
      <td>187.0</td>
      <td>1450.0</td>
      <td>79.59</td>
      <td>145.98</td>
      <td>105832.53</td>
      <td>2.46</td>
      <td>-3808.68</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>-0.14</td>
      <td>-6211.47</td>
    </tr>
    <tr>
      <th>without the band</th>
      <td>277.0</td>
      <td>2324.0</td>
      <td>75.99</td>
      <td>100.00</td>
      <td>116202.44</td>
      <td>1.89</td>
      <td>-5095.72</td>
      <td>1.84</td>
      <td>1.96</td>
      <td>-0.71</td>
      <td>4158.44</td>
    </tr>
    <tr>
      <th>without move &lt;= 0%</th>
      <td>365.0</td>
      <td>3946.0</td>
      <td>70.20</td>
      <td>62.05</td>
      <td>122433.76</td>
      <td>1.42</td>
      <td>-11084.88</td>
      <td>1.37</td>
      <td>1.52</td>
      <td>-1.18</td>
      <td>10389.76</td>
    </tr>
    <tr>
      <th>without SPX &gt; SMA10</th>
      <td>319.0</td>
      <td>2864.0</td>
      <td>73.32</td>
      <td>70.42</td>
      <td>100836.84</td>
      <td>1.32</td>
      <td>-11768.68</td>
      <td>1.39</td>
      <td>1.15</td>
      <td>-1.28</td>
      <td>-11207.16</td>
    </tr>
    <tr>
      <th>trading Thursdays</th>
      <td>237.0</td>
      <td>1862.0</td>
      <td>76.69</td>
      <td>115.50</td>
      <td>107530.72</td>
      <td>2.03</td>
      <td>-10773.60</td>
      <td>2.02</td>
      <td>2.05</td>
      <td>-0.56</td>
      <td>-4513.28</td>
    </tr>
    <tr>
      <th>keeping the morning tranches</th>
      <td>288.0</td>
      <td>3940.0</td>
      <td>76.95</td>
      <td>107.29</td>
      <td>211370.40</td>
      <td>2.40</td>
      <td>-12700.72</td>
      <td>2.30</td>
      <td>2.61</td>
      <td>-0.20</td>
      <td>99326.40</td>
    </tr>
    <tr>
      <th>without short/long ratio &lt;= 4</th>
      <td>188.0</td>
      <td>1582.0</td>
      <td>79.14</td>
      <td>143.33</td>
      <td>113370.92</td>
      <td>2.48</td>
      <td>-4888.92</td>
      <td>2.33</td>
      <td>2.83</td>
      <td>-0.12</td>
      <td>1326.92</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>role</th>
      <th>rule</th>
      <th>sharpe_contribution</th>
      <th>contribution_to_2024</th>
      <th>contribution_from_2025</th>
      <th>total_pl_effect</th>
      <th>holds_in_both_halves</th>
      <th>verdict</th>
    </tr>
    <tr>
      <th>rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ON: above 10-day SMA (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>1.28</td>
      <td>1.05</td>
      <td>1.77</td>
      <td>11207.16</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ON: move up max 0% (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>1.18</td>
      <td>1.07</td>
      <td>1.40</td>
      <td>-10389.76</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ON: VIX9D/VIX 0.87-1.02 band (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>0.71</td>
      <td>0.60</td>
      <td>0.96</td>
      <td>-4158.44</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ON: no Thursdays (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>0.56</td>
      <td>0.42</td>
      <td>0.87</td>
      <td>4513.28</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ON: forward-selected tranches (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.31</td>
      <td>-99326.40</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ON: max short/long 4, whole condor (in strategy)</td>
      <td>in strategy</td>
      <td>as set</td>
      <td>0.12</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>-1326.92</td>
      <td>True</td>
      <td>filter: pays in both halves</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SPX 5-day return, prior close, %</td>
      <td>candidate</td>
      <td>ret5_prev_pct &gt;= 0.302</td>
      <td>0.10</td>
      <td>-0.03</td>
      <td>0.34</td>
      <td>-2357.64</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>8</th>
      <td>VIX at the open vs 5 sessions ago, %</td>
      <td>candidate</td>
      <td>vix_5d_chg_pct &lt;= 6.111</td>
      <td>-0.01</td>
      <td>0.14</td>
      <td>-0.32</td>
      <td>-5336.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>9</th>
      <td>premium collected, whole condor</td>
      <td>candidate</td>
      <td>credit &lt;= 425</td>
      <td>-0.01</td>
      <td>-0.05</td>
      <td>0.05</td>
      <td>-5402.32</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>10</th>
      <td>put credit minus call credit, $</td>
      <td>candidate</td>
      <td>skew_credit &lt;= 10</td>
      <td>-0.04</td>
      <td>-0.08</td>
      <td>0.02</td>
      <td>-8596.40</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SPX SMA5 &gt; SMA9, prior close</td>
      <td>candidate</td>
      <td>sma5_gt_sma9 = 1</td>
      <td>-0.04</td>
      <td>-0.12</td>
      <td>0.14</td>
      <td>-5625.16</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SPX high-low range from 09:32 to entry, %</td>
      <td>candidate</td>
      <td>range_so_far &gt;= 0.2032</td>
      <td>-0.05</td>
      <td>-0.08</td>
      <td>0.02</td>
      <td>-6156.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>13</th>
      <td>VIX-implied daily move / 5-day average range</td>
      <td>candidate</td>
      <td>implied_vs_range5 &lt;= 2.072</td>
      <td>-0.06</td>
      <td>0.26</td>
      <td>-0.73</td>
      <td>-5739.40</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>14</th>
      <td>VIX at the open</td>
      <td>candidate</td>
      <td>vix30_open &gt;= 13.47</td>
      <td>-0.11</td>
      <td>-0.18</td>
      <td>0.00</td>
      <td>-7436.64</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SPX opening gap, %</td>
      <td>candidate</td>
      <td>gap_pct &gt;= -0.3433</td>
      <td>-0.11</td>
      <td>-0.05</td>
      <td>-0.25</td>
      <td>-7632.64</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>16</th>
      <td>random number per session (control)</td>
      <td>noise control</td>
      <td>noise &gt;= -1.337</td>
      <td>-0.11</td>
      <td>-0.09</td>
      <td>-0.15</td>
      <td>-7492.64</td>
      <td>False</td>
      <td>noise: what luck alone scores</td>
    </tr>
    <tr>
      <th>17</th>
      <td>SPX up days in the last 5 sessions</td>
      <td>candidate</td>
      <td>up_days_5 &gt;= 2</td>
      <td>-0.12</td>
      <td>-0.21</td>
      <td>0.07</td>
      <td>-5632.24</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>18</th>
      <td>VIX9D/VIX30 change over 5 sessions</td>
      <td>candidate</td>
      <td>ts_5d_chg &lt;= 0.0627</td>
      <td>-0.13</td>
      <td>0.02</td>
      <td>-0.44</td>
      <td>-6997.28</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SPX change over the 20 minutes before entry, %</td>
      <td>candidate</td>
      <td>ret_last_tranche &lt;= 0.1214</td>
      <td>-0.13</td>
      <td>-0.21</td>
      <td>-0.00</td>
      <td>-10580.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>20</th>
      <td>SPX 10-day realized vol, prior close, % (annua...</td>
      <td>candidate</td>
      <td>rv10_prev &gt;= 8.526</td>
      <td>-0.13</td>
      <td>0.23</td>
      <td>-0.91</td>
      <td>-11570.16</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>21</th>
      <td>prior day |close - open| / range</td>
      <td>candidate</td>
      <td>day_efficiency_prev &lt;= 1.026</td>
      <td>-0.13</td>
      <td>-0.14</td>
      <td>-0.15</td>
      <td>-8038.04</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SPX prior close below its 20-day high, %</td>
      <td>candidate</td>
      <td>spx_off_20d_high_pct &lt;= 0</td>
      <td>-0.14</td>
      <td>-0.21</td>
      <td>0.00</td>
      <td>-5296.52</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>23</th>
      <td>VVIX 252-day rank, prior close</td>
      <td>candidate</td>
      <td>vvix_rank252 &lt;= 33.88</td>
      <td>-0.14</td>
      <td>-0.20</td>
      <td>-0.06</td>
      <td>-8535.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>24</th>
      <td>VIX9D at the open</td>
      <td>candidate</td>
      <td>vix9d_open &gt;= 12.66</td>
      <td>-0.15</td>
      <td>-0.23</td>
      <td>0.00</td>
      <td>-8001.64</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>25</th>
      <td>VIX9D overnight gap, open vs prior close, %</td>
      <td>candidate</td>
      <td>vix9d_gap_pct &gt;= -2.531</td>
      <td>-0.15</td>
      <td>-0.06</td>
      <td>-0.32</td>
      <td>-7628.40</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>26</th>
      <td>VIX at the open vs its 20-day SMA, %</td>
      <td>candidate</td>
      <td>vix_vs_sma20_pct &lt;= 3.614</td>
      <td>-0.15</td>
      <td>-0.02</td>
      <td>-0.44</td>
      <td>-8762.04</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>27</th>
      <td>SPX prior day's high-low range, %</td>
      <td>candidate</td>
      <td>prior_day_range_pct &gt;= 0.3294</td>
      <td>-0.15</td>
      <td>-0.04</td>
      <td>-0.38</td>
      <td>-8552.40</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SPX 20-day ATR, % of prior close</td>
      <td>candidate</td>
      <td>atr20_pct &gt;= 0.6738</td>
      <td>-0.18</td>
      <td>0.01</td>
      <td>-0.59</td>
      <td>-10952.68</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>29</th>
      <td>SPX at entry vs SMA5 of prior closes, %</td>
      <td>candidate</td>
      <td>spx_entry_over_sma5 &gt;= -0.3165</td>
      <td>-0.20</td>
      <td>-0.20</td>
      <td>-0.22</td>
      <td>-11123.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>30</th>
      <td>VIX9D/VIX30 change, prior close to open</td>
      <td>candidate</td>
      <td>ts_overnight_chg &gt;= -0.005313</td>
      <td>-0.21</td>
      <td>-0.11</td>
      <td>-0.42</td>
      <td>-8427.52</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>31</th>
      <td>SPX at entry vs its 20-day SMA (backtester rul...</td>
      <td>candidate</td>
      <td>spx_vs_sma20_pct &gt;= 0.8513</td>
      <td>-0.22</td>
      <td>-0.32</td>
      <td>-0.02</td>
      <td>-14645.44</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>32</th>
      <td>VIX9D gap in [-3.3%, -0.8%) (1) or not (0)</td>
      <td>candidate</td>
      <td>vix9d_gap_band = 0</td>
      <td>-0.24</td>
      <td>-0.17</td>
      <td>-0.36</td>
      <td>-10732.16</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>33</th>
      <td>last session of the month (1) or not (0)</td>
      <td>candidate</td>
      <td>month_end = 0</td>
      <td>-0.24</td>
      <td>-0.22</td>
      <td>-0.26</td>
      <td>-9216.84</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SPX below its intraday high at entry, %</td>
      <td>candidate</td>
      <td>dd_from_hi &lt;= -0.09967</td>
      <td>-0.24</td>
      <td>-0.24</td>
      <td>-0.24</td>
      <td>-11949.76</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>35</th>
      <td>VIX9D prior close / prior day's Parkinson vol</td>
      <td>candidate</td>
      <td>ivrv_vix9d_pk1 &lt;= 4.449</td>
      <td>-0.24</td>
      <td>-0.10</td>
      <td>-0.54</td>
      <td>-10721.64</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>36</th>
      <td>monthly options-expiration week (1) or not (0)</td>
      <td>candidate</td>
      <td>opex_week = 0</td>
      <td>-0.36</td>
      <td>-0.55</td>
      <td>0.05</td>
      <td>-16632.16</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>37</th>
      <td>VIX prior close minus SPX 20-day realized vol</td>
      <td>candidate</td>
      <td>vrp20 &gt;= -1.797</td>
      <td>-0.37</td>
      <td>-0.44</td>
      <td>-0.27</td>
      <td>-16394.80</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>38</th>
      <td>IV rank of the VIX open, 90 sessions</td>
      <td>candidate</td>
      <td>ivr90 &lt;= 32.02</td>
      <td>-0.56</td>
      <td>-0.76</td>
      <td>-0.22</td>
      <td>-23775.00</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>39</th>
      <td>VIX1D at the open</td>
      <td>candidate</td>
      <td>vix1d_open &gt;= 6.806</td>
      <td>-0.69</td>
      <td>-0.99</td>
      <td>-0.23</td>
      <td>-32021.08</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
    <tr>
      <th>40</th>
      <td>VIX1D / VIX30 at the open</td>
      <td>candidate</td>
      <td>ts_1d_30_open &gt;= 0.4698</td>
      <td>-0.76</td>
      <td>-0.91</td>
      <td>-0.55</td>
      <td>-33983.08</td>
      <td>False</td>
      <td>candidate: not useful</td>
    </tr>
  </tbody>
</table>
</div>


**The noise feature scores -0.11**, and the strict bar is **+0.01**.

| removed from the finished strategy | trades | Sharpe | change | total P/L change |
|---|---|---|---|---|
| above the 10-day SMA | 2,864 | 1.32 | **-1.28** | -\$11.2k |
| max move up 0% | 3,946 | 1.42 | **-1.18** | +\$10.4k |
| the VIX9D/VIX band | 2,324 | 1.89 | **-0.71** | +\$4.2k |
| no Thursdays | 1,862 | 2.03 | **-0.56** | -\$4.5k |
| the 7 tranches (trade all 20) | 3,940 | 2.40 | -0.20 | +\$99.3k |
| short/long ratio <= 4 | 1,582 | 2.48 | -0.12 | +\$1.3k |
| *added: a 400% call stop* | 1,450 | 2.46 | *-0.14* | *-\$6.2k* |

- **Every setting pays, in both halves.** The 10-day SMA is now the most important: without it Sharpe falls to 1.32, the book loses \$11.2k
  and the drawdown triples (-\$3.8k to -\$11.8k).
- **No candidate adds in both halves.** SPX's 5-day return (+0.10) clears the noise bar overall but only from 2025 (-0.03 in 2022-24).
  SMA5 > SMA9 loses (-0.04).


    
![png](ic_analysis_files/output_79_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_a</th>
      <th>feature_b</th>
      <th>spearman</th>
      <th>corr_with_pl_a</th>
      <th>corr_with_pl_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VIX</td>
      <td>VIX9D</td>
      <td>0.96</td>
      <td>-0.06</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VIX9D gap</td>
      <td>VIX9D/VIX30 overnight</td>
      <td>0.91</td>
      <td>0.06</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>prior-day range</td>
      <td>VIX9D / Parkinson vol</td>
      <td>-0.89</td>
      <td>0.03</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S/L ratio, condor</td>
      <td>credit</td>
      <td>0.89</td>
      <td>0.26</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>range so far</td>
      <td>below intraday high</td>
      <td>-0.80</td>
      <td>0.05</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VIX9D</td>
      <td>ATR20</td>
      <td>0.80</td>
      <td>-0.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>VIX1D</td>
      <td>VIX1D/VIX30</td>
      <td>0.77</td>
      <td>-0.12</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SPX vs SMA10</td>
      <td>SPX at entry vs SMA5</td>
      <td>0.77</td>
      <td>0.09</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10-day realized vol</td>
      <td>ATR20</td>
      <td>0.77</td>
      <td>0.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VIX</td>
      <td>ATR20</td>
      <td>0.75</td>
      <td>-0.06</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>move since open</td>
      <td>below intraday high</td>
      <td>0.75</td>
      <td>-0.02</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SPX vs SMA10</td>
      <td>5-day return</td>
      <td>0.74</td>
      <td>0.09</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ATR20</td>
      <td>VRP (VIX - RV20)</td>
      <td>-0.68</td>
      <td>0.00</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>move since open</td>
      <td>range so far</td>
      <td>-0.66</td>
      <td>-0.02</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>14</th>
      <td>VIX9D</td>
      <td>VIX1D</td>
      <td>0.66</td>
      <td>-0.07</td>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10-day realized vol</td>
      <td>VIX9D</td>
      <td>0.65</td>
      <td>0.07</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>16</th>
      <td>implied / 5-day range</td>
      <td>prior-day range</td>
      <td>-0.63</td>
      <td>-0.04</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VIX vs its SMA20</td>
      <td>SPX vs SMA20</td>
      <td>-0.63</td>
      <td>-0.10</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>VIX</td>
      <td>10-day realized vol</td>
      <td>0.61</td>
      <td>-0.06</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr_with_pl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>credit</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th>S/L ratio, condor</th>
      <td>0.26</td>
    </tr>
    <tr>
      <th>SPX vs SMA20</th>
      <td>0.17</td>
    </tr>
    <tr>
      <th>below 20-day high</th>
      <td>0.16</td>
    </tr>
    <tr>
      <th>IV rank 90</th>
      <td>-0.14</td>
    </tr>
    <tr>
      <th>VIX1D</th>
      <td>-0.12</td>
    </tr>
    <tr>
      <th>VIX9D/VIX at entry</th>
      <td>-0.11</td>
    </tr>
    <tr>
      <th>VIX vs its SMA20</th>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>SPX vs SMA10</th>
      <td>0.09</td>
    </tr>
    <tr>
      <th>up days (5)</th>
      <td>0.09</td>
    </tr>
    <tr>
      <th>month end</th>
      <td>0.08</td>
    </tr>
    <tr>
      <th>put-call credit skew</th>
      <td>-0.08</td>
    </tr>
    <tr>
      <th>10-day realized vol</th>
      <td>0.07</td>
    </tr>
    <tr>
      <th>OPEX week</th>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>VIX9D / Parkinson vol</th>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>noise</th>
      <td>0.07</td>
    </tr>
    <tr>
      <th>VIX9D</th>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>VRP (VIX - RV20)</th>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>VIX9D gap</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>VIX</th>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>VIX9D/VIX30 overnight</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>VIX1D/VIX30</th>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>VIX 5-day change</th>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>range so far</th>
      <td>0.05</td>
    </tr>
    <tr>
      <th>last 20 min move</th>
      <td>0.05</td>
    </tr>
    <tr>
      <th>SPX at entry vs SMA5</th>
      <td>0.05</td>
    </tr>
    <tr>
      <th>VVIX rank</th>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>prior-day efficiency</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>VIX9D/VIX30 5-day change</th>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>5-day return</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>implied / 5-day range</th>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>SMA5 &gt; SMA9</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>prior-day range</th>
      <td>0.03</td>
    </tr>
    <tr>
      <th>move since open</th>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>gap</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>below intraday high</th>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>VIX9D gap band</th>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>ATR20</th>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>sharpe_change</th>
      <th>sharpe_to_2024_change</th>
      <th>sharpe_from_2025_change</th>
      <th>total_pl_change</th>
      <th>better_in_both_halves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strategy</th>
      <td>187.0</td>
      <td>112044.00</td>
      <td>2.60</td>
      <td>-3808.68</td>
      <td>2.44</td>
      <td>2.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ VVIX 252-day rank &lt;= 25</th>
      <td>135.0</td>
      <td>83062.76</td>
      <td>1.93</td>
      <td>-4843.48</td>
      <td>1.75</td>
      <td>2.26</td>
      <td>-0.67</td>
      <td>-0.70</td>
      <td>-0.66</td>
      <td>-28981.24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ VVIX 252-day rank &lt;= 30</th>
      <td>157.0</td>
      <td>101696.60</td>
      <td>2.41</td>
      <td>-3808.68</td>
      <td>2.23</td>
      <td>2.77</td>
      <td>-0.18</td>
      <td>-0.22</td>
      <td>-0.16</td>
      <td>-10347.40</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ VVIX 252-day rank &lt;= 35</th>
      <td>166.0</td>
      <td>106202.20</td>
      <td>2.53</td>
      <td>-3808.68</td>
      <td>2.36</td>
      <td>2.87</td>
      <td>-0.07</td>
      <td>-0.09</td>
      <td>-0.06</td>
      <td>-5841.80</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ VVIX 252-day rank &lt;= 40</th>
      <td>176.0</td>
      <td>108349.80</td>
      <td>2.56</td>
      <td>-3808.68</td>
      <td>2.40</td>
      <td>2.87</td>
      <td>-0.04</td>
      <td>-0.04</td>
      <td>-0.05</td>
      <td>-3694.20</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ VVIX 252-day rank &lt;= 50</th>
      <td>186.0</td>
      <td>111784.76</td>
      <td>2.59</td>
      <td>-3808.68</td>
      <td>2.43</td>
      <td>2.92</td>
      <td>-0.01</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-259.24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ condor credit &lt;= $450</th>
      <td>186.0</td>
      <td>109223.16</td>
      <td>2.59</td>
      <td>-3549.80</td>
      <td>2.42</td>
      <td>2.94</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>0.02</td>
      <td>-2820.84</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ SPX gap &gt;= -0.5%</th>
      <td>174.0</td>
      <td>108674.80</td>
      <td>2.55</td>
      <td>-3808.68</td>
      <td>2.48</td>
      <td>2.71</td>
      <td>-0.04</td>
      <td>0.03</td>
      <td>-0.21</td>
      <td>-3369.20</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ SPX gap &gt;= -0.25%</th>
      <td>155.0</td>
      <td>97596.52</td>
      <td>2.30</td>
      <td>-3808.68</td>
      <td>2.21</td>
      <td>2.47</td>
      <td>-0.30</td>
      <td>-0.23</td>
      <td>-0.45</td>
      <td>-14447.48</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ SPX gap &gt;= 0%</th>
      <td>111.0</td>
      <td>75546.96</td>
      <td>1.78</td>
      <td>-2440.52</td>
      <td>1.64</td>
      <td>2.06</td>
      <td>-0.81</td>
      <td>-0.80</td>
      <td>-0.86</td>
      <td>-36497.04</td>
      <td>False</td>
    </tr>
    <tr>
      <th>+ SMA5 &gt; SMA9</th>
      <td>164.0</td>
      <td>106418.84</td>
      <td>2.55</td>
      <td>-4647.88</td>
      <td>2.32</td>
      <td>3.07</td>
      <td>-0.04</td>
      <td>-0.12</td>
      <td>0.14</td>
      <td>-5625.16</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


**The features fall into a few clusters that say the same thing twice:** vol level (VIX and VIX9D: 0.96), premium (credit and the
condor's ratio: 0.89), trend (SPX vs SMA10 with SPX vs SMA5 and the 5-day return: 0.74-0.77), intraday path (range so far, below the
intraday high, move since the open: 0.75-0.80). **VIX9D/VIX at entry stands on its own**: it measures the curve's shape, not its level.
Correlation with P/L is small everywhere (credit +0.30, the ratio +0.26, SPX vs SMA20 +0.17, the rest within ±0.16).

| on top of the strategy | sessions | Sharpe | change | total P/L change | both halves |
|---|---|---|---|---|---|
| SMA5 > SMA9 | 164 | 2.55 | -0.04 | -\$5.6k | no (-0.12 in 2022-24, +0.14 in 2025-26) |
| condor credit <= \$450 | 186 | 2.59 | -0.01 | -\$2.8k | no |
| VVIX rank <= 35 | 166 | 2.53 | -0.07 | -\$5.8k | no |
| SPX gap >= 0% | 111 | 1.78 | -0.81 | -\$36.5k | no |

**Nothing tested improves the finished strategy.**

## 13. A/B with everything on: above the 10-day SMA vs the 20-day

No cascade here: **every other setting is on** (no Thursdays, max move 0%, VIX9D/VIX 0.87-1.02, the 7 tranches, ratio <= 4, held to
expiry), and only the trend rule changes: SPX at entry above its 10-day SMA (A, the strategy) against above its 20-day SMA (B), and above
both. Both SMAs are measured the backtester's way (the prior N - 1 closes and the entry price). The Sharpe difference gets a paired session
bootstrap, and two t-tests say how significant each difference is (below).


    
![png](ic_analysis_files/output_82_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A: above 10-day SMA (strategy)</th>
      <th>B: above 20-day SMA</th>
      <th>above both</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trades</th>
      <td>1450.00</td>
      <td>1778.00</td>
      <td>1320.00</td>
    </tr>
    <tr>
      <th>sessions</th>
      <td>187.00</td>
      <td>212.00</td>
      <td>170.00</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>112044.00</td>
      <td>105672.68</td>
      <td>102361.20</td>
    </tr>
    <tr>
      <th>sharpe</th>
      <td>2.60</td>
      <td>1.97</td>
      <td>2.38</td>
    </tr>
    <tr>
      <th>sharpe_to_2024</th>
      <td>2.44</td>
      <td>1.66</td>
      <td>2.16</td>
    </tr>
    <tr>
      <th>sharpe_from_2025</th>
      <td>2.92</td>
      <td>2.70</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>max_drawdown</th>
      <td>-3808.68</td>
      <td>-9093.40</td>
      <td>-3808.68</td>
    </tr>
    <tr>
      <th>win_rate_trade</th>
      <td>87.38</td>
      <td>85.38</td>
      <td>87.05</td>
    </tr>
    <tr>
      <th>win_rate_condor</th>
      <td>80.97</td>
      <td>76.60</td>
      <td>80.45</td>
    </tr>
    <tr>
      <th>avg_pl_condor</th>
      <td>154.54</td>
      <td>118.87</td>
      <td>155.09</td>
    </tr>
    <tr>
      <th>win_rate_spread</th>
      <td>87.38</td>
      <td>85.38</td>
      <td>87.05</td>
    </tr>
    <tr>
      <th>sharpe_vs_A</th>
      <td>0.00</td>
      <td>-0.63</td>
      <td>-0.21</td>
    </tr>
    <tr>
      <th>ci_low</th>
      <td>NaN</td>
      <td>-1.33</td>
      <td>-0.47</td>
    </tr>
    <tr>
      <th>ci_high</th>
      <td>NaN</td>
      <td>0.07</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>p_better_than_A</th>
      <td>NaN</td>
      <td>0.04</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>t_daily_pl</th>
      <td>NaN</td>
      <td>-0.54</td>
      <td>-2.16</td>
    </tr>
    <tr>
      <th>p_daily_pl</th>
      <td>NaN</td>
      <td>0.59</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>t_sharpe</th>
      <td>NaN</td>
      <td>-1.73</td>
      <td>-1.65</td>
    </tr>
    <tr>
      <th>p_sharpe</th>
      <td>NaN</td>
      <td>0.08</td>
      <td>0.10</td>
    </tr>
  </tbody>
</table>
</div>


    trades only the SMA10 takes: 130 (P/L $10k); only the SMA20 takes: 458 (P/L $3k)
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sharpe A</th>
      <th>sharpe B</th>
      <th>P/L A</th>
      <th>P/L B</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>2.91</td>
      <td>1.81</td>
      <td>21066.80</td>
      <td>15234.68</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>2.48</td>
      <td>2.36</td>
      <td>26732.80</td>
      <td>26941.76</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2.08</td>
      <td>1.18</td>
      <td>24584.48</td>
      <td>22361.84</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>2.69</td>
      <td>2.46</td>
      <td>27925.56</td>
      <td>29453.16</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>3.59</td>
      <td>3.57</td>
      <td>11734.36</td>
      <td>11681.24</td>
    </tr>
  </tbody>
</table>
</div>


    significance against A (the strategy's 10-day SMA); a negative t means worse than A:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sharpe_vs_A</th>
      <th>t_sharpe</th>
      <th>p_sharpe</th>
      <th>t_daily_pl</th>
      <th>p_daily_pl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B: above 20-day SMA</th>
      <td>-0.628</td>
      <td>-1.731</td>
      <td>0.083</td>
      <td>-0.543</td>
      <td>0.587</td>
    </tr>
    <tr>
      <th>above both</th>
      <td>-0.213</td>
      <td>-1.654</td>
      <td>0.098</td>
      <td>-2.160</td>
      <td>0.031</td>
    </tr>
  </tbody>
</table>
</div>


| everything on, trend rule | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | win per trade |
|---|---|---|---|---|---|---|
| **A: above the 10-day SMA (strategy)** | 1,450 | **\$112.0k** | **2.60** | **2.44 / 2.92** | **-\$3.8k** | **87.4%** |
| B: above the 20-day SMA | 1,778 | \$105.7k | 1.97 | 1.66 / 2.70 | -\$9.1k | 85.4% |
| above both | 1,320 | \$102.4k | 2.38 | 2.16 / 2.83 | -\$3.8k | 87.1% |

| year | Sharpe A (SMA10) / B (SMA20) | P/L A / B |
|---|---|---|
| 2022 (part) | **2.91** / 1.81 | **\$21.1k** / \$15.2k |
| 2023 | **2.48** / 2.36 | \$26.7k / \$26.9k |
| 2024 | **2.08** / 1.18 | **\$24.6k** / \$22.4k |
| 2025 | **2.69** / 2.46 | \$27.9k / **\$29.5k** |
| 2026 (part) | 3.59 / 3.57 | \$11.7k / \$11.7k |

- **On the 7 tranches the 10-day wins on every measure:** higher Sharpe (2.60 against 1.97) in every year, \$6.4k more P/L, and well under half
  the drawdown (-\$3.8k against -\$9.1k).
- **The 20-day's extra trades don't pay:** the 458 trades only the 20-day takes earn \$3.3k; the 130 only the 10-day takes earn \$9.7k.
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
- **The daily P/L barely differs** (t = 0.54, p = 0.59): the two rules make about the same money (\$112.0k against \$105.7k). **The 10-day's
  edge is risk, not return:** the same money with fewer bad days, hence half the drawdown and the higher Sharpe.
- **Adding the 20-day on top of the 10-day ("above both") makes significantly less money** (t = -2.16, p = 0.03): it drops the 130 trades
  only the 10-day takes, which earn \$9.7k, and adds nothing.

**The same A/B on the tranches forward selection by drawdown picks** (Section 7: 10:53, 11:13, 11:33, 12:13, 13:53, 14:13, 14:53, 15:53),
every other setting on, to check the result doesn't depend on the strategy's own tranche set.


    
![png](ic_analysis_files/output_84_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A: above 10-day SMA (strategy)</th>
      <th>B: above 20-day SMA</th>
      <th>above both</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trades</th>
      <td>1474.00</td>
      <td>1800.00</td>
      <td>1350.00</td>
    </tr>
    <tr>
      <th>sessions</th>
      <td>207.00</td>
      <td>231.00</td>
      <td>190.00</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>102826.44</td>
      <td>103055.00</td>
      <td>95840.00</td>
    </tr>
    <tr>
      <th>sharpe</th>
      <td>2.29</td>
      <td>1.97</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>sharpe_to_2024</th>
      <td>2.28</td>
      <td>1.92</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>sharpe_from_2025</th>
      <td>2.31</td>
      <td>2.06</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>max_drawdown</th>
      <td>-2582.60</td>
      <td>-7438.52</td>
      <td>-2582.60</td>
    </tr>
    <tr>
      <th>win_rate_trade</th>
      <td>87.65</td>
      <td>86.06</td>
      <td>87.41</td>
    </tr>
    <tr>
      <th>win_rate_condor</th>
      <td>80.19</td>
      <td>77.22</td>
      <td>80.00</td>
    </tr>
    <tr>
      <th>avg_pl_condor</th>
      <td>139.52</td>
      <td>114.51</td>
      <td>141.99</td>
    </tr>
    <tr>
      <th>win_rate_spread</th>
      <td>87.65</td>
      <td>86.06</td>
      <td>87.41</td>
    </tr>
    <tr>
      <th>sharpe_vs_A</th>
      <td>0.00</td>
      <td>-0.33</td>
      <td>-0.11</td>
    </tr>
    <tr>
      <th>ci_low</th>
      <td>NaN</td>
      <td>-1.04</td>
      <td>-0.44</td>
    </tr>
    <tr>
      <th>ci_high</th>
      <td>NaN</td>
      <td>0.39</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>p_better_than_A</th>
      <td>NaN</td>
      <td>0.18</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>t_daily_pl</th>
      <td>NaN</td>
      <td>0.02</td>
      <td>-1.40</td>
    </tr>
    <tr>
      <th>p_daily_pl</th>
      <td>NaN</td>
      <td>0.98</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>t_sharpe</th>
      <td>NaN</td>
      <td>-0.89</td>
      <td>-0.65</td>
    </tr>
    <tr>
      <th>p_sharpe</th>
      <td>NaN</td>
      <td>0.38</td>
      <td>0.52</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sharpe A</th>
      <th>sharpe B</th>
      <th>P/L A</th>
      <th>P/L B</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>1.74</td>
      <td>1.27</td>
      <td>15174.16</td>
      <td>12038.04</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>2.25</td>
      <td>2.65</td>
      <td>23814.32</td>
      <td>28176.12</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2.74</td>
      <td>1.80</td>
      <td>29480.44</td>
      <td>28158.84</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>2.07</td>
      <td>1.81</td>
      <td>24298.28</td>
      <td>24657.76</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>3.11</td>
      <td>3.10</td>
      <td>10059.24</td>
      <td>10024.24</td>
    </tr>
  </tbody>
</table>
</div>


| drawdown-picked tranches, everything else on | trades | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | t (Sharpe) | p | t (daily P/L) | p |
|---|---|---|---|---|---|---|---|---|---|
| **A: above the 10-day SMA** | 1,474 | \$102.8k | **2.29** | **2.28 / 2.31** | **-\$2.6k** | | | | |
| B: above the 20-day SMA | 1,800 | \$103.1k | 1.97 | 1.92 / 2.06 | -\$7.4k | -0.89 | 0.38 | +0.02 | 0.98 |
| above both | 1,350 | \$95.8k | 2.18 | 2.20 / 2.15 | -\$2.6k | -0.65 | 0.52 | -1.40 | 0.16 |

- **The 10-day still leads**, on Sharpe (2.29 against 1.97, ahead in four of five years) and above all on drawdown (-\$2.6k against -\$7.4k),
  for exactly the same money (\$102.8k against \$103.1k).
- **But on this tranche set the gap isn't significant** (t = 0.89, p = 0.38; the 10-day is better in 82% of resamples, against 96% on the
  strategy's 7). The 10-day's edge is real in direction and in drawdown, but how big it looks depends on which tranches it's paired with.

## 14. The 2026 drawdown, in the backtester's own log

Everything above is rebuilt from the 20-tranche export, which ends 2026-05-07. **The backtester's log of the live strategy** (the 7
tranches, no Thursdays, held to expiry) runs to 2026-09-28, and it shows a drawdown the rest of this notebook can't see: from a peak on
2026-05-11, **-\$13.4k by 2026-06-17**, three and a half times anything in the four years before, and **not recovered** by the end of the
log.

First, a check that the log and this notebook describe the same strategy, on the sessions both cover:

    sessions both cover: 2022-05-31 to 2026-05-07
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trades</th>
      <th>sessions</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>max_drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>backtester's log</th>
      <td>1446.0</td>
      <td>187.0</td>
      <td>111335.76</td>
      <td>2.51</td>
      <td>-3825.52</td>
    </tr>
    <tr>
      <th>the study's rebuild</th>
      <td>1450.0</td>
      <td>187.0</td>
      <td>112044.00</td>
      <td>2.53</td>
      <td>-3808.68</td>
    </tr>
  </tbody>
</table>
</div>


| 2022-05-31 to 2026-05-07 | trades | total P/L | Sharpe | max drawdown |
|---|---|---|---|---|
| the backtester's log | 1,446 | \$111.3k | 2.51 | -\$3.8k |
| this notebook's rebuild | 1,450 | \$112.0k | 2.53 | -\$3.8k |

**The rebuild matches the backtester's log almost exactly** (4 trades and \$0.7k apart over four years), so the rest of the notebook is the
live strategy. The difference is what came after 2026-05-07.


    
![png](ic_analysis_files/output_89_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peak</th>
      <th>trough</th>
      <th>recovered</th>
      <th>depth</th>
      <th>sessions_to_trough</th>
      <th>sessions_under</th>
    </tr>
    <tr>
      <th>drawdown</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DD1</th>
      <td>2026-05-11</td>
      <td>2026-06-17</td>
      <td>NaT</td>
      <td>-13391.0</td>
      <td>27</td>
      <td>100</td>
    </tr>
    <tr>
      <th>DD2</th>
      <td>2024-12-02</td>
      <td>2025-03-21</td>
      <td>2025-05-21</td>
      <td>-3826.0</td>
      <td>79</td>
      <td>121</td>
    </tr>
    <tr>
      <th>DD3</th>
      <td>2023-11-28</td>
      <td>2023-12-20</td>
      <td>2023-12-29</td>
      <td>-2543.0</td>
      <td>16</td>
      <td>22</td>
    </tr>
    <tr>
      <th>DD4</th>
      <td>2024-10-03</td>
      <td>2024-10-07</td>
      <td>2024-11-12</td>
      <td>-2338.0</td>
      <td>2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>DD5</th>
      <td>2025-05-27</td>
      <td>2025-06-02</td>
      <td>2025-06-17</td>
      <td>-1796.0</td>
      <td>4</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DD1</th>
      <th>DD2</th>
      <th>DD3</th>
      <th>DD4</th>
      <th>DD5</th>
    </tr>
    <tr>
      <th>tranche</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11:33</th>
      <td>-1526.0</td>
      <td>-779.0</td>
      <td>336.0</td>
      <td>-729.0</td>
      <td>-513.0</td>
    </tr>
    <tr>
      <th>11:53</th>
      <td>-1324.0</td>
      <td>-1700.0</td>
      <td>-61.0</td>
      <td>-699.0</td>
      <td>271.0</td>
    </tr>
    <tr>
      <th>12:13</th>
      <td>-2129.0</td>
      <td>-1336.0</td>
      <td>-486.0</td>
      <td>0.0</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>13:13</th>
      <td>-2409.0</td>
      <td>-698.0</td>
      <td>-919.0</td>
      <td>-296.0</td>
      <td>-744.0</td>
    </tr>
    <tr>
      <th>13:53</th>
      <td>-1961.0</td>
      <td>-45.0</td>
      <td>-707.0</td>
      <td>0.0</td>
      <td>-699.0</td>
    </tr>
    <tr>
      <th>14:13</th>
      <td>-1227.0</td>
      <td>155.0</td>
      <td>-38.0</td>
      <td>0.0</td>
      <td>-559.0</td>
    </tr>
    <tr>
      <th>14:53</th>
      <td>-2816.0</td>
      <td>577.0</td>
      <td>-669.0</td>
      <td>-614.0</td>
      <td>346.0</td>
    </tr>
    <tr>
      <th>book</th>
      <td>-13391.0</td>
      <td>-3826.0</td>
      <td>-2543.0</td>
      <td>-2338.0</td>
      <td>-1796.0</td>
    </tr>
  </tbody>
</table>
</div>


| drawdown (backtester's log) | peak | trough | recovered | depth |
|---|---|---|---|---|
| **2026** | **2026-05-11** | **2026-06-17** | **not by 2026-09-28** | **-\$13.4k** |
| next deepest | 2024-12-02 | 2025-03-21 | 2025-05-21 | -\$3.8k |
| third | 2023-11-28 | 2023-12-20 | 2023-12-29 | -\$2.5k |

- **Every tranche lost in it** (-\$1.2k to -\$2.8k each, 14:53 the most): it isn't one weak tranche, it is whole days going wrong.
- **After the trough the strategy traded 8 sessions to the end of September and made +\$2.1k**; the book was still \$11.3k below its peak.

**What happened: three days.**


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>before the drawdown</th>
      <th>the drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>condors</th>
      <td>723.00</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>sessions</th>
      <td>187.00</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>111335.76</td>
      <td>-13391.28</td>
    </tr>
    <tr>
      <th>avg_pl</th>
      <td>153.99</td>
      <td>-431.98</td>
    </tr>
    <tr>
      <th>win_rate_pct</th>
      <td>80.91</td>
      <td>25.81</td>
    </tr>
    <tr>
      <th>avg_win</th>
      <td>284.46</td>
      <td>216.12</td>
    </tr>
    <tr>
      <th>avg_loss</th>
      <td>-399.06</td>
      <td>-657.40</td>
    </tr>
    <tr>
      <th>full_wing_losses</th>
      <td>37.00</td>
      <td>19.00</td>
    </tr>
    <tr>
      <th>credit_per_condor</th>
      <td>325.30</td>
      <td>245.00</td>
    </tr>
    <tr>
      <th>vix_at_entry</th>
      <td>17.73</td>
      <td>17.26</td>
    </tr>
    <tr>
      <th>abs_move_after_entry_pct</th>
      <td>0.20</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>short_call_breached_pct</th>
      <td>18.40</td>
      <td>25.81</td>
    </tr>
    <tr>
      <th>short_put_breached_pct</th>
      <td>14.52</td>
      <td>48.39</td>
    </tr>
    <tr>
      <th>call_side_avg_pl</th>
      <td>86.40</td>
      <td>-109.39</td>
    </tr>
    <tr>
      <th>put_side_avg_pl</th>
      <td>67.60</td>
      <td>-322.59</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl</th>
      <th>condors</th>
      <th>full_wing</th>
      <th>call_side</th>
      <th>put_side</th>
      <th>vix</th>
      <th>spx_after_entry_pct</th>
      <th>weekday</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2026-05-12</th>
      <td>-5162.16</td>
      <td>7</td>
      <td>7</td>
      <td>-5978.58</td>
      <td>816.42</td>
      <td>19.02</td>
      <td>0.55</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>2026-06-17</th>
      <td>-4842.16</td>
      <td>7</td>
      <td>6</td>
      <td>671.42</td>
      <td>-5513.58</td>
      <td>16.85</td>
      <td>-1.15</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>2026-06-16</th>
      <td>-3622.16</td>
      <td>7</td>
      <td>5</td>
      <td>1101.42</td>
      <td>-4723.58</td>
      <td>15.94</td>
      <td>-0.34</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>2024-12-10</th>
      <td>-2440.52</td>
      <td>4</td>
      <td>2</td>
      <td>772.24</td>
      <td>-3212.76</td>
      <td>14.07</td>
      <td>-0.29</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>2023-05-23</th>
      <td>-1720.64</td>
      <td>3</td>
      <td>1</td>
      <td>594.18</td>
      <td>-2314.82</td>
      <td>17.92</td>
      <td>-0.46</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>2022-11-28</th>
      <td>-1567.64</td>
      <td>3</td>
      <td>2</td>
      <td>544.18</td>
      <td>-2111.82</td>
      <td>21.91</td>
      <td>-0.64</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>2025-05-12</th>
      <td>-1557.76</td>
      <td>2</td>
      <td>2</td>
      <td>-1743.88</td>
      <td>186.12</td>
      <td>19.24</td>
      <td>0.74</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>2022-10-05</th>
      <td>-1487.76</td>
      <td>2</td>
      <td>2</td>
      <td>-1713.88</td>
      <td>226.12</td>
      <td>29.66</td>
      <td>1.16</td>
      <td>Wednesday</td>
    </tr>
  </tbody>
</table>
</div>


| the log's worst days | P/L | condors | lost a full wing | side that broke | VIX at entry | SPX from entry to close |
|---|---|---|---|---|---|---|
| **2026-05-12** (Tue) | **-\$5.2k** | 7 | **7** | call (-\$6.0k) | 19.0 | **+0.55%** |
| **2026-06-17** (Wed) | **-\$4.8k** | 7 | **6** | put (-\$5.5k) | 16.9 | **-1.15%** |
| **2026-06-16** (Tue) | **-\$3.6k** | 7 | **5** | put (-\$4.7k) | 15.9 | -0.34% |
| 2024-12-10 (Tue) | -\$2.4k | 4 | 2 | put | 14.1 | -0.29% |
| 2023-05-23 (Tue) | -\$1.7k | 3 | 1 | put | 17.9 | -0.46% |

- **The whole drawdown is three sessions:** on 2026-05-12, 06-16 and 06-17 all seven tranches were open and SPX ran through one side of
  nearly all of them. Together they lose \$13.6k; the other three sessions of the drawdown net +\$0.2k.
- **Each of the three is worse than any day in the four years before** (the previous worst, 2024-12-10, lost \$2.4k with 4 condors open). A
  day with all seven tranches open and one side broken costs about \$5k: that is the strategy's real worst day, and in 2022-2025 it never
  happened.
- **VIX gave no warning:** it was 16-19 at entry, normal for the strategy (17.7 on average before). The moves after entry were 0.3-1.2%,
  against 0.20% on an average day.

| | before the drawdown | the drawdown |
|---|---|---|
| condors / sessions | 723 / 187 | 31 / 6 |
| win rate | 81% | **26%** |
| average loss per losing condor | -\$399 | **-\$657** |
| condors that lost a full wing | 37 (5%) | **19 (61%)** |
| average credit per condor | \$325 | **\$245** |
| SPX move from entry to close (absolute) | 0.20% | **0.53%** |
| short put breached | 15% | **48%** |

**Why 2026 hurts more: less credit for each breach.**


    
![png](ic_analysis_files/output_93_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spx</th>
      <th>credit_per_spread</th>
      <th>short_dist_pct</th>
      <th>vix</th>
      <th>breached_pct</th>
      <th>condors</th>
      <th>avg_pl_condor</th>
      <th>full_wing_losses</th>
      <th>total_pl</th>
      <th>worst_day</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022</th>
      <td>3951.05</td>
      <td>145.0</td>
      <td>0.52</td>
      <td>25.43</td>
      <td>10.00</td>
      <td>115</td>
      <td>183.19</td>
      <td>9</td>
      <td>21066.80</td>
      <td>-1567.64</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>4340.56</td>
      <td>175.0</td>
      <td>0.21</td>
      <td>16.13</td>
      <td>19.82</td>
      <td>164</td>
      <td>160.59</td>
      <td>8</td>
      <td>26336.68</td>
      <td>-1720.64</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>5309.83</td>
      <td>175.0</td>
      <td>0.16</td>
      <td>13.88</td>
      <td>22.63</td>
      <td>179</td>
      <td>137.34</td>
      <td>8</td>
      <td>24584.48</td>
      <td>-2440.52</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>6443.70</td>
      <td>150.0</td>
      <td>0.22</td>
      <td>16.77</td>
      <td>14.15</td>
      <td>212</td>
      <td>130.25</td>
      <td>12</td>
      <td>27613.44</td>
      <td>-1557.76</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>7368.93</td>
      <td>130.0</td>
      <td>0.29</td>
      <td>17.54</td>
      <td>18.26</td>
      <td>115</td>
      <td>4.03</td>
      <td>22</td>
      <td>463.80</td>
      <td>-5162.16</td>
    </tr>
  </tbody>
</table>
</div>


| year | SPX (median) | credit per spread | short strike from spot | condors | full-wing losses | avg P/L per condor | worst day |
|---|---|---|---|---|---|---|---|
| 2022 | 3,951 | \$145 | 0.52% | 115 | 9 | +\$183 | -\$1.6k |
| 2023 | 4,341 | \$175 | 0.21% | 164 | 8 | +\$161 | -\$1.7k |
| 2024 | 5,310 | \$175 | 0.16% | 179 | 8 | +\$137 | -\$2.4k |
| 2025 | 6,444 | \$150 | 0.22% | 212 | 12 | +\$130 | -\$1.6k |
| **2026 (to Sep)** | **7,369** | **\$130** | 0.29% | 115 | **22** | **+\$4** | **-\$5.2k** |

- **The credit has been shrinking since 2024** (\$175 -> \$150 -> \$130 a spread) while the wing stays 10 points: a breached spread now loses
  up to \$870 against \$825 before, and a winning one pays less. Each full-wing loss takes more winning trades to earn back (about 6.7
  winners in 2026 against 4.7 in 2023-24).
- **The strikes are not closer:** they sit further from spot in 2026 (0.29%) than in 2023-25. What changed is the size of the moves (0.25% from
  entry to close on average in 2026, against 0.18% in 2024-25) and how many hit all seven tranches the same day.
- **22 of the 59 full-wing losses in the whole log came in 2026**, 18 of them in those three days.

**What it says about the strategy:**

- **The risk was always there, it just hadn't shown up.** Seven tranches spread over three hours on one day, with no stop, are close to one
  big position: a day that runs through one side costs ~\$5k. The four years before simply had no such day on a day the filters let through.
- **The likely trigger is scheduled news.** 2026-06-16 and 06-17 were the days of a Fed (FOMC) meeting, with the decision on the 17th; the
  Thursday analysis (Section 3) found the same pattern: the strategy's worst days are days the market moves more than VIX priced, often
  macro days. This data has no economic calendar, so this is a hypothesis: **an event-day filter (no trading on FOMC decision days, and
  possibly CPI days) is the change most worth backtesting.**
- **The tranche study (Section 11) said the drawdowns were shallow:** on the rebuild they were (-\$3.8k). The log shows the tail those four
  years didn't contain, and it is the number to size the strategy on.

## 15. Summary

**The greedy cascade, step by step:**

| step | trades | P/L | Sharpe | what it does |
|---|---|---|---|---|
| raw book | 39,694 | -\$7.5k | -0.12 | wins 66% of condors, but the losses are twice the wins |
| + no Thursdays | 31,768 | \$155.6k | 0.44 | removes the weekday where the raw book loses (-\$163.0k) |
| + max move 0% | 14,416 | \$195.9k | 0.97 | enters on pullbacks, so an intraday rally doesn't run into the short call |
| + above the 10-day SMA | 7,362 | \$205.6k | 1.50 | keeps calm markets, where SPX moves less after entry |
| + VIX9D/VIX 0.87-1.02 | 4,630 | \$213.3k | 2.25 | keeps a curve that isn't bracing for a move; both tails lose on the put side |
| + the 7 tranches | 1,582 | \$113.4k | 2.48 | keeps the late morning and early afternoon; drawdown -\$15.1k -> -\$4.9k |
| **+ short/long <= 4** | **1,450** | **\$112.0k** | **2.60** | skips the richest condors the 7 still take |

**The strategy: 1,450 trades, \$112.0k, Sharpe 2.60, -\$3.8k drawdown, 87.4% of trades winning**, held to expiry; +\$599 on an average trading
day (76% of days positive), an average win of \$285 a condor against an average loss of \$399.

- **No Thursdays isn't a VIX regime:** nothing known at entry differs on Thursdays, but they move more than VIX prices (the realized range
  explains two-thirds of the loss), on a few big days every year; macro releases are the likely cause, untested here.
- **The settings work together.** The band is the worst entry filter on the raw book and adds the most at step 4; with everything on, every
  threshold is at the top of its sweep.
- **The 10-day SMA is the most important setting:** removing it costs 1.28 Sharpe and triples the drawdown. With everything on it beats the
  20-day in every year (t = 1.73 on the Sharpe difference, p = 0.08: significant at 10%, not at 5%); the edge is lower risk, not more
  money. On the drawdown-picked tranches it still leads (2.29 against 1.97, a third of the drawdown) but not significantly (p = 0.38).
- **No stop:** on this strategy every call-stop level lowers Sharpe and P/L and none reduces the drawdown, and a put stop is worse still
  (400%: Sharpe 1.91, -\$21.8k; 42% of the stopped puts would have expired winning) (Section 9). The cost is exposure to a day worse than any
  in this sample.
- **The 7 tranches are the smallest, steadiest set:** the smallest drawdown and the most P/L per \$ of drawdown of the sets tested. Adding
  11:13 and 13:33 (forward selection's peak) scores a higher Sharpe (2.73) for \$29.8k more and a deeper drawdown; the previous 15, 2.69 for
  \$68.4k more. None of the 7 should be cut: each made money every year and each one's removal lowers Sharpe.

- **The backtester's log shows a 2026 drawdown the export doesn't reach:** -\$13.4k from 2026-05-11 to 06-17, not recovered by 2026-09-28,
  from three days on which all seven tranches lost a full wing (Section 14). The rebuild matches the log to 2026-05-07; size the strategy on
  that tail, not on the -\$3.8k of 2022-2026, and backtest an event-day (FOMC/CPI) filter.

**What this notebook doesn't establish:** out-of-sample performance (apart from the walk-forward checks); the backtester's totals on these
settings (the rules were matched trade by trade against a backtest on earlier settings, with the 20-day SMA; the 10-day and the 7 tranches
haven't been checked against a backtest log); the first sessions, which need SPX closes before the export; and a put stop, which is untested.
