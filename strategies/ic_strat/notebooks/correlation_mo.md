# Why IC and MO1 / MO2 don't move together

Three 0DTE SPX iron condor books with **very similar setups**: all three sell 10-wide call and put spreads on SPX, expiring the same
day, one contract at a time, several entries a day. Yet in the backtester's portfolio view the IC strategy and the two MO books are
**almost uncorrelated**. This notebook works out why, and what it means for running them side by side.

**The short answer.**

1. **They mostly trade on different days.** IC trades 187 sessions and MO1 458, but **only 104 of them are the same day**. On a day
   only one of them trades, the other's P/L is zero, so that day adds swings to one book and nothing to the other. Most of MO1's ups and
   downs happen on days IC isn't trading.
2. **They trade on different days because their entry signals point opposite ways.** IC enters only on a **pullback** (SPX at or
   below its open); MO enters only when SPX is **stretched up** (at least 0.5% above its 5-day average). IC's other filters (its seven
   entry times, no Thursdays, the VIX9D/VIX band) remove more of MO's days.
3. **When they do trade the same minute, they're nearly the same trade** (condor P/L correlation 0.77). So the low correlation isn't
   the trades being different. It's **when** each one trades.
4. **Their big wins and losses come from different markets.** Each condor wins if SPX closes near where it was opened: IC's winning
   zone sits **below** the day's open (-0.53% to 0%), MO's **above** it (+0.13% to +0.39%). Up days hurt IC and pay MO; down days pay IC
   and hurt MO. IC sits out almost all the violent days MO trades, and the two never had a losing quarter at the same time.

**Contents.**

| § | question |
|---|---|
| 1 | The three books side by side |
| 2 | The correlation, measured like the backtester, and whether it ever spikes |
| 3 | Where the low correlation comes from: shared days, and how they move on them |
| 4 | Why they trade on different days: opposite entry signals |
| 5 | Shared days: why they still only move partly together; their worst days |
| 6 | **Why their spikes are different: what makes each one win and lose** |
| 7 | What running them together does: one portfolio, drawdowns, margin |
| 8 | Summary |

**How to read the numbers.**

- **Correlation is the backtester's portfolio correlation**: the correlation of daily P/L over every session, with days a book doesn't
  trade counted as zero.
- **IC is the strategy as traded** (`ic_analysis.ipynb`): no Thursdays, SPX at or below its open, SPX above its 10-day SMA, VIX9D/VIX
  0.87-1.02, seven entry times (11:33, 11:53, 12:13, 13:13, 13:53, 14:13, 14:53), short/long ratio <= 4, held to expiry with no stop.
  It is rebuilt from the 20-tranche export with the backtester's rules. **MO1 and MO2 are backtest logs** as exported.
- **Everything is on one calendar**: the 998 sessions the IC export covers, 2022-05-16 to 2026-05-07. The MO logs run to 2026-09;
  their trades after 2026-05-07 are left out so the three books are compared on the same days. MO1 starts trading 2022-05-25, MO2
  2022-06-21.
- **Sharpe is the backtester's** (daily P/L over \$200k per book, less a 4% risk-free rate). Intervals are 95% session-bootstrap
  intervals.

## Setup

| file | what it is |
|---|---|
| `data/strategy/ic_strat/spx/raw_baseline/IC-20-Tranche-10-Baseline.csv` | the 20-tranche IC export the IC strategy is rebuilt from (see `ic_analysis.ipynb`) |
| `data/strategy/ic_strat/spx/mo/mo1.csv` | the MO1 backtest log: "IC 10w 35Delta - signal W (SPX>=0.5% over SMA5, 12:13-14:13)" |
| `data/strategy/ic_strat/spx/mo/mo2.csv` | the MO2 backtest log: "IC 10w 35Delta - signal W + Parkinson20 >= 8 (12:13-14:13)" |
| `data/market/vix/ohlc/` | 1-minute VIX9D and VIX, to measure IC's term-structure filter on MO's trades |

None of these are in git (`data/` is git-ignored).

    998 sessions; condors: IC 725, MO1 2,790, MO2 2,162
    

## 1. The three books side by side


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO1</th>
      <th>MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>condors</th>
      <td>725</td>
      <td>2790</td>
      <td>2162</td>
    </tr>
    <tr>
      <th>trades (spreads)</th>
      <td>1450</td>
      <td>5580</td>
      <td>4324</td>
    </tr>
    <tr>
      <th>sessions traded</th>
      <td>187</td>
      <td>458</td>
      <td>350</td>
    </tr>
    <tr>
      <th>total_pl</th>
      <td>112044.0</td>
      <td>180572.8</td>
      <td>186396.44</td>
    </tr>
    <tr>
      <th>sharpe</th>
      <td>2.59583</td>
      <td>1.578353</td>
      <td>1.85775</td>
    </tr>
    <tr>
      <th>max_drawdown</th>
      <td>-3808.68</td>
      <td>-15704.2</td>
      <td>-15704.2</td>
    </tr>
    <tr>
      <th>win_rate_condor</th>
      <td>80.965517</td>
      <td>60.143369</td>
      <td>61.748381</td>
    </tr>
    <tr>
      <th>avg_pl_condor</th>
      <td>154.543448</td>
      <td>64.721434</td>
      <td>86.21482</td>
    </tr>
    <tr>
      <th>median_credit</th>
      <td>325.0</td>
      <td>470.0</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>median_short_call_pts</th>
      <td>11.45</td>
      <td>6.74</td>
      <td>7.12</td>
    </tr>
    <tr>
      <th>median_short_put_pts</th>
      <td>12.9</td>
      <td>6.94</td>
      <td>7.32</td>
    </tr>
    <tr>
      <th>entry_window</th>
      <td>11:33-14:53</td>
      <td>12:13-14:13</td>
      <td>12:13-14:13</td>
    </tr>
    <tr>
      <th>thursdays_traded</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](correlation_mo_files/output_4_1.png)
    


| | **IC** | **MO1** | **MO2** |
|---|---|---|---|
| structure | 10-wide iron condor, 0DTE SPX | same | same |
| short strikes | offsets from the export: median **11.5 pts** (call), **12.9 pts** (put) from spot | **35-delta**: median **6.7 / 6.9 pts** from spot | 35-delta: 7.1 / 7.3 pts |
| median credit | \$325 | **\$470** | \$480 |
| entry times | **7: 11:33, 11:53, 12:13, 13:13, 13:53, 14:13, 14:53** | **12:13-14:13**, every 20 min | 12:13-14:13 |
| weekdays | Mon, Tue, Wed, Fri | **every weekday** | every weekday |
| entry signal | VIX9D/VIX 0.87-1.02, **SPX at or below its open**, SPX above its 10-day SMA, short/long <= 4 | **signal W: SPX >= 0.5% above its 5-day average** | signal W + Parkinson20 >= 8 |
| exit | held to expiry | held to expiry | held to expiry |
| condors (trades) | 725 (1,450) | 2,790 (5,580) | 2,162 (4,324) |
| sessions traded | 187 | 458 | 350 |
| total P/L | \$112.0k | \$180.6k | \$186.4k |
| Sharpe | **2.60** | 1.58 | 1.86 |
| max drawdown | **-\$3.8k** | -\$15.7k | -\$15.7k |
| win rate per condor | 81% | 60% | 62% |

- **Similar trades, different choices.** All three sell the same 10-wide condor on the same index on the same day. MO sells it **much
  closer to the money** (35 delta, about 7 points from spot, for \$470 of credit against IC's \$325 at 11-13 points), and MO decides
  *when* to trade with a different signal.
- **IC is the smaller, steadier book:** a quarter as many condors as MO1, a higher Sharpe, and a drawdown a quarter the size. MO makes more
  money at one contract each, with far bigger swings.
- **MO2 is MO1 with one more filter** (Parkinson20 >= 8, a realized-volatility floor): every MO2 trade is also an MO1 trade. That is why
  MO1 and MO2 behave as one book (below).

## 2. The correlation, measured like the backtester

Each pair's correlation of daily P/L (the backtester's measure), with its 95% interval, then the same on weekly and monthly P/L.
Right: the daily correlation over a trailing six months (126 sessions), to see whether it ever spikes.


    
![png](correlation_mo_files/output_7_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>daily</th>
      <th>daily_ci_low</th>
      <th>daily_ci_high</th>
      <th>daily_rank (spearman)</th>
      <th>weekly</th>
      <th>monthly</th>
    </tr>
    <tr>
      <th>pair</th>
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
      <th>IC vs MO1</th>
      <td>0.199</td>
      <td>0.142</td>
      <td>0.256</td>
      <td>0.160</td>
      <td>0.018</td>
      <td>0.171</td>
    </tr>
    <tr>
      <th>IC vs MO2</th>
      <td>0.165</td>
      <td>0.101</td>
      <td>0.231</td>
      <td>0.123</td>
      <td>-0.007</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>MO1 vs MO2</th>
      <td>0.883</td>
      <td>0.853</td>
      <td>0.912</td>
      <td>0.881</td>
      <td>0.888</td>
      <td>0.904</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC vs MO1</th>
      <th>IC vs MO2</th>
      <th>MO1 vs MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>873.00</td>
      <td>873.00</td>
      <td>873.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.20</td>
      <td>0.17</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.15</td>
      <td>0.13</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.20</td>
      <td>0.17</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.24</td>
      <td>0.22</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.35</td>
      <td>0.32</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


| pair | daily (backtester) | 95% interval | weekly | monthly | rolling 6-month range |
|---|---|---|---|---|---|
| **IC vs MO1** | **0.20** | 0.14 to 0.26 | 0.02 | 0.17 | 0.04 to 0.35 |
| **IC vs MO2** | **0.17** | 0.10 to 0.23 | -0.01 | 0.13 | 0.04 to 0.32 |
| MO1 vs MO2 | **0.88** | 0.85 to 0.91 | 0.89 | 0.90 | 0.67 to 1.00 |

- **IC is close to uncorrelated with both MO books**, on every time scale: -0.01 to 0.20, never above 0.35 in any six-month window.
- **MO1 and MO2 are close to the same book** (0.88, never below 0.67). The comparison shows that "similar setup" *can* mean high
  correlation: it depends on whether two books trade on the same days.
- **The correlation is small but not zero**, and the interval excludes zero. When IC and MO trade the same day they do tend to move
  together (§3), so the two books aren't independent, just mostly apart.

## 3. Where the low correlation comes from

A daily correlation has two ingredients: **how often both books trade on the same day**, and **how closely they move on those days**.
On a day only one book trades, the other's P/L is zero. That day adds to one book's swings (its variance) but adds nothing to how the
two move together (their covariance). So:

**correlation ~= (correlation on shared days) x sqrt(share of book A's variance on shared days x share of book B's)**

The table measures each part; the last two rows check the approximation against the actual correlation.


    
![png](correlation_mo_files/output_10_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>pair</th>
      <th>IC vs MO1</th>
      <th>IC vs MO2</th>
      <th>MO1 vs MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>days IC trades</th>
      <td>187.000</td>
      <td>187.000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>days the other trades</th>
      <td>458.000</td>
      <td>350.000</td>
      <td>350.000</td>
    </tr>
    <tr>
      <th>days both trade</th>
      <td>104.000</td>
      <td>76.000</td>
      <td>350.000</td>
    </tr>
    <tr>
      <th>only the first</th>
      <td>83.000</td>
      <td>111.000</td>
      <td>108.000</td>
    </tr>
    <tr>
      <th>only the second</th>
      <td>354.000</td>
      <td>274.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>neither</th>
      <td>457.000</td>
      <td>537.000</td>
      <td>540.000</td>
    </tr>
    <tr>
      <th>corr on shared days</th>
      <td>0.601</td>
      <td>0.567</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>first's variance on shared days %</th>
      <td>49.522</td>
      <td>38.025</td>
      <td>76.996</td>
    </tr>
    <tr>
      <th>second's variance on shared days %</th>
      <td>20.701</td>
      <td>19.349</td>
      <td>98.708</td>
    </tr>
    <tr>
      <th>approx = shared corr x sqrt(shares)</th>
      <td>0.202</td>
      <td>0.173</td>
      <td>0.872</td>
    </tr>
    <tr>
      <th>actual correlation</th>
      <td>0.199</td>
      <td>0.165</td>
      <td>0.883</td>
    </tr>
    <tr>
      <th>days MO1 trades</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>458.000</td>
    </tr>
  </tbody>
</table>
</div>


| | IC vs MO1 | IC vs MO2 | MO1 vs MO2 |
|---|---|---|---|
| sessions both trade | **104** | **76** | **350** |
| only the first / only the second | 83 / 354 | 111 / 274 | 108 / 0 |
| **correlation on shared days** | **0.60** | **0.57** | **1.00** |
| first book's variance that falls on shared days | **50%** | 38% | 77% |
| second book's variance that falls on shared days | **21%** | 19% | 99% |
| approximation: 0.60 x sqrt(0.50 x 0.21) | **0.20** | 0.17 | 0.87 |
| actual correlation | **0.20** | 0.17 | 0.88 |

**This is the whole explanation of the number.**

- **On the days they share, IC and MO1 do move together** (0.60). They're short the same SPX on the same afternoon.
- **But those days are a small part of MO1.** Half of IC's daily swings fall on shared days, but only 21% of MO1's: most of MO1's risk is
  taken on days IC is flat. Multiply it out (0.60 x sqrt(0.50 x 0.21)) and you get **0.20, the correlation the backtester shows**.
- **MO1 vs MO2 is the opposite case:** MO2 trades on 350 days, all of them MO1 days, with the same trades (shared-day correlation
  1.00). 99% of MO2's swings and 77% of MO1's happen on shared days, so the correlation stays at 0.88.

So the question "why aren't IC and MO correlated?" becomes "**why do they trade on different days?**".

## 4. Why they trade on different days: opposite entry signals

**MO's signal W** asks for SPX at least 0.5% **above** the average of its prior 5 closes: SPX has rallied and is stretched above its
recent level. **IC's max-move filter** asks for SPX **at or below** where it opened that morning: SPX has pulled back. Each dot below is
one condor, placed by those two measures at the moment it was opened.


    
![png](correlation_mo_files/output_13_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO1</th>
      <th>MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>condors</th>
      <td>725.0</td>
      <td>2790.0</td>
      <td>2162.0</td>
    </tr>
    <tr>
      <th>SPX at/below its open (IC needs this) %</th>
      <td>100.0</td>
      <td>26.1</td>
      <td>25.7</td>
    </tr>
    <tr>
      <th>SPX &gt;= 0.5% above SMA5 (MO needs this) %</th>
      <td>44.6</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>both at once %</th>
      <td>44.6</td>
      <td>26.1</td>
      <td>25.7</td>
    </tr>
    <tr>
      <th>median move since open %</th>
      <td>-0.2</td>
      <td>0.3</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>median vs SMA5 %</th>
      <td>0.4</td>
      <td>1.1</td>
      <td>1.3</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC condors</th>
      <th>pass signal W %</th>
      <th>in MO's 12:13-14:13 window %</th>
      <th>both %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>725</td>
      <td>44.6</td>
      <td>57.5</td>
      <td>25.5</td>
    </tr>
  </tbody>
</table>
</div>


| | IC condors | MO1 condors |
|---|---|---|
| SPX at or below its open at entry (IC's rule) | 100% | **26%** |
| SPX >= 0.5% above its 5-day average (MO's rule) | **45%** | 100% |
| median move since the open | **-0.2%** | **+0.3%** |
| median SPX vs its 5-day average | +0.4% | **+1.1%** |

- **The two rules pick opposite intraday moments.** MO enters on strength: its median entry is SPX 0.3% above the open and 1.1% above
  its 5-day average. IC enters on weakness: its median entry is 0.2% below the open.
- **They aren't mutually exclusive, which is why there is some overlap.** A day can be up over the week but down since the open (the
  shaded corner): 45% of IC's condors and a quarter of MO1's sit there. IC's 10-day SMA rule pulls it towards that corner: it only trades
  when SPX is above its recent average. Those are the shared days of §3.
- **The reverse is tight too:** only 26% of IC's condors are both signal-W entries and inside MO's 12:13-14:13 window.

The move filter isn't IC's only difference. MO1's condors put through IC's filters one at a time:


    
![png](correlation_mo_files/output_15_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors left</th>
      <th>sessions left</th>
      <th>passes this filter alone %</th>
    </tr>
    <tr>
      <th>filter</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MO1 as traded</th>
      <td>2790</td>
      <td>458</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>+ one of IC's 7 entry times</th>
      <td>1591</td>
      <td>455</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>+ not a Thursday</th>
      <td>1280</td>
      <td>364</td>
      <td>80.1</td>
    </tr>
    <tr>
      <th>+ VIX9D/VIX 0.87-1.02 at entry</th>
      <td>814</td>
      <td>239</td>
      <td>64.8</td>
    </tr>
    <tr>
      <th>+ SPX above its 10-day SMA</th>
      <td>800</td>
      <td>234</td>
      <td>96.6</td>
    </tr>
    <tr>
      <th>+ short/long ratio &lt;= 4</th>
      <td>740</td>
      <td>228</td>
      <td>95.6</td>
    </tr>
    <tr>
      <th>+ SPX at or below its open</th>
      <td>184</td>
      <td>73</td>
      <td>26.1</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condors left</th>
      <th>sessions left</th>
      <th>passes this filter alone %</th>
    </tr>
    <tr>
      <th>filter</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MO2 as traded</th>
      <td>2162</td>
      <td>350</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>+ one of IC's 7 entry times</th>
      <td>1235</td>
      <td>350</td>
      <td>57.1</td>
    </tr>
    <tr>
      <th>+ not a Thursday</th>
      <td>995</td>
      <td>280</td>
      <td>80.4</td>
    </tr>
    <tr>
      <th>+ VIX9D/VIX 0.87-1.02 at entry</th>
      <td>615</td>
      <td>178</td>
      <td>63.4</td>
    </tr>
    <tr>
      <th>+ SPX above its 10-day SMA</th>
      <td>603</td>
      <td>174</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>+ short/long ratio &lt;= 4</th>
      <td>576</td>
      <td>172</td>
      <td>97.2</td>
    </tr>
    <tr>
      <th>+ SPX at or below its open</th>
      <td>135</td>
      <td>54</td>
      <td>25.7</td>
    </tr>
  </tbody>
</table>
</div>


| MO1, through IC's filters | sessions left | condors left | share of MO1's condors that pass this filter on its own |
|---|---|---|---|
| MO1 as traded | 458 | 2,790 | |
| + **one of IC's 7 entry times** | 455 | 1,591 | 57% (12:13, 13:13, 13:53 and 14:13 are shared; 12:33, 12:53 and 13:33 aren't) |
| + **no Thursdays** | 364 | 1,280 | 80% |
| + **VIX9D/VIX 0.87-1.02 at entry** | 239 | 814 | 65% |
| + SPX above its 10-day SMA | 234 | 800 | 97% |
| + short/long ratio <= 4 | 228 | 740 | 96% |
| + **SPX at or below its open** | **73** | **184** | **26%** |

- **Only 73 of MO1's 458 days (and 7% of its condors) would pass every IC filter.** The rest are days IC can't trade.
- **The move filter does most of the separating:** on its own it rejects 74% of MO1's condors.
- **IC's entry times, no Thursdays and the VIX9D/VIX band each remove a large share too.** MO trades every 20 minutes from 12:13 to 14:13,
  trades Thursdays, and doesn't look at the term structure at all.
- **The trend filters barely differ:** 97% of MO1's condors already have SPX above its 10-day average, because a market 0.5% above its
  5-day average is almost always above its 10-day too. On trend, the two books agree; on **the intraday move**, they disagree.
- **MO2 is the same picture** (54 of its 350 days pass every IC filter).

## 5. Shared days: why they still only move partly together

On the 104 days both IC and MO1 trade, their daily P/L correlation is 0.60, not 1. Two things keep it there: they don't open at the
same minutes or at the same strikes, and they're picked for different reasons even on the same day.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>condors on shared days</th>
      <td>404.00</td>
      <td>628.00</td>
    </tr>
    <tr>
      <th>condors a day</th>
      <td>3.88</td>
      <td>6.04</td>
    </tr>
    <tr>
      <th>median move since open at entry %</th>
      <td>-0.21</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>median short-strike distance (pts)</th>
      <td>11.91</td>
      <td>6.96</td>
    </tr>
    <tr>
      <th>avg P/L per shared day</th>
      <td>616.74</td>
      <td>712.32</td>
    </tr>
  </tbody>
</table>
</div>


    MO1: 104 shared days, 185 same-minute entries; both win on 61 days, both lose on 15; same sign on 73%
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>condors on shared days</th>
      <td>294.00</td>
      <td>471.00</td>
    </tr>
    <tr>
      <th>condors a day</th>
      <td>3.87</td>
      <td>6.20</td>
    </tr>
    <tr>
      <th>median move since open at entry %</th>
      <td>-0.21</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>median short-strike distance (pts)</th>
      <td>12.84</td>
      <td>7.14</td>
    </tr>
    <tr>
      <th>avg P/L per shared day</th>
      <td>679.62</td>
      <td>936.32</td>
    </tr>
  </tbody>
</table>
</div>


    MO2: 76 shared days, 136 same-minute entries; both win on 47 days, both lose on 8; same sign on 72%
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>same-minute pairs</th>
      <td>185.00</td>
    </tr>
    <tr>
      <th>condor P/L correlation</th>
      <td>0.77</td>
    </tr>
    <tr>
      <th>IC avg P/L</th>
      <td>165.27</td>
    </tr>
    <tr>
      <th>MO1 avg P/L</th>
      <td>188.47</td>
    </tr>
    <tr>
      <th>IC win rate %</th>
      <td>81.08</td>
    </tr>
    <tr>
      <th>MO1 win rate %</th>
      <td>71.35</td>
    </tr>
    <tr>
      <th>both win %</th>
      <td>70.81</td>
    </tr>
    <tr>
      <th>IC wins, MO1 loses %</th>
      <td>10.27</td>
    </tr>
    <tr>
      <th>MO1 wins, IC loses %</th>
      <td>0.54</td>
    </tr>
    <tr>
      <th>both lose %</th>
      <td>18.38</td>
    </tr>
    <tr>
      <th>IC median short-strike distance (pts)</th>
      <td>9.87</td>
    </tr>
    <tr>
      <th>MO1 median short-strike distance (pts)</th>
      <td>6.89</td>
    </tr>
  </tbody>
</table>
</div>


**On a shared day, each book trades its own moments.** IC opens about 4 condors on such a day and MO1 about 6, and only 185 of IC's 404
condors on those days are at the same minute as an MO1 condor: IC enters when SPX dips below its open, at its seven times; MO1 at every
12:13-14:13 slot where SPX is stretched up.

**When they do open the same minute, they're nearly the same trade:**

| 185 same-minute pairs | IC | MO1 |
|---|---|---|
| condor P/L correlation | **0.77** | |
| median short strike from spot | 9.9 pts | 6.9 pts |
| average P/L | \$165 | \$188 |
| win rate | 81% | 71% |

| outcome of the same-minute pair | share |
|---|---|
| both win | 71% |
| IC wins, MO1 loses | **10%** |
| MO1 wins, IC loses | **0.5%** (one pair) |
| both lose | 18% |

- **The setups are similar, and it shows here:** at the same minute the two condors win and lose together (0.77).
- **The difference is strike distance.** At the same minute MO1's shorts sit 6.9 points from spot and IC's 9.9 (12 on average across
  IC's trades), so a move that breaks MO1's short strike can still leave IC's intact (10% of pairs), but almost never the reverse
  (one pair in 185). MO collects more (\$470 against \$325) for that extra risk.
- **By day:** on the 104 shared days the two books end with the same sign 73% of the time; IC wins while MO1 loses on 20 days, and MO1
  wins while IC loses on 8.

**Their worst days.** For each book, its worst 5% of trading days, and what the others did on those days:


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>book</th>
      <th>IC</th>
      <th>MO1</th>
      <th>MO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>worst days</th>
      <td>10.0</td>
      <td>23.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>its avg P/L on them</th>
      <td>-1566.6</td>
      <td>-4061.7</td>
      <td>-3967.5</td>
    </tr>
    <tr>
      <th>MO1: traded %</th>
      <td>50.0</td>
      <td>NaN</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>MO1: avg P/L</th>
      <td>-444.8</td>
      <td>NaN</td>
      <td>-3967.5</td>
    </tr>
    <tr>
      <th>MO1: lost too %</th>
      <td>30.0</td>
      <td>NaN</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>MO2: traded %</th>
      <td>30.0</td>
      <td>69.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MO2: avg P/L</th>
      <td>-196.8</td>
      <td>-2803.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MO2: lost too %</th>
      <td>20.0</td>
      <td>69.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>IC: traded %</th>
      <td>NaN</td>
      <td>17.4</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>IC: avg P/L</th>
      <td>NaN</td>
      <td>-36.1</td>
      <td>-26.2</td>
    </tr>
    <tr>
      <th>IC: lost too %</th>
      <td>NaN</td>
      <td>8.7</td>
      <td>5.6</td>
    </tr>
  </tbody>
</table>
</div>


| | IC's 10 worst days (avg -\$1,567) | MO1's 23 worst days (avg -\$4,062) |
|---|---|---|
| IC traded / lost too | | traded on 17%, lost on **9%** (avg -\$36) |
| MO1 traded / lost too | traded on 50%, lost on **30%** (avg -\$445) | |
| MO2 traded / lost too | traded on 30%, lost on 20% | traded on 70%, lost on **70%** (avg -\$2,804) |

- **Their bad days rarely line up.** On IC's 10 worst days MO1 lost on 3, and on MO1's 23 worst days IC lost on 2; on average the other
  book gave up only \$36-445 on those days.
- **MO1 and MO2 share their bad days:** on MO1's worst days MO2 lost 70% of the time, and on MO2's worst days MO1 lost every time. That
  is what high correlation looks like in practice.

## 6. Why their spikes are different: what makes each one win and lose

Trading on different days explains the correlation number. It doesn't explain **why their big wins and big losses come from
different markets**. That comes from how a condor makes money: it keeps its credit if SPX closes between its short strikes, so
**each condor's winning zone is centred where SPX was when it was opened**. IC and MO open at different points of the day's path,
with different strike widths, so their winning zones sit in different places.


    
![png](correlation_mo_files/output_23_0.png)
    



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>short put (low edge)</th>
      <th>entry (centre)</th>
      <th>short call (high edge)</th>
      <th>width</th>
      <th>short strikes from spot, %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC</th>
      <td>-0.53</td>
      <td>-0.23</td>
      <td>-0.00</td>
      <td>0.46</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>MO1</th>
      <td>0.13</td>
      <td>0.26</td>
      <td>0.39</td>
      <td>0.25</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>MO2</th>
      <td>0.14</td>
      <td>0.29</td>
      <td>0.43</td>
      <td>0.27</td>
      <td>0.14</td>
    </tr>
  </tbody>
</table>
</div>


    IC vs MO1 median winning zones overlap by -0.14% (negative = they don't overlap)
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">IC</th>
      <th colspan="2" halign="left">MO1</th>
      <th colspan="2" halign="left">MO2</th>
    </tr>
    <tr>
      <th></th>
      <th>days</th>
      <th>avg P/L</th>
      <th>days</th>
      <th>avg P/L</th>
      <th>days</th>
      <th>avg P/L</th>
    </tr>
    <tr>
      <th>day</th>
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
      <th>below -1%</th>
      <td>6</td>
      <td>-151.0</td>
      <td>11</td>
      <td>-776.0</td>
      <td>8</td>
      <td>-439.0</td>
    </tr>
    <tr>
      <th>-1% to -0.5%</th>
      <td>30</td>
      <td>590.0</td>
      <td>26</td>
      <td>-364.0</td>
      <td>21</td>
      <td>-119.0</td>
    </tr>
    <tr>
      <th>-0.5% to -0.2%</th>
      <td>53</td>
      <td>961.0</td>
      <td>56</td>
      <td>302.0</td>
      <td>41</td>
      <td>485.0</td>
    </tr>
    <tr>
      <th>-0.2% to +0.2%</th>
      <td>74</td>
      <td>750.0</td>
      <td>117</td>
      <td>868.0</td>
      <td>83</td>
      <td>1003.0</td>
    </tr>
    <tr>
      <th>+0.2% to +0.5%</th>
      <td>14</td>
      <td>-380.0</td>
      <td>78</td>
      <td>912.0</td>
      <td>60</td>
      <td>1060.0</td>
    </tr>
    <tr>
      <th>+0.5% to +1%</th>
      <td>9</td>
      <td>-484.0</td>
      <td>111</td>
      <td>184.0</td>
      <td>84</td>
      <td>348.0</td>
    </tr>
    <tr>
      <th>above +1%</th>
      <td>1</td>
      <td>-1463.0</td>
      <td>59</td>
      <td>-194.0</td>
      <td>53</td>
      <td>-68.0</td>
    </tr>
  </tbody>
</table>
</div>


    the same, on the days IC and MO1 both traded:
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">IC</th>
      <th colspan="2" halign="left">MO1</th>
    </tr>
    <tr>
      <th></th>
      <th>days</th>
      <th>avg P/L</th>
      <th>days</th>
      <th>avg P/L</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>below -1%</th>
      <td>3</td>
      <td>-125.0</td>
      <td>3</td>
      <td>-1476.0</td>
    </tr>
    <tr>
      <th>-1% to -0.5%</th>
      <td>13</td>
      <td>705.0</td>
      <td>13</td>
      <td>-238.0</td>
    </tr>
    <tr>
      <th>-0.5% to -0.2%</th>
      <td>26</td>
      <td>1152.0</td>
      <td>26</td>
      <td>1214.0</td>
    </tr>
    <tr>
      <th>-0.2% to +0.2%</th>
      <td>48</td>
      <td>675.0</td>
      <td>48</td>
      <td>1120.0</td>
    </tr>
    <tr>
      <th>+0.2% to +0.5%</th>
      <td>8</td>
      <td>-340.0</td>
      <td>8</td>
      <td>-191.0</td>
    </tr>
    <tr>
      <th>+0.5% to +1%</th>
      <td>5</td>
      <td>-564.0</td>
      <td>5</td>
      <td>171.0</td>
    </tr>
    <tr>
      <th>above +1%</th>
      <td>1</td>
      <td>-1463.0</td>
      <td>1</td>
      <td>-3047.0</td>
    </tr>
  </tbody>
</table>
</div>


**1. Their winning zones don't overlap.** Measured against the day's 09:30 open, the median condor keeps its credit if SPX closes:

| | short put (low edge) | entry (centre) | short call (high edge) | width | short strikes from spot |
|---|---|---|---|---|---|
| **IC** | **-0.53%** | -0.23% | **0.00%** | 0.46% | 0.21% |
| **MO1** | **+0.13%** | +0.26% | **+0.39%** | 0.25% | 0.13% |

- **IC's zone sits below the open, MO's above it.** IC enters after a pullback (its median entry is 0.23% below the open), MO after a
  rally (0.26% above), and each condor is centred on its entry. The two median zones are 0.14% apart: **a close that keeps IC's credit
  in full sits outside MO's zone, and the other way round.**
- **MO's zone is about half as wide.** Its 35-delta shorts sit 0.13% from spot against IC's 0.21%, so MO needs SPX to finish closer to its
  entry. It collects more for that (\$470 against \$325).

**2. So different days pay each book, and different days hurt it.** Average daily P/L by how SPX moved from its open to its close, on
the days each book traded:

| SPX open to close | IC days | IC avg P/L | MO1 days | MO1 avg P/L |
|---|---|---|---|---|
| below -1% | 6 | -\$151 | 11 | -\$776 |
| -1% to -0.5% | 30 | **+\$590** | 26 | **-\$364** |
| -0.5% to -0.2% | 53 | **+\$961** | 56 | +\$302 |
| -0.2% to +0.2% | 74 | +\$750 | 117 | +\$868 |
| +0.2% to +0.5% | 14 | **-\$380** | 78 | **+\$912** |
| +0.5% to +1% | 9 | -\$484 | 111 | +\$184 |
| above +1% | 1 | -\$1,463 | 59 | -\$194 |

- **IC's best days are mild sell-offs** (-0.5% to -0.2%: +\$961 a day); **its pain is up days** (+0.2% and above), where SPX runs away
  from a condor centred below the open, through its short call.
- **MO's best days are flat to mildly up** (-0.2% to +0.5%: +\$868 to +\$912); **its pain is down days** (-1% to -0.5% and worse), where SPX
  falls away from a condor centred above the open, through its short put.
- **The middle pays both** (-0.2% to +0.2%), and **only a big sell-off (below -1%) hurts both**, IC much less.
- **On the days they share** (second table) the down-day split holds: on a -1% to -0.5% day IC makes +\$705 while MO1 loses \$238. On the 8
  shared +0.2% to +0.5% days both lose a little (IC -\$340, MO1 -\$191).


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>losing condors</th>
      <th>of them: call side broken %</th>
      <th>put side broken %</th>
      <th>worst 5%: put side broken %</th>
      <th>traded days: median VIX at the open</th>
      <th>traded days with VIX &gt; 20 %</th>
      <th>traded days: median range %</th>
      <th>traded days that close below the open %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC</th>
      <td>138.0</td>
      <td>54.3</td>
      <td>45.7</td>
      <td>48.6</td>
      <td>16.0</td>
      <td>19.3</td>
      <td>0.8</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>MO1</th>
      <td>1112.0</td>
      <td>60.3</td>
      <td>39.7</td>
      <td>49.3</td>
      <td>16.7</td>
      <td>26.0</td>
      <td>0.8</td>
      <td>31.2</td>
    </tr>
    <tr>
      <th>MO2</th>
      <td>827.0</td>
      <td>61.9</td>
      <td>38.1</td>
      <td>34.5</td>
      <td>17.8</td>
      <td>32.0</td>
      <td>0.9</td>
      <td>29.7</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>IC traded</th>
      <th>IC avg P/L when in</th>
      <th>MO1 traded</th>
      <th>MO1 avg P/L when in</th>
      <th>MO2 traded</th>
      <th>MO2 avg P/L when in</th>
      <th>IC and MO1 both in</th>
      <th>IC and MO2 both in</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25 widest-range days</th>
      <td>25.0</td>
      <td>1.0</td>
      <td>-1172.0</td>
      <td>8.0</td>
      <td>-339.0</td>
      <td>8.0</td>
      <td>-339.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SPX fell &gt; 1% open to close</th>
      <td>91.0</td>
      <td>6.0</td>
      <td>-151.0</td>
      <td>11.0</td>
      <td>-776.0</td>
      <td>8.0</td>
      <td>-439.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SPX rose &gt; 1% open to close</th>
      <td>86.0</td>
      <td>1.0</td>
      <td>-1463.0</td>
      <td>59.0</td>
      <td>-194.0</td>
      <td>53.0</td>
      <td>-68.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


**3. They're exposed to different markets.**

| | IC | MO1 |
|---|---|---|
| traded days that close below the open | **68%** | **31%** |
| traded days with VIX above 20 | 19% | **26%** |
| median VIX on traded days | 16.0 | 16.7 |

- **IC trades mostly down days, MO mostly up days.** 68% of IC's trading days close below their open, against 31% of MO1's. Each book's
  entry signal picks the kind of day it then trades.
- **IC is less exposed to stressed markets:** 19% of its trading days have VIX above 20, against 26% for MO1 (32% for MO2).

**4. On the market's worst days, they are almost never in together.**

| | days | IC in | MO1 in | MO2 in | IC and MO1 both in | IC and MO2 both in |
|---|---|---|---|---|---|---|
| the 25 widest-range days | 25 | 1 | 8 | 8 | **1** | **1** |
| SPX fell more than 1%, open to close | 91 | 6 | 11 | 8 | **3** | **1** |
| SPX rose more than 1%, open to close | 86 | 1 | 59 | 53 | **1** | **0** |

- **IC avoids almost all violent days.** It traded 1 of the 25 widest-range days and 6 of the 91 days SPX fell more than 1%.
- **MO trades strong up days, IC doesn't.** MO1 was in on 59 of the 86 days SPX rose more than 1% (losing only \$194 a day on average,
  because its condor sits above the open), IC on 1.
- **The two were in together on 3 big down days and 1 big up day in four years**, and on 1 of the 25 widest-range days.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>date</th>
      <th>2022Q2</th>
      <th>2022Q3</th>
      <th>2022Q4</th>
      <th>2023Q1</th>
      <th>2023Q2</th>
      <th>2023Q3</th>
      <th>2023Q4</th>
      <th>2024Q1</th>
      <th>2024Q2</th>
      <th>2024Q3</th>
      <th>2024Q4</th>
      <th>2025Q1</th>
      <th>2025Q2</th>
      <th>2025Q3</th>
      <th>2025Q4</th>
      <th>2026Q1</th>
      <th>2026Q2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC</th>
      <td>7629.0</td>
      <td>10839.0</td>
      <td>2598.0</td>
      <td>3651.0</td>
      <td>18104.0</td>
      <td>3919.0</td>
      <td>1059.0</td>
      <td>8776.0</td>
      <td>5768.0</td>
      <td>12250.0</td>
      <td>-2211.0</td>
      <td>295.0</td>
      <td>3985.0</td>
      <td>11392.0</td>
      <td>12253.0</td>
      <td>2803.0</td>
      <td>8931.0</td>
    </tr>
    <tr>
      <th>MO1</th>
      <td>-3090.0</td>
      <td>16913.0</td>
      <td>8588.0</td>
      <td>-2841.0</td>
      <td>15857.0</td>
      <td>20442.0</td>
      <td>8479.0</td>
      <td>4288.0</td>
      <td>15315.0</td>
      <td>26604.0</td>
      <td>14125.0</td>
      <td>1537.0</td>
      <td>19351.0</td>
      <td>5801.0</td>
      <td>21933.0</td>
      <td>7597.0</td>
      <td>-327.0</td>
    </tr>
    <tr>
      <th>MO2</th>
      <td>-9.0</td>
      <td>16913.0</td>
      <td>8588.0</td>
      <td>-2841.0</td>
      <td>15857.0</td>
      <td>15702.0</td>
      <td>9303.0</td>
      <td>9819.0</td>
      <td>18772.0</td>
      <td>31658.0</td>
      <td>20329.0</td>
      <td>1537.0</td>
      <td>19351.0</td>
      <td>-851.0</td>
      <td>18419.0</td>
      <td>4177.0</td>
      <td>-327.0</td>
    </tr>
  </tbody>
</table>
</div>


    days both IC and MO1 lost: 15
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO1</th>
      <th>oc_pct</th>
      <th>range_pct</th>
      <th>vix_open</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-05-12</th>
      <td>-1557.76</td>
      <td>-35.28</td>
      <td>0.64</td>
      <td>1.02</td>
      <td>19.44</td>
    </tr>
    <tr>
      <th>2023-12-13</th>
      <td>-1462.76</td>
      <td>-3047.16</td>
      <td>1.31</td>
      <td>1.43</td>
      <td>12.17</td>
    </tr>
    <tr>
      <th>2023-10-11</th>
      <td>-1277.16</td>
      <td>-3332.16</td>
      <td>0.24</td>
      <td>0.76</td>
      <td>16.52</td>
    </tr>
    <tr>
      <th>2025-02-18</th>
      <td>-1173.40</td>
      <td>-1224.40</td>
      <td>0.13</td>
      <td>0.49</td>
      <td>15.82</td>
    </tr>
    <tr>
      <th>2022-06-28</th>
      <td>-1172.16</td>
      <td>-2104.40</td>
      <td>-2.34</td>
      <td>3.21</td>
      <td>26.82</td>
    </tr>
    <tr>
      <th>2025-05-28</th>
      <td>-1037.16</td>
      <td>-3872.16</td>
      <td>-0.62</td>
      <td>0.98</td>
      <td>19.06</td>
    </tr>
    <tr>
      <th>2025-09-23</th>
      <td>-1025.16</td>
      <td>-2140.52</td>
      <td>-0.53</td>
      <td>0.81</td>
      <td>16.02</td>
    </tr>
    <tr>
      <th>2024-02-26</th>
      <td>-781.28</td>
      <td>-615.16</td>
      <td>-0.46</td>
      <td>0.56</td>
      <td>13.96</td>
    </tr>
    <tr>
      <th>2025-05-16</th>
      <td>-753.88</td>
      <td>-2716.16</td>
      <td>0.49</td>
      <td>0.86</td>
      <td>17.62</td>
    </tr>
    <tr>
      <th>2023-12-20</th>
      <td>-668.88</td>
      <td>-4167.16</td>
      <td>-1.39</td>
      <td>1.68</td>
      <td>12.35</td>
    </tr>
  </tbody>
</table>
</div>


**5. Their losing stretches come at different times.** By calendar quarter:

- **IC's one losing quarter, 2024 Q4 (-\$2.2k), was a winning quarter for MO1** (+\$14.1k).
- **MO1's losing quarters (2022 Q2, 2023 Q1, 2026 Q2 so far) were all positive for IC** (+\$7.6k, +\$3.7k, +\$8.9k).
- **In 17 quarters the two never lost in the same quarter.**

**6. The risk they do share.** On 15 days both IC and MO1 lost (out of 104 shared days), on average -\$789 for IC and -\$1,988 for MO1.
They're mostly ordinary days (median range 0.8%, 40% of them down), where SPX ended up between the two zones or swung through both, plus
the odd sharp move (2022-06-28, SPX -2.3%: IC -\$1.2k, MO1 -\$2.1k; 2023-12-20, SPX -1.4%: IC -\$0.7k, MO1 -\$4.2k). That is the cost of
running them together, and it is small next to what each earns.

**Why trading them together isn't risky:**

1. **Their winning zones sit on opposite sides of the open**, so the close that hurts one usually pays the other.
2. **They're hurt by different markets:** IC by up days, MO by down days and turbulence.
3. **IC sits out almost all the violent days MO is exposed to**, so a market shock rarely hits both.
4. **Their bad stretches don't line up:** never a losing quarter at the same time.

The one scenario that hurts both is a **sharp sell-off on a day both are in**. It happened a handful of times in four years, and IC lost
far less than MO on those days.

## 7. Running them together

Each book alone and in pairs, one contract each. Each book is given its own \$200k, so a pair is scored on \$400k; otherwise the
risk-free hurdle would make any pair look better just for running more money.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>capital</th>
      <th>total_pl</th>
      <th>sharpe</th>
      <th>sharpe_to_2024</th>
      <th>sharpe_from_2025</th>
      <th>return_on_capital %</th>
      <th>max_drawdown</th>
      <th>worst drawdown ends</th>
      <th>losing days %</th>
      <th>worst day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC</th>
      <td>200000</td>
      <td>112044.0</td>
      <td>2.59583</td>
      <td>2.442966</td>
      <td>2.92227</td>
      <td>56.022</td>
      <td>-3808.68</td>
      <td>2024-12-10</td>
      <td>23.529412</td>
      <td>-2440.52</td>
    </tr>
    <tr>
      <th>MO1</th>
      <td>200000</td>
      <td>180572.8</td>
      <td>1.578353</td>
      <td>1.651584</td>
      <td>1.430671</td>
      <td>90.2864</td>
      <td>-15704.2</td>
      <td>2023-04-13</td>
      <td>41.266376</td>
      <td>-4847.16</td>
    </tr>
    <tr>
      <th>MO2</th>
      <td>200000</td>
      <td>186396.44</td>
      <td>1.85775</td>
      <td>2.204439</td>
      <td>1.152806</td>
      <td>93.19822</td>
      <td>-15704.2</td>
      <td>2023-04-13</td>
      <td>39.428571</td>
      <td>-4847.16</td>
    </tr>
    <tr>
      <th>IC + MO1</th>
      <td>400000</td>
      <td>292616.8</td>
      <td>2.184023</td>
      <td>2.217217</td>
      <td>2.114823</td>
      <td>73.1542</td>
      <td>-13063.44</td>
      <td>2025-03-17</td>
      <td>38.632163</td>
      <td>-4909.32</td>
    </tr>
    <tr>
      <th>IC + MO2</th>
      <td>400000</td>
      <td>298440.44</td>
      <td>2.513564</td>
      <td>2.773696</td>
      <td>1.979575</td>
      <td>74.61011</td>
      <td>-12862.56</td>
      <td>2023-04-03</td>
      <td>36.008677</td>
      <td>-4909.32</td>
    </tr>
    <tr>
      <th>MO1 + MO2</th>
      <td>400000</td>
      <td>366969.24</td>
      <td>1.761272</td>
      <td>1.965135</td>
      <td>1.347494</td>
      <td>91.74231</td>
      <td>-31408.4</td>
      <td>2023-04-13</td>
      <td>41.266376</td>
      <td>-9694.32</td>
    </tr>
    <tr>
      <th>IC + MO1 + MO2</th>
      <td>600000</td>
      <td>479013.24</td>
      <td>2.123785</td>
      <td>2.297061</td>
      <td>1.771007</td>
      <td>79.83554</td>
      <td>-28121.8</td>
      <td>2023-04-03</td>
      <td>38.632163</td>
      <td>-9694.32</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](correlation_mo_files/output_30_1.png)
    


| book | capital | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | worst drawdown ends |
|---|---|---|---|---|---|---|
| IC | \$200k | \$112.0k | **2.60** | 2.44 / 2.92 | **-\$3.8k** | 2024-12-10 |
| MO1 | \$200k | \$180.6k | 1.58 | 1.65 / 1.43 | -\$15.7k | 2023-04-13 |
| MO2 | \$200k | \$186.4k | 1.86 | 2.20 / 1.15 | -\$15.7k | 2023-04-13 |
| **IC + MO1** | \$400k | \$292.6k | 2.18 | 2.22 / 2.11 | -\$13.1k | 2025-03-17 |
| **IC + MO2** | \$400k | \$298.4k | **2.51** | 2.77 / 1.98 | -\$12.9k | 2023-04-03 |
| MO1 + MO2 | \$400k | \$367.0k | 1.76 | 1.97 / 1.35 | **-\$31.4k** | 2023-04-13 |

- **Adding IC to an MO book lifts it a lot:** IC + MO2 scores 2.51 against MO2's 1.86, and its drawdown (-\$12.9k) is smaller than MO2's own
  (-\$15.7k) even though it runs both books. That is diversification: IC's good stretches cover part of MO's bad ones.
- **Adding MO to IC lowers IC's Sharpe** (2.60 -> 2.51 with MO2, 2.18 with MO1) but adds \$186k of P/L. At one contract each MO is the bigger
  book: its swings dominate the pair, so the pair looks more like MO than like IC. Sized to equal risk the picture would differ.
- **MO1 + MO2 is no diversification at all:** the drawdowns land on the same days and simply add up (-\$31.4k = two times -\$15.7k),
  and the Sharpe (1.76) sits between the two.
- **How they "don't affect each other":** they're separate trades with separate entry rules, so neither changes the other's trades. What
  they share is the market: on the few days both are in, they rise and fall together (§5). Because those days are few, combining them
  smooths the equity curve instead of doubling the risk.

### All three as one portfolio

IC, MO1 and MO2 in one account, one contract each, \$200k per strategy (\$600k). The margin table checks how much margin the portfolio
ties up on its busiest days (every condor opened that day, since all of them are open together by the afternoon), and the last table
lists the portfolio's worst days.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>capital</th>
      <th>peak day margin</th>
      <th>median margin on a trading day</th>
      <th>peak as % of capital</th>
      <th>days all of them trade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC</th>
      <td>200000.0</td>
      <td>6025.0</td>
      <td>2405.0</td>
      <td>3.012500</td>
      <td>187.0</td>
    </tr>
    <tr>
      <th>MO1</th>
      <td>200000.0</td>
      <td>4950.0</td>
      <td>3545.0</td>
      <td>2.475000</td>
      <td>458.0</td>
    </tr>
    <tr>
      <th>MO2</th>
      <td>200000.0</td>
      <td>4950.0</td>
      <td>3495.0</td>
      <td>2.475000</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>IC + MO1</th>
      <td>400000.0</td>
      <td>8675.0</td>
      <td>3720.0</td>
      <td>2.168750</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>IC + MO2</th>
      <td>400000.0</td>
      <td>8675.0</td>
      <td>3545.0</td>
      <td>2.168750</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>MO1 + MO2</th>
      <td>400000.0</td>
      <td>9900.0</td>
      <td>6415.0</td>
      <td>2.475000</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>IC + MO1 + MO2</th>
      <td>600000.0</td>
      <td>12520.0</td>
      <td>6240.0</td>
      <td>2.086667</td>
      <td>76.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IC</th>
      <th>MO1</th>
      <th>MO2</th>
      <th>portfolio</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-10-27</th>
      <td>0.0</td>
      <td>-4847.0</td>
      <td>-4847.0</td>
      <td>-9694.0</td>
    </tr>
    <tr>
      <th>2025-08-25</th>
      <td>0.0</td>
      <td>-4672.0</td>
      <td>-4672.0</td>
      <td>-9344.0</td>
    </tr>
    <tr>
      <th>2025-05-28</th>
      <td>-1037.0</td>
      <td>-3872.0</td>
      <td>-3872.0</td>
      <td>-8781.0</td>
    </tr>
    <tr>
      <th>2025-01-23</th>
      <td>0.0</td>
      <td>-4377.0</td>
      <td>-4377.0</td>
      <td>-8754.0</td>
    </tr>
    <tr>
      <th>2023-06-12</th>
      <td>0.0</td>
      <td>-4307.0</td>
      <td>-4307.0</td>
      <td>-8614.0</td>
    </tr>
    <tr>
      <th>2024-05-06</th>
      <td>0.0</td>
      <td>-4142.0</td>
      <td>-4142.0</td>
      <td>-8284.0</td>
    </tr>
    <tr>
      <th>2024-08-19</th>
      <td>0.0</td>
      <td>-4067.0</td>
      <td>-4067.0</td>
      <td>-8134.0</td>
    </tr>
    <tr>
      <th>2026-04-13</th>
      <td>0.0</td>
      <td>-4022.0</td>
      <td>-4022.0</td>
      <td>-8044.0</td>
    </tr>
    <tr>
      <th>2026-05-06</th>
      <td>0.0</td>
      <td>-3972.0</td>
      <td>-3972.0</td>
      <td>-7944.0</td>
    </tr>
    <tr>
      <th>2023-10-11</th>
      <td>-1277.0</td>
      <td>-3332.0</td>
      <td>-3332.0</td>
      <td>-7941.0</td>
    </tr>
  </tbody>
</table>
</div>


| portfolio | capital | total P/L | Sharpe | 2022-24 / 2025-26 | max drawdown | worst day |
|---|---|---|---|---|---|---|
| **IC alone** | \$200k | \$112.0k | **2.60** | 2.44 / 2.92 | **-\$3.8k** | -\$2.4k |
| IC + MO2 | \$400k | \$298.4k | 2.51 | 2.77 / 1.98 | -\$12.9k | -\$4.9k |
| IC + MO1 | \$400k | \$292.6k | 2.18 | 2.22 / 2.11 | -\$13.1k | -\$4.9k |
| IC + MO1 + MO2 | \$600k | **\$479.0k** | 2.12 | 2.30 / 1.77 | **-\$28.1k** | **-\$9.7k** |

**Adding the third book makes the portfolio worse, not better.**

- **All three earn the most money** (\$479.0k), but Sharpe falls to 2.12 and the drawdown grows to -\$28.1k, ending 2023-04-03, MO's bad
  stretch.
- **The reason is MO1 and MO2 together.** They're 0.88 correlated and trade the same trades on 350 days, so holding both is close to two
  contracts of one MO book: every MO loss counts twice. **On all ten of the portfolio's worst days MO1 and MO2 lose the same amount on
  the same day**, with IC flat on eight of them (the worst, 2025-10-27, is MO1 and MO2 -\$4.8k each, -\$9.7k in all).
- **IC + MO2 is the best combination with MO in it:** Sharpe 2.51 and a drawdown smaller than MO2 alone. MO2 is MO1 with a realized-
  volatility floor: it makes slightly more than MO1 (\$186.4k against \$180.6k) on fewer, better days.

**Margin isn't the limit at one contract each.** On the busiest day all three together tie up about \$12.5k of margin (median \$6.2k on a
trading day), about 2% of \$600k. The three books are in on the same day on 76 sessions.

**A shared \$200k would only make the numbers look better, not the portfolio.** Scored on one \$200k instead of \$600k, the three-book
Sharpe would read 2.47, but only because the same P/L is measured against less capital. The drawdown (-\$28.1k) is the same, and on
\$200k it is 14% of the account instead of 5%.

## 8. Summary

**Why IC and MO1 / MO2 aren't correlated, despite similar setups:**

1. **Same trade, different timing.** All three sell 10-wide 0DTE SPX condors, and when IC and MO1 open in the same minute their
   condors move together (0.77). The low correlation isn't about the trade.
2. **They trade on different days.** IC and MO1 share only 104 of their 187 and 458 trading days. Half of IC's daily swings but only 21% of
   MO1's fall on those shared days, so even though they move together on them (0.60), the overall daily correlation is
   0.60 x sqrt(0.50 x 0.21) = **0.20**.
3. **Because their signals point opposite ways.** IC enters when SPX is **at or below its open** (a pullback); MO enters when SPX is
   **at least 0.5% above its 5-day average** (a stretched rally). That alone rejects 74% of MO1's condors from IC's view. IC also trades
   only seven entry times, skips Thursdays and requires the VIX9D/VIX band, which MO doesn't. Only 73 of MO1's 458 days would pass every
   IC filter.
4. **Even on shared days they differ:** different minutes and different strikes (MO about 7 points from spot, IC about 10-13), so the
   same move can break MO's short strike and leave IC's intact.
5. **Their spikes differ because their winning zones sit on opposite sides of the day's open** (IC below, MO above), so up days hurt
   IC and pay MO while down days do the reverse; IC also avoids almost all the turbulent days MO trades (Section 6).

**What it means for running them together:**

- **They don't interfere:** separate entry rules mean neither book changes the other's trades.
- **They diversify:** their worst days rarely coincide, they never lost in the same quarter, and adding IC to MO2 lifts its Sharpe from
  1.86 to 2.51 and shrinks its drawdown.
- **IC alone has the best Sharpe** (2.60) and by far the smallest drawdown (-\$3.8k); adding MO adds money (\$186k with MO2) at a lower
  Sharpe, because at one contract each MO's bigger swings dominate.
- **MO1 and MO2 are effectively one book** (0.88): running both is close to doubling one position, not diversifying. All three together
  earn the most (\$479.0k) but at a Sharpe of 2.12 with a -\$28.1k drawdown.

**Caveats.**

- IC is a rebuild of the strategy with the backtester's rules (see `ic_analysis.ipynb`); MO1 and MO2 are the backtester's own logs.
- The comparison covers 2022-05 to 2026-05 only (the IC export's range).
- Correlation is measured on daily P/L as the backtester does. Within a day, the books' open positions overlap more than the daily
  number suggests on the days they share (§5).
