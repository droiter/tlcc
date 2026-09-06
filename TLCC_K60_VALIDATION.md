# TLCC K=60 — validation of the raw-close lead-lag findings

Date: 2026-09-04 (session continuation after the 12:17 `tlcc_tlc_k60` run)
Data: 197 indices in `trader_nfq_aidx.hdf` (SSE 880xxx sector/theme/sentiment
indices + major broad indices), daily closes, 2007–2026 (up to 4750 trading days).
Method family: time-lagged cross-correlation, `r[i,j,k] = corr(x_i[t], x_j[t+k])`,
k = 0..60 → *i leads j by k days* when k>0 (single-sided, overlapping-day windows,
no significance machinery in the original pipeline).

## 1. Problem found in the raw-close K=60 tables

The K=60 run on **raw close levels** produced a lead-lag table whose headline
entries are artefacts:

1. **Edge-pinning.** The net-lead statistic `d(k) = r_ab(k) − r_ba(k)` has its
   argmax at the window edge k=60 for ~99% of pairs with `|d| ≥ 0.3`
   (225/225 of `|d| ≥ 0.5`); top-|d| pairs sit at k=60 almost without exception
   (e.g. `近期新低→预计转亏` d=0.997@60, many `X→活跃可转债` d=0.7–0.85@60).
2. **Mean |d(k)| grows ~linearly with lag** on raw closes:
   mean|d(1)|≈0.002 → mean|d(60)|≈0.118. Larger k drops different ends of the
   shared window, so the two correlation legs measure *different regimes*
   (bull/bear segments of non-stationary levels). That mechanical k-dependence —
   not genuine lead-lag — dominates the statistic.
3. **Tiny-window / tiny-overlap noise.** Some headline pairs have ~37–46 shared
   days with lags of 28–44, i.e. correlations over 2–9 points that are ±1 by
   construction (`闪拉`, `闪跌`, `活跃ETF`, `活跃可转债` rows).

Consequence: best-lag tables (`_bestlag_*`), the strict-lead ordering and the
net-lead ranking from the raw-close run cannot be read as evidence of leads.

## 2. What was done

### a) Universe rerun on daily log returns — `tlcc_tlc_ret.py` (K=60)
Same 197-index panel, same single-sided TLC, but on `log(close_t/close_{t-1})`
(level trends/regimes removed). Outputs (all written):

- `tlcc_tlc_ret_k60_cube.npz`, `_bestlag_k.csv`, `_bestlag_r.csv`
- `tlcc_tlc_ret_k60_lead_ordered.csv`, `tlcc_tlc_ret_k60_net_lead.csv`
- quality-gated ranking: `tlcc_tlc_ret_k60_net_lead_q.csv`
  (overlap ≥ 250 d and L − k ≥ 100 shared points)

Artifact diagnostics on log returns: mean |d(k)| is **flat ~0.02 across k=1..60**
(0.042 at k=1, 0.022 at k=60); edge-pinned @k=60 drops from ~45% (raw, gated
pool) to **2.1%** (returns). Contemporaneous correlation rises (r0 median
0.48 → 0.60), consistent with returns co-movement being the real shared driver.

### b) Per-pair validation — `tlc_k60_validate.py` → `tlcc_tlc_k60_validated.csv`
849 candidate unordered pairs (top-600 by raw `|d|` over k≤20 + all raw
`|d(k≤60)| ≥ 0.4`) were re-tested on log returns over the same shared window:

- **half-split stability**: argmax k and sign of `d(k≤20)` reproduced in first
  vs second half (same sign, |k₁−k₂| ≤ 5, |d| ≥ 0.03 in both);
- **stationary block bootstrap** (independent block resampling, geometric blocks
  mean 15 d, B=399) null `max_{k≤20}|d(k)|`, p ≤ 0.05 required;
- agreement of raw-close vs returns-based ranking.

Outcome (849 pairs):
| verdict | count |
|---|---:|
| rejected: sign flips between halves | 385 |
| rejected: not stable / not significant | 263 |
| rejected: half-stable but bootstrap not significant | 172 |
| **VALIDATED (half-stable + p≤0.05)** | **29** |

Raw-vs-returns ranking agreement is weak (Spearman ρ = 0.12 over the 849 pairs),
and of the 695 pairs with raw `|d|max(k≤60) ≥ 0.4` (the edge-pinned population)
only 65 are still strong+stable on returns → **raw edge leads mostly do not
survive detrending**.

## 3. Surviving candidates (validated, k ≤ 20)

All validated pairs have their asymmetry at **k = 1–2 trading days** (d ≈ 0.12–0.25;
r0(returns) ≈ 0.4–0.7). Illustrative rows (full list in the CSV):

| a | b | k | d_ret | p_bs |
|---|---|---|---:|---:|
| 陆股通减 | 昨日上榜 | 1 | 0.241 | 0.0025 |
| 陆股通增 | 昨日上榜 | 1 | 0.214 | 0.0025 |
| 陆股通减 | 最近多板 | 1 | 0.213 | 0.0025 |
| 电气设备 | 通达信热股 | 1 | 0.213 | 0.0025 |
| 基金重仓 / QFII重仓 / 化工 / 沪深300 … | 通达信热股 | 1–2 | 0.15–0.20 | ≤0.01 |
| 近端次新 | 活跃ETF | 2 | −0.252 | 0.0025 |
| 高质押股 | 持续增长 | 1 | −0.119 | 0.010 |
| 户数减少 | 昨收活跃 | 2 | 0.118 | 0.048 |

**Interpretation caveats (read before trading on these):**
- “k=1” against `昨日上榜`/`昨日*`/`昨曾跌停`-family followers may be partly
  *definitional*: an index that by construction encodes day-t−1 information will
  lag any day-t index by one day. Same caution applies to `活跃可转债`,
  `历史新低` (rolling-window constructions) and the `闪拉/闪跌` group.
- Independent-block bootstrap rejects “no relation at any lag”; with r0(returns)
  ≈ 0.5–0.7 the contemporaneous co-movement is huge, so a small stable d is a
  second-order effect. Treat these as screens, not causal proof.
- 29/19306 unordered pairs pass the full screen — plausible given that these
  indices share the same underlying market.

## 4. Recommendations / next steps

1. Treat the raw-close tables (`tlcc_tlc_k60_*`) as descriptive only; use the
   log-return cube (`tlcc_tlc_ret_k60_*`) for any lead-lag inference, with the
   quality gates from §2a.
2. For the k=1–2 survivors: forward out-of-sample test (rolling re-estimation of
   d(k), then out-of-window direction checks) rather than more in-sample screens;
   and reconcile with index definitions (look-ahead within `昨日*` indices).
3. Optionally repeat the whole pipeline at K=20 on log returns — the short
   horizons are where these data carry signal; the K=60 grid mostly shows that
   nothing meaningful hides at k>20 beyond the flat ~0.02 noise floor.
4. Scripts to rerun: `python3 tlcc_tlc_ret.py 60` (≈1 min) and
   `python3 tlc_k60_validate.py` (≈20 s after cube exists).

## Files produced in this session
- `tlcc_tlc_ret_k60_cube.npz`, `tlcc_tlc_ret_k60_bestlag_k.csv`,
  `tlcc_tlc_ret_k60_bestlag_r.csv`, `tlcc_tlc_ret_k60_lead_ordered.csv`,
  `tlcc_tlc_ret_k60_net_lead.csv`, `tlcc_tlc_ret_k60_net_lead_q.csv`
- `tlcc_tlc_k60_validated.csv` (849-row validation table; verdict column)
- `tlcc_tlc_ret.py`, `tlc_k60_validate.py`
- logs: `tlcc_tlc_ret_k60.log`, `tlc_k60_validate.log`
