# -*- coding: utf-8 -*-
"""Validation of the k60 raw-close TLC lead-lag table (tlcc_tlc_k60_*).

Hypothesis driving this check: on RAW CLOSE levels the net-lead statistic
d(k) = r_ab(k) - r_ba(k) grows ~linearly with lag and its argmax is pinned at
the window edge (k=60) for ~99% of high-|d| pairs -> regime-drift artifact of
non-stationary levels, not genuine lead-lag.  We therefore re-test candidate
pairs on daily LOG-RETURNS (stationary-ish):

  1. recompute net-lead d_ret(k), k=1..60, on log returns over the SAME
     shared calendar window;
  2. within-sample stability: same statistic on first/second halves;
  3. stationary block bootstrap (independent resampling of the two series,
     geometric blocks, mean length B_L) for a max-|d| null over k<=20.

Candidates = unordered pairs (overlap>=500d) that are either in the top 600 by
|d| over k<=20 on raw closes, or have |d_raw(k=1..60)| >= 0.4 (the edge-pinned
population we want to expose as artifacts).

Output: tlcc_tlc_k60_validated.csv  (one row per analyzed pair + flags)
"""
import numpy as np
import pandas as pd
import time, sys

OUTDIR = '/workspace/tlcc'
SRC = f'{OUTDIR}/trader_nfq_aidx.hdf'
KMAX = 60          # lags considered on returns
KSIG = 20          # significance window cap for the bootstrap / headline stat
BLOCK_MEAN = 15    # mean block length (trading days) for stationary bootstrap
B = 399            # bootstrap replicates
MIN_OVERLAP = 500

t0 = time.time()
def log(s):
    print(f'[{time.time()-t0:6.0f}s] {s}', flush=True)

# ---------------- raw-side stats from the stored cube ----------------
z = np.load(f'{OUTDIR}/tlcc_tlc_k60_cube.npz', allow_pickle=True)
r = z['r']; labels = list(z['labels']); m = len(labels)
lab2i = {l: i for i, l in enumerate(labels)}
D = r[:, :, 1:] - r.transpose(1, 0, 2)[:, :, 1:]          # d(k), k=1..60
d20 = D[:, :, :20]
with np.errstate(invalid='ignore'):
    amm = np.where(np.isnan(np.abs(d20)), -1.0, np.abs(d20))
    k20 = np.nanargmax(amm, axis=2) + 1
    dv20 = np.take_along_axis(d20, (k20 - 1)[:, :, None], axis=2)[:, :, 0]
    dv20 = np.where(amm.max(axis=2) >= 0, dv20, np.nan)
    raw60 = np.nanmax(np.where(np.isnan(np.abs(D)), -1.0, np.abs(D)), axis=2)
    raw60 = np.where(raw60 >= 0, raw60, np.nan)

nl = pd.read_csv(f'{OUTDIR}/tlcc_tlc_k60_net_lead.csv')
nm = {}
for _, rw in nl.iterrows():
    nm[rw['a']] = rw['a_name']; nm[rw['b']] = rw['b_name']
ov = {}
for _, rw in nl.iterrows():
    ov[frozenset((rw['a'], rw['b']))] = int(rw['overlap_days'])

# candidate unordered pairs
pairs = []
for i in range(m):
    for j in range(i + 1, m):
        L = ov.get(frozenset((labels[i], labels[j])), 0)
        if L < MIN_OVERLAP:
            continue
        a = dv20[i, j] if not np.isnan(dv20[i, j]) else 0.0
        b = raw60[i, j] if not np.isnan(raw60[i, j]) else 0.0
        if abs(a) >= 1e-9 or abs(b) >= 1e-9:
            pairs.append((labels[i], labels[j], L, dv20[i, j], raw60[i, j], k20[i, j]))
# keep: top-600 by |dv20| OR raw60>=0.4
pairs = sorted(pairs, key=lambda p: -abs(p[3]))
top20 = pairs[:600]
rest = [p for p in pairs if not np.isnan(p[4]) and p[4] >= 0.4 and p not in top20]
pairs = top20 + rest
log(f'candidates: {len(pairs)} unordered pairs '
    f'({len(top20)} by |d20| top, {len(rest)} by |d_raw|>=0.4)')

# ---------------- load panel ----------------
log('loading close panel from trader_nfq_aidx.hdf ...')
df = pd.read_hdf(SRC, 'nfq')
wide = df['close'].unstack(['exchange', 'code']).sort_index()
Xcols = list(wide.columns)
labcol = [f'{ex}/{cd}' for (ex, cd) in Xcols]
colidx = {l: i for i, l in enumerate(labcol)}
X = wide.to_numpy(dtype=np.float64)                 # n x m close levels
dates = wide.index
n = X.shape[0]
# log-returns; NaN where a gap/pre-boundary exists
with np.errstate(divide='ignore', invalid='ignore'):
    R = np.full_like(X, np.nan)
    R[1:, :] = np.log(X[1:, :]) - np.log(X[:-1, :])
log(f'panel {n} dates x {m} cols loaded')

# ---------------- TLC helpers (on aligned arrays) ----------------
def d_curve_max(a, b, K):
    """net-lead d(k)=r_ab(k)-r_ba(k) for k=1..K; return (kstar, dstar, full d array).

    r_ab(k) = pearson(a[0:w], b[k:L]),  r_ba(k) = pearson(b[0:w], a[k:L]),
    w = L-k, computed from prefix sums of a and b.
    """
    L = len(a)
    Kk = min(K, L - 2)
    pa = np.concatenate([[0.0], np.cumsum(a)]); pb = np.concatenate([[0.0], np.cumsum(b)])
    pa2 = np.concatenate([[0.0], np.cumsum(a * a)]); pb2 = np.concatenate([[0.0], np.cumsum(b * b)])
    out = np.full(Kk + 1, np.nan)
    for k in range(1, Kk + 1):
        w = L - k
        # r_ab(k): x = a[0:w], y = b[k:L]
        sx = pa[w]; sy = pb[L] - pb[k]
        sx2 = pa2[w]; sy2 = pb2[L] - pb2[k]
        dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
        if dx <= 0 or dy <= 0:
            continue
        rab = (np.dot(a[:w], b[k:]) - sx * sy / w) / np.sqrt(dx * dy)
        # r_ba(k): x = b[0:w], y = a[k:L]
        sx = pb[w]; sy = pa[L] - pa[k]
        sx2 = pb2[w]; sy2 = pa2[L] - pa2[k]
        dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
        if dx <= 0 or dy <= 0:
            continue
        rba = (np.dot(b[:w], a[k:]) - sx * sy / w) / np.sqrt(dx * dy)
        out[k] = rab - rba
    if np.isnan(out[1:]).all():
        return 0, np.nan, out
    kk = int(np.nanargmax(np.abs(out[1:]))) + 1
    return kk, float(out[kk]), out

# ---------------- per-pair analysis ----------------
def aligned_returns(li, lj):
    i, j = colidx[li], colidx[lj]
    ri = R[:, i]; rj = R[:, j]
    msk = ~(np.isnan(ri) | np.isnan(rj))
    return ri[msk], rj[msk]

rows = []
def bs_maxd_null(a, b, K, B, mean_blk, rng):
    """independent stationary-block-bootstrap null of max_{k<=K}|d(k)|."""
    L = len(a)
    maxes = np.empty(B)
    # draw block lengths: geometric with mean mean_blk, via rng (deterministic)
    p_geom = 1.0 / mean_blk
    for rep in range(B):
        ia = np.empty(L, dtype=np.int64); ib = np.empty(L, dtype=np.int64)
        pa = pb = 0
        while pa < L:
            U = rng.random()
            bl = int(np.floor(np.log(1.0 - U) / np.log(1.0 - p_geom))) + 1
            bl = min(bl, L - pa)
            st = rng.integers(0, L)
            ia[pa:pa + bl] = (st + np.arange(bl)) % L
            pa += bl
        while pb < L:
            U = rng.random()
            bl = int(np.floor(np.log(1.0 - U) / np.log(1.0 - p_geom))) + 1
            bl = min(bl, L - pb)
            st = rng.integers(0, L)
            ib[pb:pb + bl] = (st + np.arange(bl)) % L
            pb += bl
        aa = a[ia[:L]]; bb = b[ib[:L]]
        _, _, dd = d_curve_max(aa, bb, K)
        if np.isnan(dd[1:]).all():
            maxes[rep] = 0.0
        else:
            maxes[rep] = np.nanmax(np.abs(dd[1:]))
    return maxes

rng = np.random.default_rng(20240904)
boot_candidates = []
for idx, (li, lj, L0, dv20v, raw60v, k20v) in enumerate(pairs):
    a, b = aligned_returns(li, lj)
    if len(a) < 61:
        continue
    La = len(a)
    k_ret, d_ret, darr = d_curve_max(a, b, KSIG)          # k=1..20 headline
    k_ret60, d_ret60, _ = d_curve_max(a, b, KMAX)
    r0r = float(np.corrcoef(a, b)[0, 1])
    # halves stability (returns, k=1..20)
    h = La // 2
    k1, d1, _ = d_curve_max(a[:h], b[:h], KSIG)
    k2, d2, _ = d_curve_max(a[h:], b[h:], KSIG)
    sign_ok = (not np.isnan(d1)) and (not np.isnan(d2)) and (d1 * d2 > 0)
    kclose = abs(k1 - k2) <= 5
    stable = sign_ok and kclose and abs(d1) >= 0.03 and abs(d2) >= 0.03
    # raw-close d at same k window from the stored cube (k<=20 argmax & k<=60)
    i, j = lab2i[li], lab2i[lj]
    rec = {
        'a': li, 'b': lj, 'a_name': nm[li], 'b_name': nm[lj],
        'overlap_days': L0,
        'r0_raw_close': float(r[i, j, 0]),
        'k_raw20': int(k20v) if not np.isnan(k20v) else np.nan,
        'd_raw20': (float(dv20v) if not np.isnan(dv20v) else np.nan),
        'd_raw60max': (float(raw60v) if not np.isnan(raw60v) else np.nan),
        'r0_ret': r0r,
        'k_ret_sig': k_ret, 'd_ret_sig': d_ret,
        'k_ret60': k_ret60, 'd_ret60': d_ret60,
        'h1_k': k1, 'h1_d': d1, 'h2_k': k2, 'h2_d': d2,
        'half_stable': bool(stable), 'half_sign_consistent': bool(sign_ok),
    }
    rows.append(rec)
    if (idx + 1) % 200 == 0:
        log(f'  analyzed {idx+1}/{len(pairs)} pairs')
    if stable and abs(d_ret) >= 0.10:
        boot_candidates.append(rec)
    if len(boot_candidates) >= 250:
        log('  boot-candidate cap 250 reached; stopping the scan early')
        break

res = pd.DataFrame(rows)
log(f'analyzed {len(res)} pairs; returns-screen survivors (stable halves, '
    f'|d_ret(k<=20)|>=0.10): {len(boot_candidates)}')

# ---------------- bootstrap on survivors ----------------
surv = pd.DataFrame(boot_candidates)
pvals = []
for _, rec in surv.iterrows():
    a, b = aligned_returns(rec['a'], rec['b'])
    maxes = bs_maxd_null(a, b, KSIG, B, BLOCK_MEAN, rng)
    p = float((np.sum(maxes >= abs(rec['d_ret_sig'])) + 1) / (B + 1))
    pvals.append(p)
surv['p_bs'] = pvals
surv['bs_pass'] = surv['p_bs'] <= 0.05
res = res.merge(surv[['a', 'b', 'p_bs', 'bs_pass']], on=['a', 'b'], how='left')
res['p_bs'] = res['p_bs'].astype(float)
res['bs_pass'] = res['bs_pass'].fillna(False).astype(bool)

# ---------------- verdicts & output ----------------
def verdict(row):
    if pd.isna(row['d_ret_sig']):
        return 'no_valid_returns_overlap'
    bits = []
    if row['half_stable']:
        bits.append('half-stable')
    if row['bs_pass']:
        bits.append('bootstrap<0.05')
    if row['half_stable'] and row['bs_pass']:
        return 'VALIDATED:' + ','.join(bits)
    if not row['half_sign_consistent']:
        return 'rejected:sign_flips_between_halves'
    return 'rejected:' + (','.join(bits) if bits else 'not_stable_not_significant')

res['verdict'] = res.apply(verdict, axis=1)
res = res.sort_values('d_ret_sig', key=lambda s: s.abs(), ascending=False,
                      na_position='last')
outp = f'{OUTDIR}/tlcc_tlc_k60_validated.csv'
res.to_csv(outp, index=False)
log(f'wrote {outp}  ({len(res)} rows)')

valid = res[res['verdict'].str.startswith('VALIDATED')]
log(f'\n=== VALIDATED (half-stable AND bootstrap p<=0.05): {len(valid)} pairs ===')
cols = ['a', 'a_name', 'b', 'b_name', 'overlap_days', 'r0_raw_close',
        'd_raw20', 'd_raw60max', 'r0_ret', 'k_ret_sig', 'd_ret_sig',
        'h1_k', 'h1_d', 'h2_k', 'h2_d', 'p_bs']
if len(valid):
    log(valid[cols].head(40).to_string(index=False))
else:
    log('none')

log('\n=== breakdown ===')
log(res['verdict'].value_counts().to_string())
if len(res):
    corr_ok = res.dropna(subset=['d_raw20', 'd_ret_sig'])
    if len(corr_ok):
        from scipy import stats as _st
        try:
            rho, pv = _st.spearmanr(corr_ok['d_raw20'], corr_ok['d_ret_sig'])
            log(f'spearman(raw d20, ret d_sig) over {len(corr_ok)} pairs: rho={rho:.3f} p={pv:.2e}')
        except Exception as e:
            log(f'spearman skipped: {e}')
    rawbig = res.dropna(subset=['d_raw60max'])
    rawbig = rawbig[rawbig['d_raw60max'] >= 0.4]
    if len(rawbig):
        persist = rawbig[rawbig['verdict'].str.startswith('VALIDATED') |
                         (rawbig['half_stable'] & (rawbig['d_ret_sig'].abs() >= 0.1))]
        log(f'pairs with |d_raw|max(k<=60)>=0.4: {len(rawbig)}; of them '
            f'returns-side strong/stable: {len(persist)}  -> raw edge-60 leads '
            f'mostly do NOT survive detrending' if len(persist) < 0.3 * len(rawbig)
            else f'pairs with |d_raw|max>=0.4: {len(rawbig)}; strong on returns too: {len(persist)}')
log('done')
