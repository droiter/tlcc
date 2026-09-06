# -*- coding: utf-8 -*-
"""K=120 follow-up:
A) raw close: pairs with |d|-argmax k* in 1..60, |d|>=0.3, overlap>=500.
B) log-returns: the long-horizon strong pairs (|d|>=0.3, k* in 61..120 or any
   k*<120 not at edge) -> half-split + rolling-thirds stability of the net-lead
   asymmetry d(k)=r_ab(k)-r_ba(k).
"""
import numpy as np
import pandas as pd
import time

OUTDIR = '/workspace/tlcc'
t0 = time.time()
def log(s):
    print(f'[{time.time()-t0:5.0f}s] {s}', flush=True)

def names_overlap(tag):
    nl = pd.read_csv(f'{OUTDIR}/{tag}_net_lead.csv')
    ov, nm = {}, {}
    for _, rw in nl.iterrows():
        ov[frozenset((rw['a'], rw['b']))] = int(rw['overlap_days'])
        nm[rw['a']] = rw['a_name']; nm[rw['b']] = rw['b_name']
    return ov, nm

def d_argmax_rows(tag, K):
    z = np.load(f'{OUTDIR}/{tag}_cube.npz', allow_pickle=True)
    r = z['r']; lab = list(z['labels']); m = r.shape[0]
    D = r[:, :, 1:] - r.transpose(1, 0, 2)[:, :, 1:]
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            d = D[i, j, :]
            if np.isnan(d).all():
                continue
            k = int(np.nanargmax(np.abs(d))) + 1
            out.append((lab[i], lab[j], k, float(d[k - 1])))
    return out

# ---------------- A: raw short-horizon non-edge ----------------
ov, nm = names_overlap('tlcc_tlc_k120')
rowsA = []
for a, b, k, dv in d_argmax_rows('tlcc_tlc_k120', 120):
    L = ov.get(frozenset((a, b)), 0)
    if L >= 500 and k <= 60 and abs(dv) >= 0.3:
        rowsA.append({'a': a, 'b': b, 'k*': k, 'd': dv, 'overlap_days': L})
A = pd.DataFrame(rowsA).sort_values('d', key=lambda s: s.abs(), ascending=False)
A['a_name'] = A['a'].map(nm); A['b_name'] = A['b'].map(nm)
A.to_csv(f'{OUTDIR}/tlcc_tlc_k120_short_k60.csv', index=False)
log(f'A) raw k120 short-horizon pairs (k*<=60, |d|>=0.3, ov>=500): {len(A)}')
for _, x in A.iterrows():
    sgn = 'a领先b' if x['d'] > 0 else 'b领先a'
    print(f"  {x['a_name']} -> {x['b_name']}   k*={x['k*']:2d}  d={x['d']:+.4f} ({sgn})  overlap={int(x['overlap_days'])}d")
print('saved tlcc_tlc_k120_short_k60.csv', flush=True)

# ---------------- B: ret long-horizon stability ----------------
ovr, nmr = names_overlap('tlcc_tlc_ret_k120')
rowsB = []
for a, b, k, dv in d_argmax_rows('tlcc_tlc_ret_k120', 120):
    L = ovr.get(frozenset((a, b)), 0)
    if L >= 500 and abs(dv) >= 0.3:
        rowsB.append({'a': a, 'b': b, 'k*': k, 'd': dv, 'overlap_days': L})
B = pd.DataFrame(rowsB)
log(f'B) ret k120 strong pairs (|d|>=0.3, ov>=500): {len(B)}')

# reload panel for return series
log('loading close panel ...')
df = pd.read_hdf(f'{OUTDIR}/trader_nfq_aidx.hdf', 'nfq')
wide = df['close'].unstack(['exchange', 'code']).sort_index()
Xcols = list(wide.columns)
colidx = {f'{ex}/{cd}': i for i, (ex, cd) in enumerate(Xcols)}
X = wide.to_numpy(dtype=np.float64)
with np.errstate(divide='ignore', invalid='ignore'):
    R = np.full_like(X, np.nan)
    R[1:, :] = np.log(X[1:, :]) - np.log(X[:-1, :])
del df, wide, X
log('panel ready')

def d_curve(a, b, K):
    L = len(a); Kk = min(K, L - 2)
    pa = np.concatenate([[0.0], np.cumsum(a)]); pb = np.concatenate([[0.0], np.cumsum(b)])
    pa2 = np.concatenate([[0.0], np.cumsum(a * a)]); pb2 = np.concatenate([[0.0], np.cumsum(b * b)])
    out = np.full(Kk + 1, np.nan)
    for k in range(1, Kk + 1):
        w = L - k
        sx = pa[w]; sy = pb[L] - pb[k]; sx2 = pa2[w]; sy2 = pb2[L] - pb2[k]
        dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
        if dx <= 0 or dy <= 0: continue
        rab = (np.dot(a[:w], b[k:]) - sx * sy / w) / np.sqrt(dx * dy)
        sx = pb[w]; sy = pa[L] - pa[k]; sx2 = pb2[w]; sy2 = pa2[L] - pa2[k]
        dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
        if dx <= 0 or dy <= 0: continue
        rba = (np.dot(b[:w], a[k:]) - sx * sy / w) / np.sqrt(dx * dy)
        out[k] = rab - rba
    return out

def kmax(d):
    if np.isnan(d[1:]).all(): return 0, np.nan
    k = int(np.nanargmax(np.abs(d[1:]))) + 1
    return k, float(d[k])

def aligned(a, b):
    i, j = colidx[a], colidx[b]
    ri, rj = R[:, i], R[:, j]
    msk = ~(np.isnan(ri) | np.isnan(rj))
    return ri[msk], rj[msk]

K = 120
res = []
for _, x in B.iterrows():
    a, b = aligned(x['a'], x['b'])
    L = len(a)
    # halves
    h = L // 2
    d1 = d_curve(a[:h], b[:h], K); k1, v1 = kmax(d1)
    d2 = d_curve(a[h:], b[h:], K); k2, v2 = kmax(d2)
    # thirds
    t = L // 3
    segs = [(a[:t], b[:t]), (a[t:2 * t], b[t:2 * t]), (a[2 * t:], b[2 * t:])]
    kt = []; vt = []
    for (aa, bb) in segs:
        d = d_curve(aa, bb, K); kk, vv = kmax(d)
        kt.append(kk); vt.append(vv)
    full_k = int(x['k*']); full_d = float(x['d']); sign_full = np.sign(full_d)
    def ok_sign(v): return (not np.isnan(v)) and np.sign(v) == sign_full
    halves_sign = ok_sign(v1) and ok_sign(v2)
    thirds_sign = sum(ok_sign(v) for v in vt)
    # k proximity: halves' argmax within 60 of full k*, thirds within 80 (long-horizon curves are broad)
    kh_ok = (abs(k1 - full_k) <= 60) and (abs(k2 - full_k) <= 60)
    kt_ok = sum(abs(k - full_k) <= 80 for k in kt)
    mag_ok = (abs(v1) >= 0.15 and abs(v2) >= 0.15) and all(abs(v) >= 0.15 for v in vt)
    stable = halves_sign and thirds_sign >= 2 and mag_ok and kt_ok >= 2 and kh_ok
    res.append({
        'a': x['a'], 'b': x['b'], 'overlap_days': L, 'k*': full_k, 'd': full_d,
        'h1_k': k1, 'h1_d': v1, 'h2_k': k2, 'h2_d': v2,
        't1_k': kt[0], 't1_d': vt[0], 't2_k': kt[1], 't2_d': vt[1],
        't3_k': kt[2], 't3_d': vt[2],
        'halves_sign_ok': halves_sign, 'thirds_sign_n': thirds_sign,
        'stable': stable,
    })
res = pd.DataFrame(res)
res['a_name'] = res['a'].map(nmr); res['b_name'] = res['b'].map(nmr)
res.to_csv(f'{OUTDIR}/tlcc_tlc_ret_k120_longhorizon_stability.csv', index=False)
log(f'B) analyzed {len(res)}; STABLE: {int(res["stable"].sum())} / {len(res)}')
print('\n=== STABLE pairs (half + rolling-thirds consistent) ===', flush=True)
for _, x in res[res['stable']].iterrows():
    print(f"  {x['a_name']} -> {x['b_name']}  k*={x['k*']:3d} d={x['d']:+.3f} | halves {x['h1_k']:3d}({x['h1_d']:+.2f})/{x['h2_k']:3d}({x['h2_d']:+.2f}) | thirds {x['t1_k']:3d}({x['t1_d']:+.2f})/{x['t2_k']:3d}({x['t2_d']:+.2f})/{x['t3_k']:3d}({x['t3_d']:+.2f}) | ov={int(x['overlap_days'])}d")
print('\n=== rejected (sample, first 15) ===', flush=True)
for _, x in res[~res['stable']].head(15).iterrows():
    why = []
    if not x['halves_sign_ok']: why.append('halves方向不一致')
    if x['thirds_sign_n'] < 2: why.append('thirds方向<2')
    if not (abs(x['h1_d']) >= 0.15 and abs(x['h2_d']) >= 0.15): why.append('halves幅度小')
    if not all(abs(v) >= 0.15 for v in [x['t1_d'], x['t2_d'], x['t3_d']]): why.append('thirds幅度小')
    print(f"  {x['a_name']} -> {x['b_name']}  k*={x['k*']:3d} d={x['d']:+.3f} | h {x['h1_k']:3d}({x['h1_d']:+.2f})/{x['h2_k']:3d}({x['h2_d']:+.2f}) | t {x['t1_k']:3d}({x['t1_d']:+.2f})/{x['t2_k']:3d}({x['t2_d']:+.2f})/{x['t3_k']:3d}({x['t3_d']:+.2f}) | {';'.join(why)}")
print('saved tlcc_tlc_ret_k120_longhorizon_stability.csv', flush=True)
log('done')
