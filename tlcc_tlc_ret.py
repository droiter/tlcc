# -*- coding: utf-8 -*-
"""Time-lagged cross-correlation (TLC) among the 197 indices in trader_nfq_aidx.hdf,
computed on daily LOG-RETURNS instead of raw closes.

Motivation (see tlc_k60_validate.py + TLCC_K60_VALIDATION.md): on raw close
levels the net-lead statistic d(k)=r_ab(k)-r_ba(k) grows ~linearly with lag and
its argmax is pinned at the window edge for ~99% of high-|d| pairs -> regime
drift of non-stationary levels, not genuine lead-lag.  Log-returns remove that.

Convention: r[i, j, k] = Pearson corr( ret_i[t], ret_j[t+k] ) over overlapping
days, k in 0..K  ->  i LEADS j by k days when k>0.
Outputs: {OUTP}_cube.npz / _bestlag_k.csv / _bestlag_r.csv /
         _lead_ordered.csv / _net_lead.csv   (OUTP = tlcc_tlc_ret_k{K})
"""
import sys
import numpy as np
import pandas as pd

SRC = '/workspace/tlcc/trader_nfq_aidx.hdf'
OUTDIR = '/workspace/tlcc'
K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUTP = f'{OUTDIR}/tlcc_tlc_ret_k{K}'

names = {}
try:
    for line in open(OUTDIR + '/indices_report.tsv'):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 3 and p[2] != 'name':
            names[(p[0], p[1])] = p[2]
except FileNotFoundError:
    pass

print('loading...', flush=True)
df = pd.read_hdf(SRC, 'nfq')
wide = df['close'].unstack(['exchange', 'code']).sort_index()
Xcols = list(wide.columns)
m = len(Xcols)
X = wide.to_numpy(dtype=np.float64)          # close levels, n x m
dates = wide.index
with np.errstate(divide='ignore', invalid='ignore'):
    R = np.full_like(X, np.nan)
    R[1:, :] = np.log(X[1:, :]) - np.log(X[:-1, :])   # daily log returns
X = R
del df, wide

labels = [f'{ex}/{cd}' for (ex, cd) in Xcols]
disp = []
for (ex, cd) in Xcols:
    nm = names.get((ex, cd), '')
    disp.append(f'{ex}/{cd} ' + (nm if nm else cd))
print(f'panel {X.shape[0]} dates x {m} indices (log returns)', flush=True)

def tlc_series(x, y, K):
    """x,y same length, no NaN. returns r[0..K] with r[k]=pearson(x[:L-k], y[k:])."""
    L = len(x)
    out = np.full(K + 1, np.nan)
    kk = min(K, L - 1)
    if kk < 0:
        return out
    px = np.concatenate([[0.0], np.cumsum(x)])
    py = np.concatenate([[0.0], np.cumsum(y)])
    pxx = np.concatenate([[0.0], np.cumsum(x * x)])
    pyy = np.concatenate([[0.0], np.cumsum(y * y)])
    for k in range(kk + 1):
        w = L - k
        sx = px[w]
        sy = py[L] - py[k]
        sx2 = pxx[w]
        sy2 = pyy[L] - pyy[k]
        dot = np.dot(x[:w], y[k:])
        num = dot - sx * sy / w
        dx = sx2 - sx * sx / w
        dy = sy2 - sy * sy / w
        if dx > 0 and dy > 0:
            out[k] = num / np.sqrt(dx * dy)
    return out

rcube = np.full((m, m, K + 1), np.nan)
Lmat = np.zeros((m, m), dtype=int)
print('computing pairwise TLC on log returns...', flush=True)
import time
t0 = time.time()
for i in range(m):
    xi = X[:, i]
    for j in range(m):
        if i == j:
            rcube[i, j, 0] = 1.0
            Lmat[i, j] = int(np.count_nonzero(~np.isnan(xi)))
            continue
        xj = X[:, j]
        msk = ~(np.isnan(xi) | np.isnan(xj))
        if msk.sum() < 2:
            continue
        a = xi[msk]
        b = xj[msk]
        Lmat[i, j] = len(a)
        rcube[i, j] = tlc_series(a, b, K)
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{m}  t={time.time()-t0:.0f}s', flush=True)
print(f'done in {time.time()-t0:.0f}s', flush=True)

np.savez_compressed(f'{OUTP}_cube.npz',
                    r=rcube, labels=np.array(labels), lags=np.arange(K + 1))

best_k = np.nanargmax(rcube, axis=2)
best_r = np.take_along_axis(rcube, best_k[:, :, None], axis=2)[:, :, 0]

def write_csv(fn, mat):
    hdr = 'leader\\follower,' + ','.join(labels)
    with open(fn, 'w') as fh:
        fh.write(hdr + '\n')
        for i in range(m):
            fh.write(labels[i] + ',' + ','.join(f'{v:.6g}' for v in mat[i]) + '\n')

write_csv(f'{OUTP}_bestlag_k.csv', best_k)
write_csv(f'{OUTP}_bestlag_r.csv', best_r)

ord_rows = []
for i in range(m):
    for j in range(m):
        if i == j:
            continue
        r = rcube[i, j]
        kbest = int(np.nanargmax(r[1:])) + 1
        rbest = r[kbest]
        ord_rows.append((labels[i], labels[j], disp[i], disp[j],
                         int(kbest), float(rbest), float(r[0]), float(rbest - r[0]),
                         int(Lmat[i, j])))
ord_df = pd.DataFrame(ord_rows, columns=['leader', 'follower', 'leader_name', 'follower_name',
                                         f'best_k(1..{K})', 'r_best', 'r0', 'delta_r', 'overlap_days'])
ord_df = ord_df.sort_values('r_best', ascending=False)
ord_df.to_csv(f'{OUTP}_lead_ordered.csv', index=False)

net_rows = []
for i in range(m):
    for j in range(i + 1, m):
        d = rcube[i, j, 1:] - rcube[j, i, 1:]
        kd = int(np.nanargmax(d)) + 1
        dval = float(d[kd - 1])
        net_rows.append((labels[i], labels[j], disp[i], disp[j], kd, dval,
                         float(rcube[i, j, 0]), int(Lmat[i, j])))
net_df = pd.DataFrame(net_rows, columns=['a', 'b', 'a_name', 'b_name',
                                         'k_net', 'net_d(r_ab-r_ba)', 'r0', 'overlap_days'])
net_df['abs_net'] = net_df['net_d(r_ab-r_ba)'].abs()
net_df = net_df.sort_values('abs_net', ascending=False)
net_df.to_csv(f'{OUTP}_net_lead.csv', index=False)

# ---- diagnostics: is the |d| ~ linear-in-k / edge-pinning artifact gone? ----
D = rcube[:, :, 1:] - rcube.transpose(1, 0, 2)[:, :, 1:]
mbk = np.full(K, np.nan)
for k in range(K):
    ii, jj = np.triu_indices(m, 1)
    a = D[:, :, k][ii, jj]; b = D[:, :, k][jj, ii]
    both = ~(np.isnan(a) | np.isnan(b))
    mbk[k] = np.abs(a[both]).mean()
print('\nmean |d(k)| on log returns, k=1..%d:' % K, flush=True)
for k in range(0, K, 10):
    seg = mbk[k:k + 10]
    print('  k=%2d..%-2d: %s' % (k + 1, min(k + 10, K),
          ' '.join('nan' if np.isnan(v) else f'{v:.4f}' for v in seg)), flush=True)
print(f'  slope: mean|d(1)={mbk[0]:.4f}  mean|d({K})={mbk[K-1]:.4f}', flush=True)

print('\n=== summary (log-return TLC) ===', flush=True)
r0 = rcube[:, :, 0]
tri = np.triu(r0, 1)
tri = tri[(tri > -1) & ~np.isnan(tri)]
print(f'r0: n_pairs={len(tri)} mean={tri.mean():.4f} median={np.median(tri):.4f} '
      f'min={tri.min():.4f} max={tri.max():.4f}', flush=True)
print('\ntop 15 raw lead pairs (strict-lead r, k in 1..%d):' % K, flush=True)
print(ord_df.head(15).to_string(index=False), flush=True)
print('\ntop 20 net-lead pairs (|r_ab(k)-r_ba(k)|, k in 1..%d):' % K, flush=True)
print(net_df.head(20).to_string(index=False), flush=True)
n_nz = int((net_df['net_d(r_ab-r_ba)'].abs() > 1e-9).sum())
print(f'\npairs with |net lead|>0 among {len(net_df)} = {n_nz}', flush=True)
frac_edge = float((net_df['k_net'] == K).mean())
frac_ge_half = float((net_df['k_net'] >= K // 2).mean())
print(f'frac net-lead extremes at window edge k={K}: {frac_edge:.3f} '
      f'(raw-close k60 was ~0.99 among |d|>=0.3)', flush=True)
print(f'frac net-lead extremes at k >= {K//2}: {frac_ge_half:.3f}', flush=True)
print('files written:', flush=True)
print(f'  {OUTP}_cube.npz / {OUTP}_bestlag_k.csv / {OUTP}_bestlag_r.csv', flush=True)
print(f'  {OUTP}_lead_ordered.csv / {OUTP}_net_lead.csv', flush=True)
