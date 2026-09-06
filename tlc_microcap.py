# -*- coding: utf-8 -*-
"""Dedicated hub analysis: sse/880823 微盘股.
For every other index X: d(k) = r(mic, X, k) - r(X, mic, k), k=1..K.
  d>0 at k*  -> 微盘股 net-leads X by k* days (its move anticipates X).
  d<0 at k*  -> X net-leads 微盘股.
Report on log-returns (main) + raw close (reference), with half-split sign
stability of d at k* and short-horizon (k<=20) argmax.
"""
import numpy as np
import pandas as pd
import time

OUTDIR = '/workspace/tlcc'
MIC = 'sse/880823'
K = 60
MIN_OV = 250
t0 = time.time()
def log(s):
    print(f'[{time.time()-t0:5.0f}s] {s}', flush=True)

# ---- raw cube ----
zr0 = np.load(f'{OUTDIR}/tlcc_tlc_k60_cube.npz', allow_pickle=True)
rraw = zr0['r']; lab = list(zr0['labels'])
DRAW = rraw[:, :, 1:] - rraw.transpose(1, 0, 2)[:, :, 1:]   # d_raw(k), k=1..60

# ---- returns cube ----
z = np.load(f'{OUTDIR}/tlcc_tlc_ret_k60_cube.npz', allow_pickle=True)
r = z['r']; labr = list(z['labels'])
DR = r[:, :, 1:] - r.transpose(1, 0, 2)[:, :, 1:]

# names / overlap
nl = pd.read_csv(f'{OUTDIR}/tlcc_tlc_ret_k60_net_lead.csv')
nm = {}
for _, rw in nl.iterrows():
    nm[rw['a']] = rw['a_name']; nm[rw['b']] = rw['b_name']
ovr = {}
for _, rw in nl.iterrows():
    ovr[frozenset((rw['a'], rw['b']))] = int(rw['overlap_days'])

mi = labr.index(MIC)
mi_name = nm[MIC]

# ---- load returns panel for half-split ----
log('loading panel ...')
df = pd.read_hdf(f'{OUTDIR}/trader_nfq_aidx.hdf', 'nfq')
wide = df['close'].unstack(['exchange', 'code']).sort_index()
Xcols = list(wide.columns)
colidx = {f'{ex}/{cd}': i for i, (ex, cd) in enumerate(Xcols)}
X = wide.to_numpy(dtype=np.float64)
with np.errstate(divide='ignore', invalid='ignore'):
    R = np.full_like(X, np.nan)
    R[1:, :] = np.log(X[1:, :]) - np.log(X[:-1, :])
del df, wide, X
mc = R[:, mi]

rows = []
for j in range(len(labr)):
    if j == mi:
        continue
    xj = labr[j]
    L = ovr.get(frozenset((MIC, xj)), 0)
    if L < MIN_OV:
        continue
    d = DR[mi, j, :]                 # k=1..60 (returns)
    if np.isnan(d).all():
        continue
    kstar = int(np.nanargmax(np.abs(d))) + 1
    dstar = float(d[kstar - 1])
    # short-horizon argmax k<=20
    d20 = d[:20]
    if np.isnan(d20).all():
        k20 = np.nan; d20v = np.nan
    else:
        k20 = int(np.nanargmax(np.abs(d20))) + 1
        d20v = float(d20[k20 - 1])
    r0r = float(r[mi, j, 0])
    # raw-close d at same k* and at its own argmax
    draw = DRAW[mi, j, :]
    kraw = int(np.nanargmax(np.abs(draw))) + 1 if not np.isnan(draw).all() else np.nan
    draw_v = float(draw[kstar - 1]) if kstar <= 60 else np.nan
    # half-split stability of d at k* (returns)
    xjarr = R[:, j]
    msk = ~(np.isnan(mc) | np.isnan(xjarr))
    a = mc[msk]; b = xjarr[msk]
    la = len(a)
    stable = np.nan; h1v = np.nan; h2v = np.nan
    if la >= 2 * (kstar + 30):
        h = la // 2
        def dd_at(aa, bb, k):
            L2 = len(aa); w = L2 - k
            if w < 30:
                return np.nan
            pa = np.concatenate([[0.0], np.cumsum(aa)]); pb = np.concatenate([[0.0], np.cumsum(bb)])
            pa2 = np.concatenate([[0.0], np.cumsum(aa * aa)]); pb2 = np.concatenate([[0.0], np.cumsum(bb * bb)])
            rab = rba = np.nan
            sx = pa[w]; sy = pb[L2] - pb[k]; sx2 = pa2[w]; sy2 = pb2[L2] - pb2[k]
            dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
            if dx > 0 and dy > 0:
                rab = (np.dot(aa[:w], bb[k:]) - sx * sy / w) / np.sqrt(dx * dy)
            sx = pb[w]; sy = pa[L2] - pa[k]; sx2 = pb2[w]; sy2 = pa2[L2] - pa2[k]
            dx = sx2 - sx * sx / w; dy = sy2 - sy * sy / w
            if dx > 0 and dy > 0:
                rba = (np.dot(bb[:w], aa[k:]) - sx * sy / w) / np.sqrt(dx * dy)
            return rab - rba
        h1v = dd_at(a[:h], b[:h], kstar)
        h2v = dd_at(a[h:], b[h:], kstar)
        if not (np.isnan(h1v) or np.isnan(h2v)):
            stable = bool(np.sign(h1v) == np.sign(dstar) and np.sign(h2v) == np.sign(dstar))
    direction = 'mic_leads_X' if dstar > 0 else 'X_leads_mic'
    rows.append({
        'x': xj, 'x_name': nm[xj], 'overlap_days': L,
        'dir': direction, 'k*': kstar, 'd(k*)': dstar, 'r0_ret': r0r,
        'k20': k20, 'd(k20)': d20v, 'half_stable': stable,
        'raw_d@k*': draw_v, 'raw_k*': kraw,
    })

res = pd.DataFrame(rows).sort_values('d(k*)', key=lambda s: s.abs(), ascending=False)
outp = f'{OUTDIR}/microcap_880823_hub.csv'
res.to_csv(outp, index=False)
log(f'analyzed {len(res)} partners (overlap>={MIN_OV}); saved {outp}')

for side, title in [('mic_leads_X', f'{mi_name} 领先其它 (d>0)'),
                    ('X_leads_mic', f'其它领先 {mi_name} (d<0)')]:
    sub = res[res['dir'] == side]
    print(f'\n=== {title}: {len(sub)} 对 ===', flush=True)
    sub2 = sub.sort_values('d(k*)', key=lambda s: s.abs(), ascending=False)
    print('top 18 by |d|:')
    for _, x in sub2.head(18).iterrows():
        st = '稳定' if x['half_stable'] is True else ('不稳' if x['half_stable'] is False else '—')
        print(f"  {x['x_name']}({x['x'].split('/')[1]})  k*={int(x['k*']):2d} d={x['d(k*)']:+.4f} "
              f"k20={x['k20'] if pd.isna(x['k20']) else int(x['k20'])} d20={x['d(k20)']:+.4f} "
              f"r0={x['r0_ret']:+.3f} 半样本{st} ov={int(x['overlap_days'])}d")
    if len(sub) > 18:
        print(f'  ... 其余 {len(sub)-18} 对见 CSV')

print('\n=== 短窗(k*<=20, returns) 里最强的方向对 ===', flush=True)
short = res[res['k20'].notna()].copy()
short['d20abs'] = short['d(k20)'].abs()
for side, title in [('mic_leads_X', f'{mi_name} 短窗领先 (d20>0)'),
                    ('X_leads_mic', f'短窗其它领先 (d20<0)')]:
    s2 = short[short['dir'] == side].sort_values('d20abs', ascending=False).head(10)
    print(f'--- {title} ---', flush=True)
    for _, x in s2.iterrows():
        st = '稳定' if x['half_stable'] is True else ('不稳' if x['half_stable'] is False else '—')
        print(f"  {x['x_name']}({x['x'].split('/')[1]})  k20={int(x['k20'])} d20={x['d(k20)']:+.4f} 半样本{st} ov={int(x['overlap_days'])}d")

# macro summary
stab = res['half_stable'].dropna()
print(f"\n=== 概览: 共{len(res)}对 | 方向: mic领先{int((res['dir']=='mic_leads_X').sum())}, "
      f"X领先mic{int((res['dir']=='X_leads_mic').sum())} | 半样本可用{len(stab)}对中稳定{int((stab).sum())} "
      f"({(stab).mean():.0%})", flush=True)
log('done')
