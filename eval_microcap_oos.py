# -*- coding: utf-8 -*-
"""Rolling out-of-sample forward test: does X(t) (big-cap/institutional style
indices) predict micro-cap 880823's forward h-day log return?

Candidates: from microcap_880823_hub.csv, X-leads-mic short-window rows with
sign(d(k20)) == sign(d(k*)) == negative, |d20| >= 0.08, half_stable.
Rule at day t (no look-ahead):
  - direction dir = sign(corr(X_s, mic_{s+h})) estimated on the trailing window
    [t-W, t-h]  (past only);
  - signal z = standardized X_t over the same trailing window;
  - prediction p = sign(z * dir) for mic's forward h-day log return fwd.
Metrics: OOS n, IC(z,fwd), hit rate, base rate P(fwd>0), hit-base, mean fwd
given p=+1 vs p=-1.  Also an equal-weight consensus over all candidates.
"""
import numpy as np
import pandas as pd
import time

OUTDIR = '/workspace/tlcc'
MIC = 'sse/880823'
W = 504
t0 = time.time()
def log(s):
    print(f'[{time.time()-t0:5.0f}s] {s}', flush=True)

# ---- candidates ----
hub = pd.read_csv(f'{OUTDIR}/microcap_880823_hub.csv')
cand = hub[(hub['k20'].notna())
           & (hub['d(k20)'] < 0)                 # X leads mic at short horizon
           & (np.sign(hub['d(k20)']) == np.sign(hub['d(k*)']))   # no flip
           & (hub['half_stable'] == True)
           & (hub['d(k20)'].abs() >= 0.08)
           & (hub['k20'] <= 10)].copy()
cand['h'] = cand['k20'].astype(int)
cand = cand.sort_values('d(k20)')
log(f'candidates: {len(cand)}')
for _, x in cand.iterrows():
    print(f"  {x['x_name']}({x['x'].split('/')[1]})  h={int(x['h'])} d20={x['d(k20)']:+.4f} ov={int(x['overlap_days'])}d", flush=True)

# ---- load panel ----
df = pd.read_hdf(f'{OUTDIR}/trader_nfq_aidx.hdf', 'nfq')
wide = df['close'].unstack(['exchange', 'code']).sort_index()
Xcols = list(wide.columns)
colidx = {f'{ex}/{cd}': i for i, (ex, cd) in enumerate(Xcols)}
X = wide.to_numpy(dtype=np.float64)
with np.errstate(divide='ignore', invalid='ignore'):
    R = np.full_like(X, np.nan)
    R[1:, :] = np.log(X[1:, :]) - np.log(X[:-1, :])
dates = wide.index
del df, wide, X

def aligned_pair(a, b):
    ia, ib = colidx[a], colidx[b]
    ra, rb = R[:, ia], R[:, ib]
    msk = ~(np.isnan(ra) | np.isnan(rb))
    return ra[msk], rb[msk], dates[msk]

mi_series = aligned_pair(MIC, cand.iloc[0]['x'])[0]  # placeholder, replaced below

def run_pair(xcode, h):
    a, b, dts = aligned_pair(MIC, xcode)   # a = mic, b = X
    L = len(a)
    if L < W + h + 120:
        return None
    out = {'x': xcode, 'h': h, 'n': 0, 'IC': np.nan, 'hit': np.nan,
           'base': np.nan, 'diff': np.nan,
           'm_pos': np.nan, 'm_neg': np.nan, 'date0': None, 'date1': None}
    zs = []; fwds = []
    posz = []; negz = []
    for t in range(W, L - h):
        # trailing window [t-W, t-1] for scale; corr window [t-W, t-h]
        mX = b[t - W:t].mean(); sX = b[t - W:t].std()
        if sX <= 0:
            continue
        z = (b[t] - mX) / sX
        cw0, cw1 = t - W, t - h + 1
        if cw1 - cw0 < 60:
            continue
        cc = np.corrcoef(b[cw0:cw1], a[cw0 + h:cw1 + h])[0, 1]
        if np.isnan(cc):
            continue
        fwd = float(a[t + 1:t + h + 1].sum()) if h > 1 else float(a[t + 1])
        dirr = 1.0 if cc > 0 else -1.0
        zs.append(z * dirr)
        fwds.append(fwd)
        if z * dirr > 0:
            posz.append(fwd)
        else:
            negz.append(fwd)
    if len(fwds) < 60:
        return None
    zs = np.array(zs); fwds = np.array(fwds)
    out['n'] = len(fwds)
    out['IC'] = float(np.corrcoef(zs, fwds)[0, 1])
    pred_pos = zs > 0
    hit = float((np.sign(zs) == np.sign(fwds)).mean())
    base = float((fwds > 0).mean())
    out['hit'] = hit
    out['base'] = base
    out['diff'] = hit - base
    out['m_pos'] = float(np.mean(fwds[pred_pos])) * 1e4
    out['m_neg'] = float(np.mean(fwds[~pred_pos])) * 1e4
    out['date0'] = str(dts[W]); out['date1'] = str(dts[L - h - 1])
    return out

rows = []
for _, x in cand.iterrows():
    r = run_pair(x['x'], int(x['h']))
    if r is None:
        log(f"  skip {x['x']}: too short")
        continue
    r['x_name'] = x['x_name']
    rows.append(r)
    log(f"  done {x['x_name']}: n={r['n']} IC={r['IC']:+.3f} hit={r['hit']:.3f} "
        f"base={r['base']:.3f} diff={r['diff']:+.3f}")

res = pd.DataFrame(rows)
if len(res):
    res = res.sort_values('IC', ascending=False)
    outcsv = f'{OUTDIR}/microcap_oos_test.csv'
    res.to_csv(outcsv, index=False)
    log(f'saved {outcsv}')
    print('\n=== per-predictor rolling OOS (log-return, 504d rolling, daily re-eval) ===', flush=True)
    for _, r in res.iterrows():
        print(f"{r['x_name']:<8s}({r['x'].split('/')[1]}) h={int(r['h'])}  n={int(r['n']):5d} "
              f"IC={r['IC']:+.3f}  hit={r['hit']:.1%}  base={r['base']:.1%}  "
              f"hit-base={r['diff']:+.1%}  mean_fwd(bp): 信号+ {r['m_pos']:+.2f} / 信号- {r['m_neg']:+.2f}")

    # consensus (equal-weight mean z*dir over predictors active that day)
    print('\n=== consensus (每日等权平均, 需要>=3个有效预测) ===', flush=True)
    # recompute raw signals per pair for alignment by date
    sigs = {}
    for _, x in cand.iterrows():
        a, b, dts = aligned_pair(MIC, x['x'])
        h = int(x['h']); L = len(a)
        s = np.full(L, np.nan)
        for t in range(W, L - h):
            mX = b[t - W:t].mean(); sd = b[t - W:t].std()
            if sd <= 0: continue
            cw1 = t - h + 1
            cc = np.corrcoef(b[t - W:cw1], a[t - W + h:cw1 + h])[0, 1]
            if np.isnan(cc): continue
            s[t] = ((b[t] - mX) / sd) * (1.0 if cc > 0 else -1.0)
        sigs[x['x']] = (s, dts, h)
    # consensus daily: need count of non-nan among those valid that date
    from collections import defaultdict
    acc = defaultdict(list)
    for xcode, (s, dts, h) in sigs.items():
        for tt in range(W, len(s) - h):
            if not np.isnan(s[tt]):
                acc[str(dts[tt])].append(s[tt])
    # microcap forward 1-day log return on its OWN full span (no pair alignment)
    im = colidx[MIC]
    mm = R[:, im]
    mmask = ~np.isnan(mm)
    mind = dates[mmask]; mret = mm[mmask]
    fwd_by_date = {str(mind[t]): float(mret[t + 1]) for t in range(len(mind) - 1)}
    hh = {}
    for dstr, lst in acc.items():
        if len(lst) >= 3:
            hh[dstr] = float(np.mean(lst))
    common = sorted(set(hh) & set(fwd_by_date))
    if len(common) > 100:
        zz = np.array([hh[d] for d in common]); ff = np.array([fwd_by_date[d] for d in common])
        hit = float((np.sign(zz) == np.sign(ff)).mean())
        base = float((ff > 0).mean())
        ic = float(np.corrcoef(zz, ff)[0, 1])
        mp = ff[zz > 0].mean() * 1e4; mn = ff[zz <= 0].mean() * 1e4
        print(f'consensus n={len(common)}  IC={ic:+.3f}  hit={hit:.1%}  base={base:.1%}  '
              f'hit-base={hit-base:+.1%}  mean fwd: 信号+ {mp:+.2f}bp / 信号- {mn:+.2f}bp')
        print(f'  覆盖: {common[0]} .. {common[-1]}')
        # split halves by date
        mid = common[len(common) // 2]
        for tag, lo, hi in [('前半段', common[0], mid), ('后半段', mid, common[-1])]:
            sel = [d for d in common if lo <= d <= hi]
            if len(sel) < 100:
                continue
            z2 = np.array([hh[d] for d in sel]); f2 = np.array([fwd_by_date[d] for d in sel])
            ic2 = float(np.corrcoef(z2, f2)[0, 1])
            mp2 = f2[z2 > 0].mean() * 1e4; mn2 = f2[z2 <= 0].mean() * 1e4
            print(f'  {tag} n={len(sel)} IC={ic2:+.3f} mean fwd: 信号+ {mp2:+.2f}bp / 信号- {mn2:+.2f}bp')
    else:
        print(f'consensus n={len(common)} 样本不足')
log('done')
