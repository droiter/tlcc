import numpy as np
import pandas as pd
import hdf5plugin  # noqa

SRC = '/workspace/tlcc/trader_nfq.hdf'
OUT = '/workspace/tlcc/trader_nfq_aidx.hdf'

keep = []
for fp in ['/workspace/tlcc/idx_sse.txt', '/workspace/tlcc/idx_szse.txt', '/workspace/tlcc/idx_88.txt']:
    for line in open(fp):
        line = line.rstrip('\n')
        if line:
            p = line.split('\t')
            keep.append((p[1], p[0]))  # (exchange, code)

rng = np.random.default_rng(123)
sample = [keep[i] for i in rng.choice(len(keep), size=25, replace=False)]
cols = ['open', 'close', 'high', 'low', 'vol', 'amount']
all_bad = 0
for (exch, code) in sample:
    cond = f"exchange == '{exch}' & code == '{code}'"
    a = pd.read_hdf(SRC, 'nfq', where=cond)
    b = pd.read_hdf(OUT, 'nfq', where=cond)
    a = a.reset_index(level=['exchange', 'code'], drop=True)
    b = b.reset_index(level=['exchange', 'code'], drop=True)
    assert a.index.name == 'date' and b.index.name == 'date'
    if len(a) != len(b):
        print('COUNT DIFF', exch, code, len(a), len(b))
        all_bad += 1
        continue
    joined = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
    n = len(joined)
    if n != len(a):
        print('DATE MISALIGN', exch, code, n, len(a))
        all_bad += 1
        continue
    for c in cols:
        va = joined[c + '_a'].to_numpy()
        vb = joined[c + '_b'].to_numpy()
        if not np.array_equal(va, vb):
            # tolerate -0.0/+0.0 and any float32 rounding on the value cols
            same = (va == vb) | (np.isnan(va) & np.isnan(vb))
            ndiff = int((~same).sum())
            if ndiff:
                print('VALUE DIFF', exch, code, c, ndiff, '/', n)
                all_bad += 1
                break
    print('ok', exch, code, 'rows', n, flush=True)
print('VERIFY_DONE total_issues', all_bad)
