import hdf5plugin  # noqa: F401  registers blosc
import h5py
import numpy as np
import sys, time

path = '/workspace/tlcc/trader_nfq.hdf'
out = '/workspace/tlcc/universe.tsv'

f = h5py.File(path, 'r')
t = f['nfq/table']
n = t.shape[0]

sub = t.fields(['code', 'exchange', 'date'])
agg = {}  # (code, exchange) -> [count, min_date, max_date]

CH = 2_000_000
t0 = time.time()
for start in range(0, n, CH):
    end = min(start + CH, n)
    r = sub[start:end]
    codes = r['code']
    ex = r['exchange']
    dates = r['date']
    # unique combos in this chunk: pack code and exchange into one S10 key via views
    packed = np.zeros(len(codes), dtype='S10')
    # build composite via concatenated buffer: copy code then exchange
    buf = np.empty(len(codes), dtype=[('c','S6'),('e','S4')])
    buf['c'] = codes
    buf['e'] = ex
    keys = buf.view('S10').ravel()  # layout: 6 code + 4 exchange (S6+S4 contiguous)
    uniq, inv, cnts = np.unique(keys, return_inverse=True, return_counts=True)
    for u in range(len(uniq)):
        m = inv == u
        k = uniq[u]
        key = (k[:6].decode(), k[6:].decode().strip('\x00'))
        c = int(cnts[u])
        dmin = int(dates[m].min())
        dmax = int(dates[m].max())
        if key in agg:
            e = agg[key]
            e[0] += c
            e[1] = min(e[1], dmin)
            e[2] = max(e[2], dmax)
        else:
            agg[key] = [c, dmin, dmax]
    if (start // CH) % 2 == 0:
        print(f'progress {start}/{n} uniq_so_far={len(agg)} elapsed={time.time()-t0:.0f}s', flush=True)

print('total unique instruments:', len(agg), flush=True)
with open(out, 'w') as fh:
    fh.write('code\texchange\trows\tmin_date\tmax_date\n')
    for (code, ex), (c, dmin, dmax) in sorted(agg.items()):
        fh.write(f'{code}\t{ex}\t{c}\t{dmin}\t{dmax}\n')
print('wrote', out, flush=True)
