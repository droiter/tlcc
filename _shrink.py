import sys, time
import numpy as np
import pandas as pd
import hdf5plugin  # noqa: F401  (not needed for pandas path, harmless)
import h5py

SRC = '/workspace/tlcc/trader_nfq.hdf'
OUT = '/workspace/tlcc/trader_nfq_aidx.hdf'
KEEP_FILES = ['/workspace/tlcc/idx_sse.txt', '/workspace/tlcc/idx_szse.txt', '/workspace/tlcc/idx_88.txt']

t0 = time.time()
keep = set()
for fp in KEEP_FILES:
    for line in open(fp):
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('\t')
        code, exchange = parts[0], parts[1]
        keep.add(exchange + '/' + code)
print('KEEP_SIZE', len(keep), flush=True)

# expected per-instrument counts from prior scan (authoritative reference)
expected = {}
for line in open('/workspace/tlcc/_scan_tmp.txt'):
    if line.startswith('INST '):
        _, exch, code, cnt = line.split()
        key = exch + '/' + code
        if key in keep:
            expected[key] = int(cnt)
print('KEEP_FOUND_IN_SRC', len(expected), 'missing:', sorted(keep - set(expected)), flush=True)

src_n = None
with pd.HDFStore(SRC, mode='r') as store:
    storer = store.get_storer('nfq')
    src_n = storer.nrows
print('SRC_NROWS', src_n, flush=True)

total_processed = 0
total_kept = 0
n_chunks = 0
t_last = t0
with pd.HDFStore(SRC, mode='r') as store, pd.HDFStore(OUT, mode='w', complevel=5, complib='zlib') as out:
    it = store.select('nfq', chunksize=1_000_000)
    for df in it:
        n_chunks += 1
        total_processed += len(df)
        ex = df.index.get_level_values('exchange').astype(str)
        cd = df.index.get_level_values('code').astype(str)
        m = np.isin(ex + '/' + cd, list(keep))
        sub = df[m]
        total_kept += len(sub)
        if len(sub):
            out.append('nfq', sub,
                       data_columns=['code', 'exchange', 'date'],
                       min_itemsize={'code': 6, 'exchange': 4})
        if n_chunks % 4 == 0 or total_processed == src_n:
            print('progress rows', total_processed, 'kept', total_kept,
                  't', round(time.time() - t0), 's', flush=True)
print('DONE processed', total_processed, 'kept', total_kept, 't', round(time.time() - t0), 's', flush=True)

# validate: per-instrument counts in new file
with pd.HDFStore(OUT, mode='r') as out:
    got = {}
    d = pd.read_hdf(OUT, 'nfq', columns=[])
    # count via full read of levels only
    n2 = out.get_storer('nfq').nrows
    for chunk in pd.read_hdf(OUT, 'nfq', columns=[], chunksize=1_000_000):
        ex = chunk.index.get_level_values('exchange').astype(str)
        cd = chunk.index.get_level_values('code').astype(str)
        keys = ex + '/' + cd
        uniq, cnt = np.unique(keys, return_counts=True)
        for u, c in zip(uniq, cnt):
            got[u] = got.get(u, 0) + int(c)
    mism = {k: (expected[k], got.get(k)) for k in expected if expected[k] != got.get(k)}
    extra = sorted(set(got) - set(expected))
    print('NEW_NROWS', n2)
    print('INSTR_NEW', len(got), 'MISMATCH', mism, 'EXTRA', extra[:10], flush=True)
