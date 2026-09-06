# tlcc — 指数间时滞交叉相关（TLCC）领先滞后分析

对沪深指数（通达信 SSE 880xxx 板块/概念/情绪指数 + 主要宽基）两两计算
time-lagged cross-correlation 并评估其"领先/滞后"结构。

## 约定

- `r[i,j,k] = Pearson corr( x_i[t], x_j[t+k] )`，k = 0..K，单侧窗口。
- **k>0 时 i 领先 j k 天**。
- 净领先统计量 `d(k) = r_ab(k) − r_ba(k)`：
  - `d(k) > 0` → a 领先 b；`d(k) < 0` → b 领先 a（与正/负相关无关，
    相关方向看 r 本身符号）。

## 关键结论（TLCC_K60_VALIDATION.md 全文）

1. **raw close 水平上的 TLCC 结果不可作领先证据**：净领先 |d(k)| 随 k 近似
   线性抬升、极值 ~99% 钉在窗口边缘 k=K（regime-drift/非平稳假象）。
2. **应在 log 收益上分析**：`tlcc_tlc_ret.py` 版本 mean |d(k)| 全程平坦
   ~0.02，边缘占比从 ~45% 降到 ~2%。
3. 收益版 849 个候选对的半样本 + block-bootstrap 验证：**仅 29 对**通过，
   且全部在 k=1–2 天短窗（见 `tlcc_tlc_k60_validated.csv`）。
4. 微盘股（880823）专属研究：整体是"被领先"方；最稳信号为
   机构/权重风格 → 微盘股 k=1–3（滚动样本外 IC≈+0.07，两半段方向不反），
   幅度属弱-中等（详见 `microcap_880823_hub.csv`、`microcap_oos_test.csv`）。

## 脚本

| 脚本 | 作用 |
|---|---|
| `scan_universe.py` | 扫描源 HDF 得到全部指数（code/exchange/跨度） |
| `_shrink.py` / `_verify.py` | 从源文件抽取 197 个目标指数的子集并校验 |
| `tlcc_tlc.py` | raw close TLCC，K 可调（默认 20） |
| `tlcc_tlc_ret.py` | log 收益版 TLCC（推荐），K 可调 |
| `tlc_k60_validate.py` | 候选对 returns 重算 + 半样本 + block-bootstrap 显著性 |
| `tlc_k120_followup.py` | K=120 的短窗/长窗分类与三分段稳定性 |
| `tlc_microcap.py` | 微盘股(880823) 枢纽表：谁领先它/它领先谁 |
| `eval_microcap_oos.py` | 机构→微盘股短窗信号的滚动样本外前向检验 |

## 数据

源数据 `trader_nfq_aidx.hdf`（197 指数日线 close，2007–2026）**不在仓库内**
（`.gitignore` 排除 *.hdf / *.npz / *.log）。需要时把文件放回仓库根目录即可
复跑全部脚本，例如：

```
python3 tlcc_tlc.py 60        # raw close, K=60
python3 tlcc_tlc_ret.py 120   # log returns, K=120
python3 tlc_microcap.py
python3 eval_microcap_oos.py
```

`results/` 之外的分析小表直接位于仓库根目录（见各脚本输出文件名）。
