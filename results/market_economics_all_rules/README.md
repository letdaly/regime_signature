# Volatility-managed results for every signature rule

Run 2026-09-30 with `market_economics_all_rules.py` on the stored out-of-sample weights in
`results/market_french/<market>/folds`; no model is refitted. The four rules of the published
economics table (HMM, Statistics, Statistics + log-signatures, B3 + raw signatures) keep their
position and bootstrap seeds and are reproduced exactly; the script stops if any published value
differs. Two rules are added: Statistics + raw signatures and B3 + log-signatures. Sharpe
ratios are compared with the Statistics rule by the Ledoit-Wolf studentized bootstrap test of
the market study (1,000 resamples, block length 21).

| Market | Rule | Sharpe | p vs. Statistics |
|---|---|---:|---:|
| United States | Statistics + raw sig. | 0.625 | 0.208 |
| United States | B3 + log-sig. | 0.581 | 0.763 |
| Developed ex US | Statistics + raw sig. | 0.543 | 0.016 |
| Developed ex US | B3 + log-sig. | 0.473 | 0.189 |

Across the ten Sharpe-ratio comparisons of the two markets (four signature rules and the HMM
rule against Statistics), the smallest Holm-adjusted p-value is 0.160, for Statistics + raw
signatures outside the United States. Files: `economics.csv` (every rule, both markets) and
`config.json`.
