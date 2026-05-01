# 环境时间关注实验报告

- 分析对象：`artifacts\dynamic_hlb_ctaux_full5`
- 窗口：6->1, 6->3, 9->1, 9->3

## Overall

- `full` 平均 `Ctd=0.5436`，`BStd=0.4751`
- 最佳 `only_last_k` 变体：`only_last_k_14`，平均 `Ctd=0.5306`，`BStd=0.3423`
- 最佳 `exclude_last_k` 变体：`exclude_last_k_7`，平均 `Ctd=0.5275`，`BStd=0.4171`

## Window 6->1

- 风险集规模：`eligible_n≈14.0`，未来事件数：`future_event_n≈0.6`
- Attention enrichment 排名：days_60_plus(1.024), days_31_60(0.968), days_3_7(0.947), days_8_14(0.939), last_2_days(0.938), days_15_30(0.935)
- 遮挡 Δrisk 排名：last_2_days(0.0020), days_3_7(0.0011), days_60_plus(0.0006), days_31_60(0.0005), days_8_14(0.0004), days_15_30(0.0003)
- 置乱 Δrisk 排名：last_2_days(0.0013), days_3_7(0.0005), days_8_14(0.0001), days_15_30(0.0000), days_31_60(0.0000), days_60_plus(0.0000)
- `full`：`Ctd=0.8974`，`BStd=0.0432`
- `only_last_k`：`k=2` -> Ctd=0.5641, BStd=0.0514, `k=7` -> Ctd=0.6410, BStd=0.0496, `k=14` -> Ctd=0.7692, BStd=0.0420, `k=30` -> Ctd=0.6923, BStd=0.0457
- `exclude_last_k`：`k=2` -> Ctd=0.7179, BStd=0.0421, `k=7` -> Ctd=0.7179, BStd=0.0495, `k=14` -> Ctd=0.6923, BStd=0.0476, `k=30` -> Ctd=0.6923, BStd=0.0567
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 6->3

- 风险集规模：`eligible_n≈14.0`，未来事件数：`future_event_n≈2.0`
- Attention enrichment 排名：days_60_plus(1.024), days_31_60(0.968), days_3_7(0.947), days_8_14(0.939), last_2_days(0.938), days_15_30(0.935)
- 遮挡 Δrisk 排名：last_2_days(0.0040), days_3_7(0.0021), days_60_plus(0.0014), days_31_60(0.0013), days_15_30(0.0011), days_8_14(0.0010)
- 置乱 Δrisk 排名：last_2_days(0.0017), days_3_7(0.0007), days_8_14(0.0003), days_15_30(0.0003), days_31_60(0.0002), days_60_plus(0.0000)
- `full`：`Ctd=0.5021`，`BStd=0.1536`
- `only_last_k`：`k=2` -> Ctd=0.4386, BStd=0.1857, `k=7` -> Ctd=0.3826, BStd=0.1573, `k=14` -> Ctd=0.4557, BStd=0.1431, `k=30` -> Ctd=0.4329, BStd=0.1661
- `exclude_last_k`：`k=2` -> Ctd=0.4106, BStd=0.1582, `k=7` -> Ctd=0.3856, BStd=0.1654, `k=14` -> Ctd=0.4318, BStd=0.1854, `k=30` -> Ctd=0.4164, BStd=0.1924
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 9->1

- 风险集规模：`eligible_n≈12.0`，未来事件数：`future_event_n≈0.6`
- Attention enrichment 排名：days_60_plus(1.020), days_3_7(0.944), days_31_60(0.944), last_2_days(0.941), days_15_30(0.941), days_8_14(0.940)
- 遮挡 Δrisk 排名：days_3_7(0.0011), last_2_days(0.0010), days_60_plus(0.0005), days_8_14(0.0003), days_15_30(0.0001), days_31_60(0.0001)
- 置乱 Δrisk 排名：last_2_days(0.0015), days_3_7(0.0006), days_8_14(0.0001), days_15_30(0.0000), days_60_plus(0.0000), days_31_60(0.0000)
- `full`：`Ctd=0.1515`，`BStd=0.0616`
- `only_last_k`：`k=2` -> Ctd=0.1768, BStd=0.0614, `k=7` -> Ctd=0.2096, BStd=0.0550, `k=14` -> Ctd=0.1490, BStd=0.0567, `k=30` -> Ctd=0.4015, BStd=0.0586
- `exclude_last_k`：`k=2` -> Ctd=0.2121, BStd=0.0573, `k=7` -> Ctd=0.3005, BStd=0.0726, `k=14` -> Ctd=0.2096, BStd=0.0663, `k=30` -> Ctd=0.2374, BStd=0.0740
- 结论：证据混合：模型既没有表现出明确的“只看检测日前后”特征，也不能简单归结为完全由更早长历史主导，需要结合具体窗口分别解释。

## Window 9->3

- 风险集规模：`eligible_n≈12.0`，未来事件数：`future_event_n≈2.4`
- Attention enrichment 排名：days_60_plus(1.020), days_3_7(0.944), days_31_60(0.944), last_2_days(0.941), days_15_30(0.941), days_8_14(0.940)
- 遮挡 Δrisk 排名：last_2_days(0.0025), days_3_7(0.0025), days_60_plus(0.0015), days_8_14(0.0007), days_15_30(0.0004), days_31_60(0.0004)
- 置乱 Δrisk 排名：last_2_days(0.0032), days_3_7(0.0011), days_8_14(0.0002), days_15_30(0.0001), days_31_60(0.0000), days_60_plus(0.0000)
- `full`：`Ctd=0.6082`，`BStd=1.6421`
- `only_last_k`：`k=2` -> Ctd=0.5497, BStd=1.9400, `k=7` -> Ctd=0.4732, BStd=1.6319, `k=14` -> Ctd=0.6912, BStd=1.1272, `k=30` -> Ctd=0.5732, BStd=1.5963
- `exclude_last_k`：`k=2` -> Ctd=0.7010, BStd=1.2855, `k=7` -> Ctd=0.6914, BStd=1.3810, `k=14` -> Ctd=0.6461, BStd=1.7115, `k=30` -> Ctd=0.5828, BStd=1.6411
- 结论：支持连续多日组合与滞后效应：模型对更早历史也敏感，且只保留最近2/7天无法维持性能，删除最近2/7天后仍能保留较多判别力。
