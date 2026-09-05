# 0720 — B5-3 + DWGConv-SA, batch=4 验证实验

## 实验目的

在 B5-3 基线 + DWGConvSA（空间注意力增强）的基础上，验证 **batch=4** 小批量训练效果。

## 基线

所有实验均基于 **B5-3 (0705/b5)** + **DWGConvSA**：
- Backbone: SwinV2-T
- P5 → AIFI → Y5 → HFSCC(P3m, P4m, Y5) → P3c, P4c, P5c
- P2/P3/P4 均有 proj + EMA，P2 使用 DWGConvSA refine，无 MDHIFI
- Decoder: P2_out, P3_out, P4_out（P5 检测头已移除）

## 实验配置总览

| 配置 | 实验组 | 网络结构 | 测试数据集 | batch | CLI 参数 |
|------|--------|---------|-----------|-------|---------|
| `b100.yaml` | B5-3 + DWGConv-SA, batch=4 | b0 + P2 DWGConvSA | **CST** | 4 | – |

## DWGConvSA

空间注意力增强噪声抑制：

```
x -> DWConv(3x3) -> ChannelGate(SE) -> SpatialGate(7x7) -> Proj(1x1) ─┬─> out
x ────────────────────────────────────────────> Short(1x1) ────────────┘
```

- **ChannelGate**: SE (GAP → FC → ReLU → FC → Sigmoid)，通道级选择性增强
- **SpatialGate**: `concat(avg_pool, max_pool)` → `Conv2d(2, 1, 7)` → `Sigmoid`，空间级噪声抑制
- **残差连接**: 1×1 卷积保底通路，稳定训练

## 训练命令

```bash
# b100 — B5-3 + DWGConv-SA, batch=4, CST 数据集
python scripts/train_detr.py \
  --model exp_cfg/0720/b100.yaml \
  --dataset_path datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml \
  --project 0720 \
  --name b100 \
  --epochs 30 --batch_size 4
```

## 目录结构

```
exp_cfg/0720/
├── README.md    # 本文件
└── b100.yaml    # B5-3 + DWGConv-SA, batch=4, CST
```
