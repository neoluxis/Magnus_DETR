# 0715 — 损失函数改进 & P2 路径结构对比实验

## 实验目的

1. **损失函数改进验证 (b1, b2)**：在 0705/b5 基线上验证 Inner-SIoU 和 Inner-CIoU 两种损失函数的有效性，分别在 CST 和 irdst 两个数据集上开展实验。
2. **P2 路径结构方案对比 (b3, b4)**：对比 DWGConvSA（空间注意力增强）和 DWGConvMS（多尺度自适应门控）两种 P2 噪声抑制方案，确定创新点 1 的最终实现方案。

## 基线

所有实验均基于 **0705/b5**：
- Backbone: SwinV2-T
- P5 → AIFI → Y5 → HFSCC(P3m, P4m, Y5) → P3c, P4c, P5c
- P2/P3/P4 均有 proj + EMA，P2 无 MDHIFI
- Decoder: P2_out, P3_out, P4_out（P5 检测头已移除）

## 实验配置总览

| 配置 | 实验组 | 网络结构 | 测试数据集 | CLI 参数 |
|------|--------|---------|-----------|---------|
| `b0.yaml` | 基线 | 0705/b5（无 P2 refine） | – | – |
| `b1.yaml` | 损失函数验证 | 与 b0 完全相同 | **CST** | `--iou_type inner_siou` / `inner_ciou` |
| `b2.yaml` | 损失函数验证 | 与 b0 完全相同 | **irdst** | `--iou_type inner_siou` / `inner_ciou` |
| `b3.yaml` | P2 路径结构 | b0 + P2 **DWGConvSA** | **CST + irdst** | – |
| `b4.yaml` | P2 路径结构 | b0 + P2 **DWGConvMS** | **CST + irdst** | – |

## 实验 (1): 损失函数改进验证

### 训练命令

```bash
# b1 — CST 数据集 (2 次训练: inner_siou + inner_ciou)
python scripts/train_detr.py \
  --model exp_cfg/0715/b1.yaml \
  --dataset_path datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml \
  --project 0715-loss-cst \
  --name b1-inner-siou \
  --iou_type inner_siou --inner_ratio 0.75 \
  --epochs 30 --batch_size 32

python scripts/train_detr.py \
  --model exp_cfg/0715/b1.yaml \
  --dataset_path datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml \
  --project 0715-loss-cst \
  --name b1-inner-ciou \
  --iou_type inner_ciou --inner_ratio 0.75 \
  --epochs 30 --batch_size 32

# b2 — irdst 数据集 (2 次训练: inner_siou + inner_ciou)
python scripts/train_detr.py \
  --model exp_cfg/0715/b2.yaml \
  --dataset_path datasets/IRDST_real_yolo/data.yaml \
  --project 0715-loss-irdst \
  --name b2-inner-siou \
  --iou_type inner_siou --inner_ratio 0.75 \
  --epochs 30 --batch_size 32

python scripts/train_detr.py \
  --model exp_cfg/0715/b2.yaml \
  --dataset_path datasets/IRDST_real_yolo/data.yaml \
  --project 0715-loss-irdst \
  --name b2-inner-ciou \
  --iou_type inner_ciou --inner_ratio 0.75 \
  --epochs 30 --batch_size 32
```

### 损失函数说明

| IoU 类型 | `--iou_type` | 计算方式 |
|---------|-------------|---------|
| Inner-SIoU | `inner_siou` | 在 ratio 缩放的 inner box 上计算 IoU，施加原始 box 的 SIoU 角度+距离+形状惩罚 |
| Inner-CIoU | `inner_ciou` | 在 ratio 缩放的 inner box 上计算 IoU，施加原始 box 的 CIoU 距离+长宽比惩罚 |

`--inner_ratio 0.75`（默认）表示 inner box 为原始 box 中心区域的 75% 宽高，增强中心对齐敏感性，有利于小目标检测。

## 实验 (2): P2 路径结构方案对比

### 训练命令

```bash
# b3 — DWGConvSA (CST)
python scripts/train_detr.py \
  --model exp_cfg/0715/b3.yaml \
  --dataset_path datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml \
  --project 0715-struct-cst --name b3-dwgconvsa-cst \
  --epochs 30 --batch_size 32

# b3 — DWGConvSA (irdst)
python scripts/train_detr.py \
  --model exp_cfg/0715/b3.yaml \
  --dataset_path datasets/IRDST_real_yolo/data.yaml \
  --project 0715-struct-irdst --name b3-dwgconvsa-irdst \
  --epochs 30 --batch_size 32

# b4 — DWGConvMS (CST)
python scripts/train_detr.py \
  --model exp_cfg/0715/b4.yaml \
  --dataset_path datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml \
  --project 0715-struct-cst --name b4-dwgconvms-cst \
  --epochs 30 --batch_size 32

# b4 — DWGConvMS (irdst)
python scripts/train_detr.py \
  --model exp_cfg/0715/b4.yaml \
  --dataset_path datasets/IRDST_real_yolo/data.yaml \
  --project 0715-struct-irdst --name b4-dwgconvms-irdst \
  --epochs 30 --batch_size 32
```

## tsp 提交顺序

**优先提交所有 CST 实验，完成后再提交 irdst 实验。**

```
Phase 1 — CST:
  b1 (inner_siou)
  b1 (inner_ciou)
  b3
  b4

Phase 2 — irdst:
  b2 (inner_siou)
  b2 (inner_ciou)
  b3
  b4
```

## P2 Refine 模块对比 (b3 vs b4)

### 参数量 & 计算量

输入尺寸: 640×640

| 模型 | 参数量 | FLOPs | 相比 b0 增量 |
|------|--------|-------|-------------|
| b0 (baseline) | 54,622,510 (54.62M) | 162.46G | – |
| b3 (DWGConvSA) | 54,658,000 (54.66M) | 162.53G | +35,490 (+0.06%), +0.07G (+0.04%) |
| b4 (DWGConvMS) | 54,681,966 (54.68M) | 162.72G | +59,456 (+0.11%), +0.26G (+0.16%) |

### DWGConvSA (b3)

空间注意力增强噪声抑制：

```
x -> DWConv(3x3) -> ChannelGate(SE) -> SpatialGate(7x7) -> Proj(1x1) ─┬─> out
x ────────────────────────────────────────────> Short(1x1) ────────────┘
```

- **ChannelGate**: SE (GAP → FC → ReLU → FC → Sigmoid)，通道级选择性增强
- **SpatialGate**: `concat(avg_pool, max_pool)` → `Conv2d(2, 1, 7)` → `Sigmoid`，空间级噪声抑制
- **残差连接**: 1×1 卷积保底通路，稳定训练
- 参数量增加: +35,490 (+0.06%)

### DWGConvMS (b4)

多尺度自适应门控噪声抑制：

```
                   -> DWConv(3x3) ──────────────────┐
x -> BranchGate(x) ->                               (+) -> BN -> Proj(1x1) ─┬─> out
                   -> DWConv(5x5, d=2) -> *scale ───┘                       │
x ──────────────────────────────────> Short(1x1) -> *res_scale ─────────────┘
```

- **BranchGate**: GAP → FC → ReLU → FC → 2C → Sigmoid → chunk(2)，自适应融合
- **双分支**: 3×3 保留细粒度细节 + 5×5 dilated 提供大感受野平滑
- **渐进学习**: large_scale(init=0.1) 渐进激活 5×5 分支，res_scale(init=0.1) 渐进激活残差
- 参数量增加: +59,456 (+0.11%)

## 目录结构

```
exp_cfg/0715/
├── README.md    # 本文件
├── b0.yaml      # 基线 (0705/b5)
├── b1.yaml      # 损失函数验证 — CST (架构 = b0)
├── b2.yaml      # 损失函数验证 — irdst (架构 = b0)
├── b3.yaml      # P2 DWGConvSA (b0 + DWGConvSA)
└── b4.yaml      # P2 DWGConvMS (b0 + DWGConvMS)
```
