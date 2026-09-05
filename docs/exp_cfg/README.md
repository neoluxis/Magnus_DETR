# 实验配置文档

> 按日期倒序排列。所有配置基于 **SwinV2-Tiny** 骨干网络（torchvision 预训练），使用 RTDETRDecoder 检测头。

---

## 目录

- [0715 — DWGConv P2 Refine 优化实验](#0715--dwgconv-p2-refine-优化实验)
- [0709 — B7: DWGConv P2 抑噪](#0709--b7-dwgconv-p2-抑噪)
- [0708 — B6: 移除 P2 EMA](#0708--b6-移除-p2-ema)
- [0705 — B5: 移除 P5 检测头](#0705--b5-移除-p5-检测头)
- [0704 — B3/B4: P2 无 MDHIFI + PAN 重复注入消融](#0704--b3b4-p2-无-mdhifi--pan-重复注入消融)
- [0703 — B0/B1/B2: 基线 + LiteMDHIFI + P2 输出头](#0703--b0b1b2-基线--litemdhifi--p2-输出头)
- [0702 — HFSCC P3 Gate 消融](#0702--hfscc-p3-gate-消融)
- [0630 — Upsample + MDHIFI + HFSCC Noise 组合消融](#0630--upsample--mdhifi--hfscc-noise-组合消融)
- [0628 — HFSCC + MDHIFI 布局变体](#0628--hfscc--mdhifi-布局变体)
- [0627 — B5 Stepwise Lateral 消融测试](#0627--b5-stepwise-lateral-消融测试)
- [0611 — Stepwise Lateral + Auxiliary Branch 系列](#0611--stepwise-lateral--auxiliary-branch-系列)
- [更早的基线配置（detr/ 目录）](#更早的基线配置detr-目录)
- [YOLO 系列配置（yolo/ 目录）](#yolo-系列配置yolo-目录)
- [模块缩写速查](#模块缩写速查)

---

## 0715 — DWGConv P2 Refine 优化实验

**目的**: 对比 DWGConv 两个优化变体在 P2 特征噪声抑制上的效果。B0 为基线（P2 proj 后直接 Concat，无 refine 模块）。

**骨干**: SwinV2-Tiny + EMA(P2/P3/P4/P5)
**Neck**: P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN
**Decoder**: P2, P3, P4（P5 仅作语义源和校准源，不参与检测）

| 配置 | 文件 | P2 Refine 模块 | 说明 |
|------|------|---------------|------|
| **B0** | `0715/b0.yaml` | 无 | P2 proj(96→256) 后直接 Concat，baseline |
| **B1** | `0715/b1.yaml` | **DWGConvSA** | 空间注意力增强：SE-Gate + SpatialGate(7×7) |
| **B2** | `0715/b2.yaml` | **DWGConvMS** | 多尺度自适应门控：双分支 DWConv(3×3+5×5d2) + BranchGate + BN + 渐进残差 |

### 模块参数量

| 模块 | 参数量 | 相比 DWGConv |
|------|--------|-------------|
| DWGConv (原版) | 35,392 | - |
| DWGConvSA | 35,490 | +98 (+0.28%) |
| DWGConvMS | 59,456 | +24,064 (+67.99%) |

### 模型整体参数量（640×640）

| 模型 | 参数量 | FLOPs | 相比 B0 增量 |
|------|--------|-------|-------------|
| B0 | 54.62M | 162.46G | - |
| B1 | 54.66M | 162.53G | +0.06% params, +0.04% FLOPs |
| B2 | 54.68M | 162.72G | +0.11% params, +0.16% FLOPs |

### 架构细节

**DWGConvSA**:
```
x → DWConv(3×3) → ChannelGate(SE) → SpatialGate(7×7) → Proj(1×1) ─┬→ out
x ──────────────────────────────────────→ Short(1×1) ──────────────┘
```

**DWGConvMS**:
```
                 → DWConv(3×3) ──────────────────┐
x → BranchGate →                                  (+) → BN → Proj(1×1) ─┬→ out
                 → DWConv(5×5, d=2) → *scale ────┘                     │
x ────────────────────────────────→ Short(1×1) → *res_scale ───────────┘
```

详见 `exp_cfg/0715/README.md`。

---

## 0709 — B7: DWGConv P2 抑噪

**目的**: B5 基础上，P2 路径在 1×1 Conv 后增加 DWGConv 轻量门控深度卷积。

**骨干**: SwinV2-Tiny + EMA(P2/P3/P4/P5)
**Neck**: P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN
**P2**: EMA → 1×1 Conv(96→256) → **DWGConv**(256) → Concat Up(Y3) → RepC3 → P2_out
**Decoder**: P2, P3, P4（P5 移除）

| 配置 | 文件 | 关键变化 |
|------|------|---------|
| **B7** | `0709/b7.yaml` | P2 新增 DWGConv(DWConv(3×3) + SE-Gate + 残差连接) |

---

## 0708 — B6: 移除 P2 EMA

**目的**: B5 基础上移除 P2 的 EMA 注意力模块，P2 直接使用 Permute 输出。

**骨干**: SwinV2-Tiny + EMA(**P3/P4/P5 only**)
**Neck**: P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN
**P2**: Permute → 1×1 Conv(96→256) → Concat Up(Y3) → RepC3 → P2_out（**无 EMA**）
**Decoder**: P2, P3, P4（P5 移除）

| 配置 | 文件 | 关键变化 |
|------|------|---------|
| **B6** | `0708/b6.yaml` | P2 移除 EMA，其余同 B5 |

---

## 0705 — B5: 移除 P5 检测头

**目的**: B3 基础上移除 P5 检测头，decoder 仅使用 P2/P3/P4。P5 仍通过 AIFI 提供语义信息，通过 HFSCC 提供校准信息。

**骨干**: SwinV2-Tiny + EMA(P2/P3/P4/P5)
**Neck**: P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN
**Decoder**: P2, P3, P4（**P5 移除**）

| 配置 | 文件 | 关键变化 |
|------|------|---------|
| **B5** | `0705/b5.yaml` | PAN 移除 P5_out 生成路径；HFSCC 的 P5c 仅参与 FPN top-down |

架构流程:
```
P5 proj → AIFI → Y5 ──────────────────────┐
P4 proj → MDHIFI → P4m ───────────────────┤
P3 proj → MDHIFI → P3m ───────────────────┤
P2 proj (96→256) ─────────────────────────┤
                                           ↓
              HFSCC([P3m, P4m, Y5]) → P3c, P4c, P5c
                                           ↓
    FPN: P5c → Up → Concat(P4c) → Y4 → Up → Concat(P3c) → Y3 → Up → Concat(P2) → P2_out
    PAN: P2_out → Down → Concat(Y3, P3c) → P3_out → Down → Concat(Y4, P4c) → P4_out
    (P5_out removed)
```

---

## 0704 — B3/B4: P2 无 MDHIFI + PAN 重复注入消融

**目的**: 在 B2（P2 有 MDHIFI + P2/P3/P4/P5 decoder）基础上逐步消融。

**骨干**: SwinV2-Tiny + EMA(P2/P3/P4/P5)
**Neck**: P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN
**P2**: 1×1 Conv(96→256)，**无 MDHIFI**
**Decoder**: P2, P3, P4, P5（4 尺度）

| 配置 | 文件 | 关键变化 |
|------|------|---------|
| **B3** | `0704/b3.yaml` | P2 无 MDHIFI；PAN 中 P3c/P4c 重复注入（即 P3c 同时出现在 FPN#30 和 PAN#37；P4c 同时出现在 FPN#26 和 PAN#40） |
| **B4** | `0704/b4.yaml` | B3 基础上**移除** PAN 中 P3c/P4c 的重复注入；P3c/P4c 仅在 FPN 中使用，PAN 仅使用 Y3_proj/Y4_proj |

---

## 0703 — B0/B1/B2: 基线 + LiteMDHIFI + P2 输出头

**目的**: 建立带 MDHIFI + HFSCC 的 SwinV2-Tiny 基线，并尝试 LiteMDHIFI 和 P2 输出头。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)（B0/B1）、+ EMA(P2/P3/P4/P5)（B2）

| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B0** | `0703/b0.yaml` | P3/P4 lateral → MDHIFI → HFSCC(noise=ON) → FPN+PAN；Decoder: P3/P4/P5 |
| **B1** | `0703/b1.yaml` | B0 基础上 MDHIFI → **LiteMDHIFI**（去掉 Histogram self-attention 分支） |
| **B2** | `0703/b2.yaml` | B0 基础上**增加 P2 输出头**：P2(96ch→256ch proj→MDHIFI)，Decoder: P2/P3/P4/P5（4 尺度） |

---

## 0702 — HFSCC P3 Gate 消融

**目的**: 从 0630/exp5 派生，消融 HFSCC 内部 P3 分支的 C3 gate。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)
**Neck**: P3 lateral → MDHIFI；P4 lateral plain（无 MDHIFI）；HFSCC → P3/P4 upsample 移除 → bottom-up PAN
**Decoder**: P3, P4, P5

| 配置 | 文件 | HFSCC P3 Gate | 说明 |
|------|------|--------------|------|
| **exp1** | `0702/exp1.yaml` | **有 C3 gate** | HFSCC([P3m, P4m, Y5], 4, 3, "shift", False): P3c = F3 + γ·C3·Δ3 |
| **exp2** | `0702/exp2.yaml` | **无 C3 gate** | HFSCC([P3m, P4m, Y5], 4, 3, "shift", True, **False**): P3c = F3 + γ·Δ3 |

HFSCC 公式变化:
```
exp1 (有 gate):  P3c = F3 + γ₃ · C3 · Δ₃
exp2 (无 gate):  P3c = F3 + γ₃ · Δ₃
P4/P5 不变:      P4c = F4 + γ₄·C4·Δ₄ - β₄·(1-C4)·N₄
                 P5c = F5 + γ₅·C5·Δ₅ - β₅·(1-C5)·N₅
```

---

## 0630 — Upsample + MDHIFI + HFSCC Noise 组合消融

**目的**: 系统消融 upsample、MDHIFI、HFSCC noise suppression 的组合效果。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)
**Decoder**: P3, P4, P5

| 配置 | 文件 | P3 MDHIFI | P4 MDHIFI | P3 Upsample | P4 Upsample | HFSCC Noise |
|------|------|-----------|-----------|-------------|-------------|-------------|
| **exp1** | `detr0630/b7-exp1-...` | ✓ | ✓ | ✗ | ✓ | ON |
| **exp2** | `detr0630/b7-exp2-...` | ✓ | ✗ | ✗ | ✗ | ON |
| **exp3** | `detr0630/b7-exp3-...` | ✓ | ✓ | ✓ | ✓ | ON |
| **exp4** | `detr0630/exp4.yaml` | ✓ | ✓ | ✓ | ✓ | ON（同 exp3） |
| **exp5** | `detr0630/exp5.yaml` | ✓ | ✗ | ✗ | ✓ | ON（同 0702/exp1） |

---

## 0628 — HFSCC + MDHIFI 布局变体

**目的**: 探索 HFSCC 和 MDHIFI 在不同侧枝上的布局组合。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)
**Decoder**: P3, P4, P5

| 配置 | 文件 | P3 MDHIFI | P4 MDHIFI | HFSCC Noise | 说明 |
|------|------|-----------|-----------|-------------|------|
| **b6-p3-hfscc** | `detr0628/b6-p3-hfscc.yaml` | ✓ | ✗ | ON | 仅 P3 有 MDHIFI |
| **b6-p3p4-hfscc** | `detr0628/b6-p3p4-hfscc.yaml` | ✓ | ✓ | ON | P3+P4 MDHIFI（同 0703/b0） |
| **b6-p3p4-hfscc-nonoise** | `detr0628/b6-p3p4-hfscc-nonoise.yaml` | ✓ | ✓ | **OFF** | HFSCC noise suppression 关闭 |
| **b6-plain-hfscc** | `detr0628/b6-plain-hfscc.yaml` | ✗ | ✗ | ON | 无 MDHIFI，仅 HFSCC |

---

## 0627 — B5 Stepwise Lateral 消融测试

**目的**: 在 Stepwise Lateral 架构上进行 P3 upsample 和 P4 MDHIFI 的消融测试。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)
**Head 特点**: Stepwise Lateral（P3m→Down→A4 辅助注入 P4；Y4→Down→A5 辅助注入 P5）

| 配置 | 文件 | P3 Upsample | P4 MDHIFI | 说明 |
|------|------|-------------|-----------|------|
| **test00001** | `detr0627/test00001.yaml` | ✓ | ✓ | P3/P4 双 MDHIFI + P3 upsample（完整版） |
| **test00002** | `detr0627/test00002.yaml` | ✗ | ✓ | P3 upsample 移除，P4 MDHIFI 保留 |
| **test00003** | `detr0627/test00003.yaml` | ✗ | ✗ | P3 upsample + P4 MDHIFI 均移除 |
| **test00004** | `detr0627/test00004.yaml` | ✓ | ✗ | P3 upsample 保留，P4 MDHIFI 移除 |

---

## 0611 — Stepwise Lateral + Auxiliary Branch 系列

**目的**: 系统探索 Stepwise Lateral 架构，从 B1 到 B5 逐步演进。

**骨干**: SwinV2-Tiny + EMA(P3/P4/P5)
**P3 lateral**: MDHIFI → P3m

### B1: BiDFF PAN
| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B1** | `detr111/01-b1-bidff-pan.yaml` | 标准 FPN+PAN，bottom-up 使用 plain 3×3 Conv |

### B2: Auxiliary PAN
| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B2** | `detr111/02-b2-aux-pan.yaml` | P3m → Conv(s=2) aux → P4 PAN；P3m → Conv(s=2)×2 aux → P5 PAN |

### B3: Aux P4 Plain
| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B3** | `detr111/03-b3-aux-p4-plain.yaml` | P3m → plain Conv(s=2, BN) → Concat(Y4) → P4_out |

### B4: Stepwise Lateral Plain
| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B4** | `detr111/04-b4-stepwise-lateral-plain.yaml` | P3m → Conv(s=2, A4) → P4 concat；Y4 → Conv(s=2, A5) → P5 concat |

### B5 变体系列: Stepwise Lateral + P4 MDHIFI
| 配置 | 文件 | 关键特征 |
|------|------|---------|
| **B5** | `detr111/05-b5-stepwise-lateral-p4mdhifi_corrected.yaml` | P3/P4 双 MDHIFI + stepwise lateral |
| **B5-no-P3-up** | `detr111/06-...-no-p3-upsample.yaml` | B5 基础上移除 P3 top-down upsample |
| **B5-no-P4-mdhifi** | `detr111/07-...-no-p4mdhifi.yaml` | B5 基础上移除 P4 的 MDHIFI |
| **B5-no-up** | `detr111/08-b5-no-p3p4-upsample.yaml` | B5 基础上移除 P3+P4 所有 upsample |

---

## 更早的基线配置（detr/ 目录）

这些是 Ultralytics 风格的 YAML 模型定义，实验各种 backbone + neck + head 组合。

### ResNet 系列

| 配置 | 骨干 | EMA | Neck | Decoder |
|------|------|-----|------|---------|
| `rtdetr-resnet50-last3-pretrained.yaml` | ResNet50 | ✗ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-resnet50-last3-pretrained-ema.yaml` | ResNet50 | ✓ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-resnet101-last3-pretrained.yaml` | ResNet101 | ✗ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-resnet101-last3-pretrained-ema.yaml` | ResNet101 | ✓ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-resnet101-...-ema-wtconv-biagcau-mdhifi.yaml` | ResNet101 | ✓ | AIFI→MDHIFI + BiAGCAU + WTConv | P3/P4/P5 |

### SwinV2 系列

| 配置 | 骨干 | EMA | Neck | Decoder |
|------|------|-----|------|---------|
| `rtdetr-swinv2-tiny-last3-pretrained.yaml` | SwinV2-T | ✗ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-swinv2-tiny-last3-pretrained-ema.yaml` | SwinV2-T | ✓ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-swinv2-small-last3-pretrained.yaml` | SwinV2-S | ✗ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-swinv2-small-last3-pretrained-ema.yaml` | SwinV2-S | ✓ | AIFI + RepC3 FPN+PAN | P3/P4/P5 |
| `rtdetr-swinv2-tiny-...-ema-b1-aifi-mdhifi.yaml` | SwinV2-T | ✓ | **AIFI→MDHIFI** on P5 | P3/P4/P5 |
| `rtdetr-swinv2-tiny-...-ema-b2-p3-mdhifi.yaml` | SwinV2-T | ✓ | MDHIFI on **P3 lateral** | P3/P4/P5 |
| `rtdetr-swinv2-tiny-...-ema-b3-p3-aifi.yaml` | SwinV2-T | ✓ | **AIFI** on P3 lateral | P3/P4/P5 |
| `rtdetr-swinv2-tiny-...-ema-b3-p3-hifi.yaml` | SwinV2-T | ✓ | **HIFI** on P3 lateral | P3/P4/P5 |
| `rtdetr-swinv2-tiny-...-ema-wtconv-biagcau-mdhifi.yaml` | SwinV2-T | ✓ | AIFI→MDHIFI + BiAGCAU + WTConv | P3/P4/P5 |

### 0619 BiAGCAU 系列

基于 SwinV2-Tiny + EMA(P3/P4/P5) + MDHIFI on P3 lateral。

| 配置 | BiAGCAU 位置 | 说明 |
|------|-------------|------|
| `...-0619-b1-biagcau-p3.yaml` | P3 top-down fusion only | BiAGCAU 替换 fpn_blocks[1] |
| `...-0619-b2-biagcau-topdown.yaml` | P4+P3 top-down fusions | BiAGCAU 替换 fpn_blocks[0], fpn_blocks[1] |
| `...-0619-b3-biagcau-full.yaml` | All 4 neck blocks | BiAGCAU 替换全部 4 个 RepC3 |

### 0621 Stepwise Lateral 系列

基于 SwinV2-Tiny + EMA(P3/P4/P5) + MDHIFI on P3 lateral。

| 配置 | 关键特征 |
|------|---------|
| `...-0621-b4-stepwise-lateral-plain.yaml` | 同 detr111/04 |
| `...-0621-b5-stepwise-lateral-p4mdhifi.yaml` | 同 detr111/05 |

### Aux Branch 系列

基于 SwinV2-Tiny + EMA(P3/P4/P5) + MDHIFI on P3 lateral。

| 配置 | 关键特征 |
|------|---------|
| `...-aux-p4-dilated.yaml` | P3m → Conv(s=2, dilation=2) aux → P4 |
| `...-aux-p4-plain.yaml` | P3m → plain Conv(s=2, BN) aux → P4 |
| `...-aux-p4-p4mdhifi.yaml` | P3m→aux→P4；P4_out→MDHIFI→aux→P5 |
| `...-aux-pan.yaml` | P3m→aux→P4+P5（同 detr111/02） |
| `...-bidff-pan.yaml` | 同 detr111/01 |
| `...-direct-repc3-p3-aux-p4-lateral-mdhifi.yaml` | P3/P4 双 MDHIFI，P3 无 upsample，direct RepC3 |
| `...-direct-repc3-p3-aux-p4-lateral-plain.yaml` | P3 MDHIFI，P4 plain，P3 无 upsample |
| `...-direct-repc3-p3-aux-p4-p4mdhifi.yaml` | P3/P4 双 MDHIFI，P3 无 upsample |

### 其他 backbone + neck 探索

| 配置 | 骨干 | 关键特征 |
|------|------|---------|
| `rtdetr-l.yaml` | HGNet (原版 RT-DETR-l) | HGStem + HGBlock, AIFI, FPN+PAN |
| `rtdetr-r18.yaml` | ConvNormLayer+BasicBlock(R18) | AIFI + RepC3 FPN+PAN |
| `rtdetr-p2.yaml` | BasicBlock | AIFI, P2 输出（但 decoder 使用 P3-P5） |
| `rtdetr-MRFPN.yaml` | BasicBlock | AIFI, FPN+PAN（非 MRFPN） |
| `rtdetr-EMA-MRFPN.yaml` | BasicBlock + EMA | AIFI + **MRFPN(MCAM)** 双向交叉注意力 |
| `rtdetr-ema-p2.yaml` | BasicBlock + EMA | AIFI, FPN+PAN, P2-P5 输出 |
| `rtdetr-MRFPN-EMA-p2.yaml` | BasicBlock + EMA | AIFI, FPN+PAN, P2-P5 输出 |
| `rtdetr-WTConv.yaml` | BasicBlock_WTConv | AIFI, WTConv2d downsample |
| `rtdetr-WTConv-EMA.yaml` | BasicBlock + EMA | AIFI, WTConv2d downsample |
| `rtdetr-HIFI.yaml` | BasicBlock | **HIFI**(1024,8) 替代 AIFI |
| `rtdetr-HIFI-EMA.yaml` | BasicBlock + EMA | HIFI + EMA |
| `rtdetr-HIFI-MRFPN-WTConv.yaml` | BasicBlock_WTConv | HIFI, FPN+PAN, WTConv |
| `rtdetr-HIFI-MRFPN-WTConv-EMA.yaml` | BasicBlock + EMA | HIFI, MRFPN(MCAM), WTConv |
| `rtdetr-exp1-dgwrn.yaml` | BasicBlock_DGWRN | DynamicWTConv2d downsample |
| `rtdetr-exp1-ema.yaml` | BasicBlock + EMA | DynamicWTConv2d downsample |
| `rtdetr-exp2-dgwrn-biagcau.yaml` | BasicBlock_DGWRN | BiAGCAUBlock + WTConv2d |
| `rtdetr-exp2-ema-biagcau.yaml` | BasicBlock + EMA | BiAGCAUBlock + WTConv2d |
| `rtdetr-exp3-dgwrn-biagcau-mdhifi.yaml` | BasicBlock_DGWRN | AIFI→MDHIFI + BiAGCAU + WTConv2d |
| `rtdetr-exp3-ema-biagcau-mdhifi.yaml` | BasicBlock + EMA | AIFI→MDHIFI + BiAGCAU + WTConv2d |

---

## YOLO 系列配置（yolo/ 目录）

YOLO 目标检测模型的标准配置（Ultralytics 格式），使用 `Detect` 头。

### YOLO11 系列

| 配置 | 输出尺度 | EMA | 说明 |
|------|---------|-----|------|
| `yolo11.yaml` | P3/P4/P5 | ✗ | YOLO11 标准版，C3k2 + SPPF + C2PSA |
| `yolo11-p2.yaml` | P2/P3/P4/P5 | ✗ | YOLO11 + P2 输出 |
| `yolo11-ema-p2.yaml` | P2/P3/P4/P5 | ✓ | YOLO11 + EMA(P2-P5) + P2 输出 |

### YOLO26 系列

| 配置 | 输出尺度 | 说明 |
|------|---------|------|
| `yolo26.yaml` | P3/P4/P5 | YOLO26 标准版，end2end=True, reg_max=1, C3k2+SPPF+C2PSA |
| `yolo26-p2.yaml` | P2/P3/P4/P5 | YOLO26 + P2 输出 |

### YOLOv5 系列

| 配置 | 输出尺度 | 说明 |
|------|---------|------|
| `yolov5.yaml` | P3/P4/P5 | YOLOv5 v6.0，C3 blocks + SPPF |
| `yolov5-p2.yaml` | P2/P3/P4/P5 | YOLOv5 + P2 输出 |

### YOLOv8 系列

| 配置 | 输出尺度 | 检测头 | 说明 |
|------|---------|--------|------|
| `yolov8.yaml` | P3/P4/P5 | Detect | YOLOv8.0，C2f blocks + SPPF |
| `yolov8-p2.yaml` | P2/P3/P4/P5 | Detect | YOLOv8 + P2 输出 |
| `yolov8-rtdetr.yaml` | P3/P4/P5 | **RTDETRDecoder** | YOLOv8 backbone + DETR head |

---

## 模块缩写速查

| 缩写 | 全称 | 说明 |
|------|------|------|
| **EMA** | Efficient Multi-scale Attention | 跨空间多尺度注意力模块 |
| **AIFI** | Attention-based Intra-scale Feature Interaction | 尺度内自注意力（单尺度 Transformer Encoder） |
| **HIFI** | Hierarchical Intra-scale Feature Interaction | 层级式尺度内特征交互（AIFI 变体） |
| **MDHIFI** | Multi-Directional HIFI | 带直方图注意力的 HiFi 变体 |
| **LiteMDHIFI** | Lite MDHIFI | 去掉直方图自注意力的轻量 MDHIFI |
| **HFSCC** | Hierarchical Feature Spatial Cross-scale Calibration | 跨尺度特征校准（含噪声抑制） |
| **BiAGCAU** | Bidirectional Adjacent Gated Context Aggregation Unit | 双向邻接门控上下文聚合单元 |
| **WTConv** | Wavelet Transform Convolution | 小波变换卷积 |
| **DWGConv** | Depth-Wise Gated Convolution | 深度可分离门控卷积 |
| **DWGConvSA** | DWGConv + Spatial Attention | 空间注意力增强版 |
| **DWGConvMS** | DWGConv + Multi-Scale | 多尺度自适应门控版 |
| **MCAM** | Multi-scale Cross-Attention Module | 多尺度交叉注意力（MRFPN 组件） |
| **RepC3** | Re-parameterizable C3 | 可重参数化 C3 模块 |
| **FPN** | Feature Pyramid Network | 特征金字塔网络（top-down） |
| **PAN** | Path Aggregation Network | 路径聚合网络（bottom-up） |
