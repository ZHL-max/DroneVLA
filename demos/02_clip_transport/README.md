# Demo 02: CLIPort - 语言接地的机器人操作

## 概述

这个Demo演示了CLIPort的核心思想：结合CLIP的语义理解与TransportNet的空间精度。

## 核心思想

```
CLIP (what)  +  TransportNet (where)  =  CLIPort
语义理解           空间精度              语言接地操作
```

## 架构

```
图像 ─────────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
[CLIP视觉编码] ──┐                    [TransportNet]
    │            │                         │
    ▼            ▼                         ▼
[图像特征]   [语言特征]              [空间特征图]
    │            │                         │
    └────────────┼─────────────────────────┘
                 │
                 ▼
           [注意力融合]
                 │
                 ▼
         [Pick热图] [Place热图]
```

## 运行

```bash
cd demos/02_clip_transport
python clip_transport.py
```

## 学习要点

1. **CLIP的语义理解**：如何理解"红色方块"这样的语义
2. **TransportNet的空间精度**：如何精确定位操作位置
3. **注意力融合**：如何结合what和where信息
4. **Pick-and-Place**：如何预测抓取和放置位置

## 与VLA的关系

CLIPort是VLA的早期形式：
- 输入：视觉 + 语言
- 输出：操作位置（动作）

更现代的VLA（如RT-2、OpenVLA）在此基础上：
- 使用更大的预训练模型
- 支持更复杂的动作空间
- 具有更好的泛化能力
