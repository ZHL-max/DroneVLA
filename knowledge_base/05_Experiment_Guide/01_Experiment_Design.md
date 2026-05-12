# 实验设计指南
## 如何设计和管理VLA训练实验

---

## 1. 实验设计原则

### 1.1 单变量原则

```
每次实验只改变一个变量，其他保持不变

示例：
迭代1：基线模型
迭代2：增加数据量（其他不变）
迭代3：添加注意力机制（其他不变）
迭代4：调整学习率（其他不变）

这样可以清楚知道每个改进的效果
```

### 1.2 对照实验

```
实验组：使用新方法
对照组：使用原方法

对比指标：
- 验证损失
- 成功率
- 训练时间
- 推理速度
```

### 1.3 可重复性

```python
# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 记录所有配置
config = {
    'seed': 42,
    'model': 'DroneVLA_V2',
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100,
    'data_version': 'v2.0',
}
```

---

## 2. 实验目录结构

### 标准结构

```
experiments/
├── iteration_01/
│   ├── config.yaml           # 实验配置
│   ├── checkpoints/
│   │   ├── best_model.pt     # 最佳模型
│   │   └── final_model.pt    # 最终模型
│   ├── logs/
│   │   ├── metrics.json      # 训练指标
│   │   ├── eval_results.json # 评估结果
│   │   └── train.log         # 训练日志
│   └── visualizations/
│       ├── training_curve.png
│       └── results.png
├── iteration_02/
│   └── ...
└── comparison.png            # 迭代对比图
```

### 配置文件模板

```yaml
# config.yaml
experiment:
  name: "iteration_01"
  description: "基线模型"
  date: "2026-05-13"

model:
  type: "DroneVLA_V2"
  visual_dim: 256
  use_attention: false

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: "cosine"

data:
  num_episodes: 100
  augment: false
  focus_tasks: ["navigate", "avoid"]

results:
  best_val_loss: 0.0030
  training_time: 417
  overall_success_rate: 47.5
```

---

## 3. 指标记录

### 3.1 训练指标

```python
def save_training_metrics(exp_dir, metrics):
    """保存训练指标"""
    import json

    metrics_data = {
        'total_params': metrics['total_params'],
        'best_val_loss': metrics['best_val_loss'],
        'final_train_loss': metrics['final_train_loss'],
        'epochs': metrics['epochs'],
        'device': str(metrics['device']),
        'training_time': metrics['training_time'],
    }

    with open(f"{exp_dir}/logs/metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
```

### 3.2 评估指标

```python
def save_eval_results(exp_dir, results):
    """保存评估结果"""
    import json

    with open(f"{exp_dir}/logs/eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)
```

### 3.3 指标可视化

```python
def plot_training_curve(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
```

---

## 4. 迭代优化策略

### 4.1 数据优化

```
策略1：增加数据量
- 更多训练样本
- 更多样化的场景

策略2：数据增强
- 图像噪声
- 状态扰动
- 随机裁剪

策略3：平衡数据集
- 增加弱任务数据
- 任务重采样
```

### 4.2 模型优化

```
策略1：增加模型容量
- 更大的嵌入维度
- 更深的网络

策略2：添加注意力机制
- 通道注意力
- 空间注意力
- 交叉注意力

策略3：改进架构
- 残差连接
- 多尺度特征
- 时序融合
```

### 4.3 训练优化

```
策略1：学习率调整
- 预热+衰减
- 余弦退火
- 自适应学习率

策略2：正则化
- Dropout
- 权重衰减
- 梯度裁剪

策略3：训练策略
- 课程学习
- 渐进式训练
- 多任务学习
```

---

## 5. 实验记录模板

### 实验报告模板

```markdown
# 实验报告：迭代 X

## 实验目标
- [描述本次实验要解决的问题]

## 实验配置
- 模型：[模型名称]
- 数据：[数据集描述]
- 超参数：[关键超参数]

## 实验结果
| 指标 | 迭代X-1 | 迭代X | 改进 |
|------|---------|-------|------|
| 验证损失 | 0.005 | 0.003 | -40% |
| 成功率 | 30% | 45% | +50% |

## 分析
- [分析结果]
- [成功/失败原因]

## 下一步
- [后续改进计划]
```

---

## 6. 常见问题

### Q: 实验结果不一致？
A: 检查随机种子设置，确保数据划分一致

### Q: 无法确定改进来源？
A: 每次只改一个变量，记录详细配置

### Q: 训练时间太长？
A: 使用GPU，减小数据量，简化模型

### Q: 内存不足？
A: 减小批大小，使用梯度累积

---

## 7. 工具推荐

### 实验管理工具

```
1. Weights & Biases (wandb)
   - 实验跟踪
   - 指标可视化
   - 团队协作

2. MLflow
   - 实验管理
   - 模型版本控制
   - 部署管理

3. TensorBoard
   - 训练可视化
   - 图表生成
   - 实时监控

4. 自定义脚本
   - 灵活可控
   - 无需额外依赖
   - 本项目使用
```

---

*下一节：[实验结果分析](02_Result_Analysis.md)*
