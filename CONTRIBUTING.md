# 贡献指南

感谢您对DroneVLA项目的关注！我们欢迎各种形式的贡献。

---

## 如何贡献

### 报告Bug

1. 使用 [GitHub Issues](https://github.com/ZHL-max/DroneVLA/issues) 报告
2. 描述清楚复现步骤
3. 包含系统环境信息
4. 附上错误日志

### 提出建议

1. 使用 [GitHub Discussions](https://github.com/ZHL-max/DroneVLA/discussions) 讨论
2. 说明使用场景
3. 描述期望的行为

### 提交代码

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

---

## 开发环境

```bash
# 克隆项目
git clone https://github.com/ZHL-max/DroneVLA.git
cd DroneVLA

# 创建环境
conda create -n dronevla-dev python=3.10 -y
conda activate dronevla-dev

# 安装依赖
pip install -e .
pip install pytest black flake8

# 运行测试
python tests/run_tests.py
```

---

## 代码规范

### Python风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/)
- 使用 [Black](https://github.com/psf/black) 格式化
- 使用 [Flake8](https://flake8.pycqa.org/) 检查

### 提交信息

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat(models): add diffusion policy decoder

- Implement DDPM-based action decoder
- Support variable action horizons
- Add training configuration

Closes #42
```

---

## 测试

### 运行测试

```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定测试
pytest tests/test_models.py -v
```

### 编写测试

```python
def test_new_feature():
    """测试新功能"""
    # 准备
    model = MyModel()

    # 执行
    output = model(input)

    # 验证
    assert output.shape == expected_shape
```

---

## 文档

### 更新文档

- 重要变更必须更新README.md
- 新功能需要添加使用示例
- API变更需要更新docstring

### 文档结构

```
docs/
├── VLA_Learning_Notes.md      # 学习笔记
├── World_Model_Survey.md      # 综述总结
├── Multi_Platform_Installation.md  # 安装手册
├── Hardware_Connection_Guide.md    # 硬件指南
├── Datasets_and_Training.md   # 训练指南
├── Real_World_Deployment.md   # 部署指南
└── Project_Roadmap.md         # 项目路线图
```

---

## Pull Request流程

1. 确保代码通过所有测试
2. 更新相关文档
3. 添加变更日志
4. 请求代码审查
5. 根据反馈修改

### PR模板

```markdown
## 描述
简要描述此PR的变更

## 变更类型
- [ ] 新功能
- [ ] Bug修复
- [ ] 文档更新
- [ ] 重构

## 测试
- [ ] 已运行现有测试
- [ ] 已添加新测试
- [ ] 所有测试通过

## 截图（如适用）
添加相关截图

## 相关Issue
Closes #<issue_number>
```

---

## 行为准则

- 尊重所有参与者
- 接受建设性批评
- 专注于对社区最有利的事情
- 对他人表示同理心

---

## 联系方式

- GitHub Issues: 报告Bug和功能请求
- GitHub Discussions: 一般讨论和问题

---

感谢您的贡献！
