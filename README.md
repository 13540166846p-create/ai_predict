# 双向电路预测模型

## 📊 项目简介
这是一个基于深度学习的双向电路参数预测模型，用于预测电路的输入输出参数关系。模型采用六个隐藏层的神经网络架构，能够有效拟合电路参数之间的非线性关系。

## 🎯 模型特性

### 第一种电路：5T运放电路
- **输入参数**: 7个电路设计参数（Excel文件前7列）
- **输出参数**: 5个电路性能指标（Excel文件后5列）
- **网络结构**: 六个隐藏层 [128, 256, 512, 512, 256, 128]
- **激活函数**: ReLU
- **正则化**: Batch Normalization + Dropout (0.2)

## 📁 项目结构

```
/workspace/
├── config.py                    # 配置文件（参数设置）
├── train_circuit_model.py       # 完整训练脚本
├── train_simple.py              # 简化训练脚本（推荐）
├── predict.py                   # 预测脚本
├── visualize.py                 # 可视化脚本
├── requirements.txt             # 依赖包列表
├── README.md                    # 项目说明（本文件）
└── QUICKSTART.md                # 快速入门指南
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

编辑 `config.py`，修改数据文件路径：

```python
DATA_PATH = r"E:\test_data\no.9\01_train_set\5t_opamp\source\pretrain_d_t.xlsx"
```

### 3. 训练模型

**推荐方式（使用配置文件）：**
```bash
python train_simple.py
```

**或使用完整脚本：**
```bash
python train_circuit_model.py
```

### 4. 查看结果

训练完成后会生成：
- `best_model.pth` - 最佳模型
- `final_circuit_model.pth` - 最终模型
- `training_history.png` - 训练曲线

### 5. 可视化分析

```bash
python visualize.py
```

这将生成：
- `prediction_comparison.png` - 预测值vs真实值
- `error_distribution.png` - 误差分布
- `feature_importance.png` - 特征重要性

## 🔧 使用说明

### 训练模型

#### 方式1：简化脚本（推荐新手）
```python
python train_simple.py
```
- 所有参数在 `config.py` 中配置
- 自动处理数据加载、训练、评估
- 适合快速实验

#### 方式2：完整脚本（可定制）
```python
python train_circuit_model.py
```
- 包含详细的注释和说明
- 可以直接修改代码进行深度定制
- 适合研究和开发

### 使用模型预测

#### 单个样本预测
```python
from predict import predict_circuit_outputs

# 7个输入参数
input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

# 预测5个输出参数
output = predict_circuit_outputs("best_model.pth", input_data)
print(f"预测输出: {output[0]}")
```

#### 批量预测（从Excel）
```python
from predict import batch_predict_from_excel

batch_predict_from_excel(
    model_path="best_model.pth",
    input_file="your_input.xlsx",    # 包含7列输入参数
    output_file="predictions.xlsx"   # 保存预测结果
)
```

#### 交互式预测
```python
from predict import interactive_predict

interactive_predict("best_model.pth")
# 然后输入7个参数，用空格分隔
```

## ⚙️ 参数配置

在 `config.py` 中可以调整：

### 网络结构
```python
INPUT_DIM = 7                                    # 输入维度
OUTPUT_DIM = 5                                   # 输出维度
HIDDEN_DIMS = [128, 256, 512, 512, 256, 128]    # 六个隐藏层
DROPOUT_RATE = 0.2                              # Dropout比率
```

### 训练参数
```python
EPOCHS = 200           # 训练轮数
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32        # 批次大小
PATIENCE = 20          # 早停耐心值
```

### 数据分割
```python
TEST_SIZE = 0.2   # 测试集比例（20%）
VAL_SIZE = 0.1    # 验证集比例（10%）
```

## 📊 模型架构

```
输入层 (7) 
    ↓
隐藏层1 (128) + BatchNorm + ReLU + Dropout(0.2)
    ↓
隐藏层2 (256) + BatchNorm + ReLU + Dropout(0.2)
    ↓
隐藏层3 (512) + BatchNorm + ReLU + Dropout(0.2)
    ↓
隐藏层4 (512) + BatchNorm + ReLU + Dropout(0.2)
    ↓
隐藏层5 (256) + BatchNorm + ReLU + Dropout(0.2)
    ↓
隐藏层6 (128) + BatchNorm + ReLU + Dropout(0.2)
    ↓
输出层 (5)
```

**总参数数**: 约600K+（具体取决于配置）

## 📈 评估指标

| 指标 | 说明 | 理想值 |
|-----|------|--------|
| **MSE** | 均方误差 | 越小越好 |
| **RMSE** | 均方根误差 | 越小越好 |
| **MAE** | 平均绝对误差 | 越小越好 |
| **R²** | 拟合优度 | 接近1 |

## 🎨 可视化功能

使用 `visualize.py` 生成以下图表：

1. **预测对比图**: 每个输出参数的预测值vs真实值散点图
2. **误差分布图**: 相对误差的直方图分布
3. **特征重要性**: 输入参数对输出参数的影响热图

## ✨ 训练特性

| 特性 | 说明 |
|-----|------|
| **数据归一化** | StandardScaler，自动保存和加载 |
| **早停机制** | 验证损失N个epoch不降则停止 |
| **学习率调度** | 自适应降低学习率 |
| **批归一化** | 加速训练，提高稳定性 |
| **Dropout** | 防止过拟合 |
| **GPU支持** | 自动检测并使用CUDA |

## 📋 数据要求

- **格式**: Excel (.xlsx)
- **结构**: 前7列为输入，后5列为输出
- **质量**: 无缺失值，无异常值
- **数量**: 建议至少几百个样本

## 🔍 常见问题

<details>
<summary>Q1: 找不到数据文件怎么办？</summary>

检查 `config.py` 中的 `DATA_PATH` 是否正确。注意Windows路径需要使用 `r"path"` 格式或双反斜杠。
</details>

<details>
<summary>Q2: 训练很慢怎么办？</summary>

- 如果有GPU，确保安装了CUDA版本的PyTorch
- 减小 `BATCH_SIZE` 或 `HIDDEN_DIMS`
- 减少 `EPOCHS` 数量
</details>

<details>
<summary>Q3: 模型过拟合怎么办？</summary>

- 增加 `DROPOUT_RATE`（如0.3-0.5）
- 减小模型大小（减小 `HIDDEN_DIMS`）
- 增加训练数据
- 提前停止训练
</details>

<details>
<summary>Q4: 模型欠拟合怎么办？</summary>

- 增加模型大小（增大 `HIDDEN_DIMS`）
- 增加训练轮数 `EPOCHS`
- 降低 `DROPOUT_RATE`
- 调整学习率
</details>

## 🛠️ 技术栈

| 技术 | 用途 |
|-----|------|
| **PyTorch** | 深度学习框架 |
| **Pandas** | 数据处理 |
| **Scikit-learn** | 数据预处理和评估 |
| **Matplotlib** | 可视化 |
| **NumPy** | 数值计算 |
| **OpenPyXL** | Excel文件读写 |

## 📝 下一步计划

- [ ] 实现第二种电路的预测模型
- [ ] 实现真正的双向预测（输出→输入）
- [ ] 添加超参数自动调优
- [ ] 支持更多评估指标
- [ ] Web界面支持

## 📞 获取帮助

如果遇到问题：
1. 查看 `QUICKSTART.md` 快速入门指南
2. 检查 `config.py` 配置是否正确
3. 确认数据格式符合要求
4. 查看训练日志和错误信息

## 📄 许可证

本项目仅供学习和研究使用。
