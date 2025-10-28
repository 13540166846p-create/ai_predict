# 双向电路参数预测模型

这是一个用于电路输入输出参数双向预测的深度学习模型，不涉及电路仿真，仅进行数据拟合。

## 功能特点

- **双向预测**：
  - 正向预测：从输入参数（前7列）预测输出参数（后5列）
  - 反向预测：从输出参数（后5列）预测输入参数（前7列）

- **深度神经网络**：使用多层感知机（MLP）架构，包含批归一化和Dropout正则化

- **自动化训练**：包含完整的训练、验证、评估流程

## 环境要求

Python 3.8+

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据格式

数据文件应为Excel格式（.xlsx），结构如下：
- 前7列：输入参数
- 后5列：输出参数

示例数据文件：`pretrain_d_t.xlsx`

## 使用方法

### 1. 准备数据

将数据文件 `pretrain_d_t.xlsx` 放置在项目根目录，或修改配置文件中的路径。

### 2. 训练模型

```bash
python bidirectional_circuit_predictor.py
```

训练完成后会生成：
- `models/forward_model.pth` - 正向预测模型
- `models/backward_model.pth` - 反向预测模型
- `models/scalers.pkl` - 数据标准化器
- `training_curves.png` - 训练曲线图
- `best_forward_model.pth` - 最佳正向模型
- `best_backward_model.pth` - 最佳反向模型

### 3. 使用已训练模型进行预测

```python
from bidirectional_circuit_predictor import BidirectionalPredictor
import numpy as np

# 创建预测器并加载模型
predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
predictor.load_models('models')

# 正向预测：输入 -> 输出
input_params = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
output_pred = predictor.predict_forward(input_params)
print("预测输出:", output_pred)

# 反向预测：输出 -> 输入
output_params = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
input_pred = predictor.predict_backward(output_params)
print("预测输入:", input_pred)
```

## 模型架构

- **输入层**：7维（输入参数）或 5维（输出参数）
- **隐藏层**：128 -> 256 -> 256 -> 128（可配置）
- **激活函数**：ReLU
- **正则化**：Batch Normalization + Dropout (0.2)
- **输出层**：5维（输出参数）或 7维（输入参数）

## 配置文件

可以通过修改 `config.py` 来调整模型参数：
- 数据路径和划分比例
- 模型结构（隐藏层维度、Dropout等）
- 训练参数（学习率、批次大小、训练轮数等）

## 性能指标

模型在训练完成后会自动输出以下评估指标：
- R² Score（决定系数）
- MAE（平均绝对误差）
- RMSE（均方根误差）

## 注意事项

1. 确保数据文件中没有缺失值
2. 数据会自动进行标准化处理
3. 训练过程中会自动保存最佳模型
4. 支持GPU加速（如果可用）

## 文件说明

- `bidirectional_circuit_predictor.py` - 主程序文件
- `config.py` - 配置文件
- `requirements.txt` - 依赖包列表
- `README.md` - 项目说明文档

## 数据路径配置

如果数据文件在其他位置，可以通过以下方式指定：

1. 修改 `config.py` 中的 `DATA_CONFIG['file_path']`
2. 或在代码中直接指定路径：
```python
predictor.load_data('your/path/to/pretrain_d_t.xlsx')
```

支持的路径格式：
- 相对路径：`./data/pretrain_d_t.xlsx`
- 绝对路径：`/path/to/pretrain_d_t.xlsx`
- Windows路径会自动转换：`E:\\test_data\\...` 
