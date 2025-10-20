# 快速开始指南

## 第一步：安装依赖

```bash
pip install -r requirements.txt
```

## 第二步：配置数据路径

编辑 `config.py` 文件，修改数据文件路径：

```python
DATA_PATH = r"E:\test_data\no.9\01_train_set\5t_opamp\source\pretrain_d_t.xlsx"
```

确保路径指向您的Excel数据文件。

## 第三步：训练模型

### 方式1：使用简化脚本（推荐）

```bash
python train_simple.py
```

这是最简单的方式，所有参数都在 `config.py` 中配置。

### 方式2：使用完整脚本

```bash
python train_circuit_model.py
```

这个脚本包含更多细节和注释。

## 第四步：查看结果

训练完成后，会生成以下文件：

1. **best_model.pth** - 最佳模型（验证损失最低）
2. **final_circuit_model.pth** - 最终模型
3. **training_history.png** - 训练曲线图

## 第五步：使用模型预测

### 单个样本预测

```python
from predict import predict_circuit_outputs

# 7个输入参数
input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

# 预测
output = predict_circuit_outputs("best_model.pth", input_data)
print(f"预测输出: {output[0]}")
```

### 批量预测

```python
from predict import batch_predict_from_excel

batch_predict_from_excel(
    model_path="best_model.pth",
    input_file="your_input.xlsx",  # 包含7列输入参数
    output_file="predictions.xlsx"  # 保存预测结果
)
```

## 调整模型参数

在 `config.py` 中可以调整以下参数：

### 网络结构
```python
HIDDEN_DIMS = [128, 256, 512, 512, 256, 128]  # 六个隐藏层的维度
DROPOUT_RATE = 0.2  # Dropout比率
```

### 训练参数
```python
EPOCHS = 200  # 训练轮数
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32  # 批次大小
PATIENCE = 20  # 早停耐心值
```

### 数据分割
```python
TEST_SIZE = 0.2  # 测试集比例（20%）
VAL_SIZE = 0.1  # 验证集比例（10%）
```

## 常见问题

### Q1: 找不到数据文件
**A:** 检查 `config.py` 中的 `DATA_PATH` 是否正确，注意Windows路径使用 `r"path"` 格式。

### Q2: 训练很慢
**A:** 
- 如果有GPU，确保安装了CUDA版本的PyTorch
- 可以减小 `BATCH_SIZE` 或 `HIDDEN_DIMS` 的值

### Q3: 模型过拟合
**A:** 
- 增加 `DROPOUT_RATE`（如0.3-0.5）
- 减小模型大小（减小 `HIDDEN_DIMS`）
- 增加训练数据

### Q4: 模型欠拟合
**A:** 
- 增加模型大小（增大 `HIDDEN_DIMS`）
- 增加训练轮数 `EPOCHS`
- 降低 `DROPOUT_RATE`

## 模型性能指标说明

- **MSE (均方误差)**: 越小越好
- **RMSE (均方根误差)**: 与输出参数同量纲，越小越好
- **MAE (平均绝对误差)**: 平均预测误差，越小越好
- **R² 分数**: 接近1表示拟合很好，接近0表示拟合较差

## 下一步

当您完成第一种电路的训练后，可以：
1. 调整参数优化模型性能
2. 准备第二种电路的数据和模型
3. 实现反向预测（输出参数→输入参数）

## 技术支持

如果遇到问题，请检查：
1. Python版本（建议3.7+）
2. PyTorch版本（建议1.10+）
3. 数据文件格式（Excel .xlsx格式，至少12列）
4. 数据中没有缺失值或异常值
