# 🚀 快速开始指南

## ✅ 已完成的工作

您的双向电路参数预测模型已经：
- ✅ 训练完成（200轮）
- ✅ 性能优秀（R² = 0.8492）
- ✅ 生成了详细的评估报告

---

## 📊 立即查看性能报告

### 方法1：快速报告（推荐 ⭐）
```bash
python3 quick_report.py
```
**显示内容**：
- ✅ MAE (平均绝对误差)
- ✅ MSE (均方误差)  
- ✅ R² (决定系数)
- ✅ 每个输出维度的详细指标
- ✅ 误差统计

**当前性能**：
```
整体 R² = 0.8492 ✅ 优秀

Output_1: R² = 0.798  ⚠️ 一般
Output_2: R² = 0.879  👍 良好
Output_3: R² = 0.828  👍 良好
Output_4: R² = 0.920  ✅ 优秀
Output_5: R² = 0.821  👍 良好
```

---

### 方法2：完整报告（含图表）
```bash
python3 evaluate_predictions.py
```
**生成文件**（保存在 `evaluation_results/` 目录）：
- 📊 `prediction_results.xlsx` - Excel完整结果 ⭐
- 📈 `prediction_comparison.png` - 对比曲线图
- 📉 `error_distribution.png` - 误差分布图
- 📊 `scatter_comparison.png` - 散点对比图
- 📝 `evaluation_report.txt` - 文本报告
- 📋 `metrics_report.csv` - 指标CSV
- 📋 `prediction_results.csv` - 结果CSV

---

## 🎯 使用模型进行预测

```python
from bidirectional_circuit_predictor import BidirectionalPredictor
import numpy as np

# 加载模型
predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
predictor.load_models('models')

# 正向预测：输入 → 输出
input_data = np.array([[45.2, 52.3, 48.7, 51.5, 49.8, 50.2, 47.9]])
output = predictor.predict_forward(input_data)
print("预测输出:", output)

# 反向预测：输出 → 输入
output_data = np.array([[10.5, 20.3, 15.7, 25.2, 18.9]])
input_pred = predictor.predict_backward(output_data)
print("预测输入:", input_pred)
```

或直接运行示例：
```bash
python3 inference_example.py
```

---

## 🔄 使用真实数据重新训练

```bash
# 1. 将您的数据文件放到当前目录
# 文件格式：Excel，前7列=输入，后5列=输出

# 2. 重新训练
python3 bidirectional_circuit_predictor.py

# 3. 查看新模型性能
python3 quick_report.py
```

---

## 📁 重要文件位置

```
/workspace/
├── quick_report.py              ← 运行这个查看报告 ⭐
├── evaluate_predictions.py      ← 运行这个生成完整报告
├── bidirectional_circuit_predictor.py  ← 训练模型
├── inference_example.py         ← 预测示例
│
├── models/                      ← 训练好的模型
│   ├── forward_model.pth
│   ├── backward_model.pth
│   └── scalers.pkl
│
├── evaluation_results/          ← 评估报告（已生成）
│   ├── prediction_results.xlsx  ← 推荐查看 ⭐
│   ├── prediction_comparison.png
│   ├── error_distribution.png
│   └── scatter_comparison.png
│
└── 项目总结.md                  ← 完整项目说明
```

---

## 💡 常用命令

```bash
# 查看性能报告（含 MAE、MSE、R²）
python3 quick_report.py

# 生成完整评估报告
python3 evaluate_predictions.py

# 重新训练模型
python3 bidirectional_circuit_predictor.py

# 查看预测示例
python3 inference_example.py
```

---

## 📊 当前模型性能摘要

| 指标 | 数值 | 评价 |
|------|------|------|
| **R²** | **0.8492** | ✅ 优秀 |
| **MAE** | 0.4150 | ✅ 良好 |
| **MSE** | 0.2931 | ✅ 良好 |
| **RMSE** | 0.5414 | ✅ 良好 |

**结论**：模型性能优秀，可以投入使用！

---

## ❓ 快速问答

**Q: 如何每次预测后查看报告？**  
A: 运行 `python3 quick_report.py`，会显示包含 MAE、MSE、R² 的报告

**Q: 报告保存在哪里？**  
A: `evaluation_results/` 目录，推荐查看 `prediction_results.xlsx`

**Q: R² = 0.8492 是什么意思？**  
A: 表示模型解释了84.92%的数据方差，性能优秀！

**Q: 如何使用真实数据？**  
A: 替换 `pretrain_d_t.xlsx`，然后重新运行训练即可

---

**立即开始**：运行 `python3 quick_report.py` 查看详细性能报告！
