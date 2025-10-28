"""
快速评估报告生成器
每次运行都会生成包含MAE、MSE、R²的简洁报告
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bidirectional_circuit_predictor import BidirectionalPredictor
from datetime import datetime
import os


def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)


def quick_evaluation_report(data_file='pretrain_d_t.xlsx'):
    """
    快速生成评估报告
    """
    # 加载模型
    print("\n正在加载模型...")
    predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
    predictor.load_models('models')
    
    # 加载数据
    print("正在加载数据并预测...")
    X, y = predictor.load_data(data_file, input_cols=7, output_cols=5)
    
    # 预测
    y_pred = predictor.predict_forward(X)
    
    # 打印报告
    print_separator()
    print("📊 电路参数预测模型 - 性能评估报告")
    print_separator()
    print(f"⏰ 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 数据文件: {data_file}")
    print(f"📈 样本数量: {len(X)}")
    print_separator()
    
    # 计算整体指标
    mae_overall = mean_absolute_error(y, y_pred)
    mse_overall = mean_squared_error(y, y_pred)
    rmse_overall = np.sqrt(mse_overall)
    r2_overall = r2_score(y, y_pred)
    
    print("\n【整体性能指标】")
    print_separator("-")
    print(f"  MAE  (平均绝对误差):  {mae_overall:.6f}")
    print(f"  MSE  (均方误差):      {mse_overall:.6f}")
    print(f"  RMSE (均方根误差):    {rmse_overall:.6f}")
    print(f"  R²   (决定系数):      {r2_overall:.6f} {'✅ 优秀' if r2_overall > 0.8 else '⚠️ 一般' if r2_overall > 0.6 else '❌ 较差'}")
    
    # 各输出维度指标
    print("\n【各输出维度详细指标】")
    print_separator("-")
    print(f"{'输出':<12} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12} {'评价':<8}")
    print_separator("-")
    
    for i in range(y.shape[1]):
        mae = mean_absolute_error(y[:, i], y_pred[:, i])
        mse = mean_squared_error(y[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y[:, i], y_pred[:, i])
        
        # 评价
        if r2 > 0.9:
            rating = "✅ 优秀"
        elif r2 > 0.8:
            rating = "👍 良好"
        elif r2 > 0.6:
            rating = "⚠️ 一般"
        else:
            rating = "❌ 较差"
        
        print(f"Output_{i+1:<5} {mae:<12.6f} {mse:<12.6f} {rmse:<12.6f} {r2:<12.6f} {rating}")
    
    # 误差统计
    errors = y - y_pred
    print("\n【预测误差统计】")
    print_separator("-")
    print(f"  误差均值:     {np.mean(errors):>10.6f}")
    print(f"  误差标准差:   {np.std(errors):>10.6f}")
    print(f"  最大正误差:   {np.max(errors):>10.6f}")
    print(f"  最大负误差:   {np.min(errors):>10.6f}")
    print(f"  误差中位数:   {np.median(errors):>10.6f}")
    
    # 每个输出的误差范围
    print("\n【各输出维度误差范围】")
    print_separator("-")
    for i in range(y.shape[1]):
        err_min = np.min(errors[:, i])
        err_max = np.max(errors[:, i])
        err_mean = np.mean(errors[:, i])
        err_std = np.std(errors[:, i])
        print(f"  Output_{i+1}: [{err_min:>8.4f}, {err_max:>8.4f}]  均值={err_mean:>7.4f}  标准差={err_std:>7.4f}")
    
    # 性能总结
    print("\n【性能总结】")
    print_separator("-")
    excellent_count = sum(1 for i in range(y.shape[1]) if r2_score(y[:, i], y_pred[:, i]) > 0.9)
    good_count = sum(1 for i in range(y.shape[1]) if 0.8 < r2_score(y[:, i], y_pred[:, i]) <= 0.9)
    fair_count = sum(1 for i in range(y.shape[1]) if 0.6 < r2_score(y[:, i], y_pred[:, i]) <= 0.8)
    poor_count = sum(1 for i in range(y.shape[1]) if r2_score(y[:, i], y_pred[:, i]) <= 0.6)
    
    print(f"  优秀 (R²>0.9):  {excellent_count}/{y.shape[1]} 个输出")
    print(f"  良好 (0.8<R²≤0.9): {good_count}/{y.shape[1]} 个输出")
    print(f"  一般 (0.6<R²≤0.8): {fair_count}/{y.shape[1]} 个输出")
    print(f"  较差 (R²≤0.6):  {poor_count}/{y.shape[1]} 个输出")
    
    if r2_overall > 0.8:
        print(f"\n  总体评价: ✅ 模型性能优秀 (R² = {r2_overall:.4f})")
    elif r2_overall > 0.6:
        print(f"\n  总体评价: ⚠️ 模型性能一般 (R² = {r2_overall:.4f})")
    else:
        print(f"\n  总体评价: ❌ 模型性能较差 (R² = {r2_overall:.4f})")
    
    print_separator()
    print(f"\n💡 提示: 运行 'python3 evaluate_predictions.py' 可生成详细报告和可视化图表")
    print_separator()
    
    return {
        'MAE': mae_overall,
        'MSE': mse_overall,
        'RMSE': rmse_overall,
        'R2': r2_overall
    }


def main():
    """主函数"""
    if not os.path.exists('models/forward_model.pth'):
        print("\n❌ 错误: 找不到训练好的模型!")
        print("请先运行: python3 bidirectional_circuit_predictor.py")
        return
    
    quick_evaluation_report()


if __name__ == "__main__":
    main()
