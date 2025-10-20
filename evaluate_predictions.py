"""
模型预测评估脚本
使用训练数据中的输入预测输出，并与真实输出比较
生成包含MAE、MSE、R²的详细报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bidirectional_circuit_predictor import BidirectionalPredictor
import os
from datetime import datetime


def generate_evaluation_report(predictor, data_file='pretrain_d_t.xlsx', output_dir='evaluation_results'):
    """
    生成完整的评估报告
    
    :param predictor: 训练好的预测器
    :param data_file: 数据文件路径
    :param output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("电路参数预测模型评估报告")
    print("="*80)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据文件: {data_file}")
    print("="*80)
    
    # 加载数据
    print("\n正在加载数据...")
    X, y = predictor.load_data(data_file, input_cols=7, output_cols=5)
    
    # 使用模型进行预测
    print("\n正在进行预测...")
    y_pred = predictor.predict_forward(X)
    
    print(f"\n预测完成!")
    print(f"  - 样本数量: {len(X)}")
    print(f"  - 输入维度: {X.shape[1]}")
    print(f"  - 输出维度: {y.shape[1]}")
    
    # 计算整体指标
    print("\n" + "="*80)
    print("整体性能指标")
    print("="*80)
    
    mae_overall = mean_absolute_error(y, y_pred)
    mse_overall = mean_squared_error(y, y_pred)
    rmse_overall = np.sqrt(mse_overall)
    r2_overall = r2_score(y, y_pred)
    
    print(f"\n总体性能:")
    print(f"  MAE  (平均绝对误差):     {mae_overall:.6f}")
    print(f"  MSE  (均方误差):         {mse_overall:.6f}")
    print(f"  RMSE (均方根误差):       {rmse_overall:.6f}")
    print(f"  R²   (决定系数):         {r2_overall:.6f}")
    
    # 计算每个输出维度的指标
    print("\n" + "="*80)
    print("各输出维度详细指标")
    print("="*80)
    
    output_metrics = []
    for i in range(y.shape[1]):
        mae = mean_absolute_error(y[:, i], y_pred[:, i])
        mse = mean_squared_error(y[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y[:, i], y_pred[:, i])
        
        output_metrics.append({
            'Output': f'Output_{i+1}',
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        })
        
        print(f"\nOutput_{i+1}:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.6f}")
    
    # 保存详细指标到CSV
    metrics_df = pd.DataFrame(output_metrics)
    metrics_file = os.path.join(output_dir, 'metrics_report.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n详细指标已保存至: {metrics_file}")
    
    # 计算误差统计
    print("\n" + "="*80)
    print("预测误差统计")
    print("="*80)
    
    errors = y - y_pred
    print(f"\n误差统计:")
    print(f"  最大正误差: {np.max(errors):.6f}")
    print(f"  最大负误差: {np.min(errors):.6f}")
    print(f"  误差标准差: {np.std(errors):.6f}")
    print(f"  误差均值:   {np.mean(errors):.6f}")
    
    # 保存预测结果
    print("\n" + "="*80)
    print("保存预测结果")
    print("="*80)
    
    # 创建结果DataFrame
    results_data = {}
    
    # 添加输入列
    for i in range(X.shape[1]):
        results_data[f'Input_{i+1}'] = X[:, i]
    
    # 添加真实输出列
    for i in range(y.shape[1]):
        results_data[f'True_Output_{i+1}'] = y[:, i]
    
    # 添加预测输出列
    for i in range(y_pred.shape[1]):
        results_data[f'Pred_Output_{i+1}'] = y_pred[:, i]
    
    # 添加误差列
    for i in range(y.shape[1]):
        results_data[f'Error_Output_{i+1}'] = y[:, i] - y_pred[:, i]
    
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(output_dir, 'prediction_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n完整预测结果已保存至: {results_file}")
    
    # 保存Excel格式（更易读）
    excel_file = os.path.join(output_dir, 'prediction_results.xlsx')
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='预测结果', index=False)
        metrics_df.to_excel(writer, sheet_name='性能指标', index=False)
    print(f"Excel格式结果已保存至: {excel_file}")
    
    # 绘制对比图
    print("\n" + "="*80)
    print("生成可视化图表")
    print("="*80)
    
    plot_prediction_comparison(y, y_pred, output_dir)
    plot_error_distribution(errors, output_dir)
    plot_scatter_comparison(y, y_pred, output_dir)
    
    # 生成文本报告
    generate_text_report(mae_overall, mse_overall, rmse_overall, r2_overall, 
                        output_metrics, errors, output_dir)
    
    print("\n" + "="*80)
    print("评估完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  1. {metrics_file}")
    print(f"  2. {results_file}")
    print(f"  3. {excel_file}")
    print(f"  4. {os.path.join(output_dir, 'prediction_comparison.png')}")
    print(f"  5. {os.path.join(output_dir, 'error_distribution.png')}")
    print(f"  6. {os.path.join(output_dir, 'scatter_comparison.png')}")
    print(f"  7. {os.path.join(output_dir, 'evaluation_report.txt')}")
    print("\n" + "="*80)


def plot_prediction_comparison(y_true, y_pred, output_dir):
    """绘制真实值vs预测值对比图"""
    n_outputs = y_true.shape[1]
    n_samples = min(100, y_true.shape[0])  # 只显示前100个样本
    
    fig, axes = plt.subplots(n_outputs, 1, figsize=(15, 3*n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        axes[i].plot(y_true[:n_samples, i], 'b-', label='True Value', linewidth=2, alpha=0.7)
        axes[i].plot(y_pred[:n_samples, i], 'r--', label='Predicted Value', linewidth=2, alpha=0.7)
        axes[i].set_xlabel('Sample Index')
        axes[i].set_ylabel(f'Output_{i+1}')
        axes[i].set_title(f'Output_{i+1}: True vs Predicted (First {n_samples} samples)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'prediction_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 预测对比图已保存")


def plot_error_distribution(errors, output_dir):
    """绘制误差分布图"""
    n_outputs = errors.shape[1]
    
    fig, axes = plt.subplots(1, n_outputs, figsize=(5*n_outputs, 4))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        axes[i].hist(errors[:, i], bins=50, edgecolor='black', alpha=0.7)
        axes[i].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[i].set_xlabel('Prediction Error')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Output_{i+1} Error Distribution')
        axes[i].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_err = np.mean(errors[:, i])
        std_err = np.std(errors[:, i])
        axes[i].text(0.02, 0.98, f'Mean: {mean_err:.4f}\nStd: {std_err:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 误差分布图已保存")


def plot_scatter_comparison(y_true, y_pred, output_dir):
    """绘制散点图对比"""
    n_outputs = y_true.shape[1]
    
    fig, axes = plt.subplots(1, n_outputs, figsize=(5*n_outputs, 4))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
        
        # 添加理想线 (y=x)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        
        axes[i].set_xlabel('True Value')
        axes[i].set_ylabel('Predicted Value')
        axes[i].set_title(f'Output_{i+1}: True vs Predicted')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # 添加R²值
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        axes[i].text(0.02, 0.98, f'R² = {r2:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'scatter_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 散点对比图已保存")


def generate_text_report(mae, mse, rmse, r2, output_metrics, errors, output_dir):
    """生成文本格式的评估报告"""
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("电路参数预测模型 - 详细评估报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("【整体性能指标】\n")
        f.write("-"*80 + "\n")
        f.write(f"MAE  (平均绝对误差):     {mae:.6f}\n")
        f.write(f"MSE  (均方误差):         {mse:.6f}\n")
        f.write(f"RMSE (均方根误差):       {rmse:.6f}\n")
        f.write(f"R²   (决定系数):         {r2:.6f}\n")
        f.write("\n")
        
        f.write("【各输出维度详细指标】\n")
        f.write("-"*80 + "\n")
        f.write(f"{'输出':<12} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12}\n")
        f.write("-"*80 + "\n")
        for metric in output_metrics:
            f.write(f"{metric['Output']:<12} {metric['MAE']:<12.6f} {metric['MSE']:<12.6f} "
                   f"{metric['RMSE']:<12.6f} {metric['R²']:<12.6f}\n")
        f.write("\n")
        
        f.write("【误差统计】\n")
        f.write("-"*80 + "\n")
        f.write(f"最大正误差: {np.max(errors):.6f}\n")
        f.write(f"最大负误差: {np.min(errors):.6f}\n")
        f.write(f"误差标准差: {np.std(errors):.6f}\n")
        f.write(f"误差均值:   {np.mean(errors):.6f}\n")
        f.write("\n")
        
        f.write("【各输出维度误差统计】\n")
        f.write("-"*80 + "\n")
        for i in range(errors.shape[1]):
            f.write(f"\nOutput_{i+1}:\n")
            f.write(f"  均值:   {np.mean(errors[:, i]):.6f}\n")
            f.write(f"  标准差: {np.std(errors[:, i]):.6f}\n")
            f.write(f"  最小值: {np.min(errors[:, i]):.6f}\n")
            f.write(f"  最大值: {np.max(errors[:, i]):.6f}\n")
            f.write(f"  中位数: {np.median(errors[:, i]):.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ 文本报告已保存")


def main():
    """主函数"""
    # 检查模型是否存在
    if not os.path.exists('models/forward_model.pth'):
        print("错误: 找不到训练好的模型!")
        print("请先运行 'python3 bidirectional_circuit_predictor.py' 训练模型")
        return
    
    # 加载模型
    print("正在加载模型...")
    predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
    predictor.load_models('models')
    print("模型加载成功!\n")
    
    # 生成评估报告
    generate_evaluation_report(predictor, data_file='pretrain_d_t.xlsx')


if __name__ == "__main__":
    main()
