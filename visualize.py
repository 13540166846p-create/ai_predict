"""
模型性能可视化脚本
用于分析和可视化训练好的模型性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from train_circuit_model import BidirectionalCircuitModel, CircuitTrainer
import config

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_prediction_comparison(model_path, test_data_path=None):
    """
    绘制预测值与真实值对比图
    
    Args:
        model_path: 模型文件路径
        test_data_path: 测试数据路径（如果为None，则使用训练数据）
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = BidirectionalCircuitModel(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS
    )
    trainer = CircuitTrainer(model, device)
    trainer.load_model(model_path)
    
    # 加载数据
    data_path = test_data_path if test_data_path else config.DATA_PATH
    X, y = trainer.load_data(data_path, config.INPUT_DIM, config.OUTPUT_DIM)
    
    # 预处理
    X_normalized = trainer.scaler_X.transform(X)
    y_normalized = trainer.scaler_y.transform(y)
    
    # 预测
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    model.eval()
    with torch.no_grad():
        y_pred_normalized = model(X_tensor).cpu().numpy()
    
    # 反归一化
    y_pred = trainer.scaler_y.inverse_transform(y_pred_normalized)
    y_true = y
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(config.OUTPUT_DIM, 6)):
        ax = axes[i]
        
        # 散点图
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
        
        # 理想预测线（y=x）
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')
        
        # 计算R²
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'输出参数 {i+1}\n(R² = {r2:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(config.OUTPUT_DIM, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("预测对比图已保存: prediction_comparison.png")
    plt.close()


def plot_error_distribution(model_path, test_data_path=None):
    """
    绘制预测误差分布图
    
    Args:
        model_path: 模型文件路径
        test_data_path: 测试数据路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = BidirectionalCircuitModel(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS
    )
    trainer = CircuitTrainer(model, device)
    trainer.load_model(model_path)
    
    # 加载数据
    data_path = test_data_path if test_data_path else config.DATA_PATH
    X, y = trainer.load_data(data_path, config.INPUT_DIM, config.OUTPUT_DIM)
    
    # 预测
    X_normalized = trainer.scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_normalized = model(X_tensor).cpu().numpy()
    
    y_pred = trainer.scaler_y.inverse_transform(y_pred_normalized)
    y_true = y
    
    # 计算误差
    errors = y_pred - y_true
    relative_errors = (errors / (y_true + 1e-8)) * 100  # 相对误差百分比
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(config.OUTPUT_DIM, 6)):
        ax = axes[i]
        
        # 绘制误差直方图
        ax.hist(relative_errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        
        # 添加统计信息
        mean_error = np.mean(relative_errors[:, i])
        std_error = np.std(relative_errors[:, i])
        
        ax.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'均值: {mean_error:.2f}%')
        ax.axvline(0, color='g', linestyle='-', linewidth=2, label='零误差')
        
        ax.set_xlabel('相对误差 (%)')
        ax.set_ylabel('频数')
        ax.set_title(f'输出参数 {i+1}\n(σ = {std_error:.2f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(config.OUTPUT_DIM, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("误差分布图已保存: error_distribution.png")
    plt.close()


def plot_feature_importance(model_path):
    """
    使用梯度方法估计特征重要性
    
    Args:
        model_path: 模型文件路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = BidirectionalCircuitModel(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS
    )
    trainer = CircuitTrainer(model, device)
    trainer.load_model(model_path)
    
    # 加载数据
    X, y = trainer.load_data(config.DATA_PATH, config.INPUT_DIM, config.OUTPUT_DIM)
    
    # 使用一小部分数据进行分析
    sample_size = min(100, len(X))
    X_sample = X[:sample_size]
    
    # 归一化
    X_normalized = trainer.scaler_X.transform(X_sample)
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    X_tensor.requires_grad = True
    
    # 前向传播
    model.eval()
    outputs = model(X_tensor)
    
    # 计算梯度
    importance_matrix = np.zeros((config.OUTPUT_DIM, config.INPUT_DIM))
    
    for i in range(config.OUTPUT_DIM):
        # 对每个输出计算梯度
        model.zero_grad()
        outputs[:, i].sum().backward(retain_graph=True)
        
        # 使用梯度的绝对值作为重要性度量
        importance = torch.abs(X_tensor.grad).mean(dim=0).cpu().detach().numpy()
        importance_matrix[i, :] = importance
        
        # 清除梯度
        X_tensor.grad.zero_()
    
    # 归一化重要性
    importance_matrix = importance_matrix / importance_matrix.sum(axis=1, keepdims=True)
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(config.INPUT_DIM))
    ax.set_yticks(np.arange(config.OUTPUT_DIM))
    ax.set_xticklabels([f'输入{i+1}' for i in range(config.INPUT_DIM)])
    ax.set_yticklabels([f'输出{i+1}' for i in range(config.OUTPUT_DIM)])
    
    # 添加数值标签
    for i in range(config.OUTPUT_DIM):
        for j in range(config.INPUT_DIM):
            text = ax.text(j, i, f'{importance_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('特征重要性热图\n(梯度法估计)')
    ax.set_xlabel('输入参数')
    ax.set_ylabel('输出参数')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相对重要性', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性图已保存: feature_importance.png")
    plt.close()


def generate_all_visualizations(model_path='best_model.pth'):
    """生成所有可视化图表"""
    print("开始生成可视化图表...")
    print("=" * 60)
    
    try:
        print("\n1. 生成预测对比图...")
        plot_prediction_comparison(model_path)
        
        print("\n2. 生成误差分布图...")
        plot_error_distribution(model_path)
        
        print("\n3. 生成特征重要性图...")
        plot_feature_importance(model_path)
        
        print("\n" + "=" * 60)
        print("所有可视化图表生成完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("- prediction_comparison.png  (预测值vs真实值)")
        print("- error_distribution.png     (误差分布)")
        print("- feature_importance.png     (特征重要性)")
        
    except FileNotFoundError:
        print(f"\n错误: 找不到模型文件 {model_path}")
        print("请先训练模型！")
    except Exception as e:
        print(f"\n生成可视化时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    generate_all_visualizations('best_model.pth')
