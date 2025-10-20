"""
生成示例数据文件（用于测试）
如果没有实际数据，可以使用此脚本生成模拟数据
"""

import numpy as np
import pandas as pd

def generate_sample_data(n_samples=1000, input_dim=7, output_dim=5, filename='pretrain_d_t.xlsx'):
    """
    生成示例电路参数数据
    
    :param n_samples: 样本数量
    :param input_dim: 输入维度
    :param output_dim: 输出维度
    :param filename: 保存文件名
    """
    np.random.seed(42)
    
    # 生成输入参数（模拟电路输入参数）
    # 假设输入参数在合理范围内
    X = np.random.randn(n_samples, input_dim) * 10 + 50
    
    # 生成输出参数（通过输入参数的非线性变换模拟）
    # 这样可以确保输入和输出之间有一定的关联性
    W1 = np.random.randn(input_dim, 10)
    b1 = np.random.randn(10)
    W2 = np.random.randn(10, output_dim)
    b2 = np.random.randn(output_dim)
    
    # 非线性变换
    hidden = np.tanh(X @ W1 + b1)
    y = hidden @ W2 + b2
    
    # 添加一些噪声
    y += np.random.randn(n_samples, output_dim) * 0.5
    
    # 合并数据
    data = np.hstack([X, y])
    
    # 创建列名
    input_cols = [f'Input_{i+1}' for i in range(input_dim)]
    output_cols = [f'Output_{i+1}' for i in range(output_dim)]
    columns = input_cols + output_cols
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 保存为Excel文件
    df.to_excel(filename, index=False)
    
    print(f"示例数据已生成: {filename}")
    print(f"数据形状: {df.shape}")
    print(f"\n前5行数据:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())

if __name__ == "__main__":
    generate_sample_data(n_samples=1000, input_dim=7, output_dim=5)
