"""
模型推理示例
展示如何使用训练好的模型进行预测
"""

import numpy as np
from bidirectional_circuit_predictor import BidirectionalPredictor
import os

def inference_example():
    """模型推理示例"""
    
    # 检查模型是否存在
    if not os.path.exists('models/forward_model.pth'):
        print("错误: 找不到训练好的模型!")
        print("请先运行 'python bidirectional_circuit_predictor.py' 训练模型")
        return
    
    # 创建预测器并加载模型
    print("正在加载模型...")
    predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
    predictor.load_models('models')
    print("模型加载成功!\n")
    
    # 示例1: 正向预测（输入 -> 输出）
    print("="*60)
    print("示例1: 正向预测 (输入参数 -> 输出参数)")
    print("="*60)
    
    # 准备输入数据（可以是单个样本或多个样本）
    input_params = np.array([
        [45.2, 52.3, 48.7, 51.5, 49.8, 50.2, 47.9],
        [55.1, 48.9, 52.4, 49.3, 51.7, 48.5, 52.8]
    ])
    
    print("输入参数:")
    print(input_params)
    
    # 进行正向预测
    output_pred = predictor.predict_forward(input_params)
    
    print("\n预测的输出参数:")
    print(output_pred)
    
    # 示例2: 反向预测（输出 -> 输入）
    print("\n" + "="*60)
    print("示例2: 反向预测 (输出参数 -> 输入参数)")
    print("="*60)
    
    # 准备输出数据
    output_params = np.array([
        [10.5, 20.3, 15.7, 25.2, 18.9],
        [12.3, 22.1, 17.5, 23.8, 19.4]
    ])
    
    print("输出参数:")
    print(output_params)
    
    # 进行反向预测
    input_pred = predictor.predict_backward(output_params)
    
    print("\n预测的输入参数:")
    print(input_pred)
    
    # 示例3: 双向预测一致性检验
    print("\n" + "="*60)
    print("示例3: 双向预测一致性检验")
    print("="*60)
    
    # 从输入预测输出，再从输出预测回输入
    original_input = np.array([[45.2, 52.3, 48.7, 51.5, 49.8, 50.2, 47.9]])
    
    print("原始输入参数:")
    print(original_input)
    
    # 正向预测
    predicted_output = predictor.predict_forward(original_input)
    print("\n正向预测得到的输出:")
    print(predicted_output)
    
    # 反向预测
    reconstructed_input = predictor.predict_backward(predicted_output)
    print("\n从预测输出反向预测得到的输入:")
    print(reconstructed_input)
    
    # 计算重构误差
    reconstruction_error = np.mean(np.abs(original_input - reconstructed_input))
    print(f"\n重构误差 (MAE): {reconstruction_error:.6f}")
    
    print("\n" + "="*60)
    print("推理完成!")
    print("="*60)

if __name__ == "__main__":
    inference_example()
