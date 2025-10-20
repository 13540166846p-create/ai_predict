"""
电路参数预测脚本 - 使用训练好的模型进行推理
"""

import numpy as np
import pandas as pd
import torch
from train_circuit_model import BidirectionalCircuitModel, CircuitTrainer


def predict_circuit_outputs(model_path, input_data):
    """
    使用训练好的模型预测电路输出参数
    
    Args:
        model_path: 训练好的模型文件路径
        input_data: 输入数据，可以是：
                   - numpy数组 (n_samples, 7)
                   - pandas DataFrame
                   - 单个样本列表 [7个参数]
    
    Returns:
        预测的输出参数 (n_samples, 5)
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = BidirectionalCircuitModel(input_dim=7, output_dim=5)
    trainer = CircuitTrainer(model, device)
    
    # 加载训练好的模型
    trainer.load_model(model_path)
    
    # 处理输入数据格式
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    elif isinstance(input_data, list):
        input_data = np.array([input_data])
    
    # 归一化输入
    input_normalized = trainer.scaler_X.transform(input_data)
    
    # 转换为张量
    input_tensor = torch.FloatTensor(input_normalized).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output_normalized = model(input_tensor).cpu().numpy()
    
    # 反归一化输出
    output = trainer.scaler_y.inverse_transform(output_normalized)
    
    return output


def batch_predict_from_excel(model_path, input_file, output_file):
    """
    从Excel文件批量预测
    
    Args:
        model_path: 训练好的模型文件路径
        input_file: 输入Excel文件（包含7列输入参数）
        output_file: 输出Excel文件（保存预测结果）
    """
    # 读取输入数据
    df = pd.read_excel(input_file)
    
    # 确保有7列输入
    if df.shape[1] < 7:
        raise ValueError(f"输入文件应至少包含7列，当前只有{df.shape[1]}列")
    
    # 提取前7列作为输入
    input_data = df.iloc[:, :7]
    
    print(f"正在预测 {len(input_data)} 个样本...")
    
    # 预测
    predictions = predict_circuit_outputs(model_path, input_data)
    
    # 创建输出DataFrame
    output_columns = [f'输出参数_{i+1}' for i in range(5)]
    df_output = pd.DataFrame(predictions, columns=output_columns)
    
    # 合并输入和输出
    df_result = pd.concat([df.iloc[:, :7], df_output], axis=1)
    
    # 保存结果
    df_result.to_excel(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")
    
    return df_result


def interactive_predict(model_path):
    """交互式预测"""
    print("=" * 60)
    print("电路参数预测 - 交互模式")
    print("=" * 60)
    print("请输入7个输入参数（用空格分隔）：")
    
    try:
        inputs = input().strip().split()
        if len(inputs) != 7:
            print(f"错误：需要7个参数，您输入了{len(inputs)}个")
            return
        
        # 转换为浮点数
        input_values = [float(x) for x in inputs]
        
        # 预测
        output = predict_circuit_outputs(model_path, input_values)
        
        print("\n预测结果:")
        print("-" * 60)
        for i, val in enumerate(output[0]):
            print(f"输出参数 {i+1}: {val:.6f}")
        print("-" * 60)
        
    except ValueError as e:
        print(f"输入错误: {e}")
    except Exception as e:
        print(f"预测失败: {e}")


if __name__ == "__main__":
    # 示例1：单个样本预测
    print("示例1：单个样本预测")
    print("-" * 60)
    
    model_path = "best_model.pth"
    
    # 假设的输入参数
    sample_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    try:
        output = predict_circuit_outputs(model_path, sample_input)
        print(f"输入: {sample_input}")
        print(f"预测输出: {output[0]}")
    except Exception as e:
        print(f"无法执行预测（可能模型尚未训练）: {e}")
    
    print("\n" + "=" * 60)
    
    # 示例2：批量预测（如果有输入文件）
    # batch_predict_from_excel(
    #     model_path="best_model.pth",
    #     input_file="input_data.xlsx",
    #     output_file="predictions.xlsx"
    # )
    
    # 示例3：交互式预测
    # interactive_predict("best_model.pth")
