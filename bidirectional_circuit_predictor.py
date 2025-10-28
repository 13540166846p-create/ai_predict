"""
双向电路参数预测模型
用于电路输入输出参数的双向拟合预测
- 正向预测：从输入参数预测输出参数
- 反向预测：从输出参数预测输入参数
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path


class CircuitDataset(Dataset):
    """电路数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PredictionModel(nn.Module):
    """预测模型网络结构"""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 256, 128]):
        super(PredictionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class BidirectionalPredictor:
    """双向预测器"""
    def __init__(self, input_dim=7, output_dim=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # 正向模型：输入→输出
        self.forward_model = PredictionModel(input_dim, output_dim).to(device)
        # 反向模型：输出→输入
        self.backward_model = PredictionModel(output_dim, input_dim).to(device)
        
        # 数据标准化器
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        print(f"模型已创建，使用设备: {device}")
        print(f"正向模型: {input_dim}维输入 -> {output_dim}维输出")
        print(f"反向模型: {output_dim}维输入 -> {input_dim}维输出")
    
    def load_data(self, file_path, input_cols=7, output_cols=5):
        """
        加载数据
        :param file_path: Excel文件路径
        :param input_cols: 输入列数（前N列）
        :param output_cols: 输出列数（后M列）
        """
        print(f"\n正在加载数据: {file_path}")
        
        # 处理Windows路径
        if isinstance(file_path, str):
            file_path = file_path.replace('\\', '/')
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 分离输入和输出
        X = df.iloc[:, :input_cols].values
        y = df.iloc[:, input_cols:input_cols+output_cols].values
        
        print(f"输入数据形状: {X.shape}")
        print(f"输出数据形状: {y.shape}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        预处理数据
        :param X: 输入数据
        :param y: 输出数据
        :param test_size: 测试集比例
        :param random_state: 随机种子
        """
        print("\n正在预处理数据...")
        
        # 数据标准化
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, train_loader, val_loader, epochs=200, lr=0.001, model_name="model"):
        """
        训练模型
        :param model: 要训练的模型
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param epochs: 训练轮数
        :param lr: 学习率
        :param model_name: 模型名称
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"\n开始训练 {model_name}...")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{model_name}.pth')
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        print(f"{model_name} 训练完成! 最佳验证损失: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def train(self, X, y, epochs=200, batch_size=32, lr=0.001):
        """
        训练双向模型
        :param X: 输入数据
        :param y: 输出数据
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param lr: 学习率
        """
        # 预处理数据
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        
        # 创建数据加载器 - 正向模型
        train_dataset_forward = CircuitDataset(X_train, y_train)
        test_dataset_forward = CircuitDataset(X_test, y_test)
        train_loader_forward = DataLoader(train_dataset_forward, batch_size=batch_size, shuffle=True)
        test_loader_forward = DataLoader(test_dataset_forward, batch_size=batch_size)
        
        # 创建数据加载器 - 反向模型
        train_dataset_backward = CircuitDataset(y_train, X_train)
        test_dataset_backward = CircuitDataset(y_test, X_test)
        train_loader_backward = DataLoader(train_dataset_backward, batch_size=batch_size, shuffle=True)
        test_loader_backward = DataLoader(test_dataset_backward, batch_size=batch_size)
        
        # 训练正向模型
        print("\n" + "="*60)
        print("训练正向模型 (输入 -> 输出)")
        print("="*60)
        forward_train_losses, forward_val_losses = self.train_model(
            self.forward_model, train_loader_forward, test_loader_forward, 
            epochs=epochs, lr=lr, model_name="forward_model"
        )
        
        # 训练反向模型
        print("\n" + "="*60)
        print("训练反向模型 (输出 -> 输入)")
        print("="*60)
        backward_train_losses, backward_val_losses = self.train_model(
            self.backward_model, train_loader_backward, test_loader_backward, 
            epochs=epochs, lr=lr, model_name="backward_model"
        )
        
        # 绘制训练曲线
        self.plot_training_curves(forward_train_losses, forward_val_losses, 
                                 backward_train_losses, backward_val_losses)
        
        # 评估模型
        self.evaluate(X_test, y_test)
        
        # 保存模型和标准化器
        self.save_models()
    
    def plot_training_curves(self, forward_train, forward_val, backward_train, backward_val):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(forward_train, label='训练损失')
        plt.plot(forward_val, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('正向模型训练曲线 (输入→输出)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(backward_train, label='训练损失')
        plt.plot(backward_val, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('反向模型训练曲线 (输出→输入)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("\n训练曲线已保存至: training_curves.png")
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        print("\n" + "="*60)
        print("模型评估")
        print("="*60)
        
        # 正向模型评估
        self.forward_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = self.forward_model(X_test_tensor).cpu().numpy()
        
        # 计算R²分数
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2_forward = r2_score(y_test, y_pred, multioutput='variance_weighted')
        mae_forward = mean_absolute_error(y_test, y_pred)
        rmse_forward = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n正向模型性能:")
        print(f"  R² Score: {r2_forward:.4f}")
        print(f"  MAE: {mae_forward:.6f}")
        print(f"  RMSE: {rmse_forward:.6f}")
        
        # 反向模型评估
        self.backward_model.eval()
        with torch.no_grad():
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            X_pred = self.backward_model(y_test_tensor).cpu().numpy()
        
        r2_backward = r2_score(X_test, X_pred, multioutput='variance_weighted')
        mae_backward = mean_absolute_error(X_test, X_pred)
        rmse_backward = np.sqrt(mean_squared_error(X_test, X_pred))
        
        print(f"\n反向模型性能:")
        print(f"  R² Score: {r2_backward:.4f}")
        print(f"  MAE: {mae_backward:.6f}")
        print(f"  RMSE: {rmse_backward:.6f}")
    
    def predict_forward(self, X):
        """
        正向预测：输入 -> 输出
        :param X: 输入参数
        :return: 预测的输出参数
        """
        self.forward_model.eval()
        X_scaled = self.input_scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = self.forward_model(X_tensor).cpu().numpy()
        y_pred = self.output_scaler.inverse_transform(y_pred_scaled)
        return y_pred
    
    def predict_backward(self, y):
        """
        反向预测：输出 -> 输入
        :param y: 输出参数
        :return: 预测的输入参数
        """
        self.backward_model.eval()
        y_scaled = self.output_scaler.transform(y)
        with torch.no_grad():
            y_tensor = torch.FloatTensor(y_scaled).to(self.device)
            X_pred_scaled = self.backward_model(y_tensor).cpu().numpy()
        X_pred = self.input_scaler.inverse_transform(X_pred_scaled)
        return X_pred
    
    def save_models(self, save_dir='models'):
        """保存模型和标准化器"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.forward_model.state_dict(), os.path.join(save_dir, 'forward_model.pth'))
        torch.save(self.backward_model.state_dict(), os.path.join(save_dir, 'backward_model.pth'))
        
        # 保存标准化器
        with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump({
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler
            }, f)
        
        print(f"\n模型已保存至: {save_dir}/")
    
    def load_models(self, save_dir='models'):
        """加载模型和标准化器"""
        # 加载模型
        self.forward_model.load_state_dict(torch.load(os.path.join(save_dir, 'forward_model.pth')))
        self.backward_model.load_state_dict(torch.load(os.path.join(save_dir, 'backward_model.pth')))
        
        # 加载标准化器
        with open(os.path.join(save_dir, 'scalers.pkl'), 'rb') as f:
            scalers = pickle.load(f)
            self.input_scaler = scalers['input_scaler']
            self.output_scaler = scalers['output_scaler']
        
        print(f"模型已从 {save_dir}/ 加载")


def main():
    """主函数"""
    # 数据文件路径（需要根据实际情况修改）
    # Windows路径: E:\test_data\no.9\01_train_set\5t_opamp\source\pretrain_d_t.xlsx
    # 如果在Linux环境下，需要将数据文件复制到相应位置
    data_file = "pretrain_d_t.xlsx"  # 默认在当前目录查找
    
    # 如果文件不存在，尝试其他可能的路径
    if not os.path.exists(data_file):
        possible_paths = [
            "/mnt/e/test_data/no.9/01_train_set/5t_opamp/source/pretrain_d_t.xlsx",
            "./data/pretrain_d_t.xlsx",
            "../data/pretrain_d_t.xlsx"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_file = path
                break
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件!")
        print(f"请将数据文件 'pretrain_d_t.xlsx' 放置在以下位置之一:")
        print(f"  - 当前目录: {os.getcwd()}")
        print(f"  - ./data/ 目录")
        print(f"或者修改代码中的 data_file 变量指向正确的路径")
        return
    
    # 创建双向预测器
    predictor = BidirectionalPredictor(input_dim=7, output_dim=5)
    
    # 加载数据
    X, y = predictor.load_data(data_file, input_cols=7, output_cols=5)
    
    # 训练模型
    predictor.train(X, y, epochs=200, batch_size=32, lr=0.001)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("保存的文件:")
    print("  - models/forward_model.pth (正向模型)")
    print("  - models/backward_model.pth (反向模型)")
    print("  - models/scalers.pkl (数据标准化器)")
    print("  - training_curves.png (训练曲线图)")
    print("  - best_forward_model.pth (最佳正向模型)")
    print("  - best_backward_model.pth (最佳反向模型)")


if __name__ == "__main__":
    main()
