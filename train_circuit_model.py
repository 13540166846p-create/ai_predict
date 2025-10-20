"""
双向电路预测模型训练脚本
第一种电路：5T运放电路参数预测
输入：前7列参数
输出：后5列参数
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class CircuitDataset(Dataset):
    """电路参数数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BidirectionalCircuitModel(nn.Module):
    """
    双向电路预测模型 - 六个隐藏层
    前向预测：输入参数 -> 输出参数
    """
    
    def __init__(self, input_dim=7, output_dim=5, hidden_dims=None):
        super(BidirectionalCircuitModel, self).__init__()
        
        # 默认六个隐藏层的维度配置
        if hidden_dims is None:
            hidden_dims = [128, 256, 512, 512, 256, 128]
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class CircuitTrainer:
    """电路模型训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def load_data(self, file_path, input_cols=7, output_cols=5):
        """
        加载Excel数据
        
        Args:
            file_path: Excel文件路径
            input_cols: 输入列数（前N列）
            output_cols: 输出列数（后M列）
        """
        print(f"正在加载数据: {file_path}")
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 分离输入和输出
        X = df.iloc[:, :input_cols].values
        y = df.iloc[:, input_cols:input_cols+output_cols].values
        
        print(f"输入维度: {X.shape}, 输出维度: {y.shape}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.1, batch_size=32):
        """准备训练、验证和测试数据"""
        
        # 数据归一化
        X_normalized = self.scaler_X.fit_transform(X)
        y_normalized = self.scaler_y.fit_transform(y)
        
        # 分割训练集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_normalized, y_normalized, test_size=test_size, random_state=42
        )
        
        # 从训练集中分割验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        # 创建数据加载器
        train_dataset = CircuitDataset(X_train, y_train)
        val_dataset = CircuitDataset(X_val, y_val)
        test_dataset = CircuitDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=200, lr=0.001, patience=20):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n开始训练...")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"训练损失: {train_loss:.6f}, "
                      f"验证损失: {val_loss:.6f}")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n早停触发于 epoch {epoch+1}")
                break
        
        print("=" * 60)
        print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    
    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # 反归一化
        predictions_original = self.scaler_y.inverse_transform(predictions)
        actuals_original = self.scaler_y.inverse_transform(actuals)
        
        # 计算评估指标
        mse = np.mean((predictions_original - actuals_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_original - actuals_original))
        
        # 计算每个输出的R²
        r2_scores = []
        for i in range(actuals_original.shape[1]):
            ss_res = np.sum((actuals_original[:, i] - predictions_original[:, i]) ** 2)
            ss_tot = np.sum((actuals_original[:, i] - np.mean(actuals_original[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)
        
        print("\n模型评估结果:")
        print("=" * 60)
        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"均方根误差 (RMSE): {rmse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"R² 分数 (各输出): {[f'{r2:.4f}' for r2 in r2_scores]}")
        print(f"平均 R² 分数: {np.mean(r2_scores):.4f}")
        print("=" * 60)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores)
        }
    
    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', alpha=0.8)
        plt.plot(self.val_losses, label='验证损失', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('损失 (MSE)')
        plt.title('训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练历史图已保存至: {save_path}")
        plt.close()
    
    def save_model(self, path='circuit_model.pth'):
        """保存模型和归一化器"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
        }, path)
    
    def load_model(self, path='circuit_model.pth'):
        """加载模型和归一化器"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        print(f"模型已从 {path} 加载")


def main():
    """主训练流程"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据文件路径
    data_path = r"E:\test_data\no.9\01_train_set\5t_opamp\source\pretrain_d_t.xlsx"
    
    # 创建模型
    model = BidirectionalCircuitModel(
        input_dim=7, 
        output_dim=5,
        hidden_dims=[128, 256, 512, 512, 256, 128]  # 六个隐藏层
    )
    
    print("\n模型架构:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print("=" * 60)
    
    # 创建训练器
    trainer = CircuitTrainer(model, device)
    
    # 加载数据
    X, y = trainer.load_data(data_path, input_cols=7, output_cols=5)
    
    # 准备数据
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X, y, test_size=0.2, val_size=0.1, batch_size=32
    )
    
    # 训练模型
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=200, 
        lr=0.001, 
        patience=20
    )
    
    # 加载最佳模型
    trainer.load_model('best_model.pth')
    
    # 评估模型
    metrics = trainer.evaluate(test_loader)
    
    # 绘制训练历史
    trainer.plot_training_history('training_history.png')
    
    # 保存最终模型
    trainer.save_model('final_circuit_model.pth')
    print("\n最终模型已保存至: final_circuit_model.pth")
    
    return model, trainer, metrics


if __name__ == "__main__":
    model, trainer, metrics = main()
