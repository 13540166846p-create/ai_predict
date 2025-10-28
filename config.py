"""
双向预测模型配置文件
"""

# 数据配置
DATA_CONFIG = {
    'file_path': 'pretrain_d_t.xlsx',  # 数据文件路径
    'input_cols': 7,   # 输入列数
    'output_cols': 5,  # 输出列数
    'test_size': 0.2,  # 测试集比例
    'random_state': 42  # 随机种子
}

# 模型配置
MODEL_CONFIG = {
    'input_dim': 7,
    'output_dim': 5,
    'hidden_dims': [128, 256, 256, 128],  # 隐藏层维度
    'dropout': 0.2  # Dropout比例
}

# 训练配置
TRAIN_CONFIG = {
    'epochs': 200,        # 训练轮数
    'batch_size': 32,     # 批次大小
    'learning_rate': 0.001,  # 学习率
    'weight_decay': 1e-5,    # 权重衰减
    'patience': 10           # 学习率调整耐心值
}

# 保存配置
SAVE_CONFIG = {
    'model_dir': 'models',          # 模型保存目录
    'figure_dir': 'figures',        # 图片保存目录
    'save_best_only': True          # 只保存最佳模型
}
