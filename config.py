"""
模型配置文件
在此文件中修改模型和训练参数
"""

# ============= 数据配置 =============
# 数据文件路径（Windows路径）
DATA_PATH = r"E:\test_data\no.9\01_train_set\5t_opamp\source\pretrain_d_t.xlsx"

# 输入输出维度
INPUT_DIM = 7   # 输入参数数量（Excel前7列）
OUTPUT_DIM = 5  # 输出参数数量（Excel后5列）

# ============= 模型配置 =============
# 六个隐藏层的维度配置
# 可以根据需要调整每层的神经元数量
HIDDEN_DIMS = [128, 256, 512, 512, 256, 128]

# Dropout比率（防止过拟合）
DROPOUT_RATE = 0.2

# ============= 训练配置 =============
# 训练轮数
EPOCHS = 200

# 学习率
LEARNING_RATE = 0.001

# 批次大小
BATCH_SIZE = 32

# 早停耐心值（验证损失多少个epoch不下降就停止）
PATIENCE = 20

# ============= 数据分割配置 =============
# 测试集比例
TEST_SIZE = 0.2

# 验证集比例（从剩余数据中分割）
VAL_SIZE = 0.1

# 随机种子（保证可重复性）
RANDOM_SEED = 42

# ============= 学习率调度器配置 =============
# 学习率衰减因子
LR_FACTOR = 0.5

# 学习率调度器耐心值
LR_PATIENCE = 10

# ============= 输出配置 =============
# 模型保存路径
BEST_MODEL_PATH = "best_model.pth"
FINAL_MODEL_PATH = "final_circuit_model.pth"

# 训练历史图表保存路径
TRAINING_HISTORY_PATH = "training_history.png"

# 是否使用GPU（如果可用）
USE_GPU = True

# ============= 日志配置 =============
# 打印间隔（每多少个epoch打印一次）
PRINT_INTERVAL = 10

# ============= 高级配置 =============
# 是否启用批归一化
USE_BATCH_NORM = True

# 是否启用Dropout
USE_DROPOUT = True

# 优化器类型（'adam', 'sgd', 'adamw'）
OPTIMIZER_TYPE = 'adam'

# 损失函数类型（'mse', 'mae', 'huber'）
LOSS_FUNCTION = 'mse'


# ============= 配置验证 =============
def validate_config():
    """验证配置的合理性"""
    assert len(HIDDEN_DIMS) == 6, "必须配置6个隐藏层"
    assert INPUT_DIM > 0, "输入维度必须大于0"
    assert OUTPUT_DIM > 0, "输出维度必须大于0"
    assert 0 < TEST_SIZE < 1, "测试集比例必须在0和1之间"
    assert 0 < VAL_SIZE < 1, "验证集比例必须在0和1之间"
    assert BATCH_SIZE > 0, "批次大小必须大于0"
    assert EPOCHS > 0, "训练轮数必须大于0"
    assert LEARNING_RATE > 0, "学习率必须大于0"
    print("✓ 配置验证通过")


if __name__ == "__main__":
    validate_config()
    print("\n当前配置:")
    print("=" * 60)
    print(f"数据路径: {DATA_PATH}")
    print(f"输入维度: {INPUT_DIM}")
    print(f"输出维度: {OUTPUT_DIM}")
    print(f"隐藏层维度: {HIDDEN_DIMS}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"批次大小: {BATCH_SIZE}")
    print("=" * 60)
