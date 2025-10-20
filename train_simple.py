"""
简化版训练脚本 - 使用配置文件
快速开始训练模型
"""

from train_circuit_model import BidirectionalCircuitModel, CircuitTrainer
import config
import torch


def main():
    """简化的主训练流程"""
    
    # 验证配置
    config.validate_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建模型
    model = BidirectionalCircuitModel(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dims=config.HIDDEN_DIMS
    )
    
    print("模型架构:")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数: {total_params:,}")
    print(f"隐藏层维度: {config.HIDDEN_DIMS}")
    print("=" * 60)
    
    # 创建训练器
    trainer = CircuitTrainer(model, device)
    
    try:
        # 加载数据
        print(f"\n数据文件: {config.DATA_PATH}")
        X, y = trainer.load_data(
            config.DATA_PATH,
            input_cols=config.INPUT_DIM,
            output_cols=config.OUTPUT_DIM
        )
        
        # 准备数据
        train_loader, val_loader, test_loader = trainer.prepare_data(
            X, y,
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE,
            batch_size=config.BATCH_SIZE
        )
        
        # 训练模型
        trainer.train(
            train_loader,
            val_loader,
            epochs=config.EPOCHS,
            lr=config.LEARNING_RATE,
            patience=config.PATIENCE
        )
        
        # 加载最佳模型
        trainer.load_model(config.BEST_MODEL_PATH)
        
        # 评估模型
        metrics = trainer.evaluate(test_loader)
        
        # 绘制训练历史
        trainer.plot_training_history(config.TRAINING_HISTORY_PATH)
        
        # 保存最终模型
        trainer.save_model(config.FINAL_MODEL_PATH)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳模型: {config.BEST_MODEL_PATH}")
        print(f"最终模型: {config.FINAL_MODEL_PATH}")
        print(f"训练历史图: {config.TRAINING_HISTORY_PATH}")
        print("=" * 60)
        
        return model, trainer, metrics
        
    except FileNotFoundError:
        print("\n错误: 找不到数据文件!")
        print(f"请检查路径: {config.DATA_PATH}")
        print("\n提示: 如果路径不正确，请在 config.py 中修改 DATA_PATH")
        return None, None, None
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    model, trainer, metrics = main()
