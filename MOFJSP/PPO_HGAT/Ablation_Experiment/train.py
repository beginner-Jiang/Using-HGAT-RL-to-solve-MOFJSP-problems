"""
PPO+HGAT 消融实验训练
训练 A1~A4 实验，A5 直接复制预训练模型。
"""

from common import EXPERIMENTS, train_experiment, cfg


def main():
    # 训练所有实验（A5 会在内部跳过训练，仅复制模型）
    for exp in EXPERIMENTS:
        # 对于联合策略实验（A4），临时禁用 AMP 以避免 dtype 错误
        original_amp = cfg.amp_enabled
        if not exp['use_hierarchical']:
            cfg.amp_enabled = False
            print(f"实验 {exp['name']} 为联合策略，已临时禁用 AMP (amp_enabled={cfg.amp_enabled})")

        try:
            train_experiment(exp)
        finally:
            # 恢复 AMP 设置
            cfg.amp_enabled = original_amp
            print(f"恢复 AMP 设置为 {cfg.amp_enabled}")


if __name__ == "__main__":
    main()