"""
PPO+HGAT Ablation Study Training
Trains experiments A1~A4, A5 directly copies the pre-trained model.
"""

# This is the sequential entry point of the study: it walks every entry of
# common.EXPERIMENTS through common.train_experiment, so it keeps the same environment, the
# same recipe and the same checkpoints as the managed queue (_Review/_p97_train_ablation.py with
# _p97_ablation_arm.py) and never bypasses them.  The managed queue is the advised path because
# it resumes and gates the arms; this file stays for a single-machine rerun of one session.

from common import EXPERIMENTS, train_experiment, cfg


def main():
    # Train all experiments (A5 will skip training internally and only copy the model)
    for exp in EXPERIMENTS:
        # For the joint policy experiment (A4), temporarily disable AMP to avoid dtype errors
        original_amp = cfg.amp_enabled
        if not exp['use_hierarchical']:
            cfg.amp_enabled = False
            print(f"Experiment {exp['name']} uses joint policy, temporarily disabled AMP (amp_enabled={cfg.amp_enabled})")

        try:
            train_experiment(exp)
        finally:
            # Restore AMP setting
            cfg.amp_enabled = original_amp
            print(f"Restored AMP setting to {cfg.amp_enabled}")


if __name__ == "__main__":
    main()
