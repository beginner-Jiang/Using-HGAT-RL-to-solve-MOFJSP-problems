# Project Documentation: Multi-Objective Flexible Job Shop Scheduling (MOFJSP)

This project implements various methods for solving the Multi-Objective Flexible Job Shop Scheduling Problem (MOFJSP), including heuristic rules, metaheuristics, and deep reinforcement learning approaches, and proposes the **HGAT+PPO** (Heterogeneous Graph Attention Network + Proximal Policy Optimization) framework. The project is well-structured and all code can reproduce the results reported in the paper.

---

## 1. Project Overview

This project studies the Multi-Objective Flexible Job Shop Scheduling Problem (MOFJSP) with the goal of simultaneously optimizing three objectives:
- Makespan (\(C_{\max}\))
- Load Balance (\(LB\))
- Total Tardiness (\(T_{\text{tardy}}\))

Multiple solution methods are implemented and compared to validate the effectiveness of the proposed method.

---

## 2. Project Structure

| Directory/File | Description |
|----------------|-------------|
| `mo_fjsp_instances/` | Training/validation/test datasets (self-built, 100 instances) |
| `comparison_instances/` | Comparison experiment dataset (fixed sizes, 90 instances) |
| `Heuristic/` | Six heuristic rules (SPT, EDD, LBD, FIFO, MOPNR, MWKR) |
| `Meta_Heuristic/` | Metaheuristic algorithms (MOEA/D, NSGA-II) |
| `Deep_Learning/` | Deep reinforcement learning methods (MLP+DQN, MLP+PPO, GAT+PPO) |
| `PPO_HGAT/` | Proposed HGAT+PPO method and related tools |
| `model/` | Trained model weights |
| `Figure_And_File/` | Output directory for experimental results (figures, metrics files) |
| `DataSet/` | Dataset generation scripts (`Generate_DataSet.py`, `Generate_DataSet2.py`) |
| `Compare_Experiment/` | Comparison experiment scripts (`Experiment1.py`, `Experiment2.py`) |

---

## 3. Dataset Description

### 3.1 Self-built Dataset (Training/Validation/Test)
- Generation script: `DataSet/Generate_DataSet.py`
- Configuration file: `DataSet/config.json`
- Number of instances: 100, split into training, validation, and test sets in a 60:20:20 ratio.
- Size distribution:
  - Small (10–20 jobs, 5–10 machines): 40 instances
  - Medium (21–40 jobs, 11–20 machines): 30 instances
  - Large (41–100 jobs, 21–30 machines): 30 instances
- Data format:
  - Line 1: number of jobs, number of machines
  - Lines 2 to n_jobs+1: operation data for each job (number of operations, number of optional machines, and corresponding processing times)
  - Second last line: due dates for each job
  - Last line: machine capability levels
  - Following lines: comments

### 3.2 Comparison Dataset (Fixed Sizes)
- Generation script: `DataSet/Generate_DataSet2.py`
- Configuration file: `DataSet/config2.json`
- Number of instances: 90, with 9 size configurations, 10 instances each.
- Size list:
  - Small: 10×5, 15×10, 20×10
  - Medium: 25×15, 30×20, 40×20
  - Large: 50×25, 75×25, 100×30
- Data format is the same as the self-built dataset.

---

## 4. Implementation Details

### 4.1 Heuristic Rules
Located in the `Heuristic/` directory. Each rule is a separate Python file:
- `SPT.py`: Shortest Processing Time first
- `EDD.py`: Earliest Due Date first
- `LBD.py`: Least Load first
- `FIFO.py`: First-In-First-Out (by job release time)
- `MOPNR.py`: Most Operations Remaining first
- `MWKR.py`: Most Work Remaining first

Each rule implements:
- Reading instance files
- Executing the schedule
- Computing the three objective values and total runtime
- Outputting a Gantt chart and an Excel file with metrics

### 4.2 Metaheuristic Algorithms
Located in the `Meta_Heuristic/` directory:
- `MOEA_D.py`: Multi-objective Evolutionary Algorithm based on Decomposition (MOEA/D)
- `NSGA_II.py`: Non-dominated Sorting Genetic Algorithm II (NSGA-II)

Both algorithms:
- Use an individual encoding (operation sequence + machine assignment)
- Output Pareto fronts
- Generate convergence curves, 3D Pareto front plots, and Gantt charts

### 4.3 Deep Reinforcement Learning Methods
Located in the `Deep_Learning/` directory:
- `DQN_MLP.py`: DQN + MLP encoder
- `PPO_MLP.py`: PPO + MLP encoder
- `PPO_GAT.py`: PPO + homogeneous graph attention network (Homogeneous GAT)

These methods use the same hyperparameters as in training. Trained model weights are saved in the corresponding subdirectories under `model/`.

### 4.4 Proposed Method: HGAT+PPO
Located in the `PPO_HGAT/` directory:
- `PPO_HGAT.py`: Main training script
- `Graph.py`: Generates a Gantt chart for a 100×30 instance
- `Param.py`: Parameter sensitivity analysis
- `PPO_HGAT(t).py`: Testing script
- `PPO_HGAT(v).py`: Validation script
- `Ablation_Experiment/`: Ablation study training and evaluation

**Key Features**:
- **Heterogeneous disjunctive graph**: Contains operation nodes, machine nodes, and three types of edges (conjunctive, assignment, disjunctive) to fully model the scheduling state.
- **Heterogeneous Graph Attention Network (HGAT)**: Distinguishes node and edge types to extract high-dimensional state embeddings.
- **Hierarchical reinforcement learning**: Upper layer selects operations, lower layer assigns machines, jointly trained using PPO+KL.

---

## 5. How to Run Experiments

### 5.1 Environment Setup
- Python 3.8+
- PyTorch 1.10+ (CPU or GPU recommended)
- numpy, pandas, matplotlib, openpyxl, tqdm, scipy
- Optional: numba (for acceleration)

### 5.2 Generate Datasets

Enter the `DataSet` directory and run `Generate_DataSet.py` to generate training/validation/test datasets:

`cd DataSet`
`python Generate_DataSet.py`

Run `Generate_DataSet2.py` to generate the comparison dataset:

`python Generate_DataSet2.py`

### 5.3 Train Models

- Heuristic rules do not require training.
- Metaheuristics can be run directly (example single instance):

`cd Meta_Heuristic`
`python MOEA_D.py`
`python NSGA_II.py`

- Train deep reinforcement learning methods:

`cd Deep_Learning`
`python DQN_MLP.py`      # Train DQN+MLP
`python PPO_MLP.py`      # Train PPO+MLP
`python PPO_GAT.py`      # Train PPO+GAT

- Train HGAT+PPO (proposed method):

`cd PPO_HGAT`
`python PPO_HGAT.py`     # Train the proposed method

### 5.4 Run Comparison Experiments

Enter the `Compare_Experiment` directory and execute:

`python Experiment1.py`   # Heuristic rules vs HGAT+PPO
`python Experiment2.py`   # Metaheuristics vs deep reinforcement learning methods

### 5.5 Ablation Study

Enter the `PPO_HGAT/Ablation_Experiment` directory:

`python train.py`         # Train A1–A5 variants
`python evaluate.py`      # Evaluate and generate final CSV tables

### 5.6 Parameter Sensitivity Analysis

Enter the `PPO_HGAT` directory:

`python Param.py`

### 5.7 Generate Gantt Chart

Enter the `PPO_HGAT` directory:

`python Graph.py`         # Generate Gantt chart for a 100×30 instance

---

## 6. Output Files

All experimental results are saved in the `Figure_And_File/` directory:

| Subdirectory | Contents |
|--------------|----------|
| `Heuristic/` | Gantt charts and metric Excel files for each heuristic rule |
| `Meta_Heuristic/` | Convergence curves, Pareto front plots, Gantt charts for MOEA/D and NSGA-II |
| `PPO_HGAT/Graph/` | Gantt chart (`gantt_chart_100x30.png`) |
| `PPO_HGAT/Param/` | Parameter sensitivity plots and CSV data |
| `Compare_Experiment/` | Comparison experiment Excel files (`Experiment1_Result.xlsx`, `Experiment2_Result.xlsx`) |
| `Ablation_Experiment/Final/` | Ablation study final CSV tables (Small.csv, Medium.csv, Large.csv) |
| `Ablation_Experiment/Check_Point/` | Intermediate results for each variant (pkl files) |

---

## 7. Model Files

| Method | Model Path | Description |
|--------|------------|-------------|
| MLP+DQN | `model/dqn/dqn_model.pth` | Trained DQN model |
| MLP+PPO | `model/ppo/ppo_model.pth` | Trained PPO+MLP model |
| GAT+PPO | `model/ppo_gat/ppo_gat_best.pth` | Trained PPO+GAT model |
| **HGAT+PPO** | `model/ppo_hgat/ppo_hgat_best.pth` | **Best model of the proposed method** |
| A1–A5 | `PPO_HGAT/Ablation_Experiment/model/A{1..5}/best.pth` | Trained models for ablation study variants |

---

## 8. Important Notes

- All random seeds are fixed (42) to ensure reproducibility.
- Using a GPU will significantly speed up training; if GPU memory is limited, adjust the `num_envs` parameter in the `Config`.
- Some scripts (e.g., `PPO_GAT.py`) use multiprocessing; on Windows, set `mp.set_start_method('spawn')`.
- The A5 model for the ablation study is automatically copied from the pre‑trained path if not found; no need to retrain.

---

## 9. Additional Remarks

- The self-built datasets included in the project can be regenerated with different parameters by modifying the dataset generation code.
- The experimental results and figures generated by the code may differ from those in the paper due to differences in hardware, training environment, and randomness. However, the relative performance of the proposed method remains consistent. (Only a small portion of the figures in the paper were generated using this code; those figures were further smoothed and beautified, so they are not identical to the raw outputs produced by the project.)
- Pre‑trained models are provided and can be used directly; retraining will automatically overwrite them.
- We recommend using a GPU or cloud server for training; otherwise, training may be very slow.
- Training results may vary slightly depending on the environment and the specific data used.
- **This project is currently intended for experimental and research purposes only. Real-world deployment may encounter new issues not covered by the current implementation. The project will be updated and improved periodically to address potential limitations.**

---

## 10. Citation

If you use this project's code or data, please cite the paper:

> Jiang Jize, Zheng Qiang, Zhao Dongfang. Multi-Objective Flexible Job Shop Scheduling via Heterogeneous Disjunctive Graph and Hierarchical Reinforcement Learning. *Sensors*, 2026.