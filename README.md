# Project Documentation: Multi-Objective Flexible Job Shop Scheduling (MOFJSP)

This project implements various methods for solving the Multi-Objective Flexible Job Shop Scheduling Problem (MOFJSP), including heuristic rules, metaheuristics, and deep reinforcement learning approaches, and proposes the **HGHRL** (Heterogeneous Graph and Hierarchical Reinforcement Learning) framework. The code reproduces the experiments reported in the paper, including the secondary makespan-only validation on the original Brandimarte and Lawrence instances.

The name HGHRL reflects the two core ingredients of the framework: a heterogeneous graph representation of the scheduling state and a hierarchical reinforcement-learning policy for operation selection and machine assignment.

## HGHRL at a glance

HGHRL is the complete scheduling framework proposed in this repository, rather than a single encoder or a single policy network. It combines four elements:

- a heterogeneous disjunctive graph that represents operation precedence, machine eligibility and machine contention in one state;
- a heterogeneous graph attention encoder (HGAT) that distinguishes node and edge types and uses edge attributes in message passing;
- a two-level decision structure in which the upper policy selects the next operation and the lower policy assigns an eligible machine;
- PPO with a clipped objective and an adaptive KL penalty, trained with a weighted reward over makespan, load balance and total tardiness.

In the code, `HGAT` denotes the encoder, `PPO+KL` denotes the optimizer, and `HGHRL` denotes the complete method. The baseline names such as `GAT+PPO`, `MLP+PPO` and `MLP+DQN` are kept unchanged so that the comparison remains readable. The training entry point is `HGHRL/HGHRL.py`; `HGHRL_Test.py` and `HGHRL_Validation.py` provide the corresponding test and validation workflows. The deployed checkpoint is stored at `model/hghrl/hghrl_best.pth`.

## Computational environment

The experiments were run in two complementary environments:

- **Local workstation:** Windows 11 Home 64-bit (build 26200), an Intel Core i7-13700H processor (14 cores and 20 logical threads), 16 GB of system memory, and an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB of graphics memory. The local Python environment used Python 3.9.1, PyTorch 2.4.1 with CUDA 12.1, and NumPy 1.23.1.
- **Cloud parameter-sweep node:** an AutoDL instance with an NVIDIA RTX 2080 Ti GPU (11 GB), 12 vCPUs, 43 GB of system memory, and a 50 GB data disk mounted at `/root/autodl-tmp`. The cloud-side code and outputs were kept under `/root/autodl-tmp/MOFJSP`; the parameter sweep used a Python 3.10 conda environment and a CUDA 12.1 PyTorch build.

The cloud node was used for the parallel parameter-sensitivity runs, where several configurations can be evaluated concurrently and the GPU memory budget is the limiting resource. The local workstation was used for model development, comparison experiments, public-benchmark validation and figure generation. Smaller instances can also run on CPU, but GPU execution is recommended for the full training and evaluation protocol.

---

## 1. Project Overview

This project studies the Multi-Objective Flexible Job Shop Scheduling Problem (MOFJSP) with the goal of simultaneously optimizing three objectives:
- Makespan (\(C_{\max}\))
- Load Balance (\(LB\))
- Total Tardiness (\(T_{\text{tardy}}\))

Multiple solution methods are implemented and compared to validate the effectiveness of the proposed method. A separate public-benchmark validation repeats the comparison under a makespan-only objective on the original Brandimarte and Lawrence files.

---

## 2. Project Structure

| Directory/File | Description |
|----------------|-------------|
| `DataSet/` | Dataset generation scripts, configuration files, and the self-built instances |
| `DataSet/mo_fjsp_instances/` | Training/validation/test instances (100) |
| `DataSet/comparison_instances/` | Comparison instances with fixed sizes (90) |
| `Heuristic/` | Six heuristic rules (SPT, EDD, LBD, FIFO, MOPNR, MWKR) |
| `Meta_Heuristic/` | Multi-objective metaheuristics (MOEA/D, NSGA-II, NSGA-III, RVEA) |
| `DRL/` | Deep reinforcement learning baselines (MLP+DQN, MLP+PPO, GAT+PPO, HGNN, DAN, RESCHED) |
| `HGHRL/` | Proposed HGHRL method, parameter, ablation, weight and complexity studies |
| `Compare_Experiment/` | Main comparison experiments (`Experiment1.py`, `Experiment2.py`) |
| `PublicBenchmarkValidation/` | Public makespan-only validation: common utilities, Experiment1-3 |
| `Public_DataSet/` | Original Brandimarte and Lawrence files used by the public validation |
| `model/` | Trained model weights |
| `Figure_And_File/` | Output directory for experimental results |

---

## 3. Dataset Description

### 3.1 Self-built Dataset (Training/Validation/Test)
- Generation script: `DataSet/Generate_DataSet.py`
- Configuration file: `DataSet/config.json`
- Number of instances: 100, split into training, validation and test sets in a 60:20:20 ratio.
- Size distribution: small (10-20 jobs, 5-10 machines), medium (21-40 jobs, 11-20 machines), large (41-100 jobs, 21-30 machines).
- Data format: number of jobs and machines on the first line, one job per following line, then the due dates and the machine capability levels.

### 3.2 Comparison Dataset (Fixed Sizes)
- Generation script: `DataSet/Generate_DataSet2.py`
- Configuration file: `DataSet/config2.json`
- Number of instances: 90, nine sizes with ten instances each (10x5 to 100x30).
- Data format is the same as the self-built dataset.

### 3.3 Public Dataset
- Files: Brandimarte MK01-MK10 and Lawrence la01-la40 (Hurink e-data), copied unchanged to `Public_DataSet/`.
- One split for all fifty files: Lawrence la01-la20 for adaptation, la21-la25 for validation, and 25 held-out instances for testing (Brandimarte MK01-MK10 as cross-family zero-shot and Lawrence la26-la40 as in-family held-out).
- Objective: makespan only, because the original files carry no due dates or capability levels.

---

## 4. Implementation Details

### 4.1 Heuristic Rules
Located in `Heuristic/`, one file per rule. Each rule reads an instance, builds the schedule, computes the three objectives and the runtime, and writes a Gantt chart and a metric file.

### 4.2 Metaheuristic Algorithms
Located in `Meta_Heuristic/`: MOEA/D, NSGA-II, NSGA-III and RVEA. All use an operation-sequence plus machine-assignment encoding and return a Pareto set. The public validation uses a standard genetic algorithm and tabu search in `PublicBenchmarkValidation/Experiment2/`.

### 4.3 Deep Reinforcement Learning Baselines
Located in `DRL/`:
- `DQN_MLP.py`: DQN with an MLP encoder.
- `PPO_MLP.py`: PPO with an MLP encoder.
- `PPO_GAT.py`: PPO with a homogeneous graph attention encoder.
- `HGNN.py`, `DAN.py`, `RESCHED.py`: graph-based scheduling baselines.

Trained weights are stored under `model/`.

### 4.4 Proposed Method: HGHRL
Located in `HGHRL/`:
- `HGHRL.py`: main training script and the canonical implementation of the graph environment, HGAT encoder and hierarchical policies.
- `HGHRL_Test.py` and `HGHRL_Validation.py`: evaluation-oriented entry points for checkpoint testing and validation.
- `Gantt_Chart.py`: Gantt chart for a 100x30 instance.
- `Param_Experiment/`: parameter sensitivity, joint hyperparameter validation and the proposed-method validation.
- `Ablation_Experiment/`: ablation study.
- `Multi-objective_Weight_Experiment/`: weight analysis.
- `Time_Space_Complexity_Analysis/`: time and space complexity measurements.

The implementation is organized around a single canonical environment and network definition in `HGHRL/HGHRL.py`; the experiment scripts import that definition rather than maintaining separate copies. This keeps the graph construction, feature layout, feasibility mask and checkpoint format consistent across training, comparison and public validation. The graph contains operation and machine nodes, three edge types and dynamic edge features; the encoder is HGAT; the decision process separates operation selection from machine assignment; and the optimizer is PPO with a KL penalty.

---

## 5. How to Run Experiments

### 5.1 Environment Setup
- Python 3.8+
- PyTorch 1.10+ (CPU or GPU)
- numpy, pandas, matplotlib, openpyxl, tqdm, scipy

### 5.2 Generate Datasets
```
cd DataSet
python Generate_DataSet.py
python Generate_DataSet2.py
```

### 5.3 Train Models
```
cd DRL
python DQN_MLP.py
python PPO_MLP.py
python PPO_GAT.py

cd ../HGHRL
python HGHRL.py
```

### 5.4 Main Comparison Experiments
```
cd Compare_Experiment
python Experiment1.py
python Experiment2.py
```

### 5.5 Ablation, Parameter and Complexity Studies
```
cd HGHRL
python Param_Experiment/Param.py
python Ablation_Experiment/evaluate.py
python Multi-objective_Weight_Experiment/Multi_Objective_Weight.py
python Time_Space_Complexity_Analysis/Time_Space_Complexity.py
```

### 5.6 Gantt Chart
```
cd HGHRL
python Gantt_Chart.py
```

### 5.7 Public Benchmark Validation

The public validation uses one combined split and three blocks. The proposed policy and HGNN/DAN/RESCHED are adapted on Lawrence la01-la20 with the HGAT encoder frozen (200 episodes, learning rate 1e-5); the zero-shot checkpoint is kept as a candidate and the adaptation is selected on la21-la25 with 21 rollouts. MLP+DQN, MLP+PPO and GAT+PPO keep their standard checkpoints.

```
python PublicBenchmarkValidation/common/run_public_validation.py --stage prepare
python PublicBenchmarkValidation/common/run_public_validation.py --stage train --family Combined --episodes 200 --lr 1e-5 --freeze-encoder --tag gentle --proposed-only
python PublicBenchmarkValidation/common/run_public_validation.py --stage select --family Combined
python PublicBenchmarkValidation/common/run_public_validation.py --stage evaluate --family Combined --rollouts 51
python PublicBenchmarkValidation/Experiment1/refine_public_hybrid.py --family Combined --workers 8
python PublicBenchmarkValidation/Experiment3/refine_policy.py --family Combined --workers 8
python PublicBenchmarkValidation/Experiment1/analyze_experiment1.py
python PublicBenchmarkValidation/Experiment2/run_metaheuristics.py --family Combined --evaluations 51
python PublicBenchmarkValidation/Experiment2/analyze_experiment2.py
python PublicBenchmarkValidation/Experiment3/analyze_experiment3.py
```

The three blocks are: five dispatching rules against the refined hybrid decoder; GA and TS against the deployed policy-plus-refinement decoder; and six learned baselines against the pure policy. The bounded refinement is an inference-time calibration of the decoder: it keeps the learned priority and repairs dominated machine assignments produced by the scale mismatch between the training geometry and a public instance. It is the same operator, with the same budget, used by the main rule protocol. GA/TS receive 51 schedule evaluations and the learned baselines 51 rollouts; the refinement is not given to them, so the metaheuristic and learning blocks are deployment comparisons rather than equal-budget claims.

---

## 6. Output Files

All experimental results are saved under `Figure_And_File/`:

| Subdirectory | Contents |
|--------------|----------|
| `Heuristic/` | Gantt charts and metric files for each rule |
| `Meta_Heuristic/` | Convergence curves, Pareto fronts and Gantt charts |
| `HGHRL/` | Gantt chart, parameter, ablation, weight and complexity outputs |
| `Compare_Experiment/` | Comparison experiment tables and figures |
| `PublicBenchmarkValidation/Combined/` | Public validation tables and figures for Experiment1-3 |

---

## 7. Model Files

| Method | Model Path | Description |
|--------|------------|-------------|
| MLP+DQN | `model/dqn/dqn_model.pth` | Trained DQN model |
| MLP+PPO | `model/ppo/ppo_model.pth` | Trained PPO+MLP model |
| GAT+PPO | `model/ppo_gat/ppo_gat_best.pth` | Trained PPO+GAT model |
| HGNN / DAN / RESCHED | `model/hgnn/`, `model/dan/`, `model/resched/` | Graph baseline weights |
| **HGHRL** | `model/hghrl/hghrl_best.pth` | **Best model of the proposed method** |
| Public adaptation | `Public_DataSet/PublicBenchmarkValidation/checkpoints/Lawrence/hghrl_public_best_gentle.pth` | Selected public-adaptation checkpoint |
| A1-A5 | `HGHRL/Ablation_Experiment/model/A{1..5}/best.pth` | Ablation variants |

---

## 8. Important Notes

- All random seeds are fixed (42) to ensure reproducibility.
- A GPU is recommended; if memory is limited, reduce `num_envs` in the configuration.
- Some scripts use multiprocessing; on Windows set `mp.set_start_method('spawn')`.
- The public adaptation uses one seed, 20 training instances and a short budget; it is an external transfer test, not a second full training study.
- The public results are deployment comparisons at the budgets stated in the paper appendix; they are not equal-budget superiority claims.

---

## 9. Additional Remarks

- The self-built datasets can be regenerated with different parameters through the dataset scripts.
- The figures in the paper were redrawn for presentation; the raw outputs may differ in style but not in the relative ranking of the methods.
- Pre-trained models are provided and can be used directly; retraining overwrites them.
- **This project is intended for experimental and research purposes only.**

---

## 10. Citation

The related papers have not been published yet; updates will follow.
