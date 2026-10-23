# Public Benchmark Validation

Makespan-only external validation on the original Brandimarte and Lawrence
files. All fifty files are used in one split.

## Split

- Adaptation: Lawrence `la01`-`la20` (20 instances).
- Validation: Lawrence `la21`-`la25` (5 instances).
- Test: Brandimarte `MK01`-`MK10` (10, cross-family zero-shot) and Lawrence
  `la26`-`la40` (15, in-family held-out), 25 instances in total.
- Objective: makespan only, `w = (1, 0, 0)`; one seed (`42`); the instance is
  the statistical unit.

## Three blocks

- **Experiment 1 - dispatching rules:** SPT, LBD, FIFO, MOPNR, MWKR and the
  refined hybrid decoder (five rule starts, 51 rollouts and a 4000-iteration
  bounded refinement). LPT is removed from the public rule set.
- **Experiment 2 - metaheuristics:** a standard genetic algorithm (GA) and tabu
  search (TS) over the operation-sequence / machine-assignment representation,
  with 51 schedule evaluations, compared with the deployed decoder (51 rollouts
  plus the same 4000-iteration refinement of the policy's own best schedule; no
  rule candidates).
- **Experiment 3 - learning methods:** MLP+DQN, MLP+PPO, GAT+PPO, HGNN, DAN,
  RESCHED and the pure policy at 51 rollouts.

## Adaptation

The proposed policy and HGNN/DAN/RESCHED are adapted on `la01`-`la20` with the
HGAT encoder frozen (200 episodes, learning rate `1e-5`). The zero-shot
checkpoint is kept as a candidate and the choice is made on `la21`-`la25` with
21 rollouts per instance. A 600-episode adaptation at `2e-4`, a longer unfrozen
adaptation and several weight averages were tested and did not improve the
validation gap. MLP+DQN, MLP+PPO and GAT+PPO keep their standard checkpoints.

The bounded refinement is an inference-time calibration of the decoder: it keeps
the learned priority and repairs dominated machine assignments produced by the
scale mismatch between the training geometry and a public instance. It is the
same operator, with the same budget, used by the main rule protocol. GA/TS
receive 51 schedule evaluations and the learning baselines 51 rollouts; the
refinement is not given to them, so the metaheuristic and learning blocks are
deployment comparisons rather than equal-budget claims.

## Code layout

- `PublicBenchmarkValidation/common/`: shared parser, split builder, rule
  decoder, statistics and the validation runner.
- `PublicBenchmarkValidation/Experiment1/`: hybrid rule-block refinement and
  Experiment 1 analysis.
- `PublicBenchmarkValidation/Experiment2/`: GA/TS implementations, runner and
  analysis.
- `PublicBenchmarkValidation/Experiment3/`: learned-baseline adapters, the
  policy-schedule refinement, parallel pre-computation helpers and Experiment 3
  analysis.

## Data and results

Raw public files are copied to `Public_DataSet/Brandimarte` and
`Public_DataSet/Lawrence`. The manifest, splits, checkpoints, per-instance
results and statistical tables live under
`Public_DataSet/PublicBenchmarkValidation`. Results and figures are nested as
`results/Combined/Experiment1`-`Experiment3` and
`Figure_And_File/PublicBenchmarkValidation/Combined/Experiment1`-`Experiment3`.

The original public files are never modified. For the learned methods, a
one-based working copy is generated under
`Public_DataSet/PublicBenchmarkValidation/internal/`; it only appends the two
metadata lines expected by the project's original reader (deterministic due
dates derived from processing times and neutral capability `1`). Job, machine,
eligibility and processing-time fields are unchanged.

Brandimarte reference values are the standard best-known makespans. Lawrence
values are the proven/best values recorded in the source repository's
`instances.json`; missing values are left empty rather than guessed.
