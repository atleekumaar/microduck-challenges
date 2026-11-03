# MicroDuck Arena — Project Handover

## 1. Project
MicroDuck Arena — Sprint-2m Reinforcement Learning Challenge.

Goal: Train a quadruped/robot locomotion policy that completes the 2m sprint and achieves 5/5 successful seeds.

## 2. Hardware and Environment
- OS: Windows 11 + WSL Ubuntu
- GPU: NVIDIA RTX 4050, 6GB VRAM
- RAM: 16GB
- Project: ~/projects/microduck-challenges

## 3. Training Work Completed
- PPO training with 128 parallel environments.
- Original training: 1000 iterations.
- Recovery training: additional 500 iterations.
- ONNX exports generated.
- TensorBoard logs collected.
- Arena scoring and trace recording executed.

Original run:
logs/rsl_rl/sprint_2m/2026-10-29_13-58-25_sprint-train-1000/

Checkpoint:
model_999.pt

Recovery run:
logs/rsl_rl/sprint_2m/2026-10-29_16-04-54_sprint-recovery-500/

## 4. Training Metrics
Original training:
- Mean reward: 0.15789 -> 9.14013
- Mean episode length: 20.09 -> 203.03
- Upright reward: 0.00904 -> 0.24590
- Linear velocity tracking reward: 0.00109 -> 0.04344
- XY velocity error: 0.02024 -> 0.34897
- Fell-over termination metric increased.
- Action-rate penalty curriculum: -0.1 -> -0.4

## 5. Arena Evaluation
- 21 commanded speeds: 0.20–1.20 m/s
- 5 seeds
- Original trained policy: 0/5 finishes
- All trials timed out.
- Scored at 1.05 m/s.
- Score: 50 seconds.

Trace:
logs/diagnostics/sprint-train-1000/traces/01M6H8MV8TCZRSFAY0TDFB.npz

Run metadata:
logs/diagnostics/sprint-train-1000/runs/sprint-2m/01M6H8MV8TCZRSFAY0TDFB.json

## 6. Trace Findings
- Forward displacement approximately 0.00136m.
- Robot remained almost in the same position.
- Joint movement was very small.
- No meaningful forward locomotion observed.

## 7. ONNX Validation
- Input shape: [1, 61]
- Output shape: [1, 14]
- ONNX checker passed.
- Inference outputs finite.
- Observation normalizer appears inside ONNX graph.

## 8. Known Environment Details
- Observation size: 61
- Action size: 14
- Command forward velocity at observation index 48.
- Policy action is applied relative to default joint pose.
- Arena evaluation is the final success authority.

## 9. Current Diagnosis — NOT YET CONFIRMED
Training reward improved, but velocity tracking remained weak.
The policy appears to have learned posture/standing behavior rather than effective forward locomotion.

Possible areas to investigate:
1. Training versus Arena observation parity.
2. Checkpoint versus ONNX inference parity.
3. Action scaling and joint ordering.
4. Reward shaping and curriculum.
5. Simulation timestep, control frequency and actuator parity.
6. Reset and command handling.

Do not assume any of these is the root cause without evidence.

## 10. Instructions for the Engineering Agent
- First perform a read-only audit.
- Inspect source code, configs, checkpoints, TensorBoard logs, ONNX files and Arena traces.
- Reconstruct the actual current project state.
- Identify completed work, incomplete work and confirmed failures.
- Do not overwrite existing checkpoints or logs.
- Do not start long training without approval.
- Do not change reward weights blindly.
- Compare checkpoint and ONNX outputs on identical observations.
- Produce a root-cause report with file paths and evidence.
- Propose a minimal corrective plan before implementation.
- Validate all improvements using Arena evaluation.
