# MicroDuck Arena — Sprint-2m Project Audit & Status Report

**Date**: 2026-11-04  
**Target Event**: `sprint-2m` (2 m Sprint, task `Mjlab-Sprint2m-MicroDuck`)  
**Hardware Profile**: NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM), WSL2 Ubuntu 24.04, 16 GB System RAM  
**Repository Branch**: `main` (commit `4b675ebf4d7c7b961cd78da94bfc37ca1686d321`)  
**Current Status**: **SUCCESS — 5/5 Seeds Completed in 6.849s on Arena**

---

## 1. What Has Already Been Implemented

### Environment & Challenge Registration
- **Task Registration**: The `sprint-2m` challenge is registered via `src/microduck_challenges/sprint_2m/tasks.py` and configured through `challenge.toml` and `src/microduck_challenges/sprint_2m/env.py`.
- **Policy Contract**: 
  - Observation dimension: 61 floats (gyro [3], projected gravity [3], joint positions delta from default pose [14], joint velocities [14], last action [14], commands [13: twist vx/vy/vz, head pose deltas 4D, body pose deltas 6D]).
  - Action dimension: 14 joint position targets applied relative to default stand pose at 50 Hz control frequency.
  - Joint layout: Left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), neck_pitch, head_pitch, head_yaw, head_roll, Right leg (same order).
- **Domain Randomization**:
  - Center of Mass (CoM) perturbation: ±3 mm initial, ramped to ±8 mm via curriculum.
  - Head CoM randomization: applied per-episode across neck and head assembly bodies.
  - Mass and inertia scaling: ±5% applied together.
  - Actuator friction and reflected armature randomization: ±10% on BAM friction budget and reflected rotor inertia.
  - Sensor disturbances: velocity pushes (±0.3 m/s every 3–6 s), IMU orientation mounting error (up to 6°), joint encoder calibration bias (±0.86°).
- **Reward Structure**:
  - Velocity tracking: `track_linear_velocity` (weight 2.0, Gaussian std $\sqrt{0.1} \approx 0.316$).
  - Posture & Stability: `upright` (weight 2.0, std $\sqrt{0.05} \approx 0.223$), `heading_hold` (weight 1.0, std 0.4), `pose` penalty, `head_pose_tracking` (weight 2.0), `head_pose_bias` (EMA-based).
  - Regularization: `action_rate_l2` (curriculum-ramped from -0.1 to -0.8), `dof_pos_limits`, `foot_slip`, `foot_clearance`, `foot_swing_height`, `self_collisions`.
- **Curricula**:
  - `standing_envs`: ramps relative share of standing (zero-velocity command) environments from 2% at step 0 to 25% at step $2000 \times 24$.
  - `action_rate_weight`: ramps action smoothness penalty from -0.1 to -0.8 across iterations.
  - `head_pose_bias_weight`: ramps head posture bias penalty from 0.0 to 2.0.
- **Verification Suite**:
  - 137 unit tests covering Arena actuator parity with BAM, task registrations, got-a-talent, and tug-of-war (`uv run --group workshop pytest`). All 137 tests pass.

---

## 2. Training Experiments History

### A. Historical Runs (October 2026) — Sample Starved
1. **Smoke Run (`2026-10-29_13-07-53_sprint-smoke`)**:
   - 128 envs, 2 iterations. Sanity check.
2. **Pilot Run (`2026-10-29_13-25-45_sprint-pilot-128`)**:
   - 128 envs, 100 iterations. Early posture learning.
3. **Training Run 1000 (`2026-10-29_13-58-25_sprint-train-1000`)**:
   - 128 envs, 1000 iterations (3,072,000 transitions).
   - Mean reward reached ~9.14, episode length reached ~203.
   - Result: Failed Arena evaluation (0/5 finished, 1.36 mm distance). Robot learned posture farming.
4. **Recovery Run 500 (`2026-10-29_16-04-54_sprint-recovery-500`)**:
   - 128 envs, 500 resumed iterations (1,536,000 transitions).
   - Mean reward reached ~10.42. Robot remained trapped in posture-holding minimum.

### B. Hardware Capacity Benchmark (November 2026)
- Evaluated scaling limits on local RTX 4050 6GB GPU:
  - 512 envs: 795 MB VRAM (12.9%), 2,506 FPS.
  - 1024 envs: 1,319 MB VRAM (21.5%), 5,356 FPS.
  - **2048 envs**: 2,481 MB VRAM (40.4%), 12,258 FPS — **Selected Sweet Spot (3.66 GB safe VRAM headroom)**.
  - 4096 envs: 4,933 MB VRAM (80.3%), 17,966 FPS.

### C. Scaled Training Experiment (`sprint-scale-2048-1000`) — SUCCESS
- **Configuration**: 2048 parallel environments, 24 steps per environment, 1000 iterations (49,152,000 transitions).
- **Runtime**: 59 minutes 54 seconds on local RTX 4050 GPU.
- **VRAM Utilization**: Stably held at 2,481 MiB / 6,141 MiB (40.4%).
- **Numerical Stability**: Zero NaN states, fell-over termination dropped from 1.00 to 0.125, mean episode length reached ~940 / 1000 steps.
- **All Checkpoints Saved**: `model_0.pt`, `model_250.pt`, `model_500.pt`, `model_750.pt`, `model_999.pt`.

---

## 3. Checkpoints and ONNX Artifacts

| Directory | Artifact | Source | Checkpoint | Status |
| :--- | :--- | :--- | :---: | :--- |
| `logs/rsl_rl/sprint_2m/2026-11-03_18-48-27_sprint-scale-2048-1000/` | `model_0.pt` to `model_999.pt` | RSL-RL PyTorch | 0, 250, 500, 750, 999 | Complete |
| `publish-eval-250/` | `policy.onnx`, `manifest.json` | `publish --dry-run` | 250 | Validated |
| `publish-eval-500/` | `policy.onnx`, `manifest.json` | `publish --dry-run` | 500 | Validated |
| `publish-eval-750/` | `policy.onnx`, `manifest.json` | `publish --dry-run` | 750 | Validated |
| `publish-eval-999/` | `policy.onnx`, `manifest.json` | `publish --dry-run` | 999 | **BEST MODEL** |
| `logs/reference/sprint-2m/` | `policy.onnx`, `manifest.json` | Pollen Official | Reference | Benchmark |

---

## 4. Arena Evaluation Results

All checkpoints evaluated with `arena score` across the 21 commanded speed sweep (0.20–1.20 m/s) on 5 seeds:

| Checkpoint / Policy | Optimal Command ($v_x$) | Finishes | Arena Score | Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Historical 128-env** | 0.20 m/s | **0 / 5** | 50.000s | 1.36 mm forward travel (posture farming trap) |
| **Checkpoint 250** | 0.25 m/s | **0 / 5** | 50.000s | Initiates walk, advances ~0.78m before falling |
| **Checkpoint 500** | 0.90 m/s | **5 / 5** | 26.356s | **First 100% completion!** Cautious, careful gait |
| **Checkpoint 750** | 0.45 m/s | **5 / 5** | 9.015s | Steady, smooth 2m sprint |
| **Checkpoint 999** | **0.45 m/s** | **5 / 5** | **6.849s** | **BEST: Fast, dynamic, balanced 2m sprint** |
| **Pollen Reference Baseline** | 0.55 m/s | **5 / 5** | **4.886s** | 4096-env reference recipe target |

### Checkpoint 999 Per-Seed Breakdown:
- Seed 1: **6.64s**
- Seed 2: **6.91s**
- Seed 3: **6.98s**
- Seed 4: **6.88s**
- Seed 5: **6.84s**
- **Overall Arena Score**: **6.849s (5/5 Finished)**

---

## 5. What Is Working and What Has Been Solved

1. **Sample Starvation Solved**: Scaling environments from 128 to 2048 provided $16\times$ higher sample diversity per PPO update, breaking through the posture-holding local minimum.
2. **Stable Bipedal Locomotion Discovered**: The robot successfully learned a dynamic, coordinated gait that crosses the full 2m track across all random seeds without falling.
3. **Collision-Free Checkpoint Publishing**: Checkpoint export via `publish --repo local/eval-<CKPT> --dry-run --device cpu` cleanly isolated models into separate directories without overwriting.
4. **Clean Resource Headroom**: RTX 4050 6GB GPU handled 2048 environments with 3.66 GB VRAM to spare, running at ~13,680 FPS.

---

## 6. What Remains Incomplete / Next Opportunities

1. **Gap to Reference Baseline (6.849s vs 4.886s)**:
   - Checkpoint 999 scored best at commanded speed $0.45\text{ m/s}$.
   - Higher commanded speeds ($0.55$–$0.65\text{ m/s}$) finished in 7.24s–8.59s, but commands above $0.65\text{ m/s}$ occasionally fall around 1.3–1.7m.
   - Pushing the speed beyond $0.55\text{ m/s}$ will close the gap to the 4.8s baseline.
2. **Arena Leaderboard Entry**:
   - `publish --enter` requires Hugging Face authentication and a public model repository.

---

## 7. Next Five Engineering Actions in Priority Order

1. **Action 1: User Decision on Leaderboard Submission**
   - *Goal*: Confirm if the user wishes to publish Checkpoint 999 (6.849s, 5/5 finishes) to Hugging Face and enter it on the Arena leaderboard now.

2. **Action 2: Velocity Tracking Curriculum Tuning for Higher Speeds**
   - *Goal*: Increase velocity tracking weight / broaden Gaussian kernel at speeds $> 0.60\text{ m/s}$ to encourage faster forward foot swing.

3. **Action 3: Lateral Balance & Heading Regularization Calibration**
   - *Goal*: Fine-tune `heading_hold` and foot clearance to eliminate the late-run lateral yaw drift observed at speeds $> 0.70\text{ m/s}$.

4. **Action 4: Scaled Sprint Experiment (Target: Sub-5.0s)**
   - *Goal*: Launch a follow-up 2048-env training run initialized from `model_999.pt` (or trained with optimized speed curriculum) to target sub-5.0s sprint times.

5. **Action 5: Video Replay Generation**
   - *Goal*: Record an MP4 replay using MuJoCo viewer/headless renderer of the 6.849s run for Hugging Face model card documentation.
