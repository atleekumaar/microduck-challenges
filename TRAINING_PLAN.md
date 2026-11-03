# Training Plan: 2048-Environment Scaled Locomotion Experiment

**Document Version**: 1.0  
**Challenge**: `sprint-2m` (task `Mjlab-Sprint2m-MicroDuck`)  
**Experiment Name**: `sprint-scale-2048-1000`  
**Target Transitions**: 49,152,000 ($2,048 \text{ envs} \times 24 \text{ steps} \times 1,000 \text{ iterations}$)  
**Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), WSL2 Ubuntu 24.04  
**Status**: Prepared — **AWAITING USER APPROVAL TO LAUNCH**

---

## 1. Experiment Objective & Hypothesis

### Objective
Scale environment throughput from 128 to **2048 environments** to resolve the severe sample starvation ($32\times$ deficit) identified in `PROJECT_STATUS.md`, while maintaining complete GPU VRAM safety as validated in `GPU_CAPACITY_REPORT.md`.

### Single-Variable Hypothesis
> Scaling the environment count from 128 to 2048 while preserving the nominal reward weights and curricula will increase transition volume by $16\times$ (from 3.07M to 49.15M transitions), allowing the policy to overcome early balancing local optima and develop dynamic forward stepping.

---

## 2. Exact Execution Command

The experiment will be launched locally via `uv` in background mode with stdout/stderr piped to a dedicated log file:

```bash
uv run train Mjlab-Sprint2m-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 1000 \
    --agent.seed 1 \
    --agent.logger tensorboard \
    --agent.run-name sprint-scale-2048-1000 \
    > logs/sprint-scale-2048-1000.log 2>&1 &
```

*Process tracking command:*
```bash
echo $! > logs/sprint-scale-2048-1000.pid
```

---

## 3. Configuration & Parameter Preservation

In strict compliance with the rules, **no reward weights, curriculum schedules, or environment dynamics will be modified** for this experiment.

| Parameter | Value | Setting Source | Role |
|---|---|---|---|
| `num_envs` | `2048` | CLI Override | Parallel simulation environments |
| `max_iterations` | `1000` | CLI Override | PPO update iterations |
| `seed` | `1` | CLI Override | Benchmark reproduction seed |
| `num_steps_per_env` | `24` | `env.py` (`NUM_STEPS_PER_ENV`) | Rollout horizon per update |
| `save_interval` | `250` | `env.py` (`RL_CFG.save_interval`) | Checkpoint interval (0, 250, 500, 750, 999) |
| `forward_reward` | `2.0` | `challenge.toml` (nominal) | `track_linear_velocity` weight |
| `upright_reward` | `2.0` | `challenge.toml` (nominal) | `upright` posture weight |
| `step_reward` | `3.0` | `challenge.toml` (nominal) | `air_time` gait reward |
| `heading_hold` | `1.0` | `env.py` (nominal) | Heading drift constraint |
| `action_rate_weight` | `-0.1` -> `-0.8` | `env.py` (Curriculum) | Action smoothness ramp |
| `standing_envs` | `0.02` -> `0.15` | `env.py` (Curriculum) | Zero-velocity standing curriculum |
| `batch_size` | `49,152` | $2048 \times 24$ | Transitions per PPO iteration |
| `num_mini_batches` | `4` | `env.py` (`PpoWithSymmetryCfg`) | $12,288$ samples per mini-batch |
| `learning_rate` | `1.0e-3` | `env.py` (Adaptive schedule) | Base actor/critic learning rate |

---

## 4. Runtime & Resource Budget

Based on empirical measurements from `GPU_CAPACITY_REPORT.md` (where 2048 envs achieved 12,258 FPS and 4.01 s/iteration):

| Resource | Projected Utilization | Hardware Limit | Headroom / Margin |
|---|---|---|---|
| **GPU VRAM** | **2,481 MiB (40.4%)** | 6,141 MiB | **+3,660 MiB (59.6% free)** |
| **System RAM** | **3,736 MiB** | 7,785 MiB (WSL) | **+4,049 MiB (52.0% free)** |
| **GPU Temperature** | 52°C – 62°C | 87°C (Throttling) | > 25°C thermal buffer |
| **Wall-Clock Time** | **~67 minutes** ($1000 \times 4.01\text{ s}$) | N/A | Completed in single session |
| **Storage (Disk)** | **~12 MiB** | > 50 GB free | 5 checkpoints + TB events + logs |

### Storage Footprint Breakdown:
- Checkpoints (`model_0.pt`, `model_250.pt`, `model_500.pt`, `model_750.pt`, `model_999.pt`): $5 \times 1.2\text{ MB} \approx 6.0\text{ MB}$
- TensorBoard event log (`events.out.tfevents.*`): $\approx 2.5\text{ MB}$
- Final exported ONNX (`sprint-scale-2048-1000.onnx`): $\approx 0.8\text{ MB}$
- Console training log (`logs/sprint-scale-2048-1000.log`): $\approx 1.5\text{ MB}$

---

## 5. Live Monitoring & Health Metrics

During training, progress and learning health will be inspected at each checkpoint interval (iterations 250, 500, 750, 1000):

### Progress Tracking
```bash
grep -o "Learning iteration [0-9]*/[0-9]*" logs/sprint-scale-2048-1000.log | tail -1
```

### TensorBoard Metric Inspection Script
```bash
uv run python -c "
import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 4)) for e in ea.Scalars(tag)][-5:])
" logs/rsl_rl/sprint_2m/<RUN_DIR> Train/mean_episode_length Episode_Termination/fell_over Episode_Reward/track_linear_velocity Metrics/twist/error_vel_xy
```

### Health Diagnostics Criteria
- **Healthy Indicators**:
  - `Train/mean_episode_length` climbing toward 1000 steps (20.0s).
  - `Episode_Reward/track_linear_velocity` growing steadily above 0.20.
  - `Metrics/twist/error_vel_xy` decreasing toward 0.20 m/s.
  - `Episode_Termination/fell_over` declining below 0.10.
- **Warning Indicators**:
  - `Train/mean_reward` climbing while `track_linear_velocity` stays $< 0.05$ (indicates posture farming).
  - `Episode_Termination/nan_state` $> 0$ (unstable physics).

---

## 6. Checkpoint Evaluation & Resource Isolation

### Checkpoints to be Judged:
1. `model_250.pt` (Iter 250 — 12.28M transitions) -> `publish-eval-250/policy.onnx`
2. `model_500.pt` (Iter 500 — 24.57M transitions) -> `publish-eval-500/policy.onnx`
3. `model_750.pt` (Iter 750 — 36.86M transitions) -> `publish-eval-750/policy.onnx`
4. `model_999.pt` (Iter 1000 — 49.15M transitions) -> `publish-eval-999/policy.onnx`

### Verification Findings (Empirically Confirmed on 2026-11-04):
- **Dry-Run Target**: `publish --dry-run` writes to `./publish-<name>/` where `<name>` is derived from the repository stem (e.g., `--repo local/microduck-eval-250` writes to `./publish-eval-250/`).
- **Collision Prevention**: By naming the target repo `--repo local/microduck-eval-$CKPT`, each checkpoint exports to a distinct directory (`./publish-eval-$CKPT/`), guaranteeing zero overwriting between checkpoints.
- **ONNX Validation**: Verified via `onnx.checker.check_model()`, passing all topological and tensor checks.
- **Arena Acceptance**: Verified via `arena score`, confirming full compatibility across the 21 commanded speeds (0.20–1.20 m/s) and multi-seed evaluation.
- **Resource Isolation**: Verified that `publish ... --device cpu` and `arena score` operate entirely on CPU, utilizing 0 MiB of GPU VRAM.

### Verified Evaluation Command:
```bash
RUN_DIR=$(ls -td logs/rsl_rl/sprint_2m/*_sprint-scale-2048-1000 | head -1)

for CKPT in 250 500 750 999; do
  OUT_DIR="publish-eval-${CKPT}"
  echo "======================================================="
  echo "Exporting Checkpoint ${CKPT} to ./${OUT_DIR}/"
  echo "======================================================="
  
  uv run publish \
      --run "${RUN_DIR}" \
      --checkpoint "${CKPT}" \
      --repo "local/microduck-eval-${CKPT}" \
      --dry-run \
      --allow-dirty \
      --device cpu
      
  echo "Scoring Checkpoint ${CKPT} on Arena..."
  uv run arena score "${OUT_DIR}/policy.onnx" --event sprint-2m
done
```

---

## 7. Artifact Preservation & Lineage Tracking

- **Isolation**: The run will be saved in a new, timestamped directory:
  `logs/rsl_rl/sprint_2m/<YYYY-MM-DD_HH-MM-SS>_sprint-scale-2048-1000/`
- **Existing Artifact Protection**:
  - All existing runs (`sprint-smoke`, `sprint-pilot-128`, `sprint-train-1000`, `sprint-recovery-500`, and benchmark runs) remain intact.
  - `logs/diagnostics/` and `logs/reference/` will not be modified or overwritten.
- **Provenance Auto-Capture**:
  - `train` automatically writes `provenance.json` and a git diff capture in `git/microduck-challenges.diff`.

---

## 8. Rollback, Early-Stopping & Contingency Strategy

1. **Safety Cutoff**:
   - If GPU memory exceeds 5,000 MiB (81.4%) or system RAM free falls below 500 MiB, training will be halted immediately (`kill -TERM <PID>`).
2. **Early-Stopping on Numerical Collapse**:
   - If `Episode_Termination/nan_state > 0` persists for more than 5 iterations, stop the job.
   - If mean episode length collapses to $< 10$ steps after iteration 500 without recovery over 50 iterations, propose early termination to the user.
3. **Rollback State**:
   - If the run fails, no codebase files are modified. Reverting simply requires removing the specific `sprint-scale-2048-1000` run directory from `logs/rsl_rl/sprint_2m/`.
4. **Resume Capability**:
   - If training is interrupted (e.g. system reboot), it can be resumed seamlessly:
     ```bash
     uv run train Mjlab-Sprint2m-MicroDuck \
         --env.scene.num-envs 2048 \
         --agent.resume True \
         --agent.load-run <RUN_DIR_NAME> \
         --agent.load-checkpoint model_<LAST_CKPT>.pt \
         --agent.max-iterations 1000 \
         --agent.logger tensorboard \
         --agent.run-name sprint-scale-2048-1000-resumed
     ```

---

## 9. Pre-Flight Readiness Checklist

- [x] All 137 test suite assertions passing (`pytest tests/`).
- [x] GPU VRAM verified: RTX 4050 6GB has 3.66 GB headroom at 2048 envs.
- [x] System RAM verified: WSL2 has > 4.0 GB free memory.
- [x] Disk space verified: > 50 GB free (requires < 15 MB).
- [x] No concurrent GPU or python jobs active.
- [x] Output paths and naming non-conflicting (`sprint-scale-2048-1000`).
- [x] All original files clean in git (`git status` clean).
- [ ] **User approval received.**

---

> [!CAUTION]
> Training is NOT started. Awaiting explicit user confirmation to proceed.

