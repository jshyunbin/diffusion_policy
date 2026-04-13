# CLAUDE.md

## Commit Guidelines

- Do not add Co-Authored-By lines to commit messages.

## Repo Architecture

O(N+M) design: N tasks and M methods require O(N+M) code, not O(N*M). Tasks and methods are independent — code duplication is intentional for readability.

### Task side
- **Dataset**: `torch.utils.data.Dataset`, returns samples conforming to the interface. Has `get_normalizer()` → `LinearNormalizer`.
- **EnvRunner**: takes a `Policy`, runs rollouts, returns dict of logs/metrics (wandb-compatible). Uses `AsyncVectorEnv` for parallel evaluation.
- **Config**: `config/task/<task_name>.yaml` — contains `shape_meta`, `dataset`, and `env_runner`.

### Method side
- **Policy**: inherits `BaseLowdimPolicy` or `BaseImagePolicy`. Must implement `predict_action(obs_dict)` → `{'action': ...}`, `set_normalizer(normalizer)`, and optionally `compute_loss(batch)`.
- **Workspace**: manages training/eval lifecycle. Inherits `BaseWorkspace`. All training state as object attributes (auto-saved by `save_checkpoint`). The `run()` method contains the full pipeline.
- **Config**: `config/<workspace_name>.yaml` — `_target_` points to workspace class, `task: <task_name>` selects the task.

### Interface

**Image policy** input: `{"key0": (B, To, *), "key1": (B, To, C, H, W)}` → output: `{"action": (B, Ta, Da)}`

**Image dataset** returns: `{"obs": {"key0": (To, *), ...}, "action": (Ta, Da)}`

Terminology: `To` = n_obs_steps (observation horizon), `Ta` = n_action_steps (action horizon), `T` = horizon (prediction horizon).

### Key conventions
- `Policy` handles normalization internally via its copy of `LinearNormalizer` (saved in checkpoint).
- Workspace configs use hydra: `task=<task_name>` replaces the task subtree.
- `ReplayBuffer` uses zarr format. Arrays in `data/` concatenated along time axis, `meta/episode_ends` stores episode boundaries.
- Padding at episode boundaries is handled by `SequenceSampler`.

## Environment

- Python 3.12, venv at `/media/sunho/data/hyunbin/deep-reasoning-policy/old-repo/.venv`
- Environments use gymnasium (not gym)
- GPU: RTX 3090, memory-bandwidth bottlenecked (~384 samples/sec regardless of batch size)
- wandb project: `recursive-reasoning-robot-policy`, entity: `jhyunbin`
