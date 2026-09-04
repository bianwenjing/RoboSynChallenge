# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: a `CLAUDE.md` also exists in the parent directory (`/scratch_local/wenjing/QwenV3-3Dcoding/CLAUDE.md`). It documents an unrelated project (Qwen-VL 3D code generation) and does **not** apply here.

## What this repo is

RoboSynChallenge is a benchmark/challenge repo of 10 bimanual manipulation tasks (sim2real). It is **an extension package on top of [EmbodiChain](https://dexforce.github.io/EmbodiChain/)** (**v0.2.4** — the docs say v0.2.3 but that version is missing `ReplayWrapper`; Python 3.11), not a standalone simulator. Nearly every base class (`EmbodiedEnv`, `ActionBank`, `Functor`, `LeRobotRecorder`, planners, IK/FK) lives in `embodichain.*`; this repo contributes task envs, action banks, JSON configs, assets, a few manager functors, and launch/eval scripts.

Expert demonstrations are generated procedurally (no teleop, no RL): a JSON **action graph** drives IK/motion planning to synthesize trajectories, which are recorded to LeRobot datasets and then used to train/eval policies (pi0, pi0.5, ACT, DP, Motus).

## Common commands

All commands run from the repo root; scripts `cd` there themselves.

```bash
# Collect expert demos:  <task> <random|clear> <3_0|2_1> [extra args]
bash launch/run_task.sh click_bell clear 3_0 --max_episodes 100
bash launch/run_task.sh -h                     # usage + list of task names

# Smoke-test one task without writing data
bash launch/run_task.sh mixer_operating random 3_0 --filter_dataset_saving --max_episodes 1 --headless

# Smoke-test every task (10 tasks x random+clear, headless, no saving)
bash launch/check_all_envs.sh                  # failures logged to /tmp/check_env_<task>_<setting>_*.log

# Replay a recorded trajectory
bash launch/replay_task.sh drawer_open_place random /path/to/state_action.pt --replay_mode dynamic

# Inspect domain randomization (loops env.reset + render)
bash launch/run_visualize.sh click_bell random --resets 100
bash launch/batch_run_visualize.sh random

# Collect clear + random and merge into one dataset
bash launch/collect_combined_dataset.sh click_bell 50 50 combined_v1

# Policy train / eval (per-policy wrappers -> scripts/eval_policy.py). Note the
# two different argument conventions -- see the Policies section.
bash policy/act/finetune.sh <dataset_root> <output_dir> 0 --steps 80000 --batch-size 64
bash policy/act/eval.sh click_bell random <checkpoint_path> 0 --headless True
bash policy/pi0/finetune.sh <train_config> <model_name> 0
bash policy/pi0/eval.sh click_bell random <train_config> <model_name> <ckpt_id> <gpu_id> --max_episodes 50

# Dataset format conversion (3.0 -> 2.1); run_task.sh does this automatically for format=2_1
python scripts/convert_lerobot3.0_to_2.1.py --repo-id <dataset_id> --root lerobot_dataset/<task>

# Find a safe position_range for a randomized object (IK-feasibility sweep + plot)
python scripts/analyze_rigid_spawn_range.py --gym_config configs/<task>/random/gym_config.json \
    --action_config configs/<task>/action_config.json --uid <object_uid> --list-events

# Docs (Sphinx)
cd docs && pip install -r requirements-docs.txt && rm -rf html && make html
```

There is **no test suite and no lint runner** wired up (`pyproject.toml` has an empty `[tool.black]` section). `launch/check_all_envs.sh` is the de-facto regression check — run it after touching env classes, action banks, or configs.

Extra args accepted by `run_task.sh` / `run_visualize.sh` are forwarded to EmbodiChain's arg parser: `--headless`, `--max_episodes N`, `--filter_visual_rand`, `--filter_dataset_saving`, `--preview`, `--debug_mode`.

## Architecture

### Layering

`scripts/run_env.py` (collection) and `scripts/eval_policy.py` (policy eval) are the two entry points. Both:
1. `import robosynchallenge` — this has **import side effects** and must happen before `gym.make`: it patches EmbodiChain's asset resolver (`data/asset_resolver.py`) so relative asset paths resolve against the repo root first, and swaps in this repo's `LeRobotRecorder` (`managers/datasets.py`) so relative `save_path`s land under the repo.
2. Append `robosynchallenge.managers.{actions,datasets,events,observations}` to `gym_utils.DEFAULT_MANAGER_MODULES` so functors named in JSON configs resolve here as well as in EmbodiChain.
3. Build `env_cfg` from `gym_config.json`, pass `action_config.json` through as kwargs, `gym.make(id=gym_config["id"], ...)`.

### Task envs (`robosynchallenge/tasks/<task>/`)

Each task is a pair of files:
- `<task>.py` — env classes registered with `@register_env("<Id>", max_episode_steps=N)`. The `id` in `gym_config.json` must match this string exactly.
- `action_bank.py` — an `ActionBank` subclass of `@staticmethod`s referenced by name from `action_config.json`.

Three env variants per task, all exported from `tasks/__init__.py` (which is what registers them):
- `<Task>Env` — data collection.
- `<Task>TestEnv` — policy-eval variant, subclass with different success thresholds/semantics.
- `<Task>AgentEnv` — mixes in `embodichain_tasks.tableware.base_agent_env.BaseAgentEnv`.

Tasks under `tasks/_other_tasks/` (and `configs/other_tasks/`) are competition-irrelevant; their imports are commented out in `tasks/__init__.py` and `check_all_envs.sh` skips them.

**Success semantics — two separate hooks, and they intentionally disagree in several tasks.** `compute_task_state()` returns `(success, fail, info)` and drives per-step termination during rollout; `is_task_success()` decides whether the episode counts (dataset saving in collection, the success counter in `eval_policy.py`). Collection envs commonly zero out `success` in `compute_task_state` so the scripted trajectory always plays to the end, while accumulating a latched flag (e.g. `ClickBellEnv._button_pressed`) that `is_task_success` returns. Some `*TestEnv`s hard-return `True` where success can only be judged manually. Read both methods before changing either.

### The action graph (the core abstraction)

`action_config.json` is an orchestration layer with no logic; `action_bank.py` is the computation layer. They communicate exclusively through the shared cache `env.affordance_datas`. Full detail is in `docs/tutorials/task_trajectory.md`; the essentials:

- `scope` — one directed graph per actuator (`left_arm`/`right_arm` dim 6, `left_eef`/`right_eef` dim 1), with an init strategy (`current_qpos` or `given_qpos`).
- `node` — pose/qpos generators. Decorated `@tag_node` in the bank; write results into `affordance_datas`. Most use the generic `generate_affordances_from_src` with `src_key` → `pass_processes` (`get_rotated_pose`, `get_offset_pose`, `get_fk_xpos`, `get_ik_ret`/`get_ik_qpos`, …) → `dst_key`.
- `edge` — trajectory generators. Decorated `@tag_edge`; connect `src`/`sink` node keys over `duration` steps (`plan_trajectory` via TOPPRA, `execute_open`/`execute_close` for grippers).
- `sync` — `depend_tasks` ordering constraints across the two arms and grippers.
- `misc` — `vis_graph` / `vis_gantt` debug switches (keep off for collection), `warpping` must stay on.

`@resolve_env_params` injects `env` into bank functions. `create_expert_demo_action_list` in each env maps the bank's per-part output (`left_arm`, `left_eef`, `right_arm`, `right_eef`) into a `(steps, num_envs, num_active_joints)` tensor via `env.active_joint_ids`. **Only `num_envs=1` is supported** by this mapping.

Trajectory generation can fail (IK infeasible after randomization). `run_env.py` loops: generate → if invalid, reset and retry, until `max_episodes` *successful, saved* episodes exist.

### Configs (`configs/<task>/`)

Resolution rule, shared by `launch/*.sh` and `scripts/eval_policy.py`:

```
gym_config:    configs/<task>/<setting>/gym_config.json          # setting = random | clear
action_config: configs/<task>/action_config.json                 # preferred
               configs/<task>/<setting>/action_config.json       # fallback
```

`random` enables domain randomization (lighting, camera intrinsics/extrinsics, materials/textures, robot init qpos and eef pose, object poses, distractor slots); `clear` is a fixed scene. Other `gym_config_*.json` files sitting at a task root (`_random_test`, `_table_height`, `_align_real`, `_dual`) are variants **not reachable through the standard launchers** — they must be passed explicitly to `python -m scripts.run_env --gym_config ...`.

`gym_config.json` holds `id` / `max_episodes` (number of *successful* episodes to collect) / `max_episode_steps`, plus `env.events` (randomization functors), `env.observations`, `env.dataset.lerobot`, `robot`, `sensor`, `light`, `background`, `articulation`, `rigid_object`. See `docs/tutorials/configuration.md`.

### Managers (`robosynchallenge/managers/`)

Repo-local functors layered on EmbodiChain's registry. `events.py` (~1400 lines) is the substantive one: entity-pose randomization groups, distractor slot replacement from `robosynchallenge/Distractor/ASSET_INDEX.json`, position-range visualization, debug helpers. `actions.py` and `observations.py` are currently stubs with empty `__all__` — the real terms come from EmbodiChain.

### Policies (`policy/<name>/`)

Vendored upstream training repos (pi0/pi05 = openpi, act, dp = LeRobot policies) plus a thin adapter each. `scripts/eval_policy.py` is the single generic evaluator; every policy directory supplies:
- `deploy_policy.py` — must define `get_model(usr_args)`, `eval(env, model, obs)`, `reset_model(model)`; optionally `encode_obs`/`encode_action`.
- `deploy_policy.yml` — the whole YAML is handed to `get_model` as `usr_args`.
- `eval.sh` — sets `CUDA_VISIBLE_DEVICES`/`PYTHONPATH`, then calls `scripts/eval_policy.py --config ... --overrides --key value ...`.
- `finetune.sh` — training entry point.

`policy/Your_Policy/` is the template; `docs/tutorials/policy/your_own_policy.md` is the reference. Actions are absolute joint control `[left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]` unless the task's `gym_config` declares an `actions` block. `eval.sh` assumes the sibling layout `<workspace>/{EmbodiChain,RoboSynChallenge}`; override with `EMBODICHAIN_ROOT` / `PYTHON_BIN`.

**The four policies split into two families that do not share a workflow.** Getting this wrong wastes a dataset conversion or a training run:

| | ACT / DP | PI0 / PI0.5 |
|---|---|---|
| Env setup | `cd policy/<p> && uv sync --frozen` | `GIT_LFS_SKIP_SMUDGE=1 uv sync`, then copy `src/openpi/models_pytorch/transformers_replace/*` into `.venv/.../site-packages/transformers/` |
| Dataset format | **LeRobot 3.0, read directly** via `--dataset-root` — no conversion | **LeRobot 2.1 required**, placed under `policy/<p>/training_data/` (`finetune.sh` sets `HF_LEROBOT_HOME` there) |
| Configuring a run | CLI flags only; no source edits | edit `repo_id` in the `_CONFIGS` dict of `src/openpi/training/config.py` (presets `pi0_base_robosynchallenge_full`, `pi05_base_robosynchallenge_full`) |
| Pre-training step | none | `uv run scripts/compute_norm_stats.py --config-name <cfg>` |
| `finetune.sh` args | `<dataset_root> <output_dir> [gpu] [flags…]` | `<train_config_name> <model_name> <gpu_use>` |
| `eval.sh` args | `<task> <setting> <checkpoint_path> [gpu] [opts…]` | `<task> <setting> <train_config> <model_name> [ckpt_id] [gpu] [opts…]` |
| Checkpoints | `<output_dir>/checkpoints/<id>/pretrained_model` (DDP: per-rank dirs — use `rank_0`) | `policy/<p>/checkpoints/<train_config>/<model>/<step>` |
| Memory | ACT >20 GB, DP >24 GB — single 4090/5090 | LoRA >46 GB; full >100 GB (2×A100 80G). Tune `fsdp_devices` / `batch_size`, or `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

Multi-GPU for ACT/DP is `torchrun --nproc_per_node=N scripts/train.py --distributed` with per-process `--batch-size` (recommended global batch 64); `finetune.sh` is the single-process launcher. Eval output for every policy lands in `eval_result/<task>/<policy>/<setting>/<train_config>/<model>/<ckpt|timestamp>/`, videos included.

**Released baselines** exist for ACT and DP only, as HF *model* repos containing a complete `pretrained_model/` dir — pass the download dir straight to `eval.sh`:

```bash
hf download RoboSynChallenge/ACT_sim_<task> --repo-type model --local-dir checkpoints/ACT_sim_<task>/
bash policy/act/eval.sh <task> random checkpoints/ACT_sim_<task> 0 --pytorch_device cuda --headless True
```

**Motus is documented but not present.** `docs/tutorials/policy/motus.md` is all `[TODO]` stubs and there is no `policy/motus/` directory, despite the README listing it as integrated.

To combine sim + real (the intended co-training path), merge with `lerobot-edit-dataset` from the repo root — this is what the root-level `merge_config.json` / `delete.json` job files are for:

```bash
export HF_LEROBOT_HOME=/path/to/RoboSynChallenge/
lerobot-edit-dataset --config_path merge_config.json
```

## Running the simulator on this machine (verified 2026-08-20)

**The host cannot run EmbodiChain directly.** `dexsim_engine` ships only
`manylinux_2_31` wheels (glibc >= 2.31); this host is Rocky 8 with **glibc 2.28**,
so pip reports `from versions: none`. Docker is installed but the daemon socket
is not accessible (user not in the `docker` group). The working path is
**Singularity 4.3.1** with the official image.

```bash
export SINGULARITY_CACHEDIR=/scratch_local/wenjing/.singularity_cache
singularity exec --nv \
  --bind /scratch_local/wenjing/QwenV3-3Dcoding:/workspace \
  --bind /scratch_local/wenjing/vulkan_icd:/vk \
  --bind /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin \
  --bind $HOME/.cache/embodichain_data:/root/.cache/embodichain_data \
  /scratch_local/wenjing/embodichain_cuda128.sif bash

# inside the container -- all three exports are required
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/vk/nvidia_icd.json
export PATH=/workspace/venv_embodichain/bin:$PATH
cd /workspace/RoboSynChallenge
bash launch/run_task.sh click_bell clear 3_0 --max_episodes 100 --headless
```

Always pass `--headless` — there is no X11 in the container.

**Why each bind/export is needed** (each one was a separate hard failure):

- `LD_LIBRARY_PATH` — the image ships a **0-byte stub** at
  `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`, and that directory precedes
  `--nv`'s injection dir in the default path, so the stub wins →
  `libnvidia-ml.so.1: file too short`. Prepending `/.singularity.d/libs` fixes it.
- `VK_ICD_FILENAMES` — the image's `/usr/share/vulkan/icd.d/` has intel, radeon,
  lvp and virtio ICDs but **no nvidia** one. `--nv` injects `libGLX_nvidia.so.0`
  but not an ICD manifest, so Vulkan enumerates zero devices →
  `Preferred GPU 0 not found in Vulkan candidates` / `RT_DEVICE_ERROR`. The file
  at `/scratch_local/wenjing/vulkan_icd/nvidia_icd.json` points at the injected
  library (`api_version` matches the host driver, 1.4.312).
- `nvoptix.bin` bind — `--nv` injects `.so` files only, not
  `/usr/share/nvidia/nvoptix.bin` (60 MB OptiX data blob), so
  `optixDenoiserCreate` fails with 7990 (`OPTIX_ERROR_LIBRARY_NOT_FOUND`) even
  though `libnvoptix.so.1` itself is present.

Setup state already on disk (do not redo): image
`/scratch_local/wenjing/embodichain_cuda128.sif` (19 GB SIF), venv
`/scratch_local/wenjing/QwenV3-3Dcoding/venv_embodichain` (13 GB, built from the
container's `/root/miniconda3/envs/py310`), assets in
`~/.cache/embodichain_data` (`EMBODICHAIN_DATA_ROOT` overrides; assets download
from `https://hf-mirror.com/datasets/dexforce/embodichain_data`).

**EmbodiChain must be v0.2.4, not the v0.2.3 the docs specify.**
`robosynchallenge/replay.py` imports `ReplayWrapper`, which does not exist in
v0.2.3 (its `wrapper/__init__.py` exports only `NoFailWrapper`) — collection
dies at import. v0.2.4 pins the same `dexsim_engine==0.4.3`, so switching tags
needs no reinstall. `docs/getting_started/installation.md` and the install
section's `git checkout tags/v0.2.3` are both wrong.

**`embodichain_tasks` is an undeclared, unobtainable dependency.** All 13 task
modules do `from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv`,
but the package is in neither the EmbodiChain repo, the private index, nor any
extra (`RoboSynChallenge/pyproject.toml` declares no dependencies at all). A stub
lives at `venv_embodichain/.../site-packages/embodichain_tasks/` so
`robosynchallenge.tasks` imports cleanly. Consequence: **`<Task>AgentEnv`
variants raise if used**; `<Task>Env` (collection) and `<Task>TestEnv` (eval) are
unaffected.

**`scripts/run_env.py` had a data-corruption bug (patched here).** `env.close()`
sat only in the `except` branch, so a run that finished normally never called it
— and `env.close()` is what triggers `LeRobotRecorder.finalize()` →
`dataset.finalize()`, which writes the parquet footer. Every successful
collection therefore produced truncated `data/` and `meta/episodes/` parquets
(`Parquet magic bytes not found in footer`) plus a `Segmentation fault (core
dumped)` at exit. Moving `env.close()` into a `finally` fixes both. Backup of
the original: `/tmp/run_env.py.bak`.

Sanity numbers from a verified run: `click_bell clear`, ~13 s/episode,
~18 step/s, 3 episodes x 74 frames = 222 frames, videos are **av1** 640x480 @
25 fps. Episode length always equals the action graph's summed edge durations.

## Adding a new articulated asset (drawer pipeline, 2026-08-30)

Three scripts turn an arbitrary two-body prismatic drawer URDF into a
`drawer_open_place`-ready asset. They exist because every offset in
`configs/drawer_open_place/action_config.json` is written against the official
`SlidingBoxDrawer`, and copying its constants to a different-sized drawer fails
silently (wrong grasp point, interpenetrating shells, unreachable IK).

```bash
# 0. One-off: the decomposition backend step 1 needs.
pip install coacd            # into meshcoder_env2

# 1. Geometry -> asset. Eight quantities derived, none hand-tuned.
#    Also CoACD-decomposes both links into meshes/<link>_acd.obj (~3 s each).
python scripts/make_drawer_asset.py <any_drawer.urdf> assets/MyDrawer --name my_drawer

# 1b. Asset -> gym_config. Solves init_pos so the world-frame grasp point lands
#     where --template's drawer put it, so the template's IK calibration holds.
python scripts/make_drawer_gym_config.py assets/MyDrawer/my_drawer.urdf \
    configs/drawer_open_place/my_drawer_gym_config.json

# 2. IK reachability grid (grasp height x cabinet distance). ~3 min.
python -m scripts.calibrate_drawer_grasp --asset assets/MyDrawer/my_drawer.urdf \
    --gym_config <cfg> --action_config <cfg> --num_envs 1 --headless --filter_dataset_saving

# 3. Physics search, objective = slide-joint travel fraction (not IK success).
python -m scripts.calibrate_drawer_physics --asset assets/MyDrawer/my_drawer.urdf \
    --gym_config <cfg> --action_config <cfg> --num_envs 1 --headless --filter_dataset_saving
```

`make_drawer_gym_config.py` fills the scene config from the asset: `fpath` for
both `drawer` and `drawer_background`, `init_pos` (solved, see below),
`drawer_pull_distance`, `drawer_inner_offset_xyz`, and the `attrs` whitelist.
The reachable grasp point is not a constant in the script — it is measured off
whatever `--template` you pass, so re-calibrating the template propagates. It
reproduced the hand-written `ep0_drawer_gym_config.json` with zero field
differences.

`make_drawer_asset.py` derives: link renaming (`outer_box`/`inner_box`), yaw to
align the slide axis with +X, `handle_xpos` from the handle bar behind its front
face, a CoACD convex decomposition per link, the `drawer_background` stacking
gap, `drawer_pull_distance`, and `drawer_inner_offset_xyz`. Everything lands in `<name>_constants.json`, whose
`physics` block (friction, `eef_close_ratio`, drive gains) is the search space
for step 3.

**Order matters: geometry first, then calibrate.** Re-running step 1 overwrites
what steps 2-3 wrote back — the scripts have no shared state. Calibrating on
wrong geometry wastes the run (it happened here: a calibration done against a
mis-rendered asset produced a 3-cell feasible region; after fixing the geometry
the same sweep found 9).

**Traps hit while building this, all verified:**

- **Never rotate an asset with `<visual rpy>`.** The renderer composes it
  differently from collision shapes whose coordinates you pre-rotated offline,
  so the picture and the physics disagree while `get_link_pose` still reports
  correct values. Bake the rotation into the mesh vertices, export, and leave
  every `rpy="0 0 0"`.
- **`init_rot` is dead for these tasks.** `random_drawer_pose` uses
  `randomize_entity_root_pose_group` with `relative_rotation: false`, i.e. an
  absolute orientation that overwrites it. Change `rotation_range`, or bake the
  orientation into the URDF.
- **dexsim cannot build and tear down an env twice in one process** — the second
  `gym.make` segfaults. Sweeps must either fork a subprocess per sample or, far
  better, probe inside one env: IK reachability depends only on the target pose,
  so overwriting `affordance_datas["drawer_handle_pose"]` samples the grid ~60x
  faster than rebuilding.
- **Print results before `env.close()`.** Teardown segfaults, so anything after
  it (including a `finally` block's tail) never runs.
- **`max_convex_hull_num` / `acd_method` are `MeshCfg` fields and are ignored on
  articulations** (`Key not found in ArticulationCfg` warning), so the engine
  will not decompose a link's collision mesh for you — hand it a concave tray
  whole and it gets hulled, cavity filled.
- **Decompose offline; never synthesise convex boxes from a bounding box.**
  `make_drawer_asset.py` used to build a "floor + front face + 4 walls" shell
  per link because no decomposition backend was installed. That fabricates
  surfaces the mesh does not have, and it cost two silent failures: a wall
  across the cabinet's own mouth (travel capped at 9.8 mm of a 180 mm rail) and
  a door panel placed at the handle tip (the hook gap filled in). Both needed
  their own special case to undo. `decompose()` now runs CoACD
  (`pip install coacd`) and writes a multi-`o` OBJ referenced by a single
  `<collision><mesh>` — exactly what the official `SlidingBoxDrawer` ships
  (`OuterBoxACD.obj` 13 hulls, `InnerBoxACD.obj` 15, with the handle bar and its
  two stems as separate hulls). A decomposition only produces geometry where the
  mesh has geometry, so neither failure can recur and all three special cases
  are gone. Our drawer: 8 hulls per link, `|q|` max 0.1358 vs 0.1344 for the
  box shell it replaced.

**Two bugs a second drawer exposed** (`drawer_ep0`, limit `[-0.160, +0.085]`):

- **`travel` was `|hi - lo|`.** That is the limit *span*, not the distance from
  the rest pose. For a drawer resting at 0 that can only open to -0.160, it
  gave 0.245 and a pull 36 mm past the stop. Now `abs(open_q - q_rest)`. The
  first two drawers' limits (`0…0.200`, `-0.180…0`) make both formulas agree,
  which is why it stayed hidden.
- **`drawer_pull_distance` was a dead parameter** — 0 references in
  `action_config.json`, `robosynchallenge/`, or `EmbodiChain/`. The real pull
  was hard-coded as `"offset_value": -0.14` at `action_config.json:951`, so
  every drawer got the same 0.14 regardless of its own travel — **any drawer
  with less than 0.14 m of usable travel would have jammed against the stop.**
  Root cause: the key was never in the `attrs` whitelist that
  `prepare_extra_attr` copies into `env.affordance_datas`, so the action graph
  could not see it even though the config defined it. Fixed by pointing
  `offset_value` at `"env.affordance_datas['drawer_pull_distance']"`
  (`resolve_env_attr` evals any `env.`-prefixed string, recursing into kwargs)
  and adding the key to the whitelist in all five drawer gym_configs.
  The official configs' value was changed `-0.12` → `-0.14` at the same time:
  the `-0.12` had never taken effect, and leaving it would have silently
  shortened the official task's pull by 20 mm. Verified behaviour-preserving —
  official `|q|` 0.1422 → 0.1421, while ep0_v2 and ep0 separated from a shared
  0.1358 to 0.1399 and 0.1239, each ~97 % of its own asset-derived pull.

**The `handle_xpos` rule, validated against the official asset.** `detect_handle`
walks back from the front plane to the first slice whose y-span reaches 85 % of
the tray width — that is the door panel. Everything in front of it is the handle
bar. The grasp point is then

    x = bar_rear_x - min(2 mm, gap/2)      gap = bar_rear_x - panel_x
    y, z = bar bbox centre

Run against `SlidingBoxDrawer/InnerBox.stl` this reproduces the official
hand-placed `handle_xpos = (0.105, 0, 0.1)` to **0.00 mm on all three axes**
(panel x=0.0905, bar x∈[0.107, 0.117], gap 16.5 mm → 12.0 mm behind the tip).
Use that as the regression test whenever the rule changes.

**How the gripper actually grasps** (measured, both drawers, finger links
expressed in `inner_box` local coords at peak travel):

| | official | ours, after the fix |
|---|---|---|
| `right_link7` | `[0.1121, -0.0004, 0.0290]` | `[0.0180, 0.0037, -0.0208]` |
| `right_link8` | `[0.1039, -0.0005, 0.0429]` | `[0.0087, 0.0036, -0.0049]` |
| `handle_xpos` | `[0.1050, 0.0000, 0.0350]` | `[0.0035, 0.0039, -0.0150]` |
| `|q|` max | 0.1422 / 0.2 rail | 0.1345 / 0.18 rail |

The two fingers straddle the bar **in z**, one below and one above its centre —
it is a vertical pinch on the bar, not a hook through the gap. Both drawers show
the same pattern, which is what confirms the offline mesh frame and the
simulator frame agree (this closes the earlier "horizontal bar vs vertical
handle" question — the bar is horizontal, and the pinch axis is vertical).

**Re-running step 1 changes `handle_reach`, so the cabinet must move with it.**
Moving the grasp point 7.9 mm deeper into the tray moves the world-frame target
7.9 mm away from the robot. Subtract the same delta from every `init_pos[0]`
(`drawer` *and* `drawer_background`) so the previously calibrated reachable
region still holds — otherwise the IK sweep from step 2 is invalidated.

## Conventions and gotchas

- Datasets are written to `lerobot_dataset/<task_name>/` (LeRobot 3.0 native; `format=2_1` converts in place and deletes the 3.0 copy). `run_task.sh` ends with `chmod 777 -R lerobot_dataset/` — a leftover of the Docker workflow.
- **Two dataset path conventions coexist.** Collection writes `lerobot_dataset/<task>/cobotmagic_Sim_<task>_<NNN>` (from each `gym_config`'s `save_path`), but the download docs and every training example assume the canonical LeRobot id `lerobot_dataset/RoboSynChallenge/cobotmagic_{Sim,Real}_<task>`. Move or merge locally collected runs into that layout before training.
- Gripper joints are normalized to `[0,1]` in observations/datasets; `robosynchallenge/replay.py` denormalizes against `qpos_limits` when replaying, and several envs have a `_normalize_eef_action_for_dataset` hook. Keep both sides in sync when touching gripper representation.
- Assets: prefer adding under `assets/` or `robosynchallenge/Distractor/` with repo-relative paths in configs — the patched resolver finds them before falling back to EmbodiChain's downloader. Some existing configs still hardcode `/root/workspace/RoboSynChallenge/...` texture paths from the Docker image.
- On modest GPUs (≈RTX 5060 Ti) cap collection at ~500 episodes per run and merge afterwards; larger batches destabilize collection.
- Docs are mixed-language by design: `docs/` is the published English Sphinx site (`docs/html/` is checked-in build output — regenerate, don't hand-edit), while `launch/README.md` and `scripts/README.md` are Chinese. Match the file you are editing.
- `merge_config.json` / `delete.json` at the root are one-off `lerobot-edit-dataset` job files, not project config.
- **The docs are ahead of / behind the code in three known places.** Trust the code: (1) every docs page lists 11 tasks including `open_pan`, but its env imports are commented out in `tasks/__init__.py` and its configs live under `configs/other_tasks/`, so neither the launchers nor `eval_policy.py` can resolve it — `launch/run_task.sh -h` lists the real 10; (2) `scripts/README.md` references `configs/eval/pi0_embodichain.yml`, which does not exist (the live path is `policy/<name>/deploy_policy.yml`); (3) installation docs and several config snippets still assume Docker absolute paths (`/root/workspace/...`).
