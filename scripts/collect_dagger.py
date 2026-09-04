"""DAgger / DART data collection on top of the scripted action-graph expert.

Both modes need "execute action A, but record action B":
  DART   — execute expert+noise, record the clean expert action.
  DAgger — execute the learner (or expert, per beta), record the expert action.

That is possible because EmbodiChain's LeRobotRecorder does not read the action
passed to env.step(). It reads env.rollout_buffer["actions"], which
_hook_after_sim_step fills in after each step. Overwriting that slot right after
stepping therefore changes what gets recorded without changing what was executed.

Usage
    python -m scripts.collect_dagger --gym_config configs/<task>/random/gym_config.json \
        --action_config configs/<task>/action_config.json --num_envs 1 --headless \
        --max_episodes 50 --dagger_mode dart --dart_sigma 0.02

    # DAgger needs a learner; point --policy_config at a policy deploy yml
    ... --dagger_mode dagger --beta 0.5 --policy_config policy/act/deploy_policy.yml \
        --replan_every 10
"""

import argparse
import numpy as np
import torch
import tqdm
import gymnasium as gym

import robosynchallenge  # noqa: F401  (import side effects: asset resolver + recorder)
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.utils.logger import log_info, log_warning

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]

# Functors that re-read object poses into env.affordance_datas. They are
# registered with mode="reset", so a mid-episode replan would otherwise use
# stale poses (e.g. a drawer already pulled open, a tomato already lifted).
_REFRESH_FUNCTORS = ("prepare_extra_attr", "register_info_to_env")


def refresh_affordances(env) -> None:
    """Re-run only the pose-reading reset events, leaving randomization alone."""
    base = env.unwrapped
    em = getattr(base, "event_manager", None)
    if em is None:
        return
    env_ids = torch.arange(base.num_envs, device=base.device)
    for cfg in em._mode_functor_cfgs.get("reset", []):
        name = type(getattr(cfg, "func", None)).__name__
        fn = getattr(cfg, "func", None)
        fname = name if name != "function" else getattr(fn, "__name__", "")
        if fname in _REFRESH_FUNCTORS or getattr(fn, "__name__", "") in _REFRESH_FUNCTORS:
            try:
                fn(base, env_ids, **dict(cfg.params))
            except Exception as exc:  # a failed refresh must not kill collection
                log_warning(f"refresh_affordances: {fname} failed: {exc}")


def expert_replan(env):
    """Expert action chunk from the CURRENT state, or None if IK failed.

    Works mid-episode because every task declares scope init=current_qpos, so the
    graph plans from wherever the arm currently is.
    """
    refresh_affordances(env)
    try:
        return env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
    except Exception as exc:
        log_warning(f"expert_replan failed: {exc}")
        return None


def _record_action(env, slot: int, action: torch.Tensor) -> None:
    """Overwrite what the recorder will save for this step."""
    base = env.unwrapped
    buf = getattr(base, "rollout_buffer", None)
    if buf is None or slot is None:
        return
    tgt = buf["actions"]
    if slot >= tgt.shape[1]:
        return
    tgt[:, slot, ...].copy_(action.to(tgt.device), non_blocking=True)


def rollout(env, args, learner=None) -> bool:
    """One episode. Returns False if the expert could not produce a plan."""
    chunk = expert_replan(env)
    if chunk is None or len(chunk) == 0:
        return False

    horizon = len(chunk)
    mode = args.dagger_mode
    beta = float(args.beta)
    sigma = float(args.dart_sigma)
    replan_every = max(1, int(args.replan_every))

    cursor = 0
    for t in tqdm.tqdm(range(horizon), desc=f"{mode} rollout", unit="step"):
        if mode == "dagger" and t > 0 and t % replan_every == 0:
            fresh = expert_replan(env)
            if fresh is None or len(fresh) == 0:
                log_warning(f"replan failed at t={t}; ending episode early.")
                break
            chunk, cursor = fresh, 0

        if cursor >= len(chunk):
            break
        a_expert = chunk[cursor].clone()
        cursor += 1

        if mode == "dart":
            a_exec = a_expert + torch.randn_like(a_expert) * sigma
        elif learner is not None and np.random.rand() >= beta:
            a_exec = learner(env)
        else:
            a_exec = a_expert

        slot = getattr(env.unwrapped, "current_rollout_step", None)
        env.step(a_exec)
        _record_action(env, slot, a_expert)   # execute a_exec, record a_expert
    return True


def main():
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--dagger_mode", choices=["dart", "dagger"], default="dart")
    parser.add_argument("--dart_sigma", type=float, default=0.02,
                        help="Gaussian noise std added to executed actions (radians).")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="DAgger: probability of executing the expert. 1.0 = pure expert.")
    parser.add_argument("--replan_every", type=int, default=10,
                        help="DAgger: re-query the expert every N steps.")
    parser.add_argument("--policy_config", type=str, default=None,
                        help="DAgger: deploy_policy.yml of the learner. Omit for beta=1.")
    args = parser.parse_args()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    if args.max_episodes is not None:
        gym_config["max_episodes"] = args.max_episodes
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    learner = None
    if args.dagger_mode == "dagger" and args.policy_config:
        log_warning("Learner loading is a stub: wire deploy_policy.get_model here.")

    target = int(gym_config.get("max_episodes", 1))
    log_info(f"[{args.dagger_mode}] collecting until {target} saved episodes.", color="green")
    try:
        env.reset()
        saved = 0
        while saved < target:
            ok = rollout(env, args, learner)
            env.reset(options={"save_data": ok})
            saved += int(ok)
            log_info(f"  saved {saved}/{target}", color="green")
    finally:
        env.close()   # triggers LeRobotRecorder.finalize() -> writes parquet footer


if __name__ == "__main__":
    main()
