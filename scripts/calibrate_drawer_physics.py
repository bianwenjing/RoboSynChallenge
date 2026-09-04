"""Search physics parameters against how far the drawer actually opens.

calibrate_drawer_grasp.py maximises IK success, which only proves the arm can
reach the grasp pose -- not that the fingers can drag the drawer. This script
optimises the thing that matters: travel fraction of the slide joint after
executing a full trajectory.

Gripper closing ratio lives in affordance_datas, so it can be swept inside a
single env. Friction lives in the articulation config and needs a rebuild, so it
is an outer argument -- run the script once per friction value.

    python -m scripts.calibrate_drawer_physics \
        --asset assets/UserDrawerAuto/user_drawer.urdf \
        --gym_config configs/drawer_open_place/user_drawer_gym_config.json \
        --action_config configs/drawer_open_place/action_config.json \
        --num_envs 1 --headless --filter_dataset_saving
"""
from __future__ import annotations
import argparse, json
import numpy as np
import gymnasium as gym

import robosynchallenge  # noqa: F401
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser, build_env_cfg_from_args)
import embodichain.lab.gym.utils.gym_utils as gym_utils

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions", "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events", "robosynchallenge.managers.observations"]


def npy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run_once(env, base, drawer, ratios, travel_limit):
    """Execute one trajectory, return travel fraction (0..1)."""
    acts = base.create_demo_action_list(action_sentence=0)
    if acts is None or len(acts) == 0:
        return None
    peak = 0.0
    for act in acts:
        env.step(act)
        q = abs(float(npy(drawer.get_qpos())[0, 0]))
        peak = max(peak, q)
    return peak / travel_limit


def main():
    ap = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(ap)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--ratios", type=float, nargs="*", default=[0.6, 0.8, 1.0],
                    help="gripper close ratios to try")
    ap.add_argument("--repeats", type=int, default=2, help="scenes per ratio")
    a = ap.parse_args()

    C = json.load(open(a.asset.rsplit(".", 1)[0] + "_constants.json"))
    travel_limit = abs(C.get("_travel", 0.18))

    env_cfg, gc, ac = build_env_cfg_from_args(a)
    env = gym.make(id=gc["id"], cfg=env_cfg, **ac)
    base = env.unwrapped
    drawer = base.sim.get_articulation("drawer")
    results = {}
    try:
        for r in a.ratios:
            vals = []
            for k in range(a.repeats):
                env.reset()
                for key in ("right_eef_close_ratio", "left_eef_close_ratio"):
                    if key in base.affordance_datas:
                        base.affordance_datas[key] = r
                f = run_once(env, base, drawer, r, travel_limit)
                if f is not None:
                    vals.append(f)
                print("  ratio %.2f  scene %d/%d  travel %s" %
                      (r, k + 1, a.repeats, "IK fail" if f is None else "%.1f%%" % (f * 100)),
                      flush=True)
            results[r] = float(np.mean(vals)) if vals else 0.0

        print("\nclose ratio -> mean travel fraction")
        for r, v in results.items():
            bar = "#" * int(round(v * 40))
            print("  %.2f  %6.1f%%  |%s" % (r, v * 100, bar))
        best_r = max(results, key=results.get)
        print("\nbest close ratio %.2f  (%.1f%% travel)" % (best_r, results[best_r] * 100))
        if results[best_r] < 0.05:
            print("NOTE: nothing moved the drawer. Try a different --friction run,\n"
                  "      or the handle may be too thin for this gripper to hold.")
        C.setdefault("physics", {})["eef_close_ratio"] = round(best_r, 3)
        C["travel_by_ratio"] = {str(k): round(v, 4) for k, v in results.items()}
        json.dump(C, open(a.asset.rsplit(".", 1)[0] + "_constants.json", "w"), indent=2)
        print("written back to constants.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
