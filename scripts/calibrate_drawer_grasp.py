"""Find a reachable grasp pose for a generated drawer by probing IK directly.

IK reachability depends only on the target pose, not on where the object
physically sits. So one env suffices: reset once, then for each candidate offset
overwrite drawer_handle_pose in affordance_datas and ask the action graph for a
trajectory. Rebuilding the env per candidate is ~60x slower and segfaults --
dexsim cannot tear down and rebuild a scene in one process.
"""
from __future__ import annotations
import argparse, json
import xml.etree.ElementTree as ET
import numpy as np
import gymnasium as gym

import robosynchallenge  # noqa: F401
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser, build_env_cfg_from_args)
import embodichain.lab.gym.utils.gym_utils as gym_utils

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions", "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events", "robosynchallenge.managers.observations"]

HANDLE_KEY = "drawer_handle_pose"


def write_back(urdf, cfg_path, dz, dx):
    tree = ET.parse(urdf); root = tree.getroot()
    for j in root.findall("joint"):
        if j.get("name") == "fixed_handle":
            o = j.find("origin"); xyz = [float(v) for v in o.get("xyz").split()]
            o.set("xyz", "%.6f %.6f %.6f" % (xyz[0], xyz[1], xyz[2] + dz))
    tree.write(urdf, xml_declaration=True, encoding="utf-8")
    d = json.load(open(cfg_path))
    for a in d.get("articulation", []):
        a["init_pos"] = [round(a["init_pos"][0] + dx, 4), a["init_pos"][1], a["init_pos"][2]]
    json.dump(d, open(cfg_path, "w"), indent=4)
    return d["articulation"][0]["init_pos"][0]


def main():
    ap = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(ap)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--dz", type=float, nargs=3, default=(-0.03, 0.03, 5))
    ap.add_argument("--dx", type=float, nargs=3, default=(-0.08, 0.08, 5))
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()

    env_cfg, gc, ac = build_env_cfg_from_args(a)
    env = gym.make(id=gc["id"], cfg=env_cfg, **ac)
    base = env.unwrapped
    zs = np.linspace(a.dz[0], a.dz[1], int(a.dz[2]))
    xs = np.linspace(a.dx[0], a.dx[1], int(a.dx[2]))
    grid = np.zeros((len(zs), len(xs)), int)
    try:
        for t in range(a.trials):
            env.reset()
            ref = base.affordance_datas.get(HANDLE_KEY)
            if ref is None:
                raise SystemExit(HANDLE_KEY + " missing -- is handle_xpos in the URDF?")
            ref = np.array(ref, dtype=float).reshape(4, 4)
            for i, dz in enumerate(zs):
                for j, dx in enumerate(xs):
                    M = ref.copy(); M[0, 3] += dx; M[2, 3] += dz
                    base.affordance_datas[HANDLE_KEY] = M
                    acts = base.create_demo_action_list(action_sentence=0)
                    if acts is not None and len(acts) > 0:
                        grid[i, j] += 1
            base.affordance_datas[HANDLE_KEY] = ref
            print("  scene %d/%d  cumulative hits %d" % (t + 1, a.trials, grid.sum()), flush=True)
        # Report and write back BEFORE closing: dexsim segfaults during teardown,
    # so anything after env.close() never runs.
        print("\\nIK success out of %d scenes  (rows: grasp dz, cols: cabinet dx)" % a.trials)
        print("        " + " ".join("%+.3f" % x for x in xs))
        for i, dz in enumerate(zs):
            print("  %+.3f " % dz + " ".join("%6d" % grid[i, j] for j in range(len(xs))))
        bi, bj = np.unravel_index(grid.argmax(), grid.shape)
        best = int(grid[bi, bj])
        print("\\nbest cell: dz=%+.4f dx=%+.4f  ->  %d/%d" % (zs[bi], xs[bj], best, a.trials))
        if best > 0:
            new_x = write_back(a.asset, a.gym_config, float(zs[bi]), float(xs[bj]))
            print("written back: handle z offset %+.4f, init_pos.x -> %.4f" % (zs[bi], new_x))
        else:
            print("WARNING: nothing reachable. Widen --dz/--dx or check asset orientation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
