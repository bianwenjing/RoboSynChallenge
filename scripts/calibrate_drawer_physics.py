"""Search the gripper drive against how far the drawer fails to close.

The earlier objective -- how far the drawer OPENS -- turned out to measure the
wrong half of the task. Every drawer tested opens to within 3 % of its commanded
pull; what separates a usable asset from an unusable one is whether the fingers
keep hold through the push-back. drawer_ep0 opens to 0.146 of a commanded 0.150
and then stalls with 77 mm still open, leaving the duck outside the cabinet
footprint the success check measures.

So the objective here is the RESIDUAL opening at the end of the episode.

What is searched, and why these knobs:

  gripper stiffness / max_effort   The gripper is position-controlled with
      finite stiffness, so grip force is stiffness x tracking error, capped by
      max_effort. The official 450 / 15 N was set for an 18 mm handle bar on a
      0.20 kg tray. A thinner bar on a heavier tray needs more: drawer_ep0's
      fingers are pushed from 4.0 mm to 27.8 mm open mid-push. Raising
      eef_close_ratio cannot help -- it is already at its 1.0 ceiling.
  push-back duration              The push edge has a fixed 50-step budget
      regardless of how far it has to travel or how heavy the tray is.

These interact non-monotonically -- 6000 / 200 N closes WORSE than 2000 / 60 N,
because too stiff a grip destabilises the contact -- so they are searched rather
than derived from a formula.

Each sample needs its own process: dexsim segfaults on a second gym.make, and
drive_pros is read at build time.

    python -m scripts.calibrate_drawer_physics \
        --gym_config configs/drawer_ep0/clear/gym_config.json \
        --action_config configs/drawer_ep0/action_config.json \
        --episodes 3 --headless --filter_dataset_saving
"""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (stiffness, max_effort) for the two right-gripper joints. The first entry is
# the official value; the rest scale grip force for thinner bars / heavier trays.
GRIP_GRID = [(450.0, 15.0), (1500.0, 45.0), (3000.0, 90.0)]
PUSH_STEPS = [50, 80]          # push-back edge budget; 50 is the official value


def patch_gym(src, stiffness, effort, out):
    cfg = json.load(open(src))
    dp = cfg["robot"]["drive_pros"]
    for key in ("right_joint[7-8]", "left_joint[7-8]"):
        if key in dp["stiffness"]:
            dp["stiffness"][key] = stiffness
            # critical damping for a unit-ish inertia keeps the finger from
            # ringing when the grip force jumps
            dp["damping"][key] = round(2.0 * (stiffness ** 0.5), 2)
            dp["max_effort"][key] = effort
    json.dump(cfg, open(out, "w"), indent=1)


def patch_action(src, steps, out):
    cfg = json.load(open(src))
    n = [0]

    def walk(o):
        if isinstance(o, dict):
            if "right_drawer_push_close" in o and isinstance(o["right_drawer_push_close"], dict):
                e = o["right_drawer_push_close"]
                if "duration" in e:
                    e["duration"] = steps
                    n[0] += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    json.dump(cfg, open(out, "w"), indent=1)
    return n[0]


def worker(args):
    """Run `--episodes` trajectories, print the residual-opening statistics."""
    import gymnasium as gym
    import robosynchallenge  # noqa: F401
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser, build_env_cfg_from_args)
    import embodichain.lab.gym.utils.gym_utils as gym_utils
    gym_utils.DEFAULT_MANAGER_MODULES += [
        "robosynchallenge.managers.actions", "robosynchallenge.managers.datasets",
        "robosynchallenge.managers.events", "robosynchallenge.managers.observations"]

    cfg, gc, ac = build_env_cfg_from_args(args)
    env = gym.make(id=gc["id"], cfg=cfg, **ac)
    b = env.unwrapped
    npy = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    drw = b.sim.get_articulation("drawer")
    duck = b.sim.get_rigid_object("duck")

    resid, dist = [], []
    for _ in range(args.episodes):
        acts = None
        for _ in range(10):
            env.reset()
            acts = b.create_demo_action_list(action_sentence=0)
            if acts is not None and len(acts) > 0:
                break
        if acts is None or len(acts) == 0:
            continue
        for act in acts:
            env.step(act)
        resid.append(abs(float(npy(drw.get_qpos())[0, 0])))
        T = npy(drw.get_link_pose("outer_box", to_matrix=True)[0])
        dw = npy(duck.get_local_pose(to_matrix=True)[0])[:3, 3]
        dist.append(float(np.linalg.norm(dw[:2] - T[:2, 3])))
    if resid:
        ok = sum(1 for d in dist if d <= 0.1)
        print("WORKER residual_med=%.4f dist_med=%.4f ok=%d/%d"
              % (float(np.median(resid)), float(np.median(dist)), ok, len(dist)), flush=True)
    else:
        print("WORKER residual_med=nan dist_med=nan ok=0/0", flush=True)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym_config", required=True)
    ap.add_argument("--action_config", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--filter_dataset_saving", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args, rest = ap.parse_known_args()

    if args.worker:
        import argparse as _a
        from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
        p = _a.ArgumentParser(); add_env_launcher_args_to_parser(p)
        p.add_argument("--episodes", type=int, default=3)
        p.add_argument("--worker", action="store_true")
        worker(p.parse_args())
        return

    tmp = tempfile.mkdtemp(prefix="drawercal_")
    rows = []
    for (st, ef), steps in itertools.product(GRIP_GRID, PUSH_STEPS):
        g = os.path.join(tmp, f"gym_{st:.0f}_{ef:.0f}.json")
        a = os.path.join(tmp, f"act_{steps}.json")
        patch_gym(os.path.join(REPO, args.gym_config), st, ef, g)
        if patch_action(os.path.join(REPO, args.action_config), steps, a) == 0:
            print("!! push-back edge 'right_drawer_push_close' not found")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--gym_config", g, "--action_config", a,
               "--num_envs", str(args.num_envs), "--episodes", str(args.episodes)]
        if args.headless:
            cmd.append("--headless")
        if args.filter_dataset_saving:
            cmd.append("--filter_dataset_saving")
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout
        line = next((l for l in out.splitlines() if l.startswith("WORKER")), "")
        vals = dict(kv.split("=") for kv in line.split()[1:]) if line else {}
        r = float(vals.get("residual_med", "nan"))
        d = float(vals.get("dist_med", "nan"))
        ok = vals.get("ok", "0/0")
        rows.append((st, ef, steps, r, d, ok))
        print("  stiffness %6.0f  effort %5.0f  push %3d ->  residual %.4f  dist %.4f  %s"
              % (st, ef, steps, r, d, ok), flush=True)

    good = [x for x in rows if x[3] == x[3]]
    if not good:
        print("\nno usable sample"); return
    best = min(good, key=lambda x: (x[3], x[4]))
    print("\nbest: stiffness=%.0f max_effort=%.0f push_steps=%d"
          "   residual %.4f  duck-drawer %.4f  %s" % (best[0], best[1], best[2], best[3], best[4], best[5]))
    print("\nApply to the gym_config's robot.drive_pros:")
    print('  "stiffness":  {"right_joint[7-8]": %.1f}' % best[0])
    print('  "damping":    {"right_joint[7-8]": %.2f}' % (2.0 * best[0] ** 0.5))
    print('  "max_effort": {"right_joint[7-8]": %.1f}' % best[1])
    if best[2] != PUSH_STEPS[0]:
        print("And the action_config's right_drawer_push_close duration: %d" % best[2])


if __name__ == "__main__":
    main()
