"""Fill a drawer_open_place gym_config from an asset built by make_drawer_asset.py.

This is the last manual step of the drawer pipeline. Everything it writes is
derived, not typed:

  articulation[].fpath        the new URDF, for `drawer` and `drawer_background`
  drawer.init_pos             x/y solved so the world-frame grasp point lands
                              where the TEMPLATE's drawer put it -- that pose is
                              the one the IK reachability sweep calibrated, so a
                              deeper drawer must be shifted rather than inherit
                              init_pos verbatim. z is taken from the template
                              UNCHANGED: it is the shelf height on the table, not
                              a function of the drawer. Solving z too would match
                              the handle heights and leave a short cabinet
                              floating (the official cabinet is 0.2 m tall with a
                              mid-height handle; ours is 0.09 m with a low one,
                              which lifted it 73 mm off its stack).
  drawer_background.init_pos  the same, dropped by the asset's shell height
  drawer_pull_distance        the asset's usable travel x PULL_FRACTION
  drawer_inner_offset_xyz     the tray cavity centre, for the place step
  attrs whitelist             gains `drawer_pull_distance`; without it the key
                              never reaches env.affordance_datas and the action
                              graph silently falls back to nothing

The reachable grasp point is not a constant in this file: it is measured from
whatever template you point at, so re-calibrating the template propagates.

Usage
    python scripts/make_drawer_gym_config.py assets/MyDrawer/my_drawer.urdf \
        configs/drawer_open_place/my_drawer_gym_config.json \
        [--template configs/drawer_open_place/user_drawer_gym_config.json]
"""
from __future__ import annotations
import argparse, json, os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vec(s):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(3)


def handle_offset(urdf_path, init_q=0.0, scale=(1.0, 1.0, 1.0)):
    """Grasp point in the outer_box frame: joint origin + handle origin, shifted
    by the slide joint's spawn position.

    The handle rides on the tray, so an asset that spawns closed at a non-zero
    q has its handle displaced by axis * q. Ignoring that put drawer_ep0's
    grasp target 85 mm off (its closed pose is q=+0.085, not 0)."""
    root = ET.parse(urdf_path).getroot()

    def org(jname):
        j = root.find(f"joint[@name='{jname}']")
        if j is None:
            raise SystemExit(f"{urdf_path}: no joint '{jname}'")
        o = j.find("origin")
        return _vec(o.get("xyz")) if o is not None else np.zeros(3)

    axis = np.zeros(3)
    sj = root.find("joint[@name='slide_rails']")
    if sj is not None and sj.find("axis") is not None:
        axis = _vec(sj.find("axis").get("xyz"))
    local = org("slide_rails") + org("fixed_handle") + axis * float(init_q)
    # body_scale is applied by the simulator to the whole articulation, so the
    # handle does not sit where the URDF says unless it is folded in here. The
    # official config squashes its drawer to 35 % height; inheriting that value
    # while solving against unscaled coordinates put the grasp target 17-42 mm
    # above the real handle on every asset generated from that template.
    return local * np.asarray(scale, dtype=float)


def find_entity(cfg, uid):
    for e in cfg.get("articulation", []):
        if e.get("uid") == uid:
            return e
    raise SystemExit(f"no articulation with uid '{uid}'")


def resolve(fpath):
    """Config paths are repo-relative; fall back to EmbodiChain's asset cache."""
    p = os.path.join(REPO, fpath)
    if os.path.exists(p):
        return p
    cache = os.environ.get("EMBODICHAIN_DATA_ROOT",
                           os.path.expanduser("~/.cache/embodichain_data"))
    p = os.path.join(cache, fpath)
    if os.path.exists(p):
        return p
    raise SystemExit(f"cannot resolve asset path: {fpath}")


def world_handle(init_pos, init_rot, local):
    """EmbodiChain composes init_rot as intrinsic-free euler xyz in degrees."""
    return np.asarray(init_pos) + R.from_euler("xyz", np.deg2rad(init_rot)).as_matrix() @ local


def walk_named(o):
    """Yield every {"name": ..., "value": ...} node, at any depth."""
    if isinstance(o, dict):
        if isinstance(o.get("name"), str) and "value" in o:
            yield o
        for v in o.values():
            yield from walk_named(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_named(v)


def walk_attrs(o, uid):
    if isinstance(o, dict):
        if o.get("entity_cfg", {}).get("uid") == uid and isinstance(o.get("attrs"), list):
            yield o
        for v in o.values():
            yield from walk_attrs(v, uid)
    elif isinstance(o, list):
        for v in o:
            yield from walk_attrs(v, uid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", help="URDF emitted by make_drawer_asset.py")
    ap.add_argument("out", help="gym_config to write")
    ap.add_argument("--template",
                    default="configs/drawer_open_place/user_drawer_gym_config.json",
                    help="scene to copy: events, cameras, lights, distractor slots")
    ap.add_argument("--grasp-from", default=None,
                    help="config whose drawer pose supplies the calibrated grasp "
                         "point (default: --template). Split them when the scene "
                         "you want and the pose you validated live in different "
                         "configs: the official random scene carries the 16 "
                         "randomisation events, but its drawer sits 25 mm further "
                         "out than the pose the IK sweep validated -- far enough "
                         "that the left arm misses the tray on the place step "
                         "(duck ended ~156 mm away, 0/3 episodes saved).")
    a = ap.parse_args()

    urdf = os.path.abspath(a.urdf)
    consts = os.path.join(os.path.dirname(urdf),
                          os.path.splitext(os.path.basename(urdf))[0] + "_constants.json")
    if not os.path.exists(consts):
        raise SystemExit(f"missing {consts} -- run make_drawer_asset.py first")
    c = json.load(open(consts))
    cfg = json.load(open(os.path.join(REPO, a.template)))

    # --- the pose the calibrated grasp point occupies, in world coords ---
    gsrc = json.load(open(os.path.join(REPO, a.grasp_from))) if a.grasp_from else cfg
    tmpl = find_entity(gsrc, "drawer")
    tmpl_q = (tmpl.get("init_qpos") or [0.0])[0]
    tmpl_scale = tmpl.get("body_scale") or (1.0, 1.0, 1.0)
    tmpl_local = handle_offset(resolve(tmpl["fpath"]), tmpl_q, tmpl_scale)
    target = world_handle(tmpl["init_pos"], tmpl["init_rot"], tmpl_local)
    print(f"scene    {a.template}")
    print(f"grasp    {a.grasp_from or a.template}")
    print(f"template asset {tmpl['fpath']}")
    print(f"  body_scale {list(tmpl_scale)}")
    print(f"  handle offset (outer_box frame, scaled) {np.round(tmpl_local, 4).tolist()}")
    print(f"  -> calibrated world grasp point {np.round(target, 4).tolist()}")

    # --- solve this asset's init_pos so its grasp point lands there ---
    new_q = (c.get("init_qpos") or [0.0])[0]
    # Assets from make_drawer_asset.py are already at their intended size, so
    # they are placed unscaled rather than inheriting the template's squash.
    new_scale = (1.0, 1.0, 1.0)
    new_local = handle_offset(urdf, new_q, new_scale)
    scene_drawer = find_entity(cfg, "drawer")
    rot = scene_drawer["init_rot"]
    init = target - R.from_euler("xyz", np.deg2rad(rot)).as_matrix() @ new_local
    init = [round(float(x), 4) for x in init]
    init[2] = round(float(scene_drawer["init_pos"][2]), 4)   # shelf height, not solved
    check = world_handle(init, rot, new_local)
    print(f"new asset {os.path.relpath(urdf, REPO)}")
    print(f"  handle offset (outer_box frame) {np.round(new_local, 4).tolist()}")
    print(f"  -> init_pos {init}   (grasp lands at {np.round(check, 4).tolist()})")
    print(f"     x/y solved to the template's grasp point; z = template shelf height")
    assert np.allclose(check[:2], target[:2], atol=5e-4), "init_pos x/y solve failed"

    rel = os.path.relpath(urdf, REPO)
    bg_z = round(init[2] + float(c["background_z_offset"]), 4)
    for e in cfg["articulation"]:
        e["fpath"] = rel
        e["body_scale"] = list(new_scale)
        e["init_pos"] = init if e["uid"] == "drawer" else [init[0], init[1], bg_z]
        # Spawn closed. The closed pose is not always q=0: drawer_ep0's limit is
        # [-0.160, +0.085] and at q=0 its tray juts 105 mm past the cabinet
        # front, which put the duck outside the footprint the success check
        # measures against (0/84 episodes saved).
        if c.get("init_qpos"):
            e["init_qpos"] = c["init_qpos"]
    print(f"  -> drawer_background init_pos [{init[0]}, {init[1]}, {bg_z}]"
          f"   (shell height {c['shell_height']})")
    if c.get("init_qpos"):
        print(f"  -> init_qpos {c['init_qpos']}   (closed pose)")

    upd = {"drawer_pull_distance": c["drawer_pull_distance"],
           "drawer_inner_offset_xyz": c["drawer_inner_offset_xyz"]}
    seen = set()
    for n in walk_named(cfg):
        if n["name"] in upd:
            print(f"  -> {n['name']:<26} {n['value']} -> {upd[n['name']]}")
            n["value"] = upd[n["name"]]
            seen.add(n["name"])
    for k in upd.keys() - seen:
        print(f"  !! '{k}' not present in template -- action graph will not see it")

    # The drawer's randomisation RANGE stays with the scene template, not the
    # grasp source. Reachability is a property of the arm, and the benchmark's
    # own range ([-0.01,-0.18]..[0.05,0]) is proven reachable. This used to be
    # copied from --grasp-from, narrowed to y +-0.10 m, because the benchmark's
    # range produced IK failures -- but those failures came from the body_scale
    # squash dropping the handle 17-42 mm, not from the range. At true scale,
    # 25/25 executed episodes succeed under the wide range; the only cost is
    # ~30 % of attempts rejected at IK before execution, which is cheap.
    # Only the ORIENTATION fields follow the grasp source, since the grasp
    # pipeline assumes the handle faces the robot.
    if a.grasp_from:
        gev = gsrc.get("env", {}).get("events", {}).get("random_drawer_pose")
        sev = cfg.get("env", {}).get("events", {}).get("random_drawer_pose")
        if gev and sev:
            for k in ("rotation_range", "relative_rotation"):
                if k in gev.get("params", {}):
                    sev["params"][k] = gev["params"][k]
                    print(f"  -> random_drawer_pose.{k} <- {a.grasp_from}")
            print(f"  -> random_drawer_pose.position_range kept from scene: "
                  f"{sev['params'].get('position_range')}")

    # Without this the key never reaches env.affordance_datas.
    for grp in walk_attrs(cfg, "drawer"):
        if "drawer_pull_distance" not in grp["attrs"]:
            grp["attrs"].append("drawer_pull_distance")
            print("  -> attrs whitelist += drawer_pull_distance")

    out = os.path.join(REPO, a.out) if not os.path.isabs(a.out) else a.out
    with open(out, "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(out, REPO) if out.startswith(REPO) else out}")


if __name__ == "__main__":
    main()
