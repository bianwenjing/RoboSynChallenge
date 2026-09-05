"""Turn any two-body prismatic drawer URDF into a drawer_open_place-ready asset.

Everything the task needs is derived from geometry -- no hand tuning:

  outer_box / inner_box   renamed from the prismatic joint's parent / child
  collision geometry      CoACD convex decomposition of each baked mesh, written
                          as a multi-`o` OBJ (what the official SlidingBoxDrawer
                          ships). Never synthesise convex boxes from a bounding
                          box: that invents surfaces the mesh does not have.
  handle_xpos             just behind the handle bar's rear face, so the fingers
                          close around the bar rather than in front of its tip
  slide axis              rotated to +X (the convention action_config is written
                          against) by baking an rpy onto every visual/collision
  drawer_pull_distance    -(travel * PULL_FRACTION)
  drawer_inner_offset_xyz centre of the tray cavity, expressed w.r.t. outer_box

Usage
    python scripts/make_drawer_asset.py <in.urdf> <out_dir> [--name user_drawer]
"""
from __future__ import annotations
import argparse, json, math, os, shutil
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

PULL_FRACTION = 0.80   # use 80% of the rail travel, leaving margin at the stop
# How far the arm can drag a handle before the fingers lose it is a property of
# the ROBOT and of the push-back edge's fixed 50-step budget, not of the drawer.
# A pure fraction let drawer_ep0 (0.245 m rail) ask for 0.196 m; the drawer
# opened fine but never came back to closed (q_end 0.034-0.159 vs 0.000 for the
# drawers that work), so the duck stayed outside the cabinet footprint the
# success check measures. 0.15 is just above the two values known to work
# (official 0.140, ep0_v2 0.144).
PULL_MAX = 0.15

# Where the duck is released, as a position inside the OPENED tray. Calibrated
# once against the action graph's hard-coded (0.20, -0.10, 0.15) on a drawer
# that works, then reused for every asset -- see the RULE comment in main().
PLACE_X_FRAC  = 0.668   # along the opened tray's length
PLACE_Y_MAX   = 0.05    # absolute cap on the lateral offset (m)
PLACE_Y_FRAC  = -0.50   # of the half-width, toward the tray centre.
# The official graph's -0.10 m sits ON the tray wall, which alone equals the
# 0.1 m success radius -- it only works because the duck rolls inward on the
# trays it was tuned for. Depending on that roll is an assumption a differently
# shaped tray breaks, so aim at mid-width instead. Verified neutral: the drawer
# it was calibrated against still collects 6/6.
PLACE_Z_CLEAR = 0.066   # above the tray rim
HOOK_DEPTH    = 0.012  # how far below the top rim the grasp point sits (m)

# World x at which the official SlidingBoxDrawer puts its handle. That pose is
# known-reachable for the right arm, so a deeper or shallower drawer should be
# shifted until its own handle lands there rather than inheriting the official
# init_pos verbatim.
TARGET_HANDLE_X = 0.573


def _vec(s, d=(0.0, 0.0, 0.0)):
    return np.array([float(x) for x in s.split()]) if s else np.array(d, float)


def rot_z(v, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def detect_handle(mesh_baked, front_x, width_frac=0.85, eps=0.0008):
    """Locate the door panel and the handle bar, and pick the grasp depth.

    A tray front is two features stacked along +x: a full-width door panel, then
    a narrower bar sticking out of it. Walking back from the front plane, the
    first slice whose y-span reaches `width_frac` of the tray width is the panel;
    everything in front of it is the bar.

    The grasp point does NOT go on the bar's outer tip -- the fingers would close
    in front of it. It goes just behind the bar's rear face, in the hook gap, so
    closing wraps the bar. Reproduces the official SlidingBoxDrawer's hand-placed
    handle_xpos (0.105, 0, 0.1) to the millimetre; see _selftest_official().

    Only the grasp point is used downstream -- collision geometry comes from
    the convex decomposition, not from these bounds.

    Returns (grasp_xyz, bar_bbox, panel_x, why) or (None, None, None, why).
    """
    V = np.asarray(mesh_baked.vertices)
    full_y = float(np.ptp(V[:, 1]))
    full_z = float(np.ptp(V[:, 2]))
    # The door panel is the first slice back from the front that is wide AND
    # tall. Width alone is not enough: drawer_ep0_v2_var4 mounts its handle bar
    # on two brackets that together span 86 % of the tray width, so a width-only
    # test mistook the handle band for the panel and then read the bar as a
    # 0.3 mm "recessed pull". A panel also fills most of the tray's height,
    # which a handle band never does.
    panel_x = None
    for x in np.arange(front_x, front_x - 0.06, -0.0005):
        sl = V[(V[:, 0] >= x - eps) & (V[:, 0] <= x + eps)]
        if (len(sl) > 3 and np.ptp(sl[:, 1]) > width_frac * full_y
                and np.ptp(sl[:, 2]) > 0.50 * full_z):
            panel_x = float(x)
            break
    if panel_x is None:
        return None, None, None, "no full-width front panel found", None

    bar = V[V[:, 0] > panel_x + eps]
    # A real bar spans a useful part of the tray and stands proud enough for the
    # fingers to close around. Anything smaller is a moulding or a panel edge --
    # drawer_walnut's raised centre section is 4 x 4 mm and 1.5 mm proud, and
    # taking it for a handle put the grasp point on a flat door.
    if len(bar) >= 8:
        _lo, _hi = bar.min(axis=0), bar.max(axis=0)
        plausible = ((_hi[1] - _lo[1]) > 0.15 * full_y
                     and (_hi[0] - panel_x) > 0.004
                     and (_hi[2] - _lo[2]) > 0.004)
    else:
        plausible = False
    if plausible:
        blo, bhi = bar.min(axis=0), bar.max(axis=0)
        gap = float(blo[0] - panel_x)          # hook depth between panel and bar
        clearance = min(0.002, max(gap, 0.0) / 2.0)
        grasp = np.array([blo[0] - clearance,
                          (blo[1] + bhi[1]) / 2.0,
                          (blo[2] + bhi[2]) / 2.0])
        why = ("protruding bar: panel x=%.4f, bar x[%.4f,%.4f] out %.1f mm, gap %.1f mm, "
               "%.0fx%.0f mm -> grasp %.1f mm behind tip"
               % (panel_x, blo[0], bhi[0], 1000 * (bhi[0] - panel_x), 1000 * gap,
                  1000 * (bhi[1] - blo[1]), 1000 * (bhi[2] - blo[2]),
                  1000 * (front_x - grasp[0])))
        return grasp, (blo, bhi), panel_x, "bar|" + why, None

    # No material in front of the panel. The handle may still be there, RECESSED
    # into a pocket -- drawer_walnut's brass pull sits inside a shallow inset, so
    # the protruding-bar search finds only a few-millimetre sliver of the panel
    # edge. Look INSIDE the panel instead: between its rear and front planes,
    # the frontmost slice whose y-span is a modest fraction of the tray width is
    # the pull, and the fingers reach it through the pocket.
    rear = None
    for x in np.arange(panel_x - 0.002, panel_x - 0.06, -0.0005):
        sl = V[(V[:, 0] >= x - eps) & (V[:, 0] <= x + eps)]
        if (len(sl) > 3 and np.ptp(sl[:, 1]) > width_frac * full_y
                and np.ptp(sl[:, 2]) > 0.50 * full_z):
            rear = float(x)
            break
    if rear is None:
        return None, None, None, "flat front panel, no handle of either kind", None
    inner = V[(V[:, 0] > rear + eps) & (V[:, 0] < panel_x - eps)]
    inner = inner[np.abs(inner[:, 1]) < 0.45 * full_y] if len(inner) else inner
    if len(inner) < 8:
        return None, None, None, "no pull found inside the front panel", None
    blo, bhi = inner.min(axis=0), inner.max(axis=0)
    # Inside the pocket, the pull bar is the thin layer; the rest is void. Take
    # the frontmost slice's z-extent as the bar and treat the surrounding
    # opening as clearance, otherwise the grasp point lands inside the bar
    # itself and the fingers have nothing to close into.
    tip = inner[inner[:, 0] > bhi[0] - 0.003]
    if len(tip) >= 4 and np.ptp(tip[:, 2]) < 0.9 * (bhi[2] - blo[2]):
        bar_lo = np.array([blo[0], tip[:, 1].min(), tip[:, 2].min()])
        bar_hi = np.array([bhi[0], tip[:, 1].max(), tip[:, 2].max()])
    else:
        bar_lo, bar_hi = blo.copy(), bhi.copy()
    # Hook behind the bar, in the clearance between it and the pocket floor.
    grasp = np.array([max(rear + 0.002, bar_lo[0] - 0.002),
                      (bar_lo[1] + bar_hi[1]) / 2.0,
                      (bar_lo[2] + bar_hi[2]) / 2.0])
    why = ("recessed|recessed pull: panel x[%.4f,%.4f], pull x[%.4f,%.4f] %.0fx%.0f mm, "
           "pocket %.1f mm deep, bar %.0f mm tall -> hook behind the bar"
           % (rear, panel_x, blo[0], bhi[0], 1000 * (bhi[1] - blo[1]),
              1000 * (bhi[2] - blo[2]), 1000 * (panel_x - rear),
              1000 * (bar_hi[2] - bar_lo[2])))
    pocket = dict(rear=rear, front=panel_x,
                  lo=np.array([rear, blo[1], blo[2]]),
                  hi=np.array([panel_x, bhi[1], bhi[2]]),
                  bar_lo=bar_lo, bar_hi=bar_hi)
    return grasp, (blo, bhi), panel_x, why, pocket


def panel_boxes(panel_lo, panel_hi, pk):
    """Explicit convex boxes for a door panel that carries a RECESSED pull.

    CoACD merges a shallow pocket into the slab around it -- verified on
    drawer_walnut, where the grasp point 16 mm inside the recess landed inside a
    hull, so the fingers met a flat door whatever the wrist orientation. The
    pocket's position and size are known exactly from the mesh, so build that
    one region by hand: a frame of four boxes around the opening, plus the pull
    bar, leaving the void genuinely empty.  Returns [(centre, size), ...].
    """
    def bx(a, b):
        return ([(a[i] + b[i]) / 2 for i in range(3)],
                [max(b[i] - a[i], 0.002) for i in range(3)])
    x0, x1 = pk["rear"], pk["front"]
    (py0, pz0), (py1, pz1) = (panel_lo[1], panel_lo[2]), (panel_hi[1], panel_hi[2])
    (ky0, kz0), (ky1, kz1) = (pk["lo"][1], pk["lo"][2]), (pk["hi"][1], pk["hi"][2])
    out = [
        bx((x0, py0, kz1), (x1, py1, pz1)),      # above the opening
        bx((x0, py0, pz0), (x1, py1, kz0)),      # below it
        bx((x0, ky1, kz0), (x1, py1, kz1)),      # one side
        bx((x0, py0, kz0), (x1, ky0, kz1)),      # the other
        bx(pk["bar_lo"], pk["bar_hi"]),          # the pull itself
    ]
    return [(c, z) for c, z in out if all(v > 0.0015 for v in z)]


def decompose(mesh, out_path, threshold=0.05, max_hulls=32, clip_front=None):
    """Convex-decompose a mesh and write it as a multi-`o` OBJ.

    Physics engines only take convex shapes. A drawer or a cabinet is a concave
    shell: handed over whole, it is hulled and the cavity fills solid -- the tray
    cannot slide in and nothing can be placed inside.

    Earlier versions of this script synthesised convex boxes from the bounding
    box instead (no decomposition backend was installed). That fabricates
    surfaces the mesh does not have, and it cost two silent failures: a wall
    across the cabinet's mouth, which capped travel at 9.8 mm of a 180 mm rail,
    and a door panel placed at the handle tip, which filled the gap the fingers
    close into. A decomposition can't invent either -- it only produces geometry
    where the mesh has geometry. This is also what the official SlidingBoxDrawer
    ships (`*ACD.obj`, 13 and 15 hulls).

    Returns the number of hulls written.
    """
    import coacd
    coacd.set_log_level("error")
    if clip_front is not None:
        # Everything from clip_front forward is built explicitly by
        # panel_boxes(); decomposing it too would refill the pocket.
        V = np.asarray(mesh.vertices)
        keep = np.asarray(mesh.faces)[(V[np.asarray(mesh.faces)][:, :, 0]
                                       <= clip_front + 1e-4).all(axis=1)]
        if len(keep) >= 4:
            mesh = trimesh.Trimesh(vertices=V, faces=keep).process()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if len(F) < 4:
        trimesh.Scene().export(out_path)
        return 0
    parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=threshold,
                            max_convex_hull=max_hulls, preprocess_mode="auto")
    scene = trimesh.Scene()
    for i, (v, f) in enumerate(parts):
        scene.add_geometry(trimesh.Trimesh(np.asarray(v), np.asarray(f)),
                           geom_name=f"hull_{i}")
    scene.export(out_path)
    return len(parts)


def load_mesh(urdf_dir, fname):
    p = os.path.join(urdf_dir, fname)
    return trimesh.load(p, force="mesh", process=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf"); ap.add_argument("out_dir")
    ap.add_argument("--name", default="user_drawer")
    a = ap.parse_args()

    src_dir = os.path.dirname(os.path.abspath(a.urdf))
    root = ET.parse(a.urdf).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")

    pj = next((j for j in joints if j.get("type") == "prismatic"), None)
    if pj is None:
        raise SystemExit("no prismatic joint found -- not a sliding drawer")

    outer_n = pj.find("parent").get("link")
    inner_n = pj.find("child").get("link")
    axis = _vec(pj.find("axis").get("xyz"), (1, 0, 0))
    lim = pj.find("limit")
    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
    j_org = _vec(pj.find("origin").get("xyz") if pj.find("origin") is not None else None)

    # Direction the tray physically travels when opening, in outer_box frame.
    open_q = lo if abs(lo) > abs(hi) else hi
    # The CLOSED pose is the limit end opposite the opening direction -- not
    # necessarily q=0. drawer_ep0's limit is [-0.160, +0.085]: at q=0 its tray
    # already juts 105 mm past the cabinet front, so the duck came to rest
    # outside the cabinet footprint and the success check (duck within 0.1 m of
    # the outer_box origin) could never pass. At q=+0.085 the tray body sits
    # flush and only the handle protrudes, which is the real rest state.
    # Emitting init_qpos makes the asset spawn closed; travel is then the full
    # limit span. Limits like [-0.18, 0] or [0, 0.2] close at 0 and hide this.
    closed_q = hi if open_q == lo else lo
    travel = abs(hi - lo)
    # NORMALISE THE JOINT so q=0 is the closed pose, which is what the whole
    # action graph assumes (its place target, push-back edge and success check
    # are all written against a drawer that rests at 0). drawer_ep0's limit is
    # [-0.160, +0.085]: carrying that through as init_qpos=+0.085 left the tray
    # 105 mm proud of the cabinet at spawn and the push-back never returned it,
    # so the duck came to rest outside the footprint the success check measures.
    # Baking the closed offset into the joint origin instead makes the asset
    # structurally identical to the drawers that already work.
    # Limits become [-travel, 0] or [0, travel], matching the opening sign.
    open_dir = axis * open_q
    open_dir = open_dir / (np.linalg.norm(open_dir) or 1.0)

    # Rotate about z so the opening direction becomes +X.
    yaw = -math.degrees(math.atan2(open_dir[1], open_dir[0]))
    yaw = round(yaw / 90.0) * 90.0                       # snap to a right angle
    print(f"prismatic '{pj.get('name')}'  parent={outer_n} child={inner_n}")
    print(f"  axis={axis}  limit=[{lo}, {hi}]  travel={travel:.3f} m"
          f"  closed at q={closed_q:+.4f} -> normalised to 0, limits [{lo}, {hi}]")
    print(f"  open dir={np.round(open_dir,3)}  ->  bake yaw {yaw:+.0f} deg to align +X")

    def mesh_of(link):
        g = links[link].find("collision/geometry/mesh") or links[link].find("visual/geometry/mesh")
        return g.get("filename")

    inner_file, outer_file = mesh_of(inner_n), mesh_of(outer_n)
    inner_m, outer_m = load_mesh(src_dir, inner_file), load_mesh(src_dir, outer_file)

    # Tray bbox, rotated into the baked frame, then shifted by the joint origin.
    ib = np.array([rot_z(v, yaw) for v in inner_m.bounds])
    ilo, ihi = ib.min(0), ib.max(0)
    jo = rot_z(j_org, yaw) + rot_z(axis, yaw) * closed_q   # q=0 -> closed
    lo, hi = (-travel, 0.0) if open_q < closed_q else (0.0, travel)

    # RULE: grasp the protruding handle if the mesh has one; only fall back to
    # "front face, one finger-width below the rim" when the panel is flat.
    inner_baked = inner_m.copy()
    Tz = np.eye(4)
    _c, _s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    Tz[:3, :3] = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])
    inner_baked.apply_transform(Tz)
    handle, handle_bbox, panel_x, why, pocket = detect_handle(inner_baked, float(ihi[0]))
    kind = why.split("|", 1)[0] if handle is not None and "|" in why else "bar"
    if handle is not None:
        why = why.split("|", 1)[1]
    if handle is None:
        handle = np.array([ihi[0], (ilo[1] + ihi[1]) / 2.0, ihi[2] - HOOK_DEPTH])
        panel_x = float(ihi[0]); pocket = None
        print(f"  -> no handle detected ({why}); using front-face fallback")
    else:
        print(f"  -> handle detected [{kind}]: {why}")
    pull = -min(travel * PULL_FRACTION, PULL_MAX)
    # RULE: the place target is where the duck is released above the OPEN tray.
    # It is expressed as the same relative position inside the opened tray that
    # the official action graph's hard-coded (0.20, -0.10, 0.15) occupies for a
    # drawer already known to work, so that drawer's numbers are reproduced
    # exactly while a differently shaped tray gets its own.
    #   x  PLACE_X_FRAC along the opened tray's length
    #   y  PLACE_Y_FRAC of the half-width, just outside the near wall
    #   z  PLACE_Z_CLEAR above the tray rim (the duck is dropped, not set down)
    # A tray-cavity centre at q=0 -- what this used to emit -- is the wrong
    # frame twice over: the duck is placed while the drawer is OPEN, and the
    # value was never read by anything (drawer_inner_offset_xyz had no
    # consumer until the place offsets were parameterised).
    tray_lo = jo + ilo
    tray_hi = jo + ihi
    open_lo = tray_lo + np.array([abs(pull), 0, 0])
    open_hi = tray_hi + np.array([abs(pull), 0, 0])
    # The lateral offset must be capped in ABSOLUTE terms, not just scaled: the
    # success check is a 0.1 m radius in XY, so a fraction of a wide tray eats
    # the whole budget on its own. drawer_v2_variant's 312 mm tray put the duck
    # 78 mm off-centre, leaving 63 mm for everything else, and it never passed.
    _y = PLACE_Y_FRAC * (open_hi[1] - open_lo[1]) / 2.0
    inner_centre = np.array([
        open_lo[0] + PLACE_X_FRAC * (open_hi[0] - open_lo[0]),
        float(np.clip(_y, -PLACE_Y_MAX, PLACE_Y_MAX)),
        open_hi[2] + PLACE_Z_CLEAR,
    ])

    print(f"  tray bbox (baked) {np.round(ilo,3)} ~ {np.round(ihi,3)}")
    print(f"  -> handle_xpos           {np.round(handle,4)}")
    _cap = " capped by PULL_MAX" if travel * PULL_FRACTION > PULL_MAX else ""
    print(f"  -> drawer_pull_distance  {pull:+.4f}   ({PULL_FRACTION:.0%} of {travel:.3f}{_cap})")
    print(f"  -> drawer_inner_offset   {np.round(inner_centre,4)}")

    # RULE: shift the cabinet so this drawer's handle reaches the same world x
    # as the official asset's, instead of copying the official init_pos.
    handle_reach = jo[0] + handle[0]          # handle offset from outer_box, baked frame
    suggested_x = TARGET_HANDLE_X + handle_reach   # init_rot z=180 mirrors +X -> -X
    print(f"  -> handle reach          {handle_reach:+.4f} m from outer_box")
    print(f"  -> suggested outer_box x {suggested_x:.4f}  (so handle lands at {TARGET_HANDLE_X})")

    # Bake the yaw into the vertices and export. Relying on <visual rpy> instead
    # would leave the renderer free to compose that rotation differently from the
    # collision boxes (whose coordinates are pre-rotated here), and the two paths
    # then disagree on screen.
    os.makedirs(os.path.join(a.out_dir, "meshes"), exist_ok=True)
    T = np.eye(4)
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    T[:3, :3] = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    baked, baked_m = {}, {}
    for f, m in ((outer_file, outer_m), (inner_file, inner_m)):
        stem = os.path.splitext(os.path.basename(f))[0]
        out_name = f"{stem}_baked.obj"
        mm = m.copy(); mm.apply_transform(T)
        mm.export(os.path.join(a.out_dir, "meshes", out_name))
        baked[f] = out_name; baked_m[f] = mm
    print(f"  -> baked {yaw:+.0f} deg into mesh vertices: {list(baked.values())}")

    rpy = "0 0 0"   # rotation is baked into the vertices

    def body(link_name, mesh_file, inertial, baked_mesh, pk=None):
        """One link: the baked visual mesh plus its collision geometry.

        Normally that is a CoACD decomposition of the whole mesh. When the link
        carries a recessed pull, the door panel is clipped off before decomposing
        and rebuilt from explicit boxes, because CoACD fills the pocket in.
        """
        vis = baked[mesh_file]
        acd = os.path.splitext(vis)[0].replace("_baked", "") + "_acd.obj"
        n = decompose(baked_mesh, os.path.join(a.out_dir, "meshes", acd),
                      clip_front=(pk["rear"] if pk else None))
        cols = [f'''    <collision><origin xyz="0 0 0" rpy="{rpy}"/>
      <geometry><mesh filename="meshes/{acd}"/></geometry></collision>''']
        extra = 0
        if pk is not None:
            b = np.array(baked_mesh.bounds)
            for c, z in panel_boxes(b[0], b[1], pk):
                cols.append(f'''    <collision><origin xyz="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" rpy="0 0 0"/>
      <geometry><box size="{z[0]:.5f} {z[1]:.5f} {z[2]:.5f}"/></geometry></collision>''')
                extra += 1
        print(f"  -> {link_name}: {n} convex hulls -> meshes/{acd}"
              + (f"  + {extra} explicit panel boxes (pocket kept open)" if extra else ""))
        return f'''  <link name="{link_name}">
{inertial}    <visual><origin xyz="0 0 0" rpy="{rpy}"/>
      <geometry><mesh filename="meshes/{vis}"/></geometry></visual>
{chr(10).join(cols)}
  </link>''', n

    def inertial_of(link):
        el = links[link].find("inertial")
        if el is None:
            return '    <inertial><mass value="0.3"/><inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial>\n'
        c = rot_z(_vec(el.find("origin").get("xyz") if el.find("origin") is not None else None), yaw)
        mass = el.find("mass").get("value"); I = el.find("inertia").attrib
        ixx, iyy = (I["iyy"], I["ixx"]) if abs(yaw) % 180 == 90 else (I["ixx"], I["iyy"])
        return (f'    <inertial><origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
                f'      <mass value="{mass}"/>\n'
                f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{I["izz"]}"/></inertial>\n')

    # Both links get a real convex decomposition of their own baked mesh, so the
    # cabinet mouth stays open and the handle keeps its gap without either being
    # special-cased -- the mesh already says where the material is.
    outer_body, n_out = body("outer_box", outer_file, inertial_of(outer_n), baked_m[outer_file])
    inner_body, n_in = body("inner_box", inner_file, inertial_of(inner_n),
                            baked_m[inner_file], pk=pocket)

    # RULE: drawer_background is a second, kinematic copy stacked underneath to
    # look like a two-drawer cabinet. Its z must follow this asset's shell
    # height, not the official 0.075 gap, or the two bodies interpenetrate.
    ob = np.array([rot_z(v, yaw) for v in outer_m.bounds])
    shell_h = float(ob.max(0)[2] - ob.min(0)[2])
    bg_gap = shell_h + 0.005          # official leaves ~5 mm between the layers
    print(f"  -> shell height {shell_h:.4f} -> background z offset {-bg_gap:.4f}")
    # A protruding bar is pinched vertically (fingers above and below it). A
    # recessed pull cannot be: the pocket is only as tall as the pull, so a pair
    # of fingers straddling in z hits the door and stops ~15 mm short. Rolling
    # the grasp frame 90 deg about x turns the pinch on its side, so the finger
    # pair enters edge-on through the pocket opening.
    handle_rpy = ("1.5707963268 -1.5707963268 0" if kind == "recessed"
                  else "0 -1.5707963268 0")
    print(f"  -> grasp frame rpy      {handle_rpy}   ({'edge-on into pocket' if kind=='recessed' else 'vertical pinch'})")
    new_axis = rot_z(axis, yaw); new_axis = np.round(new_axis, 6)
    urdf = f'''<?xml version="1.0"?>
<!-- Generated by scripts/make_drawer_asset.py from {os.path.basename(a.urdf)}.
     Geometry baked +{yaw:.0f} deg about z so the slide axis is +X, matching the
     convention every offset in drawer_open_place/action_config.json assumes.
     handle_xpos is derived from the tray geometry, not hand-placed.
     Collision shapes are a CoACD decomposition of the same baked meshes. -->
<robot name="{a.name}">
{outer_body}

{inner_body}

  <link name="handle_xpos"/>

  <joint name="slide_rails" type="prismatic">
    <origin xyz="{jo[0]:.6f} {jo[1]:.6f} {jo[2]:.6f}" rpy="0 0 0"/>
    <parent link="outer_box"/><child link="inner_box"/>
    <axis xyz="{new_axis[0]:g} {new_axis[1]:g} {new_axis[2]:g}"/>
    <limit lower="{lo}" upper="{hi}" effort="25.0" velocity="1.0" acceleration="5.0"/>
  </joint>

  <joint name="fixed_handle" type="fixed">
    <origin xyz="{handle[0]:.6f} {handle[1]:.6f} {handle[2]:.6f}" rpy="{handle_rpy}"/>
    <parent link="inner_box"/><child link="handle_xpos"/>
  </joint>
</robot>
'''
    out_urdf = os.path.join(a.out_dir, f"{a.name}.urdf")
    open(out_urdf, "w").write(urdf)
    # Keep the official +/- spread but recentre it on the suggested x.
    half = 0.075
    consts = {"drawer_pull_distance": round(float(pull), 4),
              "drawer_inner_offset_xyz": [round(float(v), 4) for v in inner_centre],
              "init_pos_x": round(float(suggested_x + half), 4),
              "position_range": [[-2 * half, -0.18, 0.0], [0.0, 0.0, 0.0]],
              "handle_local_z": round(float(handle[2]), 4),
              "front_face_z_range": [round(float(ilo[2]), 4), round(float(ihi[2]), 4)],
              "_handle_reach": round(float(handle_reach), 4),
              "background_z_offset": round(-bg_gap, 4),
              "shell_height": round(shell_h, 4),
              # Physics knobs. Copied from the official asset as a starting point;
              # they govern whether the gripper can actually drag the drawer and
              # are what calibrate_drawer_physics.py searches over.
              "physics": {
                  "static_friction": 5.0,
                  "dynamic_friction": 4.8,
                  "eef_close_ratio": 1.0,
                  "drive_stiffness": 5.0,
                  "drive_damping": 1.5,
              },
              "_travel": round(travel, 4), "_baked_yaw_deg": yaw}
    open(os.path.join(a.out_dir, f"{a.name}_constants.json"), "w").write(json.dumps(consts, indent=2))
    print(f"\nwrote {out_urdf}\nwrote {a.name}_constants.json")


if __name__ == "__main__":
    main()
