"""Export a mesh-free RP1.3.2 model for reference kinematics and projection.

The compiled source model is the authority for every body pose, joint pivot,
axis, range and foot capsule. This export preserves those kinematics and real
inertials, but deliberately omits visual meshes and non-foot collision shapes.
It is for FK and reference projection, not an equivalent dynamics simulation.
No model files are modified and no ROS/device dependencies are imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _values(values):
    """Round-trip binary64 values through XML without losing FK precision."""
    return " ".join(format(float(value), ".17g") for value in np.asarray(values).ravel())


def _name(mujoco, model, object_type, object_id):
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    if value is None:
        raise ValueError(f"RP1 object {object_type}/{object_id} is unnamed")
    return value


def export_kinematic_xml(model) -> str:
    """Return a self-contained lightweight MJCF for a compiled RP1 model.

    Includes all body/joint names and same-named 28 motors. Moving-body mass,
    inertia and inertial frame are copied exactly; only foot collision capsule
    geoms survive. Compiled geom ``pos/quat/size`` are used directly, never the
    original ``fromto`` fields or visual mesh offsets.
    """
    import mujoco

    if (model.nq, model.nv, model.nu) != (35, 34, 28):
        raise ValueError("expected a floating-base RP1 with 28 hinge joints")
    root = ET.Element("mujoco", model="rp1.3.2")
    root.append(ET.Comment(
        "Kinematic FK/reference projection export. Real inertials preserved; "
        "visual meshes and non-foot collisions omitted. Not a dynamics replacement."
    ))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="false")
    ET.SubElement(root, "option", timestep=format(float(model.opt.timestep), ".17g"),
                  gravity=_values(model.opt.gravity))
    world = ET.SubElement(root, "worldbody")
    children = {parent: [] for parent in range(model.nbody)}
    for body_id in range(1, model.nbody):
        children[int(model.body_parentid[body_id])].append(body_id)
    exported_foot_capsules = 0

    def append_body(body_id, parent_xml):
        nonlocal exported_foot_capsules
        body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        element = ET.SubElement(parent_xml, "body", name=body_name,
                                pos=_values(model.body_pos[body_id]),
                                quat=_values(model.body_quat[body_id]))
        mass = float(model.body_mass[body_id])
        joint_ids = range(int(model.body_jntadr[body_id]),
                          int(model.body_jntadr[body_id] + model.body_jntnum[body_id]))
        if mass > 0.0:
            if np.any(model.body_inertia[body_id] <= 0.0):
                raise ValueError(f"{body_name} has non-positive compiled inertia")
            ET.SubElement(element, "inertial", pos=_values(model.body_ipos[body_id]),
                          quat=_values(model.body_iquat[body_id]),
                          mass=format(mass, ".17g"), diaginertia=_values(model.body_inertia[body_id]))
        elif model.body_jntnum[body_id]:
            raise ValueError(f"{body_name} is a moving body with no real inertia")
        for joint_id in joint_ids:
            joint_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_type = model.jnt_type[joint_id]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                ET.SubElement(element, "freejoint", name=joint_name)
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                attrs = {
                    "name": joint_name, "type": "hinge",
                    "pos": _values(model.jnt_pos[joint_id]),
                    "axis": _values(model.jnt_axis[joint_id]),
                    "limited": "true" if model.jnt_limited[joint_id] else "false",
                    "range": _values(model.jnt_range[joint_id]),
                    "ref": format(float(model.qpos0[model.jnt_qposadr[joint_id]]), ".17g"),
                }
                ET.SubElement(element, "joint", **attrs)
            else:
                raise ValueError(f"unsupported RP1 joint type for {joint_name}: {joint_type}")
        first = int(model.body_geomadr[body_id])
        for geom_id in range(first, first + int(model.body_geomnum[body_id])):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not geom_name.startswith(("left_foot_", "right_foot_")):
                continue
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError(f"foot projection expects a capsule: {geom_name}")
            ET.SubElement(element, "geom", name=geom_name, type="capsule",
                          pos=_values(model.geom_pos[geom_id]),
                          quat=_values(model.geom_quat[geom_id]),
                          size=_values(model.geom_size[geom_id, :2]),
                          contype="0", conaffinity="0", rgba="0.55 0.6 0.65 1")
            exported_foot_capsules += 1
        for child in children[body_id]:
            append_body(child, element)

    for body_id in children[0]:
        append_body(body_id, world)
    if exported_foot_capsules != 14:
        raise ValueError(f"expected RP1's fourteen foot capsules, found {exported_foot_capsules}")
    actuators = ET.SubElement(root, "actuator")
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("RP1 kinematic export requires joint motor transmissions")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        ET.SubElement(actuators, "motor", name=actuator_name, joint=joint_name,
                      gear=_values(model.actuator_gear[actuator_id]))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _foot_capsules(model, mujoco):
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("left_foot_", "right_foot_"))
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE
    }


def _sole_min(model, data, geoms):
    heights = {}
    for side in ("left", "right"):
        heights[side] = min(
            float(data.geom_xpos[gid, 2]
                  - abs(data.geom_xmat[gid].reshape(3, 3)[2, 2]) * model.geom_size[gid, 1]
                  - model.geom_size[gid, 0])
            for name, gid in geoms.items() if name.startswith(side + "_")
        )
    return heights


def assert_fk_equivalence(full_model, lightweight_model, random_qposes=30, *, seed=142, tolerance=1e-9):
    """Verify all bodies and actual capsule soles over arbitrary valid poses.

    ``random_qposes`` can be a sample count or an explicit (N,35) array. Returns
    maximum errors for audit; raises AssertionError on any FK mismatch.
    """
    import mujoco

    for attr in ("nq", "nv", "nu", "nbody", "njnt"):
        if getattr(full_model, attr) != getattr(lightweight_model, attr):
            raise AssertionError(f"model {attr} differs")
    if lightweight_model.nmesh or lightweight_model.ntex:
        raise AssertionError("lightweight projection model contains visual assets")
    bodies = []
    for body_id in range(full_model.nbody):
        name = _name(mujoco, full_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        lightweight_id = mujoco.mj_name2id(lightweight_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if lightweight_id < 0:
            raise AssertionError(f"body missing from lightweight model: {name}")
        bodies.append((body_id, lightweight_id))
    for joint_id in range(full_model.njnt):
        name = _name(mujoco, full_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        light_id = mujoco.mj_name2id(lightweight_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if light_id != joint_id or full_model.jnt_qposadr[joint_id] != lightweight_model.jnt_qposadr[light_id]:
            raise AssertionError(f"joint order/address differs: {name}")
    full_geoms, light_geoms = _foot_capsules(full_model, mujoco), _foot_capsules(lightweight_model, mujoco)
    if set(full_geoms) != set(light_geoms):
        raise AssertionError("foot capsule names differ")
    if isinstance(random_qposes, (int, np.integer)):
        if random_qposes < 1:
            raise ValueError("need at least one randomized pose")
        rng = np.random.default_rng(seed)
        poses = np.tile(full_model.qpos0, (int(random_qposes), 1))
        poses[:, :3] = rng.uniform([-2.0, -2.0, 0.3], [2.0, 2.0, 1.5], (len(poses), 3))
        quaternions = rng.normal(size=(len(poses), 4))
        poses[:, 3:7] = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        for joint_id in range(1, full_model.njnt):
            adr = full_model.jnt_qposadr[joint_id]
            bounds = full_model.jnt_range[joint_id] if full_model.jnt_limited[joint_id] else [-np.pi, np.pi]
            poses[:, adr] = rng.uniform(bounds[0], bounds[1], len(poses))
    else:
        poses = np.asarray(random_qposes, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != full_model.nq or not np.all(np.isfinite(poses)):
            raise ValueError("explicit poses must have shape (N,35) and be finite")
    full_data, light_data = mujoco.MjData(full_model), mujoco.MjData(lightweight_model)
    metrics = {"max_body_position_error_m": 0.0, "max_body_rotation_matrix_error": 0.0,
               "max_foot_capsule_position_error_m": 0.0, "max_foot_capsule_rotation_matrix_error": 0.0,
               "max_sole_height_error_m": 0.0, "pose_count": len(poses), "tolerance": tolerance}
    for pose in poses:
        full_data.qpos[:], light_data.qpos[:] = pose, pose
        mujoco.mj_forward(full_model, full_data)
        mujoco.mj_forward(lightweight_model, light_data)
        for fid, lid in bodies:
            metrics["max_body_position_error_m"] = max(metrics["max_body_position_error_m"],
                                                       float(np.max(np.abs(full_data.xpos[fid] - light_data.xpos[lid]))))
            metrics["max_body_rotation_matrix_error"] = max(metrics["max_body_rotation_matrix_error"],
                                                           float(np.max(np.abs(full_data.xmat[fid] - light_data.xmat[lid]))))
        for name, fid in full_geoms.items():
            lid = light_geoms[name]
            metrics["max_foot_capsule_position_error_m"] = max(metrics["max_foot_capsule_position_error_m"],
                                                              float(np.max(np.abs(full_data.geom_xpos[fid] - light_data.geom_xpos[lid]))))
            metrics["max_foot_capsule_rotation_matrix_error"] = max(metrics["max_foot_capsule_rotation_matrix_error"],
                                                                   float(np.max(np.abs(full_data.geom_xmat[fid] - light_data.geom_xmat[lid]))))
        full_sole, light_sole = _sole_min(full_model, full_data, full_geoms), _sole_min(lightweight_model, light_data, light_geoms)
        metrics["max_sole_height_error_m"] = max(metrics["max_sole_height_error_m"],
                                                 max(abs(full_sole[side] - light_sole[side]) for side in full_sole))
    if any(value > tolerance for key, value in metrics.items() if key.startswith("max_")):
        raise AssertionError(f"kinematic export changed FK: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help="external RP1.3.2 MJCF; defaults to PN_RP1_MODEL")
    parser.add_argument("--output", type=Path, help="optional exported mesh-free MJCF file")
    parser.add_argument("--poses", type=int, default=30)
    args = parser.parse_args()
    import mujoco
    from pn_rp1_paths import resolve_model_path

    full = mujoco.MjModel.from_xml_path(str(resolve_model_path(args.model)))
    xml = export_kinematic_xml(full)
    light = mujoco.MjModel.from_xml_string(xml)
    result = assert_fk_equivalence(full, light, args.poses)
    result.update({"full_mesh_count": full.nmesh, "light_mesh_count": light.nmesh,
                   "xml_bytes": len(xml.encode("utf-8"))})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(xml, encoding="utf-8")
        result["output"] = str(args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
