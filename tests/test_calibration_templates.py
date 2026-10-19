"""Keep the published current calibration exact and neutral opt-in.

All preparation tests use temporary files and fake model/build helpers. They
do not load a PN SDK, ROS, MuJoCo, recordings or a robot connection.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
RP1 = ROOT / "mocapapi_python/rp1"
CURRENT = RP1 / "pn_axis_to_rp1_28dof_current.json"
DEFAULT = RP1 / "pn_axis_to_rp1_28dof.json"
NEUTRAL = RP1 / "pn_axis_to_rp1_28dof_neutral.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def walk_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item)
    else:
        yield value


def without_model_path(calibration):
    result = copy.deepcopy(calibration)
    result["source_calibration"].pop("robot_model_path")
    return result


def test_published_defaults_are_identical_and_keep_current_dimensions():
    assert DEFAULT.read_bytes() == CURRENT.read_bytes()
    source = read_json(CURRENT)["source_calibration"]
    assert source["source_root_reference_position_m"] == [0., 0., 1.0095348389594085]
    assert source["source_tpose_world"]["Hips"][0][2] == 1.0095348389594085
    for name in ("LeftLeg", "RightLeg", "LeftFoot", "RightFoot"):
        assert source["actor_median_lengths_m"][name] == pytest.approx(.48, abs=.002)
    assert source["root_position_scale"] == .78
    assert source["mapping_profile"] == "pitch"


def test_archived_neutral_dimensions_are_separate_from_the_current_default():
    neutral, current = read_json(NEUTRAL), read_json(CURRENT)
    source = neutral["source_calibration"]
    assert source["source_root_reference_position_m"] == [0., 0., .91]
    for name in ("LeftLeg", "RightLeg", "LeftFoot", "RightFoot"):
        assert source["actor_median_lengths_m"][name] == .43
    assert source["actor_median_lengths_m"] != current["source_calibration"]["actor_median_lengths_m"]


@pytest.mark.parametrize("path", [DEFAULT, CURRENT, NEUTRAL], ids=["default", "current", "neutral"])
def test_templates_keep_analytical_standard_tpose_and_valid_relative_model(path):
    calibration = read_json(path)
    source = calibration["source_calibration"]
    assert "analytical" in source["source_pose_definition"]
    assert source["standard_tpose_verification"]["recorded_pose_reference_used"] is False
    assert source["standard_tpose_verification"]["max_target_position_error_m"] < 1e-12
    assert source["standard_tpose_verification"]["max_target_orientation_error_rad"] < 1e-12
    assert set(source["source_tpose_local_rotation_wxyz"]) == set(source["required_joints"])
    assert all(rotation == [1., 0., 0., 0.]
               for rotation in source["source_tpose_local_rotation_wxyz"].values())
    relative = Path(source["robot_model_path"])
    assert not relative.is_absolute()
    model = (path.parent / relative).resolve(strict=True)
    assert model.is_relative_to(ROOT)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == source["robot_model_sha256"]


@pytest.mark.parametrize("path", [DEFAULT, CURRENT, NEUTRAL], ids=["default", "current", "neutral"])
def test_templates_do_not_distribute_private_recording_provenance(path):
    calibration = read_json(path)
    provenance = calibration["source_calibration"]["length_provenance"]
    assert {"recording", "recording_sha256", "frame_count"}.isdisjoint(provenance)
    assert provenance.get("pose_values_used") is False
    for value in walk_values(calibration):
        if isinstance(value, str):
            assert "/home/" not in value
            assert "/Users/" not in value
            assert "dataset.npz" not in value


def test_current_baseline_keeps_limits_solver_contact_and_torso_settings():
    calibration = read_json(CURRENT)
    preset = read_json(ROOT / "configs/presets/baseline.json")
    for section in ("output_limits", "solver", "contact"):
        assert calibration[section] == preset[section]
    assert calibration["torso_compensation"]["enabled"] is preset["torso_compensation_enabled"]
    assert calibration["output_limits"]["joint_acceleration_enabled"] is False
    assert calibration["contact"]["enabled"] is False
    assert calibration["torso_compensation"]["enabled"] is False


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pn_rp1_template_preparation_test", ROOT / "tools/prepare_calibration.py")
    tool = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, tool)
    spec.loader.exec_module(tool)
    repository = tmp_path / "repository with spaces"
    source = repository / "mocapapi_python"
    rp1 = source / "rp1"
    rp1.mkdir(parents=True)
    model = repository / "assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml"
    model.parent.mkdir(parents=True)
    model.write_text('<mujoco model="rp1.3.2"/>')
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    templates = {}
    for name, original in (("current", CURRENT), ("neutral", NEUTRAL)):
        value = read_json(original)
        value["source_calibration"]["robot_model_sha256"] = model_hash
        destination = rp1 / original.name
        destination.write_text(json.dumps(value))
        templates[name] = destination
    outputs = (rp1 / DEFAULT.name, templates["current"])
    outputs[0].write_bytes(templates["current"].read_bytes())
    preset = repository / "configs/presets/baseline.json"
    preset.parent.mkdir(parents=True)
    preset.write_bytes((ROOT / "configs/presets/baseline.json").read_bytes())
    for name, value in {"ROOT": repository, "SOURCE": source, "DEFAULT_PRESET": preset,
                        "DEFAULT_OUTPUTS": outputs, "TEMPLATE_PATHS": templates}.items():
        monkeypatch.setattr(tool, name, value)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delenv("PN_RP1_MODEL", raising=False)
    paths = ModuleType("pn_rp1_paths")
    paths.resolve_model_path = mock.Mock(side_effect=lambda selected, **kwargs: Path(selected) if selected else model)
    calibration_module = ModuleType("pn_rp1_calibration")
    calibration_module.build_calibration = mock.Mock(side_effect=AssertionError("unexpected calibration rebuild"))
    monkeypatch.setitem(sys.modules, "pn_rp1_paths", paths)
    monkeypatch.setitem(sys.modules, "pn_rp1_calibration", calibration_module)
    for module in ("mocap_api", "pn_sdk", "rclpy", "mujoco", "mink"):
        monkeypatch.setitem(sys.modules, module, None)
    return SimpleNamespace(tool=tool, root=repository, templates=templates, outputs=outputs,
                           model=model, preset=preset, paths=paths,
                           build=calibration_module.build_calibration)


def test_no_arguments_copy_current_without_importing_or_rebuilding_geometry(preparation, monkeypatch, capsys):
    tool = preparation.tool
    before = read_json(preparation.templates["current"])
    neutral_before = preparation.templates["neutral"].read_bytes()
    # Even importing the numerical builder is forbidden in the copy path.
    monkeypatch.setitem(sys.modules, "pn_rp1_calibration", None)
    tool.main([])
    report = json.loads(capsys.readouterr().out)
    for output in preparation.outputs:
        assert without_model_path(read_json(output)) == without_model_path(before)
    assert preparation.outputs[0].read_bytes() == preparation.outputs[1].read_bytes()
    assert preparation.templates["neutral"].read_bytes() == neutral_before
    assert report["template"] == "current"
    assert report["geometry_regenerated"] is False
    assert report["preset"] is None
    assert report["sdk_opened"] is report["ros_started"] is False


def test_output_only_preserves_current_template_settings_without_a_default_overlay(preparation):
    current = read_json(preparation.templates["current"])
    current["solver"]["task_gain"] = .314159
    current["output_limits"]["root_linear_acceleration_limit_m_s2"] = 47.5
    preparation.templates["current"].write_text(json.dumps(current))
    before = {path: path.read_bytes() for path in (*preparation.outputs, preparation.templates["neutral"])}
    output = preparation.root / "outputs/custom/current.json"
    preparation.tool.main(["--output", str(output)])
    result = read_json(output)
    assert without_model_path(result) == without_model_path(current)
    assert (output.parent / result["source_calibration"]["robot_model_path"]).resolve() == preparation.model
    assert {path: path.read_bytes() for path in before} == before
    preparation.build.assert_not_called()


def test_explicit_preset_overlays_parameters_without_reconstructing_tpose(preparation):
    current = read_json(preparation.templates["current"])
    overlay = {"solver": {"task_gain": .4321}, "contact": {"enabled": True},
               "output_limits": {"root_linear_acceleration_limit_m_s2": 42.},
               "torso_compensation_enabled": True}
    preset = preparation.root / "explicit-preset.json"
    preset.write_text(json.dumps(overlay))
    output = preparation.root / "outputs/overlaid.json"
    preparation.tool.main(["--preset", str(preset), "--output", str(output)])
    result = read_json(output)
    assert without_model_path({"source_calibration": result["source_calibration"]}) == \
        without_model_path({"source_calibration": current["source_calibration"]})
    for section in ("solver", "contact", "output_limits"):
        assert result[section] == {**current[section], **overlay[section]}
    assert result["torso_compensation"] == {**current["torso_compensation"], "enabled": True}
    preparation.build.assert_not_called()


def test_neutral_template_requires_an_independent_output_and_leaves_defaults_intact(preparation):
    before = {path: path.read_bytes() for path in (*preparation.outputs, preparation.templates["neutral"])}
    output = preparation.root / "outputs/neutral-trial.json"
    preparation.tool.main(["--template", "neutral", "--output", str(output)])
    assert without_model_path(read_json(output)) == without_model_path(read_json(preparation.templates["neutral"]))
    assert {path: path.read_bytes() for path in before} == before
    preparation.build.assert_not_called()


@pytest.mark.parametrize("selection", ["missing-output", "default", "current", "neutral-archive", "current-to-neutral"])
def test_neutral_selection_cannot_implicitly_replace_default_or_archived_files(preparation, selection):
    arguments = ["--template", "neutral"]
    if selection == "default":
        arguments += ["--output", str(preparation.outputs[0])]
    elif selection == "current":
        arguments += ["--output", str(preparation.outputs[1])]
    elif selection == "neutral-archive":
        arguments += ["--output", str(preparation.templates["neutral"])]
    elif selection == "current-to-neutral":
        arguments = ["--output", str(preparation.templates["neutral"])]
    before = {path: path.read_bytes() for path in (*preparation.outputs, preparation.templates["neutral"])}
    with pytest.raises(SystemExit) as raised:
        preparation.tool.main(arguments)
    assert raised.value.code == 2
    assert {path: path.read_bytes() for path in before} == before
    preparation.build.assert_not_called()


@pytest.mark.parametrize("regeneration", ["changed-model", "recording-lengths"])
def test_regeneration_uses_selected_template_scale_mapping_and_settings(preparation, regeneration):
    current = read_json(preparation.templates["current"])
    current["source_calibration"]["mapping_profile"] = "yaw"
    current["source_calibration"]["root_position_scale"] = .81234
    current["source_calibration"]["actor_median_lengths_m"]["LeftLeg"] = .5123
    current["solver"]["task_gain"] = .2345
    preparation.templates["current"].write_text(json.dumps(current))
    generated = copy.deepcopy(current)
    for section in ("output_limits", "solver", "contact", "torso_compensation"):
        generated[section] = {"generated_defaults": True}
    preparation.build.side_effect = None
    preparation.build.return_value = generated
    output = preparation.root / "outputs/regenerated.json"
    arguments = ["--output", str(output)]
    if regeneration == "changed-model":
        selected = preparation.root / "custom-model.xml"
        selected.write_text('<mujoco model="changed"/>')
        arguments += ["--model", str(selected)]
    else:
        selected = preparation.root / "user-owned-lengths.npz"
        selected.write_bytes(b"fake recording; the builder is mocked")
        arguments += ["--recording-lengths", str(selected)]
    preparation.tool.main(arguments)
    preparation.build.assert_called_once()
    options = preparation.build.call_args.kwargs
    assert options["mapping_profile"] == "yaw"
    assert options["root_position_scale"] == .81234
    if regeneration == "changed-model":
        assert options["model_path"] == selected
        assert options["actor_lengths"] == current["source_calibration"]["actor_median_lengths_m"]
        assert options.get("recording_path") is None
    else:
        assert options["model_path"] == preparation.model
        assert options["recording_path"] == selected
        assert options.get("actor_lengths") is None
    result = read_json(output)
    for section in ("output_limits", "solver", "contact", "torso_compensation"):
        assert result[section] == current[section]
