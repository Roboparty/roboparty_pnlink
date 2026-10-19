#!/usr/bin/env python3
"""Prepare PN/RP1 calibration from the current or archived neutral template.

No SDK, ROS, network connection or recorded human pose is used by default.
The default preserves the shipped current calibration's numerical values and
analytical T-pose. Recording lengths or a changed model regenerate geometry;
write custom calibrations to ignored outputs/ with --output. Selecting the
archived neutral template requires an independent output file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mocapapi_python'
DEFAULT_PRESET = ROOT / 'configs/presets/baseline.json'
DEFAULT_OUTPUTS = (
    SOURCE / 'rp1/pn_axis_to_rp1_28dof.json',
    SOURCE / 'rp1/pn_axis_to_rp1_28dof_current.json',
)
TEMPLATE_PATHS = {
    'current': SOURCE / 'rp1/pn_axis_to_rp1_28dof_current.json',
    'neutral': SOURCE / 'rp1/pn_axis_to_rp1_28dof_neutral.json',
}
ALLOWED_PRESET_KEYS = {
    'description', 'output_limits', 'solver', 'contact',
    'torso_compensation_enabled',
}


def load_preset(path: Path) -> dict:
    preset = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(preset, dict):
        raise ValueError('preset must be a JSON object')
    unknown = set(preset) - ALLOWED_PRESET_KEYS
    if unknown:
        raise ValueError(f'unsupported preset fields: {sorted(unknown)}')
    for section in ('output_limits', 'solver', 'contact'):
        if section in preset and not isinstance(preset[section], dict):
            raise ValueError(f'preset.{section} must be a JSON object')
    if ('torso_compensation_enabled' in preset
            and type(preset['torso_compensation_enabled']) is not bool):
        raise ValueError('preset.torso_compensation_enabled must be boolean')
    return preset


def apply_preset(calibration: dict, preset: dict) -> dict:
    result = copy.deepcopy(calibration)
    for section in ('output_limits', 'solver', 'contact'):
        if section in preset:
            result[section] = {
                **result.get(section, {}), **copy.deepcopy(preset[section]),
            }
    if 'torso_compensation_enabled' in preset:
        result.setdefault('torso_compensation', {})['enabled'] = preset['torso_compensation_enabled']
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', choices=tuple(TEMPLATE_PATHS), default='current',
                        help='calibration base; current preserves the shipped live values, neutral requires --output')
    parser.add_argument('--model', type=Path,
                        default=Path(os.environ['PN_RP1_MODEL']) if os.environ.get('PN_RP1_MODEL') else None,
                        help='RP1.3.2 MJCF; defaults to PN_RP1_MODEL or the bundled model')
    parser.add_argument('--preset', type=Path,
                        help='optional parameter overlay, e.g. configs/presets/baseline.json; omitted preserves the template settings')
    parser.add_argument('--recording-lengths', type=Path,
                        help='optional user-owned PN NPZ: scalar median bone lengths only; no recorded pose reference')
    parser.add_argument('--output', type=Path,
                        help='write only this calibration; current without --output replaces the two default aliases')
    args = parser.parse_args(argv)
    if args.template == 'neutral' and args.output is None:
        parser.error('--template neutral requires --output to an independent file under outputs/')
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output == TEMPLATE_PATHS['neutral'].expanduser().resolve():
            parser.error('the archived neutral template cannot be overwritten; select a different --output')
        if args.template == 'neutral' and output in (path.resolve() for path in DEFAULT_OUTPUTS):
            parser.error('--template neutral requires an independent --output, separate from the current default aliases')
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        template_path = TEMPLATE_PATHS[args.template].expanduser().resolve(strict=True)
        template = json.loads(template_path.read_text(encoding='utf-8'))
        source_calibration = template['source_calibration']
        sys.path.insert(0, str(SOURCE))
        from pn_rp1_paths import resolve_model_path
        model_path = resolve_model_path(
            args.model, configured_path=source_calibration['robot_model_path'],
            config_path=template_path)
        preset_path = args.preset.expanduser().resolve(strict=True) if args.preset is not None else None
        recording = (args.recording_lengths.expanduser().resolve(strict=True)
                     if args.recording_lengths is not None else None)
        preset = load_preset(preset_path) if preset_path is not None else None
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        regenerated = recording is not None or model_hash != source_calibration['robot_model_sha256']
        if regenerated:
            from pn_rp1_calibration import build_calibration
            dimensions = ({'recording_path': recording} if recording is not None
                          else {'actor_lengths': copy.deepcopy(source_calibration['actor_median_lengths_m'])})
            calibration = build_calibration(
                model_path=model_path,
                mapping_profile=source_calibration['mapping_profile'],
                root_position_scale=source_calibration['root_position_scale'],
                **dimensions,
            )
            for section in ('output_limits', 'solver', 'contact', 'torso_compensation'):
                if section in template:
                    calibration[section] = copy.deepcopy(template[section])
        else:
            calibration = copy.deepcopy(template)
        if preset is not None:
            calibration = apply_preset(calibration, preset)
        outputs = (args.output.expanduser().resolve(),) if args.output else DEFAULT_OUTPUTS
        for output in outputs:
            calibration['source_calibration']['robot_model_path'] = os.path.relpath(model_path, output.parent)
            encoded = json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + '.tmp')
            temporary.write_text(encoded, encoding='utf-8')
            temporary.replace(output)
    except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f'Calibration preparation failed: {exc}\n')
    print(json.dumps({
        'outputs': [str(path) for path in outputs],
        'preset': str(preset_path) if preset_path is not None else None, 'robot': 'rp1.3.2',
        'template': args.template, 'geometry_regenerated': regenerated,
        'mapping_profile': calibration['source_calibration']['mapping_profile'],
        'root_position_scale': calibration['source_calibration']['root_position_scale'],
        'source_pose': 'analytical standard T-pose',
        'actor_dimensions': ('user recording scalar medians' if recording else
                             'selected template dimensions'),
        'sdk_opened': False, 'ros_started': False,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
