#!/usr/bin/env python3
"""Check the source/approved SDK export without printing matched secret values."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    '.so', '.dll', '.dylib', '.stl', '.obj', '.mcap', '.bag', '.npz', '.onnx', '.pt', '.pth', '.exe',
    '.csv', '.bvh', '.jsonl', '.db3', '.urdf', '.mjcf', '.usd', '.usda', '.usdc', '.ply',
    '.glb', '.gltf', '.fbx', '.step', '.stp',
}
LOCAL_DIRECTORIES = {'.git', '.runtime', '.pytest_cache', '__pycache__', 'external',
                     'local_assets', 'outputs', 'output', 'recordings', 'models',
                     'build', 'dist', 'htmlcov'}
VENDOR_MODULES = {'mocap_api.py', 'mocap_main_base.py'}
SDK_SOURCE_PATHS = {'mocapapi_python/mocap_api.py', 'mocapapi_python/mocap_main_base.py'}
SDK_BINARY_SHA256 = {
    'mocapapi_python/lib/librobotapi_x86-64.so': 'bee36adb6e070e8c013f51bdeac2a880e81626ec0cb31a86ccb1cae0a9bc5a8f',
    'mocapapi_python/lib/librobotapi_arm64.so': '8c69886cd16e628904ba96496e2bf9fec41f3ac7817a07366f3c09e985992244',
}
MODEL_ENTRY = 'assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml'
MODEL_SHA256 = '3de357cc73c7ccce0b08ba96ea8d1eee62e734d6811ebe492a6a8ae13c768232'
PATTERNS = {
    'private home path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'GitHub access token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'URL credentials': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-export-hashes', action='store_true',
                        help='also verify the initial export snapshot, before subsequent development')
    args = parser.parse_args()
    manifest_path = ROOT / 'RELEASE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    files = dict(manifest['files'])
    model_manifest = json.loads((ROOT / 'RP1_MODEL_MANIFEST.json').read_text())
    model_files = model_manifest['files']
    entry = ROOT / MODEL_ENTRY
    model_tree = ET.parse(entry).getroot()
    compiler = model_tree.find('compiler')
    referenced = {MODEL_ENTRY}
    for mesh in model_tree.findall('./asset/mesh'):
        mesh_path = (entry.parent / compiler.get('meshdir', '.') / mesh.get('file')).resolve()
        referenced.add(mesh_path.relative_to(ROOT).as_posix())
    if (len(referenced) != 30 or set(model_files) != referenced
            or model_tree.get('model') != 'rp1.3.2' or model_tree.findall('.//include')
            or hashlib.sha256(entry.read_bytes()).hexdigest() != MODEL_SHA256):
        print('FAIL: the RP1 model or its reviewed 29-mesh closure changed.')
        return 1
    # Inspect new source files even before Git is initialized. Local asset and
    # generated-data directories are intentionally excluded; tracked files in
    # those directories are still included by git ls-files below.
    for directory, children, filenames in os.walk(ROOT, followlinks=False):
        children[:] = [name for name in children
                       if name not in LOCAL_DIRECTORIES and not name.startswith('.venv')]
        for name in filenames:
            path = Path(directory) / name
            if (name == manifest_path.name or name == '.env'
                    or (name.startswith('.env.') and name != '.env.example')
                    or (path.parent.name == 'rp1' and name.startswith('pn_axis_to_rp1_'))):
                continue
            files.setdefault(path.relative_to(ROOT).as_posix(), None)
    # If a user later initializes Git, check tracked additions as well as the
    # original allowlist, even if an ignored binary was force-added.
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                                capture_output=True, check=True)
        for relative in result.stdout.decode().split('\0'):
            if relative and relative != manifest_path.name:
                files.setdefault(relative, None)
    errors = []
    for entry in ('install.sh', 'run.sh'):
        if not os.access(ROOT / entry, os.X_OK):
            errors.append((entry, 'entry point is not executable; restore its Git executable bit'))
    verified_sdk_binaries = 0
    verified_model_files = 0
    for relative, expected in sorted(files.items()):
        path = ROOT / relative
        if not path.is_file():
            errors.append((relative, 'missing source file'))
            continue
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        if relative in model_files:
            if sha256 != model_files[relative]['sha256']:
                errors.append((relative, 'RP1 asset differs from the reviewed model manifest'))
            else:
                verified_model_files += 1
            if args.verify_export_hashes and expected and sha256 != expected['sha256']:
                errors.append((relative, 'changed since export; review and regenerate manifest'))
            if relative != MODEL_ENTRY:
                continue
        if relative in SDK_BINARY_SHA256:
            if sha256 != SDK_BINARY_SHA256[relative] or not data.startswith(b'\x7fELF'):
                errors.append((relative, 'PN SDK binary changed; explicit SDK review/update is required'))
            else:
                verified_sdk_binaries += 1
            if args.verify_export_hashes and expected and sha256 != expected['sha256']:
                errors.append((relative, 'changed since export; review and regenerate manifest'))
            continue
        if ((path.suffix.lower() in FORBIDDEN_SUFFIXES and relative not in model_files)
                or (relative.startswith('assets/') and relative not in model_files) or '.so.' in path.name
                or '.git' in Path(relative).parts
                or (path.name in VENDOR_MODULES and relative not in SDK_SOURCE_PATHS)):
            errors.append((relative, 'unapproved binary/model/data/history should not be published'))
        if args.verify_export_hashes and expected and hashlib.sha256(data).hexdigest() != expected['sha256']:
            errors.append((relative, 'changed since export; review and regenerate manifest'))
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            errors.append((relative, 'unreviewed binary file'))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append((relative, label))
    for relative, label in errors:
        print(f'FAIL {relative}: {label}')
    if errors:
        return 1
    print(f'PASS: {len(files)} source/tracked files; {verified_sdk_binaries} approved PN SDK binaries verified by SHA-256.')
    print(f'RP1.3.2: {verified_model_files} approved model files (one XML and 29 meshes) verified by SHA-256.')
    print('No unapproved assets/binaries/data files or matched text credentials. Binary inspection is limited to the approved hashes.')
    print('Scope: this is a finite source scan, not a full Git-history or binary audit.')
    if not (ROOT / 'LICENSE').is_file():
        print('PENDING: the code owner must choose the project license before public release.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
