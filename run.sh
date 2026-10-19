#!/usr/bin/env bash
set -e
PN_RP1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PN_RP1_ROOT"
if [[ -x /usr/bin/python3 ]]; then
    exec /usr/bin/python3 tools/setup_environment.py "$@"
fi
exec python3 tools/setup_environment.py "$@"
