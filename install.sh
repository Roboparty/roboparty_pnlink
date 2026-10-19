#!/usr/bin/env bash
set -e
PN_RP1_INSTALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PN_RP1_INSTALL_ROOT/run.sh" setup "$@"
