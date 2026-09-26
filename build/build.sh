#!/usr/bin/env bash
# Compile the Mojo sources into the ctypes-loadable shared library.
#
# One compilation unit: `capi.mojo` re-exports all of `linalg` over the C ABI,
# and Mojo build cost is essentially fixed per unit (see MOJO_NOTES.md 6).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/dist/libmojokarateclub.so"

# `pixi run` sets MODULAR_HOME; a bare `mojo` invocation does not, and without
# it every import fails with "unable to locate module 'std'" (MOJO_NOTES.md 0).
if [[ -z "${MODULAR_HOME:-}" ]]; then
    export MODULAR_HOME="$(dirname "$(dirname "$(command -v mojo)")")/share/max"
fi

mkdir -p "$root/dist"

# `-o` takes a FILE path, not a directory (MOJO_NOTES.md 0).
# `-I src/mojokarateclub` is required: capi.mojo does a bare `import linalg`,
# which must resolve to the sibling linalg.mojo. Without it every symbol
# fails with "package 'linalg' has no declaration ...".
exec mojo build --emit shared-lib -I "$root/src/mojokarateclub" \
  -o "$out" "$root/src/mojokarateclub/capi.mojo"
