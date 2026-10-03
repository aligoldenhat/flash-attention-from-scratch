#!/usr/bin/env bash
# Run clang-format (check only) and clang-tidy on the project's CUDA sources.
#
#   scripts/lint.sh          check formatting + run clang-tidy
#   scripts/lint.sh --fix    reformat files in place, then run clang-tidy
#
# Why not `clang-tidy -p build/debug`? compile_commands.json holds nvcc
# commands, and clang-tidy (unlike clangd) doesn't read .clangd to strip the
# nvcc-only flags. So we pass clang-compatible flags after `--` instead.
# Keep these in sync with the Add: section of .clangd.
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the newest installed version (clang 21 can't fully parse CUDA 13 + GCC 15).
pick() { for c in "$@"; do command -v "$c" >/dev/null && { echo "$c"; return; }; done; echo "$1"; }
CLANG_TIDY=${CLANG_TIDY:-$(pick clang-tidy-22 clang-tidy)}
CLANG_FORMAT=${CLANG_FORMAT:-$(pick clang-format-22 clang-format)}

GTEST_INC=build/debug/_deps/googletest-src/googletest/include  # downloaded by `cmake --preset debug`
CUDA_FLAGS=(
    -xcuda -std=c++20 -Icsrc -isystem "$GTEST_INC"
    --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_86
    --no-cuda-version-check -Wno-unknown-cuda-version
)

mapfile -t all_files < <(find csrc tests/cpp \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \) | sort)
# Headers are checked through the .cu files that include them (HeaderFilterRegex).
mapfile -t cu_files < <(printf '%s\n' "${all_files[@]}" | grep '\.cu$')

# Run all tools even if one fails; fail at the end if any did.
status=0

echo "== $CLANG_FORMAT"
if [[ "${1:-}" == "--fix" ]]; then
    "$CLANG_FORMAT" -i "${all_files[@]}" || status=1
else
    "$CLANG_FORMAT" --dry-run --Werror "${all_files[@]}" || status=1
fi

if [[ ! -d $GTEST_INC ]]; then
    echo "GoogleTest headers missing: run 'cmake --preset debug' first" >&2
    exit 1
fi
echo "== $CLANG_TIDY (CUDA sources)"
"$CLANG_TIDY" --quiet "${cu_files[@]}" -- "${CUDA_FLAGS[@]}" || status=1

# bindings.cpp is plain C++ against PyTorch: ask the venv's torch for its include paths
# (the same ones setup.py uses) instead of hard-coding them.
PY=${PY:-.venv/bin/python}
if [[ -x $PY ]]; then
    echo "== $CLANG_TIDY (csrc/bindings.cpp)"
    mapfile -t torch_inc < <("$PY" -c "
import sysconfig, torch.utils.cpp_extension as c
for p in c.include_paths() + [sysconfig.get_paths()['include']]: print('-isystem' + p)")
    "$CLANG_TIDY" --quiet csrc/bindings.cpp -- -xc++ -std=c++20 -Icsrc "${torch_inc[@]}" \
        -isystem/usr/local/cuda/include -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H ||
        status=1
else
    echo "skipping csrc/bindings.cpp: no Python venv at $PY" >&2
fi

exit "$status"
