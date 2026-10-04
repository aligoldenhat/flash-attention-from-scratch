# Common tasks. Python lives in .venv (uv); the CMake build (build/) is the standalone driver.
#
#   make build      build + install the Python extension (editable)
#   make test       GoogleTest (ctest) + pytest against the float32 reference
#   make sanitize   compute-sanitizer memcheck + racecheck (+ synccheck, initcheck)
#   make bench      full benchmark sweep -> bench/results/<gpu>.csv, then plot + table
#   make profile    Nsight Compute report of the kernel (needs sudo for GPU counters)
#   make tune       tile-configuration sweep for every (d, causal) slot
#   make lint       clang-format + clang-tidy
#   make dev        strict-warning CMake builds of the standalone driver (debug, release, asan)
#
# Override the GPU architecture for the rented GPUs, e.g.  make build ARCH=8.9  (RTX 4090)
# or  make build ARCH=12.0  (RTX 5090). Default: the GPU in this machine.

PY      ?= .venv/bin/python
ARCH    ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
SAN      = compute-sanitizer --error-exitcode 1
DRIVER   = build/release/fa_dev
JOBS    ?= 4

.PHONY: build test sanitize bench plot profile nsys tune lint dev clean

build:
	TORCH_CUDA_ARCH_LIST="$(ARCH)" MAX_JOBS=$(JOBS) uv pip install --python $(PY) -e . --no-build-isolation

test: dev
	ctest --test-dir build/release --output-on-failure
	$(PY) -m pytest -q

dev:
	cmake --preset debug && cmake --build --preset debug
	cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=$(subst .,,$(ARCH)) && cmake --build --preset release
	cmake --preset asan && cmake --build --preset asan

# Sanitizers run on the standalone driver (fast, only our kernel in the process) for shapes
# that hit every code path: odd N (partial Q and K/V tiles), both head dims, causal on/off,
# and on the pytest suite filtered to our kernel (catches binding-level mistakes too).
SAN_SHAPES = "1 2 257 64 0" "1 2 257 64 1" "2 3 200 128 0" "2 3 200 128 1" "1 1 1 64 1" "1 2 1000 128 1"
SAN_VARIANTS = base opt fp16acc
sanitize: dev
	@for tool in memcheck racecheck synccheck initcheck; do \
	  for variant in $(SAN_VARIANTS); do \
	    for shape in $(SAN_SHAPES); do \
	      echo "== $$tool $$variant $$shape"; \
	      $(SAN) --tool $$tool $(DRIVER) $$shape 1 $$variant | tail -n 2 || exit 1; \
	    done; \
	  done; \
	  echo "== $$tool fa_tests"; \
	  $(SAN) --tool $$tool build/release/fa_tests --gtest_brief=1 | tail -n 2 || exit 1; \
	done
	$(SAN) --tool memcheck --kernel-name kns=flash_fwd_kernel $(PY) -m pytest -q -x tests -k "not 2048 and not deterministic"
	$(SAN) --tool racecheck --kernel-name kns=flash_fwd_kernel $(PY) -m pytest -q -x tests -k "test_matches_reference and (100 or 257) "

bench:
	$(PY) bench/benchmark.py
	$(PY) bench/plot.py

plot:
	$(PY) bench/plot.py

profile: dev
	profile/ncu.sh d128 4 16 4096 128 0
	profile/ncu.sh d64 4 32 4096 64 0

nsys:
	profile/nsys.sh 4096 128 0

tune:
	@for v in opt fp16acc; do for slot in "64 0" "64 1" "128 0" "128 1"; do bench/tune.sh $$v $$slot; echo; done; done

lint:
	scripts/lint.sh

clean:
	rm -rf build fa/_C*.so *.egg-info
