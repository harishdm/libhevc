# Agent Guide for libhevc

This guide provides architectural context, core development rules, build instructions, and verification workflows for AI coding assistants working in the `libhevc` repository. Specialized runbooks and procedures are provided via skills in `_agents/skills/`.

---

## 1. Project Overview & Codebase Architecture

`libhevc` is an optimized HEVC (ITU-T H.265 / ISO/IEC 23008-2) video decoder and encoder implementation originally developed by Ittiam Systems and maintained for Android and Linux environments.

### Core Directory Layout
* **`common/`**: Shared core signal processing and SIMD primitives (interpolation filters, intra/inter prediction, inverse transforms, deblocking, SAO, buffer/DPB management). Architecture-optimized SIMD implementations reside in `common/x86/` (SSSE3, SSE4.2, AVX2), `common/arm/` (NEON), and `common/arm64/` (AArch64).
* **`decoder/`**: HEVC video decoder (`ihevcd_*`): NAL and slice parsing, CTB/TU reconstruction pipeline, multithreading job queue, and architecture function selectors.
* **`encoder/`**: HEVC video encoder (`ihevce_*`): Motion estimation, rate control, mode decision, and entropy coding (CABAC).
* **`examples/`**: Standalone CLI applications (`hevcdec`, `hevcenc`).
* **`tests/`**: GTest unit tests and Google Benchmark performance tests (`tests/common/`), along with bitstream regression test harnesses (`tests/decoder/`, `tests/encoder/`).
* **`cmake/`**: CMake helper modules and cross-compilation toolchains (`cmake/toolchains/aarch32_toolchain.cmake`, `cmake/toolchains/aarch64_toolchain.cmake`).

---

## 2. Core Architectural Rule: Function Selector Pattern

The codec dispatches performance-critical SIMD routines via `func_selector_t` defined in [`decoder/ihevcd_function_selector.h`](decoder/ihevcd_function_selector.h) and `ihevc_func_selector_t` defined in [`common/ihevc_function_selector.h`](common/ihevc_function_selector.h).

When function pointers are added, modified, or removed, **all five architecture initializer files must be updated consistently** in `decoder/` (and in `common/` if exposed in `ihevc_func_selector_t`):
1. Generic C: `decoder/ihevcd_function_selector_generic.c` (`common/ihevc_function_selector_generic.c`)
2. x86 SSSE3: `decoder/x86/ihevcd_function_selector_ssse3.c` (`common/x86/ihevc_function_selector_ssse3.c`)
3. x86 SSE4.2: `decoder/x86/ihevcd_function_selector_sse42.c` (`common/x86/ihevc_function_selector_sse42.c`)
4. ARMv7 NEON: `decoder/arm/ihevcd_function_selector_a9q.c` (`common/arm/ihevc_function_selector_a9q.c`)
5. ARMv8 AArch64: `decoder/arm64/ihevcd_function_selector_av8.c` (`common/arm64/ihevc_function_selector_av8.c`)

When adding new SIMD or HBD kernels, also update the corresponding function pointer getters in `tests/common/ihevc_*_utils.cc` so unit tests and benchmarks exercise the new implementations.

### Invariants:
* **No uninitialized pointers:** Leaving a function pointer uninitialized causes runtime NULL pointer dereferences. If an architecture lacks an optimized SIMD kernel, initialize the pointer to the generic C function (or to the baseline SIMD tier, e.g. SSSE3 for SSE4.2).
* **Only dispatch SIMD routines:** Do not add entries to `func_selector_t` for internal helper functions that lack SIMD variants; keep those as static or inline functions.
* **Remove dead selectors:** Clean up function pointers and their initializers across all 5 files when refactoring.

---

## 3. Environment & Build Instructions

* **Out-of-tree builds required:** Never build inside the source tree (CMake will error). Use an out-of-tree directory (e.g. `/tmp/build_<arch>` or `build/<arch>`). On network-mounted filesystems, building in a local directory like `/tmp/build_<arch>` improves build speed.
* **Avoid recursive search:** Avoid running recursive `find` or `grep` across the whole repository on remote/network filesystems. Use `git grep`, `git ls-files`, or targeted file lookups.
* **CMake test & benchmark options:** Both `ENABLE_TESTS` (GoogleTest unit and regression tests) and `ENABLE_BENCHMARKS` (Google Benchmark SIMD performance tests) default to `ON`.

### Build Commands

```bash
# Native x64 Build (unit tests, regression tests, and benchmarks)
cmake -B /tmp/build_x64 -S . -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_x64

# ARM64 (AArch64) Cross-Compilation (requires aarch64-linux-gnu-gcc and qemu-aarch64)
cmake -B /tmp/build_arm64 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64 \
  -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_arm64

# ARM32 (AArch32 / armv7-a) Cross-Compilation (requires arm-linux-gnueabihf-gcc and qemu-arm)
cmake -B /tmp/build_arm32 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch32_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-arm \
  -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_arm32
```

---

## 4. Testing & Verification Checklist

Run unit and regression tests using CTest with `--output-on-failure`:

```bash
# Run all unit and regression tests
ctest --test-dir /tmp/build_x64 --output-on-failure

# Run test subsets
ctest --test-dir /tmp/build_x64 --output-on-failure -R "Dec"     # Decoder regression
ctest --test-dir /tmp/build_x64 --output-on-failure -R "Enc"     # Encoder regression
ctest --test-dir /tmp/build_x64 --output-on-failure -R "ihevc_"  # SIMD unit tests
```

Run SIMD performance benchmarks directly (each benchmark also validates SIMD output against the generic C reference `ARCH_NA` prior to measurement):

```bash
# Run a benchmark executable with optional filter
/tmp/build_x64/ihevc_deblk_benchmark --benchmark_filter="BM_DeblkLuma"
/tmp/build_x64/ihevc_hbd_sao_benchmark --benchmark_filter="10bit"

# Run cross-compiled benchmarks under QEMU
qemu-aarch64 /tmp/build_arm64/ihevc_itrans_recon_benchmark
qemu-arm /tmp/build_arm32/ihevc_luma_inter_pred_benchmark
```

### Pre-Commit Verification Checklist
Before submitting any changes, ensure:
1. Code compiles without warnings (`-Wall` clean), including unit test and benchmark targets.
2. All tests pass on native **x64** (`ctest --test-dir /tmp/build_x64 --output-on-failure`).
3. All tests pass on **ARM64** via QEMU (`ctest --test-dir /tmp/build_arm64 --output-on-failure`).
4. All tests pass on **ARM32** via QEMU (`ctest --test-dir /tmp/build_arm32 --output-on-failure`).
5. Relevant SIMD benchmarks (`ihevc_*_benchmark` / `ihevc_hbd_*_benchmark`) run without reference mismatch errors when SIMD kernels are added or modified.
6. Bitstream regression MD5 checksums match golden references unless explicitly updating for standard compliance.

---

## 5. Git Commit Guidelines

* **Commit subject format:** Use subsystem prefixes (`[decoder] ...`, `[encoder] ...`, `[common] ...`, `[DecTests] ...`, `[EncTests] ...`, `tests: ...`, `docs: ...`).
* **Atomic commits:** Separate refactoring and dead-code cleanup from functional changes.
* **Clean diffs:** Ensure no temporary debug prints (`printf`), commented-out code, or untracked scratch files remain.

---

## 6. Available Skills (`_agents/skills/`)

Specialized runbooks and checklists are available as on-demand skills:
* **`libhevc-reviewer`**: In-depth code review protocol covering function selector integrity across all 5 architectures, 10-bit High Bit Depth pointer and stride safety, SIMD alignment and memory bounds, test/benchmark coverage, and golden checksum validation.
* **`libhevc-tester`**: Testing and benchmarking runbook covering CTest, GTest filtering (`--gtest_filter`), Google Benchmark execution (`--benchmark_filter`), running cross-compiled binaries under QEMU, and standalone `hevcdec`/`hevcenc` execution.
* **`libhevc-formatter`**: Code style guide for C library code (4 spaces, Allman braces, Ittiam Hungarian naming prefixes) and C++17 test/benchmark code (`tests/common/`), plus automated pre-commit hygiene checks (`git diff --check`).
