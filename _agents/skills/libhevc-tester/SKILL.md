---
name: libhevc-tester
description: >-
  Use this skill to build, execute, and debug unit tests, regression tests, and Google Benchmark performance tests for the libhevc codebase across multiple target architectures (native x64, ARM64, and ARM32 via QEMU). Covers CTest test suites, GTest filtering, SIMD benchmarking, regression tests, and bitstream verification.
---

# Tester Skill for libhevc

This skill provides step-by-step procedures for compiling, executing, and debugging unit tests, regression tests, and SIMD performance benchmarks for `libhevc` across all supported CPU architectures.

---

## 1. Multi-Architecture Build Matrix

Always build out-of-tree (e.g. `/tmp/build_<arch>` or `build/<arch>`). On remote/network filesystems, local directories like `/tmp/build_<arch>` offer significantly faster compilation. Both `ENABLE_TESTS` (GoogleTest) and `ENABLE_BENCHMARKS` (Google Benchmark) default to `ON`.

### Native x64 Build
```bash
cmake -B /tmp/build_x64 -S . -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_x64
```

### ARM64 (AArch64) Cross-Compilation
Requires `aarch64-linux-gnu-gcc` and `qemu-aarch64`:
```bash
cmake -B /tmp/build_arm64 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64 \
  -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_arm64
```

### ARM32 (AArch32 / armv7-a) Cross-Compilation
Requires `arm-linux-gnueabihf-gcc` and `qemu-arm`:
```bash
cmake -B /tmp/build_arm32 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch32_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-arm \
  -DENABLE_TESTS=ON -DENABLE_BENCHMARKS=ON -G Ninja
ninja -C /tmp/build_arm32
```

---

## 2. Test Execution with CTest

Always pass `--output-on-failure` to ensure error details and stack traces are displayed on failure.

```bash
# Run all unit and regression tests
ctest --test-dir /tmp/build_x64 --output-on-failure
ctest --test-dir /tmp/build_arm64 --output-on-failure
ctest --test-dir /tmp/build_arm32 --output-on-failure

# Run only Decoder regression tests (8-bit and 10-bit bitstreams)
ctest --test-dir /tmp/build_x64 --output-on-failure -R "Dec"

# Run only Encoder regression tests
ctest --test-dir /tmp/build_x64 --output-on-failure -R "Enc"

# Run only SIMD primitive unit tests
ctest --test-dir /tmp/build_x64 --output-on-failure -R "ihevc_"
```

---

## 3. Targeted Test Debugging

When a test fails, isolate and debug it individually:

### 1. Specific Unit Test Executable
Run the test binary directly and use `--gtest_filter`:
```bash
# Run specific test cases within a unit test binary
/tmp/build_x64/ihevc_deblk_test --gtest_filter="*LumaVert*"
/tmp/build_x64/ihevc_itrans_recon_test --gtest_filter="*4x4*"
/tmp/build_x64/ihevc_sao_test --gtest_filter="*EdgeOffset*"
```

### 2. Verbose CTest Output
Run a single regression test with full CTest verbosity (`-V`):
```bash
ctest --test-dir /tmp/build_x64 -V -R "bbb_10b_176x144_yuv420"
```

### 3. Direct QEMU Execution for Cross-Compiled Targets
Run cross-compiled binaries directly under QEMU emulation:
```bash
# ARM64 under QEMU
qemu-aarch64 /tmp/build_arm64/ihevc_itrans_recon_test --gtest_filter="*4x4*"
qemu-aarch64 /tmp/build_arm64/hevc_dec_tests --gtest_filter="*bbb_10b*"

# ARM32 under QEMU
qemu-arm /tmp/build_arm32/ihevc_sao_test
qemu-arm /tmp/build_arm32/hevc_dec_tests --gtest_filter="*bbb_8b*"
```

---

## 4. SIMD Performance Benchmarks (Google Benchmark)

SIMD benchmark executables (`ihevc_*_benchmark` for 8-bit and `ihevc_hbd_*_benchmark` for 8/10-bit High Bit Depth) are built when `ENABLE_BENCHMARKS=ON`. They are standalone binaries in the build directory (not invoked by `ctest`).

Every benchmark automatically verifies SIMD output against the generic C reference (`ARCH_NA`) before entering the timing loop, reporting `SkipWithError("Output mismatch between SIMD and C reference")` on any discrepancy.

### Available Benchmark Binaries
* **Deblocking:** `ihevc_deblk_benchmark`, `ihevc_hbd_deblk_benchmark`
* **SAO:** `ihevc_sao_benchmark`, `ihevc_hbd_sao_benchmark`
* **Inverse Transform & Reconstruction:** `ihevc_itrans_recon_benchmark`, `ihevc_hbd_itrans_recon_benchmark`, `ihevc_chroma_itrans_recon_benchmark`, `ihevc_itrans_res_benchmark`, `ihevc_recon_benchmark`, `ihevc_chroma_recon_benchmark`
* **Intra Prediction:** `ihevc_luma_intra_pred_benchmark`, `ihevc_hbd_luma_intra_pred_benchmark`, `ihevc_chroma_intra_pred_benchmark`, `ihevc_hbd_chroma_intra_pred_benchmark`
* **Inter Prediction:** `ihevc_luma_inter_pred_benchmark`, `ihevc_hbd_luma_inter_pred_benchmark`, `ihevc_chroma_inter_pred_benchmark`, `ihevc_hbd_chroma_inter_pred_benchmark`
* **Padding & Weighted Prediction:** `ihevc_padding_benchmark`, `ihevc_weighted_pred_benchmark`

### Running & Filtering Benchmarks
```bash
# Run all benchmarks in an executable
/tmp/build_x64/ihevc_deblk_benchmark

# Filter by kernel name, block size, bit depth, or architecture (C, SSSE3, SSE42, ARMV8, A9Q)
/tmp/build_x64/ihevc_luma_inter_pred_benchmark --benchmark_filter="BM_LumaInterPred/horz/16x16"
/tmp/build_x64/ihevc_hbd_sao_benchmark --benchmark_filter="BM_HbdSaoEdgeLuma/.*10bit"

# List all registered benchmark cases without running them
/tmp/build_x64/ihevc_sao_benchmark --benchmark_list_tests=true

# Quick smoke run with short minimum measurement time
/tmp/build_x64/ihevc_itrans_recon_benchmark --benchmark_min_time=0.05s

# Run cross-compiled benchmarks under QEMU
qemu-aarch64 /tmp/build_arm64/ihevc_deblk_benchmark --benchmark_filter="ARMV8"
qemu-arm /tmp/build_arm32/ihevc_sao_benchmark --benchmark_filter="A9Q"
```

---

## 5. Standalone Bitstream Validation (`hevcdec`)

To verify decoded bitstream frames or compute MD5 checksums directly:

```bash
# Decode bitstream to YUV 4:2:0 planar
/tmp/build_x64/hevcdec \
  --input path/to/input.hevc \
  --output /tmp/output.yuv \
  --save_output 1 \
  --chroma_format YUV_420P

# Decode and compute output MD5 checksum
/tmp/build_x64/hevcdec \
  --input path/to/input.hevc \
  --chksum /tmp/output.md5 \
  --save_chksum 1

# Multi-threaded decoding (e.g. 10-bit)
/tmp/build_x64/hevcdec \
  --input path/to/10bit_input.hevc \
  --output /tmp/output_10bit.yuv \
  --save_output 1 \
  --num_cores 4
```

---

## 6. Pre-Commit Validation Checklist

Before submitting code, ensure:
1. Native x64 build compiles cleanly without warnings (`-Wall`) for library, test, and benchmark targets.
2. Native x64 test suite passes (`ctest --test-dir /tmp/build_x64 --output-on-failure`).
3. ARM64 build and tests pass via QEMU (`ctest --test-dir /tmp/build_arm64 --output-on-failure`).
4. ARM32 build and tests pass via QEMU (`ctest --test-dir /tmp/build_arm32 --output-on-failure`).
5. Relevant SIMD benchmarks (`ihevc_*_benchmark` / `ihevc_hbd_*_benchmark`) execute without reference mismatch errors.
