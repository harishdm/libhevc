---
name: libhevc-tester
description: >-
  Use this skill to build, execute, and debug tests for the libhevc codebase across multiple target architectures (native x64, ARM64, and ARM32 via QEMU). Covers CTest test suites, GTest filtering, regression tests, and bitstream verification.
---

# Tester Skill for libhevc

This skill provides step-by-step procedures for compiling, executing, and debugging unit and regression tests for `libhevc` across all supported CPU architectures.

---

## 1. Multi-Architecture Build Matrix

Always build out-of-tree (e.g. `/tmp/build_<arch>` or `build/<arch>`). On remote/network filesystems, local directories like `/tmp/build_<arch>` offer significantly faster compilation.

### Native x64 Build
```bash
cmake -B /tmp/build_x64 -S . -DENABLE_TESTS=ON -G Ninja
ninja -C /tmp/build_x64
```

### ARM64 (AArch64) Cross-Compilation
Requires `aarch64-linux-gnu-gcc` and `qemu-aarch64`:
```bash
cmake -B /tmp/build_arm64 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64 \
  -DENABLE_TESTS=ON -G Ninja
ninja -C /tmp/build_arm64
```

### ARM32 (AArch32 / armv7-a) Cross-Compilation
Requires `arm-linux-gnueabihf-gcc` and `qemu-arm`:
```bash
cmake -B /tmp/build_arm32 -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch32_toolchain.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-arm \
  -DENABLE_TESTS=ON -G Ninja
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

# Run only DSP primitive unit tests
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

## 4. Standalone Bitstream Validation (`hevcdec`)

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

## 5. Pre-Commit Validation Checklist

Before submitting code, ensure:
1. Native x64 build compiles cleanly without warnings (`-Wall`).
2. Native x64 test suite passes (`ctest --test-dir /tmp/build_x64 --output-on-failure`).
3. ARM64 build and tests pass via QEMU (`ctest --test-dir /tmp/build_arm64 --output-on-failure`).
4. ARM32 build and tests pass via QEMU (`ctest --test-dir /tmp/build_arm32 --output-on-failure`).
