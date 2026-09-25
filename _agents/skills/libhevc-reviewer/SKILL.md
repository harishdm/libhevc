---
name: libhevc-reviewer
description: >-
  Use this skill to conduct thorough code reviews of modifications, refactorings, or new features in the libhevc repository. Covers function selector integrity across all architectures, 10-bit High Bit Depth pointer and stride safety, SIMD alignment and memory bounds, unit test and benchmark coverage, golden checksum parity, and commit hygiene.
---

# Code Reviewer Skill for libhevc

This skill provides a structured code review protocol for changes submitted to the `libhevc` codebase. Use this checklist and inspection workflow when reviewing patches, commits, or pull requests.

---

## 1. Review Workflow

1. **Examine the Change Diff:**
   ```bash
   # Working tree changes vs HEAD
   git diff HEAD

   # Recent commit
   git show --stat -p HEAD
   ```

2. **Verify Function Selector Integrity (`func_selector_t` & `ihevc_func_selector_t`):**
   * If any function pointer in [`decoder/ihevcd_function_selector.h`](file:///work/codecs/libhevc/decoder/ihevcd_function_selector.h) (`func_selector_t`) or [`common/ihevc_function_selector.h`](file:///work/codecs/libhevc/common/ihevc_function_selector.h) (`ihevc_func_selector_t`) was added, removed, or modified:
     * Check that **all 5 architecture initializers** were updated consistently in `decoder/` (and in `common/` if part of `ihevc_func_selector_t`):
       1. [`decoder/ihevcd_function_selector_generic.c`](file:///work/codecs/libhevc/decoder/ihevcd_function_selector_generic.c) / [`common/ihevc_function_selector_generic.c`](file:///work/codecs/libhevc/common/ihevc_function_selector_generic.c)
       2. [`decoder/x86/ihevcd_function_selector_ssse3.c`](file:///work/codecs/libhevc/decoder/x86/ihevcd_function_selector_ssse3.c) / [`common/x86/ihevc_function_selector_ssse3.c`](file:///work/codecs/libhevc/common/x86/ihevc_function_selector_ssse3.c)
       3. [`decoder/x86/ihevcd_function_selector_sse42.c`](file:///work/codecs/libhevc/decoder/x86/ihevcd_function_selector_sse42.c) / [`common/x86/ihevc_function_selector_sse42.c`](file:///work/codecs/libhevc/common/x86/ihevc_function_selector_sse42.c)
       4. [`decoder/arm/ihevcd_function_selector_a9q.c`](file:///work/codecs/libhevc/decoder/arm/ihevcd_function_selector_a9q.c) / [`common/arm/ihevc_function_selector_a9q.c`](file:///work/codecs/libhevc/common/arm/ihevc_function_selector_a9q.c)
       5. [`decoder/arm64/ihevcd_function_selector_av8.c`](file:///work/codecs/libhevc/decoder/arm64/ihevcd_function_selector_av8.c) / [`common/arm64/ihevc_function_selector_av8.c`](file:///work/codecs/libhevc/common/arm64/ihevc_function_selector_av8.c)
     * **Update test/benchmark getters:** When adding new SIMD or HBD kernels, verify that the corresponding function getters in `tests/common/ihevc_*_utils.cc` (e.g. `GetHbd*Fn(arch, ...)`) are updated so unit tests and benchmarks dispatch the new kernels.
     * **No NULL pointers:** If an architecture lacks an optimized SIMD kernel, verify it initializes the pointer to the generic C function (or to the baseline SIMD tier, e.g. SSSE3 for SSE4.2).
     * **No static helpers in selector:** Confirm that internal helper functions without SIMD versions are NOT exposed in `func_selector_t`; they must remain static or inline.
     * **Dead code elimination:** When removing or refactoring functions, confirm that unused selector members and initializers have been completely removed across all 5 files.

3. **Verify High Bit Depth (HBD) & Dynamic Bit Depth Safety:**
   * **Bit depth dynamism:** Bit depth is per-sequence (`i4_bit_depth_luma`, `i4_bit_depth_chroma`) with pixel size abstraction (`pixel_size_y`, `pixel_size_uv` = 1 byte for 8-bit, 2 bytes for 10-bit).
   * **Pointer arithmetic & strides:**
     * Identify whether strides (`src_strd`, `dst_strd`) are in **bytes** or in **pixels**.
     * If stride is in bytes, advancing `UWORD16 *pu2_src` by `strd` jumps `2 * strd` bytes (a common bug). Verify proper casting: `(UWORD8 *)pu2_src + strd` or `pu2_src + (strd / sizeof(UWORD16))`.
     * If stride is in pixels, verify `pu2_src + strd`.
   * **Dynamic range clipping:**
     * 8-bit: `CLIP_U8(val)` or `CLIP3(0, 255, val)`.
     * 10-bit: `CLIP_U10(val)` or `CLIP3(0, (1 << bit_depth) - 1, val)`.
   * **Overflow prevention:** Ensure intermediate arithmetic, transform shift constants, and rounding offsets `(1 << (shift - 1))` do not overflow 16-bit or 32-bit registers with 10-bit inputs.

4. **Verify Memory Bounds & SIMD Safety:**
   * **Spatial neighbors & padding:** Routines accessing spatial neighbors (intra prediction, deblocking, SAO) must not access negative offsets beyond allocated padding margins (`PAD_LEFT`, `PAD_TOP`, `PAD_RIGHT`, `PAD_BOT`).
   * **Block boundaries:** SIMD operations (128-bit SSE/NEON, 256-bit AVX2) must not over-read or over-write past the valid block width and height, especially for small blocks (4x4, 4x8) or tile/slice edges.
   * **Alignment:** Do not assume 16-byte or 32-byte alignment unless explicitly guaranteed. Verify that unaligned load/store intrinsics (`_mm_loadu_si128`, `vld1q_u8`, etc.) are used for unaligned buffers.

5. **Verify Regression, Unit Test & Benchmark Coverage:**
   * Verify all regression tests pass without changing MD5 checksums.
   * If MD5 checksums change, confirm that an explicit technical justification (e.g. standard conformance bugfix) is documented in the commit, and golden references are updated in the same commit.
   * Confirm that every new or modified SIMD primitive has an accompanying unit test (`tests/common/ihevc_*_test.cc`) and benchmark (`tests/common/ihevc_*_benchmark.cc` or `ihevc_hbd_*_benchmark.cc`) comparing against the generic C reference (`ARCH_NA`).
   * **Shared test utility hygiene:** Ensure shared headers in `tests/common/` (`TestCommon.h`, `func_selector.h`, `ihevc_*_utils.h`) do not unconditionally include `<gtest/gtest.h>` (guard GTest-specific helpers with `#if __has_include(<gtest/gtest.h>)`), so benchmark targets can link `libhevc_dsp_test_utils` without a GTest dependency.
   * **Build system sync:** When adding new unit tests, benchmarks, or shared test utility files, verify that both [`tests/common/common.cmake`](file:///work/codecs/libhevc/tests/common/common.cmake) and [`tests/common/Android.bp`](file:///work/codecs/libhevc/tests/common/Android.bp) are updated in sync.

6. **Verify Git & Commit Hygiene:**
   * Commit subject uses conventional prefix: `[decoder]`, `[encoder]`, `[common]`, `[DecTests]`, `[EncTests]`, `tests:`, `docs:`.
   * Clean diff: No debugging prints (`printf`), commented-out dead code, or untracked scratch files.
   * Atomic commit: Refactoring and dead-code removal are separated from functional changes.
