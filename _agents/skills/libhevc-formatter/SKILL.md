---
name: libhevc-formatter
description: >-
  Use this skill to inspect, clean, and format code changes in the libhevc repository. Enforces Ittiam/libhevc coding standards, indentation, Hungarian-style type and pointer naming conventions, Doxygen comments, and removes trailing whitespace or residual debug code.
---

# Code Formatter & Hygiene Skill for libhevc

This skill defines the coding style, naming conventions, and file hygiene checks required for code in the `libhevc` repository.

---

## 1. Code Formatting Standards

* **Indentation:** 4 spaces per indent level. Do NOT use tabs (`\t`).
* **Brace Style:** Allman style — open `{` and close `}` braces belong on their own lines for function definitions, structs, and control blocks (`if`, `for`, `while`, `switch`):
  ```c
  if(ps_codec->u4_pic_wd > 0)
  {
      ps_codec->u4_pic_ht = u4_ht;
  }
  else
  {
      return IV_FAIL;
  }
  ```
* **Switch Statements:** Indent `case` labels by 4 spaces within `switch`, and indent case bodies by another 4 spaces:
  ```c
  switch(e_arch)
  {
      case ARCH_X86_SSSE3:
          ihevcd_init_function_ptr_ssse3(ps_func_selector);
          break;
      default:
          ihevcd_init_function_ptr_generic(ps_func_selector);
          break;
  }
  ```
* **Pointer Alignment:** Asterisk attaches to the variable name, not the type:
  ```c
  codec_t *ps_codec;
  UWORD8 *pu1_dst;
  void *pv_buf;
  ```

---

## 2. Naming Conventions

The codebase uses standard Ittiam Hungarian-style prefixes for types and variables:

### Primitive Types
* `WORD8` / `UWORD8`: Signed / unsigned 8-bit integer
* `WORD16` / `UWORD16`: Signed / unsigned 16-bit integer
* `WORD32` / `UWORD32`: Signed / unsigned 32-bit integer
* `CHAR`: 8-bit character

### Variable Prefixes
Prefix       | Meaning                     | Example
:----------- | :-------------------------- | :------------------
`ps_`        | Pointer to struct           | `ps_codec`, `ps_buf_mgr`
`pu1_`       | Pointer to `UWORD8`         | `pu1_src`, `pu1_dst`
`pu2_`       | Pointer to `UWORD16`        | `pu2_src_10b`, `pu2_dst`
`pi4_`       | Pointer to `WORD32`         | `pi4_coeff`
`pv_`        | Pointer to `void`           | `pv_codec`, `pv_buf`
`u4_`        | `UWORD32` scalar            | `u4_width`, `u4_height`
`i4_`        | `WORD32` scalar             | `i4_bit_depth_luma`
`u2_` / `i2_`| `UWORD16` / `WORD16` scalar | `u2_mv_x`, `i2_mv_y`
`u1_` / `i1_`| `UWORD8` / `WORD8` scalar   | `u1_is_intra`
`s_`         | Struct instance (non-ptr)   | `s_func_selector`
`e_`         | Enum variable               | `e_processor_arch`
`au4_`       | Array of `UWORD32`          | `au4_status[BUF_MGR_MAX_CNT]`
`api4_`      | Array of pointers to `WORD32`| `api4_coeff_addr[MAX_CTB]`

---

## 3. Headers and Comment Blocks

### Doxygen Function Headers
All exported and key internal functions must include standard Doxygen comment blocks:
```c
/**
*******************************************************************************
* @brief
*   Brief one-line summary of function purpose.
*
* @par Description:
*   Detailed description of algorithm, constraints, or caveats.
*
* @param[in] ps_codec
*   Pointer to codec context.
* @param[out] pu1_dst
*   Destination frame buffer.
*
* @returns
*   IV_SUCCESS on success, error code otherwise.
*******************************************************************************
*/
```

### Include Header Hierarchy
Maintain standard include order:
1. Standard C library headers (`<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<assert.h>`)
2. Core framework types and interfaces (`ihevc_typedefs.h`, `iv.h`, `ivd.h`)
3. Core shared headers (`ihevc_defs.h`, `ihevc_macros.h`, `ihevc_structs.h`)
4. Module headers (`ihevcd_defs.h`, `ihevcd_structs.h`, `ihevcd_function_selector.h`)

---

## 4. Pre-Commit Hygiene & Cleanup Checks

Run these commands before committing changes to ensure no residual artifacts:

```bash
# 1. Check for trailing whitespace and whitespace errors
git diff --check

# 2. Check for leftover debug printf statements in modified files
git diff -U0 HEAD | grep -E "^\+[ ]*printf"

# 3. Check for leftover commented-out code blocks or temporary #if 0
git diff -U0 HEAD | grep -E "^\+[ ]*(/\*|//|#if 0)"

# 4. Check for untracked scratch files
git status --porcelain
```
