/******************************************************************************
*
* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
******************************************************************************/


/**
*******************************************************************************
* @file
*  ihevc_hbd_resi_trans_x86_intr.c
*
* @brief
*  Contains function definitions for residual and  forward transform
*  coded in x86 intrinsics
*
*
* @author
*
*
* @par List of Functions:
*  - ihevc_resi_trans_4x4_ttype1()
*  - ihevc_resi_trans_4x4()
*  - ihevc_resi_trans_8x8()
*  - ihevc_resi_trans_16x16()
*  - ihevc_resi_trans_32x32()
*
* @remarks
*  None
*
*******************************************************************************
*/


/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_resi_trans.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"
#include "ihevc_debug.h"
#include <immintrin.h>

static const WORD16 g_ai2_ihevc_hbd_trans_8_odd[8][8] =
{
    { 89,  75,  89,  75,  89,  75,  89,  75 },
    { 50,  18,  50,  18,  50,  18,  50,  18 },
    { 75, -18,  75, -18,  75, -18,  75, -18 },
    {-89, -50, -89, -50, -89, -50, -89, -50 },
    { 50, -89,  50, -89,  50, -89,  50, -89 },
    { 18,  75,  18,  75,  18,  75,  18,  75 },
    { 18, -50,  18, -50,  18, -50,  18, -50 },
    { 75, -89,  75, -89,  75, -89,  75, -89 }
};

static const WORD16 g_ai2_ihevc_hbd_trans_16_odd[32][8] =
{
    { 90,  87,  90,  87,  90,  87,  90,  87 },
    { 80,  70,  80,  70,  80,  70,  80,  70 },
    { 57,  43,  57,  43,  57,  43,  57,  43 },
    { 25,   9,  25,   9,  25,   9,  25,   9 },

    { 87,  57,  87,  57,  87,  57,  87,  57 },
    { 9,  -43,   9, -43,  9,  -43,   9, -43 },
    {-80, -90, -80, -90, -80, -90, -80, -90 },
    {-70, -25, -70, -25, -70, -25, -70, -25 },

    { 80,   9,  80,   9,  80,   9,  80,   9 },
    {-70, -87, -70, -87, -70, -87, -70, -87 },
    {-25,  57, -25,  57, -25,  57, -25,  57 },
    { 90,  43,  90,  43,  90,  43,  90,  43 },

    { 70, -43,  70, -43,  70, -43,  70, -43 },
    {-87,   9, -87,   9, -87,   9, -87,   9 },
    { 90,  25,  90,  25,  90,  25,  90,  25 },
    {-80, -57, -80, -57, -80, -57, -80, -57 },

    { 57, -80,  57, -80,  57, -80,  57, -80 },
    {-25,  90, -25,  90, -25,  90, -25,  90 },
    { -9, -87,  -9, -87,  -9, -87,  -9, -87 },
    { 43,  70,  43,  70,  43,  70,  43,  70 },

    { 43, -90,  43, -90,  43, -90,  43, -90 },
    { 57,  25,  57,  25,  57,  25,  57,  25 },
    {-87,  70, -87,  70, -87,  70, -87,  70 },
    {  9, -80,   9, -80,   9, -80,   9, -80 },

    { 25, -70,  25, -70,  25, -70,  25, -70 },
    { 90, -80,  90, -80,  90, -80,  90, -80 },
    { 43,   9,  43,   9,  43,   9,  43,   9 },
    {-57,  87, -57,  87, -57,  87, -57,  87 },

    {  9, -25,   9, -25,   9, -25,   9, -25 },
    { 43, -57,  43, -57,  43, -57,  43, -57 },
    { 70, -80,  70, -80,  70, -80,  70, -80 },
    { 87, -90,  87, -90,  87, -90,  87, -90 }
};

static const WORD32 g_ai4_ihevc_hbd_trans_32_even[2][4] =
{
    { 83,  36,  83,  36 },
    { 36,  83,  36,  83 }
};
/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform type 1
 * on input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 4x4 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_4x4_ttype1_sse42(UWORD16 *pu2_src,
                                        UWORD16 *pu2_pred,
                                        WORD32 *pi4_temp,
                                        WORD16 *pi2_dst,
                                        WORD32 i4_src_strd,
                                        WORD32 i4_pred_strd,
                                        WORD32 i4_dst_strd_chr_flag,
                                        UWORD8 u1_bit_depth)
{
    WORD32 add, shift;
    WORD32 trans_size;
    UWORD32  u4_blk_sad = 0;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    __m128i src1_8x16b, src2_8x16b, src3_8x16b, src4_8x16b;
    __m128i prd1_8x16b, prd2_8x16b, prd3_8x16b, prd4_8x16b;
    __m128i tmp1_8x16b, tmp2_8x16b, tmp3_8x16b, tmp4_8x16b;
    __m128i tmp5_8x16b, tmp6_8x16b, tmp7_8x16b;
    __m128i c0_4x32b, c1_4x32b, c2_4x32b, c3_4x32b;
    __m128i const74_4x32b, const29_4x32b, const55_4x32b;
    __m128i add_4x32b, sad_8x16b;
    __m128i dst1_8x16b, dst3_8x16b, dst2_8x16b, dst4_8x16b;

    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    trans_size = TRANS_SIZE_4;
    log2_trans_size = 2;

    /* Residue + Forward Transform 1st stage */
    shift = log2_trans_size - 1 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    add_4x32b = _mm_set1_epi32 (add);
    const74_4x32b = _mm_set1_epi32 (74);
    const29_4x32b = _mm_set1_epi32 (29);
    const55_4x32b = _mm_set1_epi32 (55);

    {
        /* Load 4 source rows: R0, R1, R2, R3*/
        src1_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src));
        src2_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + i4_src_strd));
        src3_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
        src4_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

        /* Load 4 pred rows: P0, P1, P2, P3 */
        prd1_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred ));
        prd2_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + i4_pred_strd));
        prd3_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
        prd4_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

        tmp1_8x16b = _mm_unpacklo_epi16 (src1_8x16b, src2_8x16b);
        tmp2_8x16b = _mm_unpacklo_epi16 (src3_8x16b, src4_8x16b);
        tmp3_8x16b = _mm_unpacklo_epi16 (prd1_8x16b, prd2_8x16b);
        tmp4_8x16b = _mm_unpacklo_epi16 (prd3_8x16b, prd4_8x16b);

        src1_8x16b = _mm_unpacklo_epi32 (tmp1_8x16b, tmp2_8x16b);
        prd1_8x16b = _mm_unpacklo_epi32 (tmp3_8x16b, tmp4_8x16b);
        src3_8x16b = _mm_unpackhi_epi32 (tmp1_8x16b, tmp2_8x16b);
        prd3_8x16b = _mm_unpackhi_epi32 (tmp3_8x16b, tmp4_8x16b);

        tmp1_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
        tmp3_8x16b = _mm_sub_epi16 (src3_8x16b, prd3_8x16b);

        /* SAD Computation */
        tmp2_8x16b  = _mm_abs_epi16 (tmp1_8x16b);
        tmp4_8x16b  = _mm_abs_epi16 (tmp3_8x16b);
        sad_8x16b   = _mm_add_epi16 (tmp2_8x16b, tmp4_8x16b);

        tmp2_8x16b = _mm_srli_si128 (tmp1_8x16b, 8);
        tmp4_8x16b = _mm_srli_si128 (tmp3_8x16b, 8);

        /* 16-32 bit conversion */
        tmp1_8x16b = _mm_cvtepi16_epi32 (tmp1_8x16b); /*    pu1_src[0] - pu1_pred[0]    */
        tmp2_8x16b = _mm_cvtepi16_epi32 (tmp2_8x16b); /*    pu1_src[1] - pu1_pred[1]    */
        tmp3_8x16b = _mm_cvtepi16_epi32 (tmp3_8x16b); /*    pu1_src[2] - pu1_pred[2]    */
        tmp4_8x16b = _mm_cvtepi16_epi32 (tmp4_8x16b); /*    pu1_src[3] - pu1_pred[3]    */

        tmp5_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
        c2_4x32b   = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
        c0_4x32b   = _mm_add_epi32 (tmp1_8x16b, tmp4_8x16b);
        c1_4x32b   = _mm_add_epi32 (tmp2_8x16b, tmp4_8x16b);

        tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp4_8x16b);

        c3_4x32b = _mm_mullo_epi32 (tmp3_8x16b, const74_4x32b);

        /*  for k = 0   */
        tmp1_8x16b = _mm_mullo_epi32 (const29_4x32b, c0_4x32b);
        tmp2_8x16b = _mm_mullo_epi32 (const55_4x32b, c1_4x32b);

        /*  for k = 1   */
        tmp3_8x16b = _mm_mullo_epi32 (const74_4x32b, tmp5_8x16b);

        /*  for k = 2   */
        tmp4_8x16b = _mm_mullo_epi32 (const29_4x32b, c2_4x32b);
        tmp5_8x16b = _mm_mullo_epi32 (const55_4x32b, c0_4x32b);

        /*  for k = 3   */
        tmp6_8x16b = _mm_mullo_epi32 (const55_4x32b, c2_4x32b);
        tmp7_8x16b = _mm_mullo_epi32 (const29_4x32b, c1_4x32b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
        tmp2_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp5_8x16b);
        tmp4_8x16b = _mm_sub_epi32 (tmp6_8x16b, tmp7_8x16b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, c3_4x32b);
        tmp2_8x16b = _mm_sub_epi32 (tmp2_8x16b, c3_4x32b);
        tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, c3_4x32b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
        tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, add_4x32b);
        tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
        tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);

        dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
        dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
        dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
        dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

        _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + trans_size), dst2_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst3_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst4_8x16b);

        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);

        u4_blk_sad += _mm_cvtsi128_si32 (sad_8x16b);
        u4_blk_sad = (u4_blk_sad & 0xFFFF);

    }

    {

        /* Forward transform 2nd stage */
        shift = log2_trans_size + 6; // log2(iHeight) + 6
        add = 1 << (shift - 1);
        add_4x32b = _mm_set1_epi32 (add);

        /* Load 4 source rows: R0, R1, R2, R3*/
        src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp));
        src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + trans_size));
        src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 2 * trans_size));
        src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 3 * trans_size));

        tmp1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
        tmp2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
        tmp3_8x16b = _mm_unpackhi_epi32 (src1_8x16b, src2_8x16b);
        tmp4_8x16b = _mm_unpackhi_epi32 (src3_8x16b, src4_8x16b);

        src1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
        src2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
        src3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
        src4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

        tmp5_8x16b = _mm_add_epi32 (src1_8x16b, src2_8x16b);
        c2_4x32b   = _mm_sub_epi32 (src1_8x16b, src2_8x16b);
        c0_4x32b   = _mm_add_epi32 (src1_8x16b, src4_8x16b);
        c1_4x32b   = _mm_add_epi32 (src2_8x16b, src4_8x16b);

        tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, src4_8x16b);
        c3_4x32b = _mm_mullo_epi32 (const74_4x32b, src3_8x16b);

        /*  for k = 0   */
        tmp1_8x16b = _mm_mullo_epi32 (const29_4x32b, c0_4x32b);
        tmp2_8x16b = _mm_mullo_epi32 (const55_4x32b, c1_4x32b);

        /*  for k = 1   */
        tmp3_8x16b = _mm_mullo_epi32 (const74_4x32b, tmp5_8x16b);

        /*  for k = 2   */
        tmp4_8x16b = _mm_mullo_epi32 (const29_4x32b, c2_4x32b);
        tmp5_8x16b = _mm_mullo_epi32 (const55_4x32b, c0_4x32b);

        /*  for k = 3   */
        tmp6_8x16b = _mm_mullo_epi32 (const55_4x32b, c2_4x32b);
        tmp7_8x16b = _mm_mullo_epi32 (const29_4x32b, c1_4x32b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
        tmp2_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp5_8x16b);
        tmp4_8x16b = _mm_sub_epi32 (tmp6_8x16b, tmp7_8x16b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, c3_4x32b);
        tmp2_8x16b = _mm_sub_epi32 (tmp2_8x16b, c3_4x32b);
        tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, c3_4x32b);

        tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
        tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, add_4x32b);
        tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
        tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);

        dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
        dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
        dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
        dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

        dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
        dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);
        dst3_8x16b = _mm_packs_epi32 (dst3_8x16b, dst3_8x16b);
        dst4_8x16b = _mm_packs_epi32 (dst4_8x16b, dst4_8x16b);

        _mm_storel_epi64 ((__m128i *)(pi2_dst), dst1_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 1 * i4_dst_strd), dst2_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 2 * i4_dst_strd), dst3_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 3 * i4_dst_strd), dst4_8x16b);

    }

    return u4_blk_sad;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform on
 * input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 4x4 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_4x4_sse42(UWORD16 *pu2_src,
                          UWORD16 *pu2_pred,
                          WORD32 *pi4_temp,
                          WORD16 *pi2_dst,
                          WORD32 i4_src_strd,
                          WORD32 i4_pred_strd,
                          WORD32 i4_dst_strd_chr_flag,
                          UWORD8 u1_bit_depth)
{
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad=0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    __m128i src1_8x16b, src2_8x16b, src3_8x16b, src4_8x16b;
    __m128i prd1_8x16b, prd2_8x16b, prd3_8x16b, prd4_8x16b;
    __m128i tmp1_8x16b, tmp2_8x16b, tmp3_8x16b, tmp4_8x16b;
    __m128i tmp5_8x16b, tmp6_8x16b, tmp7_8x16b, tmp8_8x16b;
    __m128i mask2_8x16b;
    __m128i e1_8x16b;
    __m128i o1_8x16b;
    __m128i coeff1_8x16b, coeff2_8x16b, coeff3_8x16b, coeff4_8x16b;
    __m128i add_4x32b, chroma_mask_8x16b, sad_8x16b;
    __m128i dst1_8x16b, dst3_8x16b, dst2_8x16b, dst4_8x16b;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_4;
    log2_trans_size = 2;

    chroma_mask_8x16b = _mm_set_epi32 (0x0, 0x0, 0x0D0C0908, 0x05040100);
    mask2_8x16b = _mm_set_epi32 (0x0D0C0F0E, 0x09080B0A, 0x05040706, 0x01000302);

    if (0 == chroma_flag)
    {
        /* Load 4 source rows: R0, R1, R2, R3*/
        src1_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src));
        src2_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + i4_src_strd));
        src3_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
        src4_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

        /* Load 4 pred rows: P0, P1, P2, P3 */
        prd1_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred ));
        prd2_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + i4_pred_strd));
        prd3_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
        prd4_8x16b = _mm_loadl_epi64 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

        src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
        src2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
        prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd2_8x16b);
        prd2_8x16b = _mm_unpacklo_epi32 (prd3_8x16b, prd4_8x16b);

        tmp1_8x16b = _mm_unpacklo_epi64 (src1_8x16b, src2_8x16b);
        tmp2_8x16b = _mm_unpacklo_epi64 (prd1_8x16b, prd2_8x16b);
        tmp3_8x16b = _mm_unpackhi_epi64 (src1_8x16b, src2_8x16b);
        tmp4_8x16b = _mm_unpackhi_epi64 (prd1_8x16b, prd2_8x16b);

        tmp1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp2_8x16b);
        tmp2_8x16b = _mm_sub_epi16 (tmp3_8x16b, tmp4_8x16b);

        /* SAD Computation */
        tmp3_8x16b = _mm_abs_epi16 (tmp1_8x16b);
        tmp4_8x16b = _mm_abs_epi16 (tmp2_8x16b);

        tmp2_8x16b = _mm_shuffle_epi8 (tmp2_8x16b, mask2_8x16b);

        e1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp2_8x16b);
        o1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp2_8x16b);

        /*  for k = 0, 2, 1, 3  */
        coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
        coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
        coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
        coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

        /* SAD Computation */
        sad_8x16b = _mm_add_epi16 (tmp3_8x16b, tmp4_8x16b);

        /*  for k = 0, 2, 1, 3  */
        dst1_8x16b = _mm_madd_epi16 (e1_8x16b, coeff1_8x16b);
        dst3_8x16b = _mm_madd_epi16 (e1_8x16b, coeff2_8x16b);
        dst2_8x16b = _mm_madd_epi16 (o1_8x16b, coeff3_8x16b);
        dst4_8x16b = _mm_madd_epi16 (o1_8x16b, coeff4_8x16b);

        _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + trans_size), dst2_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst3_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst4_8x16b);

        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi16 (sad_8x16b, sad_8x16b);

        u4_blk_sad += _mm_cvtsi128_si32 (sad_8x16b);
        u4_blk_sad = (u4_blk_sad & 0xFFFF);

    }
    else
    {
        /* Load 4 source rows: R0, R1, R2, R3*/
        src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src));
        src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + i4_src_strd));
        src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
        src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

        /* Load 4 pred rows: P0, P1, P2, P3 */
        prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred ));
        prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + i4_pred_strd));
        prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
        prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

        src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
        src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
        src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
        src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

        prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
        prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
        prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
        prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

        src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
        src2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
        prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd2_8x16b);
        prd2_8x16b = _mm_unpacklo_epi32 (prd3_8x16b, prd4_8x16b);

        tmp1_8x16b = _mm_unpacklo_epi64 (src1_8x16b, src2_8x16b);
        tmp2_8x16b = _mm_unpacklo_epi64 (prd1_8x16b, prd2_8x16b);
        tmp3_8x16b = _mm_unpackhi_epi64 (src1_8x16b, src2_8x16b);
        tmp4_8x16b = _mm_unpackhi_epi64 (prd1_8x16b, prd2_8x16b);

        tmp1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp2_8x16b);
        tmp2_8x16b = _mm_sub_epi16 (tmp3_8x16b, tmp4_8x16b);

        tmp2_8x16b = _mm_shuffle_epi8 (tmp2_8x16b, mask2_8x16b);

        e1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp2_8x16b);
        o1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp2_8x16b);

        /*  for k = 0, 2, 1, 3  */
        coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
        coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
        coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
        coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

        dst1_8x16b = _mm_madd_epi16 (e1_8x16b, coeff1_8x16b);
        dst3_8x16b = _mm_madd_epi16 (e1_8x16b, coeff2_8x16b);
        dst2_8x16b = _mm_madd_epi16 (o1_8x16b, coeff3_8x16b);
        dst4_8x16b = _mm_madd_epi16 (o1_8x16b, coeff4_8x16b);

        _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + trans_size), dst2_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst3_8x16b);
        _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst4_8x16b);

    }

    {
        __m128i e1_4x32b, e2_4x32b;
        __m128i o1_4x32b, o2_4x32b;
        __m128i coeff3_4x32b, coeff4_4x32b;

        pi4_temp = pi4_tmp_orig;
        /* Forward Transform 2nd stage */
        shift = 9 + (u1_bit_depth - 8); // log2(iHeight) + 6
        add = 1 << (shift - 1);
        add_4x32b = _mm_set1_epi32 (add);

        /* Load 4 source rows: R0, R1, R2, R3*/
        src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp));
        src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + trans_size));
        src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 2 * trans_size));
        src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 3 * trans_size));

        tmp1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
        tmp2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
        tmp3_8x16b = _mm_unpackhi_epi32 (src1_8x16b, src2_8x16b);
        tmp4_8x16b = _mm_unpackhi_epi32 (src3_8x16b, src4_8x16b);

        src1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
        src2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
        src3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
        src4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

        /*  e[k] = pi4_temp[k] + pi4_temp[3 - k];   */
        e1_4x32b = _mm_add_epi32 (src1_8x16b, src4_8x16b);
        e2_4x32b = _mm_add_epi32 (src2_8x16b, src3_8x16b);

        /*  o[k] = pi4_temp[k] - pi4_temp[3 - k];   */
        o1_4x32b = _mm_sub_epi32 (src1_8x16b, src4_8x16b);
        o2_4x32b = _mm_sub_epi32 (src2_8x16b, src3_8x16b);

        /*  k = 1   */
        coeff3_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_4_intr[1]);
        coeff4_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_4_intr[2]);

        tmp1_8x16b = _mm_slli_epi32 (e1_4x32b, 6);
        tmp2_8x16b = _mm_slli_epi32 (e2_4x32b, 6);
        tmp5_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff3_4x32b);
        tmp6_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff4_4x32b);
        tmp7_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff4_4x32b);
        tmp8_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff3_4x32b);

        tmp3_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
        tmp4_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
        tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
        tmp6_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp8_8x16b);

        tmp1_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
        tmp2_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);
        tmp3_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
        tmp4_8x16b = _mm_add_epi32 (tmp6_8x16b, add_4x32b);

        dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
        dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
        dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
        dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

        dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
        dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);
        dst3_8x16b = _mm_packs_epi32 (dst3_8x16b, dst3_8x16b);
        dst4_8x16b = _mm_packs_epi32 (dst4_8x16b, dst4_8x16b);

        _mm_storel_epi64 ((__m128i *)(pi2_dst), dst1_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 1 * i4_dst_strd), dst2_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 2 * i4_dst_strd), dst3_8x16b);
        _mm_storel_epi64 ((__m128i *)(pi2_dst + 3 * i4_dst_strd), dst4_8x16b);

    }

    return u4_blk_sad;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform on
 * input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 8x8 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 8x8
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

UWORD32 ihevc_hbd_resi_trans_8x8_sse42(UWORD16 *pu2_src,
                          UWORD16 *pu2_pred,
                          WORD32 *pi4_temp,
                          WORD16 *pi2_dst,
                          WORD32 i4_src_strd,
                          WORD32 i4_pred_strd,
                          WORD32 i4_dst_strd_chr_flag,
                          UWORD8 u1_bit_depth)
{
    WORD32 i;
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad=0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;

    __m128i src1_8x16b, src2_8x16b, src3_8x16b, src4_8x16b;
    __m128i prd1_8x16b, prd2_8x16b, prd3_8x16b, prd4_8x16b;
    __m128i tmp1_8x16b, tmp2_8x16b, tmp3_8x16b, tmp4_8x16b;
    __m128i tmp5_8x16b, tmp6_8x16b, tmp7_8x16b, tmp8_8x16b;
    __m128i mask2_8x16b;
    __m128i e1_8x16b, e2_8x16b;
    __m128i o1_8x16b, o2_8x16b;
    __m128i ee12_8x16b, eo12_8x16b;
    __m128i coeff1_8x16b, coeff2_8x16b, coeff3_8x16b, coeff4_8x16b;
    __m128i coeff5_8x16b, coeff6_8x16b, coeff7_8x16b, coeff8_8x16b;
    __m128i add_4x32b, chroma_mask_8x16b, sad_8x16b;
    __m128i dst1_8x16b, dst3_8x16b, dst2_8x16b, dst4_8x16b;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_8;

    sad_8x16b = _mm_setzero_si128 ();
    chroma_mask_8x16b = _mm_set_epi32 (0x0, 0x0, 0x0D0C0908, 0x05040100);
    mask2_8x16b = _mm_set_epi32 (0x0D0C0F0E, 0x09080B0A, 0x05040706, 0x01000302);

    if (0 == chroma_flag)
    {
        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred ));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

            tmp1_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp2_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);
            tmp3_8x16b = _mm_sub_epi16 (src3_8x16b, prd3_8x16b);
            tmp4_8x16b = _mm_sub_epi16 (src4_8x16b, prd4_8x16b);

            tmp5_8x16b = _mm_unpacklo_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp6_8x16b = _mm_unpacklo_epi32 (tmp2_8x16b, tmp4_8x16b);
            tmp7_8x16b = _mm_unpackhi_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp8_8x16b = _mm_unpackhi_epi32 (tmp2_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_unpacklo_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp2_8x16b = _mm_unpackhi_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_unpacklo_epi32 (tmp7_8x16b, tmp8_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (tmp7_8x16b, tmp8_8x16b);

            /* SAD Computation */
            tmp5_8x16b = _mm_abs_epi16 (tmp5_8x16b);
            tmp6_8x16b = _mm_abs_epi16 (tmp6_8x16b);
            tmp7_8x16b = _mm_abs_epi16 (tmp7_8x16b);
            tmp8_8x16b = _mm_abs_epi16 (tmp8_8x16b);

            tmp3_8x16b = _mm_shuffle_epi8 (tmp3_8x16b, mask2_8x16b);
            tmp4_8x16b = _mm_shuffle_epi8 (tmp4_8x16b, mask2_8x16b);

            e1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp4_8x16b);
            e2_8x16b = _mm_add_epi16 (tmp2_8x16b, tmp3_8x16b);

            o1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp4_8x16b);
            o2_8x16b = _mm_sub_epi16 (tmp2_8x16b, tmp3_8x16b);

            e2_8x16b = _mm_shuffle_epi8 (e2_8x16b, mask2_8x16b);

            /*  for k = 0, 2, 4, 6  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

            ee12_8x16b = _mm_add_epi16 (e1_8x16b, e2_8x16b);
            eo12_8x16b = _mm_sub_epi16 (e1_8x16b, e2_8x16b);

            /* SAD Computation */
            tmp5_8x16b = _mm_add_epi16 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi16 (tmp7_8x16b, tmp8_8x16b);
            tmp5_8x16b = _mm_add_epi16 (tmp5_8x16b, tmp7_8x16b);
            sad_8x16b  = _mm_add_epi16 (sad_8x16b, tmp5_8x16b);

            /*  for k = 0, 2, 4, 6  */
            dst1_8x16b = _mm_madd_epi16 (ee12_8x16b, coeff1_8x16b);
            dst3_8x16b = _mm_madd_epi16 (ee12_8x16b, coeff2_8x16b);
            dst2_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff3_8x16b);
            dst4_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff4_8x16b);

            _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 4 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 6 * trans_size), dst4_8x16b);

            /*  for k = 1, 3, 5, 7  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o1_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o2_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o1_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o2_8x16b, coeff8_8x16b);

            dst1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            dst2_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            dst3_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            dst4_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 1 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 5 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 7 * trans_size), dst4_8x16b);

            pu2_src  += 4 * i4_src_strd;
            pu2_pred += 4 * i4_pred_strd;
            pi4_temp += 4;
        }

        tmp1_8x16b = _mm_cvtepi16_epi32 (sad_8x16b);
        tmp2_8x16b = _mm_srli_si128(sad_8x16b, 8);
        tmp2_8x16b = _mm_cvtepi16_epi32 (tmp2_8x16b);

        sad_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
        sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);

        u4_blk_sad = _mm_cvtsi128_si32 (sad_8x16b);
    }
    else
    {
        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred ));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp1_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp2_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp3_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp4_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            tmp5_8x16b = _mm_unpacklo_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp6_8x16b = _mm_unpackhi_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp7_8x16b = _mm_unpacklo_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp8_8x16b = _mm_unpackhi_epi32 (tmp3_8x16b, tmp4_8x16b);

            tmp7_8x16b = _mm_shuffle_epi8 (tmp7_8x16b, mask2_8x16b);
            tmp8_8x16b = _mm_shuffle_epi8 (tmp8_8x16b, mask2_8x16b);

            e1_8x16b = _mm_add_epi16 (tmp5_8x16b, tmp8_8x16b);
            e2_8x16b = _mm_add_epi16 (tmp6_8x16b, tmp7_8x16b);

            o1_8x16b = _mm_sub_epi16 (tmp5_8x16b, tmp8_8x16b);
            o2_8x16b = _mm_sub_epi16 (tmp6_8x16b, tmp7_8x16b);

            e2_8x16b = _mm_shuffle_epi8 (e2_8x16b, mask2_8x16b);

            /*  for k = 0, 2, 4, 6  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

            ee12_8x16b = _mm_add_epi16 (e1_8x16b, e2_8x16b);
            eo12_8x16b = _mm_sub_epi16 (e1_8x16b, e2_8x16b);

            dst1_8x16b = _mm_madd_epi16 (ee12_8x16b, coeff1_8x16b);
            dst3_8x16b = _mm_madd_epi16 (ee12_8x16b, coeff2_8x16b);
            dst2_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff3_8x16b);
            dst4_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff4_8x16b);

            _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 4 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 6 * trans_size), dst4_8x16b);

            /*  for k = 1, 3, 5, 7  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o1_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o2_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o1_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o2_8x16b, coeff8_8x16b);

            dst1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            dst2_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            dst3_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            dst4_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 1 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 5 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 7 * trans_size), dst4_8x16b);

            pu2_src  += 4 * i4_src_strd;
            pu2_pred += 4 * i4_pred_strd;
            pi4_temp += 4;
        }
    }

    {
        __m128i e1_4x32b, e2_4x32b, e3_4x32b, e4_4x32b;
        __m128i o1_4x32b, o2_4x32b, o3_4x32b, o4_4x32b;
        __m128i ee1_4x32b, ee2_4x32b;
        __m128i eo1_4x32b, eo2_4x32b;
        __m128i coeff1_4x32b, coeff2_4x32b, coeff3_4x32b, coeff4_4x32b;

        /* Forward Transform 2nd stage */
        pi4_temp = pi4_tmp_orig;
        shift = 11 + (u1_bit_depth - 8); // log2(iHeight) + 6
        add = 1 << (shift - 1);
        add_4x32b = _mm_set1_epi32 (add);

        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + trans_size));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 2 * trans_size));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 3 * trans_size));

            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + trans_size));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + 2 * trans_size));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + 3 * trans_size));

            tmp1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (src1_8x16b, src2_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (src3_8x16b, src4_8x16b);

            src1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            src2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            src3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            src4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd2_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (prd3_8x16b, prd4_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (prd1_8x16b, prd2_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (prd3_8x16b, prd4_8x16b);

            prd1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            prd4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            /*  e[k] = pi4_temp[k] + pi4_temp[7 - k];   */
            e1_4x32b = _mm_add_epi32 (src1_8x16b, prd4_8x16b);
            e2_4x32b = _mm_add_epi32 (src2_8x16b, prd3_8x16b);
            e3_4x32b = _mm_add_epi32 (src3_8x16b, prd2_8x16b);
            e4_4x32b = _mm_add_epi32 (src4_8x16b, prd1_8x16b);

            /*  o[k] = pi4_temp[k] - pi4_temp[7 - k];   */
            o1_4x32b = _mm_sub_epi32 (src1_8x16b, prd4_8x16b);
            o2_4x32b = _mm_sub_epi32 (src2_8x16b, prd3_8x16b);
            o3_4x32b = _mm_sub_epi32 (src3_8x16b, prd2_8x16b);
            o4_4x32b = _mm_sub_epi32 (src4_8x16b, prd1_8x16b);

            /*  ee[k] = e[k] + e[3 - k] */
            ee1_4x32b = _mm_add_epi32 (e1_4x32b, e4_4x32b);
            ee2_4x32b = _mm_add_epi32 (e2_4x32b, e3_4x32b);

            /*  eo[k] = e[k] - e[3 - k] */
            eo1_4x32b = _mm_sub_epi32 (e1_4x32b, e4_4x32b);
            eo2_4x32b = _mm_sub_epi32 (e2_4x32b, e3_4x32b);


            /*  k = 2   */
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_8_intr[1]);
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_8_intr[2]);

            /*  k = 0, 4    */
            tmp1_8x16b = _mm_slli_epi32 (ee1_4x32b, 6);
            tmp2_8x16b = _mm_slli_epi32 (ee2_4x32b, 6);

            /*  k = 2, 6    */
            tmp5_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff3_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff4_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff4_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff3_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, add_4x32b);

            tmp3_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp6_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp8_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp6_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);
            dst3_8x16b = _mm_packs_epi32 (dst3_8x16b, dst3_8x16b);
            dst4_8x16b = _mm_packs_epi32 (dst4_8x16b, dst4_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 2 * i4_dst_strd), dst2_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 4 * i4_dst_strd), dst3_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 6 * i4_dst_strd), dst4_8x16b);

            /*  k = 1, 3, 5, 7  */
            coeff1_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_8_intr[5]));
            coeff2_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_8_intr[3]));
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_8_intr[6]));
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_8_intr[4]));

            /*  k = 1, 3    */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff1_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff2_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff3_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff4_4x32b);

            tmp5_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff2_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff4_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff1_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff3_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 1 * i4_dst_strd), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 3 * i4_dst_strd), dst2_8x16b);

            /*  k = 5, 7    */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff3_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff1_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff4_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff2_4x32b);

            tmp5_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff4_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff3_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff2_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff1_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp7_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 5 * i4_dst_strd), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 7 * i4_dst_strd), dst2_8x16b);

            pi4_temp  +=  4 * trans_size;
            pi2_dst += 4;
        }
    }

    return u4_blk_sad;
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform on
 * input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 16x16 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 16x16
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_16x16_sse42(UWORD16 *pu2_src,
                            UWORD16 *pu2_pred,
                            WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 i4_src_strd,
                            WORD32 i4_pred_strd,
                            WORD32 i4_dst_strd_chr_flag,
                            UWORD8 u1_bit_depth)
{
    WORD32 i;
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad = 0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    __m128i src1_8x16b, src2_8x16b, src3_8x16b, src4_8x16b;
    __m128i prd1_8x16b, prd2_8x16b, prd3_8x16b, prd4_8x16b;
    __m128i tmp1_8x16b, tmp2_8x16b, tmp3_8x16b, tmp4_8x16b;
    __m128i tmp5_8x16b, tmp6_8x16b, tmp7_8x16b, tmp8_8x16b;
    __m128i mask1_8x16b, mask2_8x16b, mask3_8x16b;
    __m128i e1_8x16b, e2_8x16b, e3_8x16b, e4_8x16b;
    __m128i o1_8x16b, o2_8x16b, o3_8x16b, o4_8x16b;
    __m128i o_tmp1_8x16b, o_tmp2_8x16b, o_tmp3_8x16b, o_tmp4_8x16b;
    __m128i ee12_8x16b, ee34_8x16b, eo12_8x16b, eo34_8x16b;
    __m128i eee12_8x16b, eeo12_8x16b;
    __m128i coeff1_8x16b, coeff2_8x16b, coeff3_8x16b, coeff4_8x16b;
    __m128i coeff5_8x16b, coeff6_8x16b, coeff7_8x16b, coeff8_8x16b;
    __m128i add_4x32b, chroma_mask_8x16b, sad_8x16b;
    __m128i dst1_8x16b, dst3_8x16b, dst2_8x16b, dst4_8x16b;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_16;
    log2_trans_size = 4;
    /* Residue + Forward Transform 1st stage */
    shift = (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);


    add_4x32b = _mm_set1_epi32 (add);
    sad_8x16b = _mm_setzero_si128 ();
    chroma_mask_8x16b = _mm_set_epi32 (0x0, 0x0, 0x0D0C0908, 0x05040100);
    mask1_8x16b = _mm_set_epi32 (0x01000302, 0x05040706, 0x09080B0A, 0x0D0C0F0E);
    mask2_8x16b = _mm_set_epi32 (0x0D0C0F0E, 0x09080B0A, 0x05040706, 0x01000302);
    mask3_8x16b = _mm_set_epi32 (0x05040706, 0x01000302, 0x0D0C0F0E, 0x09080B0A);

    if (0 == chroma_flag)
    {
        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred ));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

            tmp1_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp2_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);
            tmp3_8x16b = _mm_sub_epi16 (src3_8x16b, prd3_8x16b);
            tmp4_8x16b = _mm_sub_epi16 (src4_8x16b, prd4_8x16b);

            /* Load 4 source rows: R4, R5, R6, R7*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 3 * i4_src_strd));

            /* Load 4 pred rows: P4, P5, P6, P7 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 3 * i4_pred_strd));

            tmp5_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp6_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);
            tmp7_8x16b = _mm_sub_epi16 (src3_8x16b, prd3_8x16b);
            tmp8_8x16b = _mm_sub_epi16 (src4_8x16b, prd4_8x16b);

            tmp5_8x16b = _mm_shuffle_epi8 (tmp5_8x16b, mask1_8x16b);
            tmp6_8x16b = _mm_shuffle_epi8 (tmp6_8x16b, mask1_8x16b);
            tmp7_8x16b = _mm_shuffle_epi8 (tmp7_8x16b, mask1_8x16b);
            tmp8_8x16b = _mm_shuffle_epi8 (tmp8_8x16b, mask1_8x16b);

            e1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp5_8x16b);
            e2_8x16b = _mm_add_epi16 (tmp2_8x16b, tmp6_8x16b);
            e3_8x16b = _mm_add_epi16 (tmp3_8x16b, tmp7_8x16b);
            e4_8x16b = _mm_add_epi16 (tmp4_8x16b, tmp8_8x16b);

            o1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp5_8x16b);
            o2_8x16b = _mm_sub_epi16 (tmp2_8x16b, tmp6_8x16b);
            o3_8x16b = _mm_sub_epi16 (tmp3_8x16b, tmp7_8x16b);
            o4_8x16b = _mm_sub_epi16 (tmp4_8x16b, tmp8_8x16b);

            /* SAD Computation */
            tmp1_8x16b = _mm_abs_epi16 (tmp1_8x16b);
            tmp2_8x16b = _mm_abs_epi16 (tmp2_8x16b);
            tmp3_8x16b = _mm_abs_epi16 (tmp3_8x16b);
            tmp4_8x16b = _mm_abs_epi16 (tmp4_8x16b);

            tmp5_8x16b = _mm_abs_epi16 (tmp5_8x16b);
            tmp6_8x16b = _mm_abs_epi16 (tmp6_8x16b);
            tmp7_8x16b = _mm_abs_epi16 (tmp7_8x16b);
            tmp8_8x16b = _mm_abs_epi16 (tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_add_epi16 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi16 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_add_epi16 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp2_8x16b);
            tmp2_8x16b = _mm_add_epi16 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp2_8x16b);

            tmp2_8x16b = _mm_cvtepi16_epi32 (tmp1_8x16b);
            tmp1_8x16b = _mm_srli_si128 (tmp1_8x16b, 8);
            tmp1_8x16b = _mm_cvtepi16_epi32 (tmp1_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            sad_8x16b  = _mm_add_epi32 (sad_8x16b, tmp1_8x16b);

            tmp5_8x16b = _mm_unpacklo_epi32 (e1_8x16b, e3_8x16b);
            tmp6_8x16b = _mm_unpacklo_epi32 (e2_8x16b, e4_8x16b);
            tmp7_8x16b = _mm_unpackhi_epi32 (e1_8x16b, e3_8x16b);
            tmp8_8x16b = _mm_unpackhi_epi32 (e2_8x16b, e4_8x16b);

            e1_8x16b = _mm_unpacklo_epi32 (tmp5_8x16b, tmp6_8x16b);
            e2_8x16b = _mm_unpackhi_epi32 (tmp5_8x16b, tmp6_8x16b);
            e3_8x16b = _mm_unpacklo_epi32 (tmp7_8x16b, tmp8_8x16b);
            e4_8x16b = _mm_unpackhi_epi32 (tmp7_8x16b, tmp8_8x16b);

            e3_8x16b = _mm_shuffle_epi8 (e3_8x16b, mask2_8x16b);
            e4_8x16b = _mm_shuffle_epi8 (e4_8x16b, mask2_8x16b);

            /*  for k = 0, 4, 8, 12 */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

            ee12_8x16b = _mm_add_epi16 (e1_8x16b, e4_8x16b);
            ee34_8x16b = _mm_add_epi16 (e2_8x16b, e3_8x16b);
            eo12_8x16b = _mm_sub_epi16 (e1_8x16b, e4_8x16b);
            eo34_8x16b = _mm_sub_epi16 (e2_8x16b, e3_8x16b);

            ee34_8x16b = _mm_shuffle_epi8 (ee34_8x16b, mask2_8x16b);

            eee12_8x16b = _mm_add_epi16 (ee12_8x16b, ee34_8x16b);
            eeo12_8x16b = _mm_sub_epi16 (ee12_8x16b, ee34_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (eee12_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (eee12_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (eeo12_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (eeo12_8x16b, coeff4_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, add_4x32b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 4 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 8 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 12 * trans_size), dst4_8x16b);

            /*  for k = 2, 6, 10, 14    */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp7_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 6 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 10 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 14 * trans_size), dst4_8x16b);

            tmp1_8x16b = _mm_unpacklo_epi32 (o1_8x16b, o3_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (o2_8x16b, o4_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (o1_8x16b, o3_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (o2_8x16b, o4_8x16b);

            o1_8x16b = _mm_unpacklo_epi32 (tmp1_8x16b, tmp2_8x16b);
            o2_8x16b = _mm_unpackhi_epi32 (tmp1_8x16b, tmp2_8x16b);
            o3_8x16b = _mm_unpacklo_epi32 (tmp3_8x16b, tmp4_8x16b);
            o4_8x16b = _mm_unpackhi_epi32 (tmp3_8x16b, tmp4_8x16b);

            /*  for k = 1, 3    */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 5, 7    */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[8]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[9]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[10]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[11]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[12]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[13]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[14]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[15]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 1 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 9, 11   */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[16]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[17]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[18]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[19]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[20]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[21]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[22]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[23]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 5 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 7 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 13, 15  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[24]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[25]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[26]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[27]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[28]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[29]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[30]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[31]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 9 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 11 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 13 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 15 * trans_size), dst2_8x16b);

            pu2_src  += 4 * i4_src_strd;
            pu2_pred += 4 * i4_pred_strd;
            pi4_temp += 4;
        }

        sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);
        sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);

        u4_blk_sad = _mm_cvtsi128_si32 (sad_8x16b);
    }

    else
    {
        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred ));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp1_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp2_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 8 + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 8 + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp3_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp4_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 16));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 16 + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 16 + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 16 + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 16));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 16 + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 16 + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 16 + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp5_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp6_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 24));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 24 + i4_src_strd));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 24 + 2 * i4_src_strd));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_src + 24 + 3 * i4_src_strd));

            /* Load 4 pred rows: P0, P1, P2, P3 */
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 24));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 24 + i4_pred_strd));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 24 + 2 * i4_pred_strd));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pu2_pred + 24 + 3 * i4_pred_strd));

            src1_8x16b = _mm_shuffle_epi8 (src1_8x16b, chroma_mask_8x16b);
            src2_8x16b = _mm_shuffle_epi8 (src2_8x16b, chroma_mask_8x16b);
            src3_8x16b = _mm_shuffle_epi8 (src3_8x16b, chroma_mask_8x16b);
            src4_8x16b = _mm_shuffle_epi8 (src4_8x16b, chroma_mask_8x16b);

            prd1_8x16b = _mm_shuffle_epi8 (prd1_8x16b, chroma_mask_8x16b);
            prd2_8x16b = _mm_shuffle_epi8 (prd2_8x16b, chroma_mask_8x16b);
            prd3_8x16b = _mm_shuffle_epi8 (prd3_8x16b, chroma_mask_8x16b);
            prd4_8x16b = _mm_shuffle_epi8 (prd4_8x16b, chroma_mask_8x16b);

            src1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src3_8x16b);
            src2_8x16b = _mm_unpacklo_epi32 (src2_8x16b, src4_8x16b);
            prd1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd3_8x16b);
            prd2_8x16b = _mm_unpacklo_epi32 (prd2_8x16b, prd4_8x16b);

            tmp7_8x16b = _mm_sub_epi16 (src1_8x16b, prd1_8x16b);
            tmp8_8x16b = _mm_sub_epi16 (src2_8x16b, prd2_8x16b);

            tmp5_8x16b = _mm_shuffle_epi8 (tmp5_8x16b, mask3_8x16b);
            tmp6_8x16b = _mm_shuffle_epi8 (tmp6_8x16b, mask3_8x16b);
            tmp7_8x16b = _mm_shuffle_epi8 (tmp7_8x16b, mask3_8x16b);
            tmp8_8x16b = _mm_shuffle_epi8 (tmp8_8x16b, mask3_8x16b);

            e1_8x16b = _mm_add_epi16 (tmp1_8x16b, tmp7_8x16b);
            e2_8x16b = _mm_add_epi16 (tmp2_8x16b, tmp8_8x16b);
            e3_8x16b = _mm_add_epi16 (tmp3_8x16b, tmp5_8x16b);
            e4_8x16b = _mm_add_epi16 (tmp4_8x16b, tmp6_8x16b);

            o_tmp1_8x16b = _mm_sub_epi16 (tmp1_8x16b, tmp7_8x16b);
            o_tmp2_8x16b = _mm_sub_epi16 (tmp2_8x16b, tmp8_8x16b);
            o_tmp3_8x16b = _mm_sub_epi16 (tmp3_8x16b, tmp5_8x16b);
            o_tmp4_8x16b = _mm_sub_epi16 (tmp4_8x16b, tmp6_8x16b);

            tmp5_8x16b = _mm_unpacklo_epi32 (e1_8x16b, e2_8x16b);
            tmp6_8x16b = _mm_unpackhi_epi32 (e1_8x16b, e2_8x16b);
            tmp7_8x16b = _mm_unpacklo_epi32 (e3_8x16b, e4_8x16b);
            tmp8_8x16b = _mm_unpackhi_epi32 (e3_8x16b, e4_8x16b);

            tmp7_8x16b = _mm_shuffle_epi8 (tmp7_8x16b, mask2_8x16b);
            tmp8_8x16b = _mm_shuffle_epi8 (tmp8_8x16b, mask2_8x16b);

            coeff1_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[0]);
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[4]);
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[2]);
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const *)g_ai2_ihevc_trans_8_intr[6]);

            ee12_8x16b = _mm_add_epi16 (tmp5_8x16b, tmp8_8x16b);
            ee34_8x16b = _mm_add_epi16 (tmp6_8x16b, tmp7_8x16b);
            eo12_8x16b = _mm_sub_epi16 (tmp5_8x16b, tmp8_8x16b);
            eo34_8x16b = _mm_sub_epi16 (tmp6_8x16b, tmp7_8x16b);

            ee34_8x16b = _mm_shuffle_epi8 (ee34_8x16b, mask2_8x16b);

            eee12_8x16b = _mm_add_epi16 (ee12_8x16b, ee34_8x16b);
            eeo12_8x16b = _mm_sub_epi16 (ee12_8x16b, ee34_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (eee12_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (eee12_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (eeo12_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (eeo12_8x16b, coeff4_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, add_4x32b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 4 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 8 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 12 * trans_size), dst4_8x16b);


            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_8_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (eo12_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (eo34_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp7_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 2 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 6 * trans_size), dst2_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 10 * trans_size), dst3_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 14 * trans_size), dst4_8x16b);

            o1_8x16b = _mm_unpacklo_epi32 (o_tmp1_8x16b, o_tmp2_8x16b);
            o3_8x16b = _mm_unpacklo_epi32 (o_tmp3_8x16b, o_tmp4_8x16b);
            o2_8x16b = _mm_unpackhi_epi32 (o_tmp1_8x16b, o_tmp2_8x16b);
            o4_8x16b = _mm_unpackhi_epi32 (o_tmp3_8x16b, o_tmp4_8x16b);

            /*  for k = 1, 3    */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[0]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[1]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[2]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[3]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[4]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[5]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[6]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[7]));

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 5, 7    */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[8]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[9]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[10]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[11]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[12]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[13]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[14]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[15]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 1 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 3 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 9, 11   */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[16]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[17]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[18]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[19]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[20]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[21]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[22]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[23]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 5 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 7 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            /*  for k = 13, 15  */
            coeff1_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[24]));
            coeff2_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[25]));
            coeff3_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[26]));
            coeff4_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[27]));
            coeff5_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[28]));
            coeff6_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[29]));
            coeff7_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[30]));
            coeff8_8x16b = _mm_loadu_si128 ((__m128i const*) (g_ai2_ihevc_hbd_trans_16_odd[31]));

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 9 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 11 * trans_size), dst2_8x16b);

            tmp1_8x16b = _mm_madd_epi16 (o1_8x16b, coeff1_8x16b);
            tmp2_8x16b = _mm_madd_epi16 (o2_8x16b, coeff2_8x16b);
            tmp3_8x16b = _mm_madd_epi16 (o3_8x16b, coeff3_8x16b);
            tmp4_8x16b = _mm_madd_epi16 (o4_8x16b, coeff4_8x16b);
            tmp5_8x16b = _mm_madd_epi16 (o1_8x16b, coeff5_8x16b);
            tmp6_8x16b = _mm_madd_epi16 (o2_8x16b, coeff6_8x16b);
            tmp7_8x16b = _mm_madd_epi16 (o3_8x16b, coeff7_8x16b);
            tmp8_8x16b = _mm_madd_epi16 (o4_8x16b, coeff8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            _mm_storeu_si128 ((__m128i *)(pi4_temp + 13 * trans_size), dst1_8x16b);
            _mm_storeu_si128 ((__m128i *)(pi4_temp + 15 * trans_size), dst2_8x16b);

            pu2_src  += 4 * i4_src_strd;
            pu2_pred += 4 * i4_pred_strd;
            pi4_temp += 4;
        }
    }

    {
        __m128i src5_8x16b, src6_8x16b, src7_8x16b, src8_8x16b;
        __m128i prd5_8x16b, prd6_8x16b, prd7_8x16b, prd8_8x16b;
        __m128i e1_4x32b, e2_4x32b, e3_4x32b, e4_4x32b;
        __m128i e5_4x32b, e6_4x32b, e7_4x32b, e8_4x32b;
        __m128i o1_4x32b, o2_4x32b, o3_4x32b, o4_4x32b;
        __m128i o5_4x32b, o6_4x32b, o7_4x32b, o8_4x32b;
        __m128i ee1_4x32b, ee2_4x32b, ee3_4x32b, ee4_4x32b;
        __m128i eo1_4x32b, eo2_4x32b, eo3_4x32b, eo4_4x32b;
        __m128i eee1_4x32b, eee2_4x32b, eeo1_4x32b, eeo2_4x32b;
        __m128i coeff1_4x32b, coeff2_4x32b, coeff3_4x32b, coeff4_4x32b;
        __m128i coeff5_4x32b, coeff6_4x32b, coeff7_4x32b, coeff8_4x32b;


        /* Forward Transform 2nd stage */
        shift = 13; //log2_trans_size - 1 + log2_trans_size + 6;
        add = 1 << (shift - 1);
        pi4_temp = pi4_tmp_orig;
        add_4x32b = _mm_set1_epi32 (add);

        for(i = 0; i < trans_size; i += 4)
        {
            /* Load 4 source rows: R0, R1, R2, R3*/
            src1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp));
            src2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + trans_size));
            src3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 2 * trans_size));
            src4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 3 * trans_size));

            src5_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4));
            src6_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + trans_size));
            src7_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + 2 * trans_size));
            src8_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 4 + 3 * trans_size));

            tmp1_8x16b = _mm_unpacklo_epi32 (src1_8x16b, src2_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (src3_8x16b, src4_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (src1_8x16b, src2_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (src3_8x16b, src4_8x16b);

            src1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            src2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            src3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            src4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_unpacklo_epi32 (src5_8x16b, src6_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (src7_8x16b, src8_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (src5_8x16b, src6_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (src7_8x16b, src8_8x16b);

            src5_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            src6_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            src7_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            src8_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            /* Load 4 source rows: R4, R5, R6, R7*/
            prd1_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 8));
            prd2_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 8 + trans_size));
            prd3_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 8 + 2 * trans_size));
            prd4_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 8 + 3 * trans_size));

            prd5_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 12));
            prd6_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 12 + trans_size));
            prd7_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 12 + 2 * trans_size));
            prd8_8x16b = _mm_loadu_si128 ((__m128i const*) (pi4_temp + 12 + 3 * trans_size));

            tmp1_8x16b = _mm_unpacklo_epi32 (prd1_8x16b, prd2_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (prd3_8x16b, prd4_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (prd1_8x16b, prd2_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (prd3_8x16b, prd4_8x16b);

            prd1_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd2_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd3_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            prd4_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_unpacklo_epi32 (prd5_8x16b, prd6_8x16b);
            tmp2_8x16b = _mm_unpacklo_epi32 (prd7_8x16b, prd8_8x16b);
            tmp3_8x16b = _mm_unpackhi_epi32 (prd5_8x16b, prd6_8x16b);
            tmp4_8x16b = _mm_unpackhi_epi32 (prd7_8x16b, prd8_8x16b);

            prd5_8x16b = _mm_unpacklo_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd6_8x16b = _mm_unpackhi_epi64 (tmp1_8x16b, tmp2_8x16b);
            prd7_8x16b = _mm_unpacklo_epi64 (tmp3_8x16b, tmp4_8x16b);
            prd8_8x16b = _mm_unpackhi_epi64 (tmp3_8x16b, tmp4_8x16b);

            /*  e[k] = pi4_temp[k] + pi4_temp[15 - k];  */
            e1_4x32b = _mm_add_epi32 (src1_8x16b, prd8_8x16b);
            e2_4x32b = _mm_add_epi32 (src2_8x16b, prd7_8x16b);
            e3_4x32b = _mm_add_epi32 (src3_8x16b, prd6_8x16b);
            e4_4x32b = _mm_add_epi32 (src4_8x16b, prd5_8x16b);

            e5_4x32b = _mm_add_epi32 (src5_8x16b, prd4_8x16b);
            e6_4x32b = _mm_add_epi32 (src6_8x16b, prd3_8x16b);
            e7_4x32b = _mm_add_epi32 (src7_8x16b, prd2_8x16b);
            e8_4x32b = _mm_add_epi32 (src8_8x16b, prd1_8x16b);

            /*  o[k] = pi4_temp[k] - pi4_temp[15 - k];  */
            o1_4x32b = _mm_sub_epi32 (src1_8x16b, prd8_8x16b);
            o2_4x32b = _mm_sub_epi32 (src2_8x16b, prd7_8x16b);
            o3_4x32b = _mm_sub_epi32 (src3_8x16b, prd6_8x16b);
            o4_4x32b = _mm_sub_epi32 (src4_8x16b, prd5_8x16b);

            o5_4x32b = _mm_sub_epi32 (src5_8x16b, prd4_8x16b);
            o6_4x32b = _mm_sub_epi32 (src6_8x16b, prd3_8x16b);
            o7_4x32b = _mm_sub_epi32 (src7_8x16b, prd2_8x16b);
            o8_4x32b = _mm_sub_epi32 (src8_8x16b, prd1_8x16b);

            /*  ee[k] = e[k] + e[7 - k] */
            ee1_4x32b = _mm_add_epi32 (e1_4x32b, e8_4x32b);
            ee2_4x32b = _mm_add_epi32 (e2_4x32b, e7_4x32b);
            ee3_4x32b = _mm_add_epi32 (e3_4x32b, e6_4x32b);
            ee4_4x32b = _mm_add_epi32 (e4_4x32b, e5_4x32b);

            /*  eo[k] = e[k] - e[7 - k] */
            eo1_4x32b = _mm_sub_epi32 (e1_4x32b, e8_4x32b);
            eo2_4x32b = _mm_sub_epi32 (e2_4x32b, e7_4x32b);
            eo3_4x32b = _mm_sub_epi32 (e3_4x32b, e6_4x32b);
            eo4_4x32b = _mm_sub_epi32 (e4_4x32b, e5_4x32b);

            /*  k = 4, 12   */
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_16_even[3]);
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_16_even[4]);

            /*  eee[k] = ee[k] + ee[3 - k]  */
            eee1_4x32b = _mm_add_epi32 (ee1_4x32b, ee4_4x32b);
            eee2_4x32b = _mm_add_epi32 (ee2_4x32b, ee3_4x32b);

            /*  eeo[k] = ee[k] - ee[3 - k]  */
            eeo1_4x32b = _mm_sub_epi32 (ee1_4x32b, ee4_4x32b);
            eeo2_4x32b = _mm_sub_epi32 (ee2_4x32b, ee3_4x32b);

            tmp1_8x16b = _mm_slli_epi32 (eee1_4x32b, 6);
            tmp2_8x16b = _mm_slli_epi32 (eee2_4x32b, 6);
            tmp5_8x16b = _mm_mullo_epi32 (eeo1_4x32b, coeff3_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (eeo2_4x32b, coeff4_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (eeo1_4x32b, coeff4_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (eeo2_4x32b, coeff3_4x32b);

            tmp3_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp6_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp3_8x16b, add_4x32b);
            tmp2_8x16b = _mm_add_epi32 (tmp4_8x16b, add_4x32b);
            tmp3_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);
            tmp4_8x16b = _mm_add_epi32 (tmp6_8x16b, add_4x32b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst3_8x16b = _mm_srai_epi32 (tmp2_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp3_8x16b, shift);
            dst4_8x16b = _mm_srai_epi32 (tmp4_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);
            dst3_8x16b = _mm_packs_epi32 (dst3_8x16b, dst3_8x16b);
            dst4_8x16b = _mm_packs_epi32 (dst4_8x16b, dst4_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 4 * i4_dst_strd), dst2_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 8 * i4_dst_strd), dst3_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 12 * i4_dst_strd), dst4_8x16b);

            /*  k = 2, 6    */
            coeff1_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[1]));
            coeff2_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[2]));
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[6]));
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[5]));

            tmp1_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff1_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff2_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (eo3_4x32b, coeff3_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (eo4_4x32b, coeff4_4x32b);

            tmp5_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff2_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff4_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (eo3_4x32b, coeff1_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (eo4_4x32b, coeff3_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp7_8x16b = _mm_add_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 2 * i4_dst_strd), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 6 * i4_dst_strd), dst2_8x16b);

            /*  k = 10, k = 14  */
            tmp1_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff3_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff1_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (eo3_4x32b, coeff4_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (eo4_4x32b, coeff2_4x32b);

            tmp5_8x16b = _mm_mullo_epi32 (eo1_4x32b, coeff4_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (eo2_4x32b, coeff3_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (eo3_4x32b, coeff2_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (eo4_4x32b, coeff1_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, add_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp5_8x16b = _mm_sub_epi32 (tmp5_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp7_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);
            tmp5_8x16b = _mm_add_epi32 (tmp5_8x16b, tmp7_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);
            dst2_8x16b = _mm_srai_epi32 (tmp5_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);
            dst2_8x16b = _mm_packs_epi32 (dst2_8x16b, dst2_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 10 * i4_dst_strd), dst1_8x16b);
            _mm_storel_epi64 ((__m128i *)(pi2_dst + 14 * i4_dst_strd), dst2_8x16b);

            /*  k = 1   */
            coeff1_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[0]));
            coeff2_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[1]));
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[2]));
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[3]));

            coeff5_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[4]));
            coeff6_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[5]));
            coeff7_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[6]));
            coeff8_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_odd[7]));

            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff1_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff2_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff3_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff4_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff5_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff6_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff7_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff8_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 1 * i4_dst_strd), dst1_8x16b);

            /*  k = 3   */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff2_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff5_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff8_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff6_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff3_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff1_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff4_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff7_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_sub_epi32 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 3 * i4_dst_strd), dst1_8x16b);

            /*  k = 5   */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff3_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff8_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff4_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff2_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff7_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff5_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff1_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff6_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp3_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp8_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 5 * i4_dst_strd), dst1_8x16b);

            /*  k = 7   */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff4_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff6_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff2_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff8_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff1_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff7_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff3_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff5_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_sub_epi32 (tmp6_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp4_8x16b, tmp3_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 7 * i4_dst_strd), dst1_8x16b);

            /*  k = 9   */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff5_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff3_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff7_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff1_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff8_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff2_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff6_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff4_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp7_8x16b, tmp3_8x16b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 9 * i4_dst_strd), dst1_8x16b);

            /*  k = 11  */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff6_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff1_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff5_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff7_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff2_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff4_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff8_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff3_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_sub_epi32 (tmp6_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 11 * i4_dst_strd), dst1_8x16b);

            /*  k = 13  */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff7_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff4_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff1_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff3_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff6_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff8_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff5_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff2_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_sub_epi32 (tmp6_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_sub_epi32 (tmp8_8x16b, tmp4_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 13 * i4_dst_strd), dst1_8x16b);

            /*  k = 15  */
            tmp1_8x16b = _mm_mullo_epi32 (o1_4x32b, coeff8_4x32b);
            tmp2_8x16b = _mm_mullo_epi32 (o2_4x32b, coeff7_4x32b);
            tmp3_8x16b = _mm_mullo_epi32 (o3_4x32b, coeff6_4x32b);
            tmp4_8x16b = _mm_mullo_epi32 (o4_4x32b, coeff5_4x32b);
            tmp5_8x16b = _mm_mullo_epi32 (o5_4x32b, coeff4_4x32b);
            tmp6_8x16b = _mm_mullo_epi32 (o6_4x32b, coeff3_4x32b);
            tmp7_8x16b = _mm_mullo_epi32 (o7_4x32b, coeff2_4x32b);
            tmp8_8x16b = _mm_mullo_epi32 (o8_4x32b, coeff1_4x32b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp5_8x16b);
            tmp2_8x16b = _mm_add_epi32 (tmp2_8x16b, tmp6_8x16b);
            tmp3_8x16b = _mm_add_epi32 (tmp3_8x16b, tmp7_8x16b);
            tmp4_8x16b = _mm_add_epi32 (tmp4_8x16b, tmp8_8x16b);

            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, add_4x32b);

            tmp1_8x16b = _mm_sub_epi32 (tmp1_8x16b, tmp2_8x16b);
            tmp3_8x16b = _mm_sub_epi32 (tmp3_8x16b, tmp4_8x16b);
            tmp1_8x16b = _mm_add_epi32 (tmp1_8x16b, tmp3_8x16b);

            dst1_8x16b = _mm_srai_epi32 (tmp1_8x16b, shift);

            dst1_8x16b = _mm_packs_epi32 (dst1_8x16b, dst1_8x16b);

            _mm_storel_epi64 ((__m128i *)(pi2_dst + 15 * i4_dst_strd), dst1_8x16b);

            pi4_temp  +=  4 * trans_size;
            pi2_dst += 4;
        }
    }

    return u4_blk_sad;
}
/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform on
 * input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 32x32 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 32x32
 *
 * @param[out] pi2_dst
 *  Output 32x32 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_32x32_sse42(UWORD16 *pu2_src,
                            UWORD16 *pu2_pred,
                            WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 i4_src_strd,
                            WORD32 i4_pred_strd,
                            WORD32 i4_dst_strd_chr_flag,
                            UWORD8 u1_bit_depth)

{
    WORD32 i;
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad = 0 ;
    WORD32 dst_strd;
    WORD16 temp_array[1024];
    WORD32 MEM_ALIGN16 temp_array1[1024];
    WORD16 *o_temp;
    WORD32 *o_temp1_ptr;

    WORD32 *g_ai2_ihevc_trans_32_intr_8_ptr;
    WORD32 *g_ai2_ihevc_trans_32_intr_16_ptr;

    __m128i m_temp_reg_0, m_temp_reg_1, m_temp_reg_2, m_temp_reg_3, m_temp_reg_4, m_temp_reg_5, m_temp_reg_6, m_temp_reg_7;
    __m128i m_temp_reg_8, m_temp_reg_9, m_temp_reg_10, m_temp_reg_11, m_temp_reg_12, m_temp_reg_13, m_temp_reg_14,m_temp_reg_15;
    __m128i m_temp_reg_18, m_temp_reg_19, m_temp_reg_20, m_temp_reg_21;
    __m128i sad_8x16b, add_4x32b, minusone_4x32b;
    __m128i temp_res_0, temp_res_1, temp_res_2, temp_res_3, temp_res_4, temp_res_5, temp_res_6, temp_res_7;
    __m128i res_r0_r3_r1_r2, res_r7_r4_r6_r5,res_r8_r11_r9_r10,res_r15_r12_r14_r13, res_r16_r19_r17_r18, res_r23_r20_r22_r21, res_r24_r27_r25_r26, res_r31_r28_r29_r30;
    __m128i reg_e0_e3_e1_e2, reg_o0_o3_o1_o2, reg_e7_e4_e6_e5, reg_o7_o4_o6_o5, reg_e8_e11_e9_e10, reg_o8_o11_o9_o10, reg_e15_e12_e14_e13, reg_o15_o12_o14_o13;
    __m128i reg_ee0_ee3_ee1_ee2, reg_eo0_eo3_eo1_eo2, reg_ee7_ee4_ee6_ee5, reg_eo7_eo4_eo6_eo5;
    __m128i reg_eee0_eee3_eee1_eee2, reg_eeo0_eeo3_eeo1_eeo2;
    __m128i reg_eeee0_eeee1, reg_eeeo0_eeeo1;
    __m128i mask1_8x16b, mask2_8x16b;

    mask1_8x16b     = _mm_set_epi32 (0x0B0A0D0C, 0x09080F0E, 0x05040302, 0x07060100);
    mask2_8x16b     = _mm_set_epi32 (0x0F0E0B0A, 0x0D0C0908, 0x07060302, 0x05040100);
    sad_8x16b       = _mm_setzero_si128 ();
    minusone_4x32b  = _mm_set1_epi32 (-1);

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_32;
    dst_strd = i4_dst_strd_chr_flag >> 16;

    /* Residue + Forward Transform 1st stage */
    o_temp = temp_array;
    o_temp1_ptr = temp_array1;

    shift = 2 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);
    add_4x32b = _mm_set1_epi32(add);

    g_ai2_ihevc_trans_32_intr_8_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_8;
    g_ai2_ihevc_trans_32_intr_16_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_16;

/* unrolling outer loop */
    for(i = 0; i < trans_size; i+=2)
    {
/* row =0 */
        m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pu2_src));                        /* k = 0-8  */
        m_temp_reg_1 = _mm_loadu_si128((__m128i*)(pu2_pred));                       /* k = 0-8  */

/* row =1 */
        m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pu2_src + i4_src_strd));          /* k = 0-8  */
        m_temp_reg_3 = _mm_loadu_si128((__m128i*)(pu2_pred + i4_pred_strd));        /* k = 0-8  */

/* row =0 */
        m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pu2_src + 8));                    /* k = 8-16  */
        m_temp_reg_5 = _mm_loadu_si128((__m128i*)(pu2_pred + 8));                   /* k = 8-16  */

/* row =1 */
        m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pu2_src + i4_src_strd + 8));      /* k = 8-16  */
        m_temp_reg_7 = _mm_loadu_si128((__m128i*)(pu2_pred + i4_pred_strd + 8));    /* k = 8-16  */

/* pu1_src */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_8 = _mm_shuffle_epi8 (m_temp_reg_0, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_9 = _mm_shuffle_epi8 (m_temp_reg_4, mask1_8x16b);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_10 = _mm_shuffle_epi8 (m_temp_reg_2, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_11 = _mm_shuffle_epi8 (m_temp_reg_6, mask1_8x16b);

/* pu1_pred */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_18 = _mm_shuffle_epi8 (m_temp_reg_1, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_19 = _mm_shuffle_epi8 (m_temp_reg_5, mask1_8x16b);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_20 = _mm_shuffle_epi8 (m_temp_reg_3, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_21 = _mm_shuffle_epi8 (m_temp_reg_7, mask1_8x16b);

/*residue computation */

    /* row 0 */
        /* r0 r3 r1 r2 r7 r4 r6 r5 */
        temp_res_0 = _mm_sub_epi16(m_temp_reg_8, m_temp_reg_18);
        /* r8 r11 r9 r10 r15 r12 r14 r13 */
        temp_res_1 = _mm_sub_epi16(m_temp_reg_9, m_temp_reg_19);

    /* row 1 */
        /* r0 r3 r1 r2 r7 r4 r6 r5 */
        temp_res_2 = _mm_sub_epi16(m_temp_reg_10, m_temp_reg_20);
        /* r8 r11 r9 r10 r15 r12 r14 r13 */
        temp_res_3 = _mm_sub_epi16(m_temp_reg_11, m_temp_reg_21);

/* row =0 */
        m_temp_reg_8 = _mm_loadu_si128((__m128i*)(pu2_src + 16));                   /* k = 16-24  */
        m_temp_reg_9 = _mm_loadu_si128((__m128i*)(pu2_pred + 16));                  /* k = 16-24  */

/* row =1 */
        m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pu2_src + i4_src_strd + 16));    /* k = 16-24  */
        m_temp_reg_11 = _mm_loadu_si128((__m128i*)(pu2_pred + i4_pred_strd + 16));  /* k = 16-24  */

/* row =0 */
        m_temp_reg_12 = _mm_loadu_si128((__m128i*)(pu2_src + 24));                  /* k = 24-32  */
        m_temp_reg_13 = _mm_loadu_si128((__m128i*)(pu2_pred + 24));                 /* k = 24-32  */

/* row =1 */
        m_temp_reg_14 = _mm_loadu_si128((__m128i*)(pu2_src + i4_src_strd + 24));      /* k = 24-32  */
        m_temp_reg_15 = _mm_loadu_si128((__m128i*)(pu2_pred + i4_pred_strd + 24));    /* k = 24-32  */

/* pu1_src */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_8 = _mm_shuffle_epi8 (m_temp_reg_8, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_12 = _mm_shuffle_epi8 (m_temp_reg_12, mask1_8x16b);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_10 = _mm_shuffle_epi8 (m_temp_reg_10, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_14 = _mm_shuffle_epi8 (m_temp_reg_14, mask1_8x16b);

/* pu1_pred */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_9 = _mm_shuffle_epi8 (m_temp_reg_9, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_13 = _mm_shuffle_epi8 (m_temp_reg_13, mask1_8x16b);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_11 = _mm_shuffle_epi8 (m_temp_reg_11, mask1_8x16b);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_15 = _mm_shuffle_epi8 (m_temp_reg_15, mask1_8x16b);

/*residue computation */

    /* row 0 */
        /* r16 r19 r17 r16 r23 r20 r22 r21 */
        temp_res_4 = _mm_sub_epi16(m_temp_reg_8, m_temp_reg_9);
        /* r24 r27 r25 r26 r31 r28 r29 r30 */
        temp_res_5 = _mm_sub_epi16(m_temp_reg_12, m_temp_reg_13);

    /* row 1 */
        /* r16 r19 r17 r16 r23 r20 r22 r21 */
        temp_res_6 = _mm_sub_epi16(m_temp_reg_10, m_temp_reg_11);
        /* r24 r27 r25 r26 r31 r28 r29 r30 */
        temp_res_7 = _mm_sub_epi16(m_temp_reg_14, m_temp_reg_15);

        /* SAD Calculation */
        m_temp_reg_0 = _mm_abs_epi16(temp_res_0);
        m_temp_reg_1 = _mm_abs_epi16(temp_res_1);
        m_temp_reg_2 = _mm_abs_epi16(temp_res_2);
        m_temp_reg_3 = _mm_abs_epi16(temp_res_3);
        m_temp_reg_4 = _mm_abs_epi16(temp_res_4);
        m_temp_reg_5 = _mm_abs_epi16(temp_res_5);
        m_temp_reg_6 = _mm_abs_epi16(temp_res_6);
        m_temp_reg_7 = _mm_abs_epi16(temp_res_7);

        m_temp_reg_0 = _mm_add_epi16(m_temp_reg_0, m_temp_reg_1);
        m_temp_reg_2 = _mm_add_epi16(m_temp_reg_2, m_temp_reg_3);
        m_temp_reg_4 = _mm_add_epi16(m_temp_reg_4, m_temp_reg_5);
        m_temp_reg_6 = _mm_add_epi16(m_temp_reg_6, m_temp_reg_7);

        m_temp_reg_0 = _mm_add_epi16(m_temp_reg_0, m_temp_reg_2);
        m_temp_reg_2 = _mm_add_epi16(m_temp_reg_4, m_temp_reg_6);
        m_temp_reg_0 = _mm_add_epi16(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_2 = _mm_cvtepi16_epi32 (m_temp_reg_0);
        m_temp_reg_0 = _mm_srli_si128 (m_temp_reg_0, 8);
        m_temp_reg_0 = _mm_cvtepi16_epi32 (m_temp_reg_0);

        m_temp_reg_0 = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
        sad_8x16b    = _mm_add_epi32(sad_8x16b, m_temp_reg_0);

/* Residue Re-ordering */
        /* r0 r3 r0 r3 r1 r2 r1 r2 */
        res_r0_r3_r1_r2 = _mm_unpacklo_epi32(temp_res_0, temp_res_2);
        res_r7_r4_r6_r5 = _mm_unpackhi_epi32(temp_res_0, temp_res_2);

        res_r8_r11_r9_r10   = _mm_unpacklo_epi32(temp_res_1, temp_res_3);
        res_r15_r12_r14_r13 = _mm_unpackhi_epi32(temp_res_1, temp_res_3);

        res_r16_r19_r17_r18 = _mm_unpacklo_epi32(temp_res_4, temp_res_6);
        res_r23_r20_r22_r21 = _mm_unpackhi_epi32(temp_res_4, temp_res_6);

        res_r24_r27_r25_r26 = _mm_unpacklo_epi32(temp_res_5, temp_res_7);
        res_r31_r28_r29_r30 = _mm_unpackhi_epi32(temp_res_5, temp_res_7);

/* e[] and o[] calculations */
        /* e0 e3 e0 e3 e1 e2 e1 e2 */
        reg_e0_e3_e1_e2 = _mm_add_epi16(res_r0_r3_r1_r2, res_r31_r28_r29_r30);
        reg_o0_o3_o1_o2 = _mm_sub_epi16(res_r0_r3_r1_r2, res_r31_r28_r29_r30);
        reg_e7_e4_e6_e5 = _mm_add_epi16(res_r7_r4_r6_r5, res_r24_r27_r25_r26);
        reg_o7_o4_o6_o5 = _mm_sub_epi16(res_r7_r4_r6_r5, res_r24_r27_r25_r26);

        reg_e8_e11_e9_e10 = _mm_add_epi16(res_r8_r11_r9_r10, res_r23_r20_r22_r21);
        reg_o8_o11_o9_o10 = _mm_sub_epi16(res_r8_r11_r9_r10, res_r23_r20_r22_r21);

        reg_e15_e12_e14_e13 = _mm_add_epi16(res_r15_r12_r14_r13, res_r16_r19_r17_r18);
        reg_o15_o12_o14_o13 = _mm_sub_epi16(res_r15_r12_r14_r13, res_r16_r19_r17_r18);

/* EVEN PART CALCULATION */
        {
            __m128i coeff0_8_8x16b, coeff16_24_8x16b, coeff4_8x16b, coeff12_8x16b, coeff20_8x16b, coeff28_8x16b;
            __m128i src0_8_4x32b, src16_24_4x32b, src8_4x32b, src24_4x32b, src4_4x32b, src12_4x32b, src20_4x32b, src28_4x32b;
            __m128i src4_12_4x32b, src20_28_4x32b, src2_6_4x32b, src10_14_4x32b, src18_22_4x32b, src26_30_4x32b;
            __m128i src2_4x32b, src6_4x32b, src10_4x32b, src14_4x32b, src18_4x32b, src22_4x32b, src26_4x32b, src30_4x32b;
            __m128i temp0_4x32b, temp1_4x32b, temp2_4x32b, temp3_4x32b, temp4_4x32b, temp5_4x32b, temp6_4x32b, temp7_4x32b;
            __m128i coeff0_8x16b, coeff2_8x16b, coeff3_8x16b, coeff1_8x16b, coeff5_8x16b, coeff6_8x16b, coeff7_8x16b;

    /* ee[] and eo[] calculation */
            /* ee0 ee3 ee0 ee3 ee1 ee2 ee1 ee2 */
            reg_ee0_ee3_ee1_ee2 = _mm_add_epi16(reg_e0_e3_e1_e2, reg_e15_e12_e14_e13);
            reg_eo0_eo3_eo1_eo2 = _mm_sub_epi16(reg_e0_e3_e1_e2, reg_e15_e12_e14_e13);
            reg_ee7_ee4_ee6_ee5 = _mm_add_epi16(reg_e7_e4_e6_e5, reg_e8_e11_e9_e10);
            reg_eo7_eo4_eo6_eo5 = _mm_sub_epi16(reg_e7_e4_e6_e5, reg_e8_e11_e9_e10);

    /* eee[] and eeo[] calculations */
            /* eee0 eee3 eee0 eee3 eee1 eee2 eee1 eee2 */
            reg_eee0_eee3_eee1_eee2 = _mm_add_epi16(reg_ee0_ee3_ee1_ee2, reg_ee7_ee4_ee6_ee5);
            reg_eeo0_eeo3_eeo1_eeo2 = _mm_sub_epi16(reg_ee0_ee3_ee1_ee2, reg_ee7_ee4_ee6_ee5);

    /* eeee[] and eeee[] calculation */
            temp0_4x32b = _mm_srli_si128 (reg_eee0_eee3_eee1_eee2, 8);
            temp2_4x32b = _mm_cvtepi16_epi32 (reg_eee0_eee3_eee1_eee2);
            temp3_4x32b = _mm_cvtepi16_epi32 (temp0_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64 (temp2_4x32b, temp3_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64 (temp2_4x32b, temp3_4x32b);

            /*  k = 8, 24   */
            coeff0_8_8x16b = _mm_loadu_si128 ((__m128i *) &g_ai4_ihevc_hbd_trans_32_even[0]);
            coeff16_24_8x16b = _mm_loadu_si128 ((__m128i *) &g_ai4_ihevc_hbd_trans_32_even[1]);

            reg_eeee0_eeee1 = _mm_hadd_epi32 (temp0_4x32b, temp1_4x32b);
            reg_eeeo0_eeeo1 = _mm_hsub_epi32 (temp0_4x32b, temp1_4x32b);

            temp0_4x32b = _mm_slli_epi32 (reg_eeee0_eeee1, 6);
            temp1_4x32b = _mm_mullo_epi32 (reg_eeeo0_eeeo1, coeff0_8_8x16b);
            temp2_4x32b = _mm_mullo_epi32 (reg_eeeo0_eeeo1, coeff16_24_8x16b);

            src0_8_4x32b    = _mm_hadd_epi32 (temp0_4x32b, temp1_4x32b);
            src16_24_4x32b  = _mm_hsub_epi32 (temp0_4x32b, temp2_4x32b);

            src0_8_4x32b    = _mm_add_epi32 (src0_8_4x32b, add_4x32b);
            src16_24_4x32b  = _mm_add_epi32 (src16_24_4x32b, add_4x32b);

            src0_8_4x32b    = _mm_srai_epi32 (src0_8_4x32b, shift);
            src16_24_4x32b  = _mm_srai_epi32 (src16_24_4x32b, shift);

            src8_4x32b      = _mm_srli_si128 (src0_8_4x32b, 8);
            src24_4x32b     = _mm_srli_si128 (src16_24_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp),src0_8_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 16 * trans_size),src16_24_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 8 * trans_size),src8_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 24 * trans_size),src24_4x32b);

    /* COL 4, 12, 20, 28 calculations */
            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[2][0]);
            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[3][0]);
            coeff20_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[4][0]);
            coeff28_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[5][0]);

            src4_4x32b  = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff4_8x16b);
            src12_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff12_8x16b);
            src20_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff20_8x16b);
            src28_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff28_8x16b);

            temp0_4x32b = _mm_unpacklo_epi64(src4_4x32b, src12_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src4_4x32b, src12_4x32b);

            src4_12_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            temp2_4x32b = _mm_unpacklo_epi64(src20_4x32b, src28_4x32b);
            temp3_4x32b = _mm_unpackhi_epi64(src20_4x32b, src28_4x32b);

            src20_28_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src20_28_4x32b  = _mm_add_epi32 (src20_28_4x32b, add_4x32b);
            src4_12_4x32b   = _mm_add_epi32 (src4_12_4x32b, add_4x32b);

            src20_28_4x32b  = _mm_srai_epi32 (src20_28_4x32b, shift);
            src4_12_4x32b   = _mm_srai_epi32 (src4_12_4x32b, shift);

            src28_4x32b = _mm_srli_si128(src20_28_4x32b, 8);
            src12_4x32b = _mm_srli_si128(src4_12_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 4 * trans_size),src4_12_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 20 * trans_size),src20_28_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 12 * trans_size),src12_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 28 * trans_size),src28_4x32b);

    /* COL 2, 6, 10, 14, 18, 22, 26, 30 calculations */
        /* K = 2 and K = 6 */
            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[6][0]);
            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[7][0]);
            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[8][0]);
            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[9][0]);

            temp0_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff0_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff1_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff2_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff3_8x16b);

            src2_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            src6_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src2_4x32b, src6_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src2_4x32b, src6_4x32b);

            src2_6_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src2_6_4x32b = _mm_add_epi32 (src2_6_4x32b, add_4x32b);

            src2_6_4x32b = _mm_srai_epi32 (src2_6_4x32b, shift);

            src6_4x32b = _mm_srli_si128(src2_6_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 2 * trans_size),src2_6_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 6 * trans_size),src6_4x32b);

        /* K = 10 and K = 14 */
            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[10][0]);
            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[11][0]);
            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[12][0]);
            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[13][0]);

            temp4_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff4_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff5_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff6_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff7_8x16b);

            src10_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            src14_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            temp4_4x32b = _mm_unpacklo_epi64(src10_4x32b, src14_4x32b);
            temp5_4x32b = _mm_unpackhi_epi64(src10_4x32b, src14_4x32b);

            src10_14_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            src10_14_4x32b = _mm_add_epi32 (src10_14_4x32b, add_4x32b);

            src10_14_4x32b = _mm_srai_epi32 (src10_14_4x32b, shift);

            src14_4x32b = _mm_srli_si128(src10_14_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 10 * trans_size),src10_14_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 14 * trans_size),src14_4x32b);

        /* K = 18 and K = 22 */
            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[14][0]);
            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[15][0]);
            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[16][0]);
            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[17][0]);

            temp0_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff0_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff1_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff2_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff3_8x16b);

            src18_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            src22_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src18_4x32b, src22_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src18_4x32b, src22_4x32b);

            src18_22_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src18_22_4x32b = _mm_add_epi32 (src18_22_4x32b, add_4x32b);

            src18_22_4x32b = _mm_srai_epi32 (src18_22_4x32b, shift);

            src22_4x32b = _mm_srli_si128(src18_22_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 18 * trans_size),src18_22_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 22 * trans_size),src22_4x32b);

        /* K = 10 and K = 14 */
            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[18][0]);
            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[19][0]);
            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[20][0]);
            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[21][0]);

            temp4_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff4_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff5_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_eo0_eo3_eo1_eo2, coeff6_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_eo7_eo4_eo6_eo5, coeff7_8x16b);

            src26_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            src30_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            temp4_4x32b = _mm_unpacklo_epi64(src26_4x32b, src30_4x32b);
            temp5_4x32b = _mm_unpackhi_epi64(src26_4x32b, src30_4x32b);

            src26_30_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            src26_30_4x32b = _mm_add_epi32 (src26_30_4x32b, add_4x32b);

            src26_30_4x32b = _mm_srai_epi32 (src26_30_4x32b, shift);

            src30_4x32b = _mm_srli_si128(src26_30_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 26 * trans_size),src26_30_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 30 * trans_size),src30_4x32b);

        }

        /* ODD Part Calculation */
        {
            __m128i temp0_4x32b, temp1_4x32b, temp2_4x32b, temp3_4x32b, temp4_4x32b, temp5_4x32b, temp6_4x32b, temp7_4x32b;
            __m128i coeff0_8x16b, coeff2_8x16b, coeff3_8x16b, coeff4_8x16b, coeff1_8x16b, coeff5_8x16b, coeff6_8x16b, coeff7_8x16b;
            __m128i coeff10_8x16b, coeff11_8x16b, coeff12_8x16b, coeff13_8x16b, coeff14_8x16b, coeff15_8x16b, coeff16_8x16b, coeff17_8x16b;
            __m128i temp10_4x32b, temp11_4x32b, temp12_4x32b, temp13_4x32b;// temp14_4x32b, temp15_4x32b, temp16_4x32b, temp17_4x32b;
            __m128i src1_4x32b, src3_4x32b, src5_4x32b, src7_4x32b, src9_4x32b, src11_4x32b;
            __m128i src1_3_4x32b, src5_7_4x32b, src9_11_4x32b;

        /* K = 1 and K = 3 */
            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[0][0]);
            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[1][0]);
            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[2][0]);
            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[3][0]);
            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[4][0]);
            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[5][0]);
            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[6][0]);
            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[7][0]);

            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff0_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff1_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff2_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff3_8x16b);
            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff4_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff5_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff6_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff7_8x16b);

            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);
            src1_4x32b   = _mm_add_epi32(temp10_4x32b, temp11_4x32b);
            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);
            src3_4x32b   = _mm_add_epi32(temp12_4x32b, temp13_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src1_4x32b, src3_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src1_4x32b, src3_4x32b);

            src1_3_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src1_3_4x32b = _mm_add_epi32 (src1_3_4x32b, add_4x32b);

            src1_3_4x32b = _mm_srai_epi32 (src1_3_4x32b, shift);

            src3_4x32b = _mm_srli_si128(src1_3_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 1 * trans_size),src1_3_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 3 * trans_size),src3_4x32b);

        /* K = 5 and K = 7 */
            coeff10_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[8][0]);
            coeff11_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[9][0]);
            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[10][0]);
            coeff13_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[11][0]);
            coeff14_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[12][0]);
            coeff15_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[13][0]);
            coeff16_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[14][0]);
            coeff17_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[15][0]);

            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff10_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff11_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff12_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff13_8x16b);
            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff14_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff15_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff16_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff17_8x16b);

            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);
            src5_4x32b   = _mm_add_epi32(temp10_4x32b, temp11_4x32b);
            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);
            src7_4x32b   = _mm_add_epi32(temp12_4x32b, temp13_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);

            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src5_7_4x32b = _mm_add_epi32 (src5_7_4x32b, add_4x32b);

            src5_7_4x32b = _mm_srai_epi32 (src5_7_4x32b, shift);

            src7_4x32b = _mm_srli_si128(src5_7_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 5 * trans_size),src5_7_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 7 * trans_size),src7_4x32b);

        /* K = 9 and K = 11 */
            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[16][0]);
            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[17][0]);
            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[18][0]);
            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[19][0]);
            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[20][0]);
            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[21][0]);
            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[22][0]);
            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[23][0]);

            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff0_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff1_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff2_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff3_8x16b);
            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff4_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff5_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff6_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff7_8x16b);

            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);
            src9_4x32b   = _mm_add_epi32(temp10_4x32b, temp11_4x32b);
            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);
            src11_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src9_4x32b, src11_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src9_4x32b, src11_4x32b);

            src9_11_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src9_11_4x32b = _mm_add_epi32 (src9_11_4x32b, add_4x32b);

            src9_11_4x32b = _mm_srai_epi32 (src9_11_4x32b, shift);

            src11_4x32b = _mm_srli_si128(src9_11_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 9 * trans_size),src9_11_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 11 * trans_size),src11_4x32b);

        /* K = 13 and K = 15 */
            coeff10_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[24][0]);
            coeff11_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[25][0]);
            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[26][0]);
            coeff13_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[27][0]);
            coeff14_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[28][0]);
            coeff15_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[29][0]);
            coeff16_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[30][0]);
            coeff17_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[31][0]);

            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff10_8x16b);
            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff11_8x16b);
            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff12_8x16b);
            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff13_8x16b);
            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff14_8x16b);
            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff15_8x16b);
            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff16_8x16b);
            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff17_8x16b);

            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);
            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);
            src5_4x32b   = _mm_add_epi32(temp10_4x32b, temp11_4x32b);
            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);
            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);
            src7_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);

            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);
            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);

            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src5_7_4x32b = _mm_add_epi32 (src5_7_4x32b, add_4x32b);

            src5_7_4x32b = _mm_srai_epi32 (src5_7_4x32b, shift);

            src7_4x32b = _mm_srli_si128(src5_7_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 13 * trans_size),src5_7_4x32b);
            _mm_storel_epi64((__m128i *)(pi4_temp + 15 * trans_size),src7_4x32b);

        /* K = 17 and K = 19 */
            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[0][8]);

            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[1][8]);

            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[2][8]);

            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[3][8]);


            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[4][8]);

            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[5][8]);

            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[6][8]);

            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[7][8]);


            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff0_8x16b);

            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff1_8x16b);

            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff2_8x16b);

            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff3_8x16b);


            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff4_8x16b);

            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff5_8x16b);

            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff6_8x16b);

            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff7_8x16b);


            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src1_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src3_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src1_4x32b, src3_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src1_4x32b, src3_4x32b);


            src1_3_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src1_3_4x32b = _mm_add_epi32 (src1_3_4x32b, add_4x32b);

            src1_3_4x32b = _mm_srai_epi32 (src1_3_4x32b, shift);

            src3_4x32b = _mm_srli_si128(src1_3_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 17 * trans_size),src1_3_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 19 * trans_size),src3_4x32b);


        /* K = 21 and K = 23 */

            coeff10_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[8][8]);

            coeff11_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[9][8]);

            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[10][8]);

            coeff13_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[11][8]);


            coeff14_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[12][8]);

            coeff15_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[13][8]);

            coeff16_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[14][8]);

            coeff17_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[15][8]);


            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff10_8x16b);

            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff11_8x16b);

            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff12_8x16b);

            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff13_8x16b);


            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff14_8x16b);

            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff15_8x16b);

            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff16_8x16b);

            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff17_8x16b);


            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src5_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src7_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);


            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src5_7_4x32b = _mm_add_epi32 (src5_7_4x32b, add_4x32b);

            src5_7_4x32b = _mm_srai_epi32 (src5_7_4x32b, shift);

            src7_4x32b = _mm_srli_si128(src5_7_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 21 * trans_size),src5_7_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 23 * trans_size),src7_4x32b);


        /* K = 25 and K = 27 */

            coeff0_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[16][8]);

            coeff1_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[17][8]);

            coeff2_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[18][8]);

            coeff3_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[19][8]);


            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[20][8]);

            coeff5_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[21][8]);

            coeff6_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[22][8]);

            coeff7_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[23][8]);


            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff0_8x16b);

            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff1_8x16b);

            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff2_8x16b);

            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff3_8x16b);


            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff4_8x16b);

            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff5_8x16b);

            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff6_8x16b);

            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff7_8x16b);


            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src9_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src11_4x32b = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src9_4x32b, src11_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src9_4x32b, src11_4x32b);


            src9_11_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src9_11_4x32b = _mm_add_epi32 (src9_11_4x32b, add_4x32b);

            src9_11_4x32b = _mm_srai_epi32 (src9_11_4x32b, shift);

            src11_4x32b = _mm_srli_si128(src9_11_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 25 * trans_size),src9_11_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 27 * trans_size),src11_4x32b);


        /* K = 29 and K = 31 */

            coeff10_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[24][8]);

            coeff11_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[25][8]);

            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[26][8]);

            coeff13_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[27][8]);


            coeff14_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[28][8]);

            coeff15_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[29][8]);

            coeff16_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[30][8]);

            coeff17_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_odd[31][8]);


            temp0_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff10_8x16b);

            temp1_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff11_8x16b);

            temp2_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff12_8x16b);

            temp3_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff13_8x16b);


            temp4_4x32b = _mm_madd_epi16(reg_o0_o3_o1_o2, coeff14_8x16b);

            temp5_4x32b = _mm_madd_epi16(reg_o7_o4_o6_o5, coeff15_8x16b);

            temp6_4x32b = _mm_madd_epi16(reg_o8_o11_o9_o10, coeff16_8x16b);

            temp7_4x32b = _mm_madd_epi16(reg_o15_o12_o14_o13, coeff17_8x16b);


            temp10_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            temp11_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src5_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src7_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);

            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            src5_7_4x32b = _mm_add_epi32 (src5_7_4x32b, add_4x32b);

            src5_7_4x32b = _mm_srai_epi32 (src5_7_4x32b, shift);

            src7_4x32b = _mm_srli_si128(src5_7_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 29 * trans_size),src5_7_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 31 * trans_size),src7_4x32b);
        }


        pu2_src  += 2 * i4_src_strd;
        pu2_pred += 2 * i4_pred_strd;
        pi4_temp += 2;
    }

    sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);
    sad_8x16b = _mm_hadd_epi32 (sad_8x16b, sad_8x16b);

    u4_blk_sad = _mm_cvtsi128_si32 (sad_8x16b);

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = 13; // log2(iHeight) + 6
    add = 1 << (shift - 1);
    add_4x32b = _mm_set1_epi32(add);

    for(i = 0; i < TRANS_SIZE_32; i+=4)
    {
        {
            __m128i m_temp_reg_16, m_temp_reg_17, m_temp_reg_18, m_temp_reg_19;
            __m128i m_temp_reg_20, m_temp_reg_21, m_temp_reg_22, m_temp_reg_23;
            __m128i m_temp_reg_24, m_temp_reg_25, m_temp_reg_26, m_temp_reg_27;
            __m128i m_temp_reg_31, m_temp_reg_30, m_temp_reg_29, m_temp_reg_28, m_temp_reg_32;
            __m128i m_temp_reg_46, m_temp_reg_47, m_temp_reg_48, m_temp_reg_49;
            __m128i m_temp_reg_40, m_temp_reg_41, m_temp_reg_42, m_temp_reg_44;
            __m128i coeff1_4x32b, coeff2_4x32b, coeff3_4x32b, coeff4_4x32b;

/* row = 0, 1, 2, 3 */
            m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pi4_temp));                   /* k = 0-3  */
            m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size));        /* k = 0-3  */
            m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size));      /* k = 0-3  */
            m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size));      /* k = 0-3  */

/* row = 0, 1, 2, 3 */
            m_temp_reg_8  = _mm_loadu_si128((__m128i*)(pi4_temp + 4));              /* k = 4-7  */
            m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 4));   /* k = 4-7  */
            m_temp_reg_12 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 4)); /* k = 4-7  */
            m_temp_reg_14 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 4)); /* k = 4-7  */

            /* K = 0,1,2,3 */
            m_temp_reg_16 = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_17 = _mm_unpacklo_epi32(m_temp_reg_4, m_temp_reg_6);
            m_temp_reg_18 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_19 = _mm_unpackhi_epi32(m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_4  = _mm_unpacklo_epi64(m_temp_reg_16, m_temp_reg_17);
            m_temp_reg_20 = _mm_unpackhi_epi64(m_temp_reg_16, m_temp_reg_17);
            m_temp_reg_5  = _mm_unpacklo_epi64(m_temp_reg_18, m_temp_reg_19);
            m_temp_reg_21 = _mm_unpackhi_epi64(m_temp_reg_18, m_temp_reg_19);

            /* K = 4,5,6,7 */
            m_temp_reg_0  = _mm_unpacklo_epi32 (m_temp_reg_8, m_temp_reg_10);
            m_temp_reg_1  = _mm_unpacklo_epi32(m_temp_reg_12, m_temp_reg_14);
            m_temp_reg_24 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_10);
            m_temp_reg_25 = _mm_unpackhi_epi32(m_temp_reg_12, m_temp_reg_14);

            m_temp_reg_6  = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_22 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_7  = _mm_unpacklo_epi64(m_temp_reg_24, m_temp_reg_25);
            m_temp_reg_23 = _mm_unpackhi_epi64(m_temp_reg_24, m_temp_reg_25);

/* row = 0, 1, 2, 3 */
            m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pi4_temp + 24));                  /* k = 24-27  */
            m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 24));       /* k = 24-27  */
            m_temp_reg_8 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 24));     /* k = 24-27  */
            m_temp_reg_9 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 24));     /* k = 24-27  */

/* row = 0, 1, 2, 3 */
            m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pi4_temp+28));                   /* k = 28-31  */
            m_temp_reg_32 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size+28));        /* k = 28-31  */
            m_temp_reg_24 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size+28));      /* k = 28-31  */
            m_temp_reg_25 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size+28));      /* k = 28-31  */


            /* K = 24,25,26,27 */
            m_temp_reg_1  = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_3  = _mm_unpacklo_epi32(m_temp_reg_8, m_temp_reg_9);
            m_temp_reg_11 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_9  = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_9);

            m_temp_reg_31 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_3);
            m_temp_reg_27 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_3);
            m_temp_reg_30 = _mm_unpacklo_epi64(m_temp_reg_11, m_temp_reg_9);
            m_temp_reg_26 = _mm_unpackhi_epi64(m_temp_reg_11, m_temp_reg_9);

            /* K = 28,29,30,31 */
            m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_10, m_temp_reg_32);
            m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_24, m_temp_reg_25);
            m_temp_reg_8 = _mm_unpackhi_epi32(m_temp_reg_10, m_temp_reg_32);
            m_temp_reg_9 = _mm_unpackhi_epi32(m_temp_reg_24, m_temp_reg_25);

            m_temp_reg_29 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_25 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_28 = _mm_unpacklo_epi64(m_temp_reg_8, m_temp_reg_9);
            m_temp_reg_24 = _mm_unpackhi_epi64(m_temp_reg_8, m_temp_reg_9);

/* pu1_src[31 - k] and pu1_pred[31 - k]*/

            /* for k=0 to 7 */
            /* e[k] = pi2_tmp[k] + pi2_tmp[31 - k]; */
            m_temp_reg_16  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_24);   /* for k=0 */
            m_temp_reg_17  = _mm_add_epi32 (m_temp_reg_20, m_temp_reg_28);  /* for k=1 */
            m_temp_reg_18  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_25);   /* for k=2 */
            m_temp_reg_19  = _mm_add_epi32 (m_temp_reg_21, m_temp_reg_29);  /* for k=3 */

            m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_26);   /* for k=4 */
            m_temp_reg_13  = _mm_add_epi32 (m_temp_reg_22, m_temp_reg_30);  /* for k=5 */
            m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_7, m_temp_reg_27);   /* for k=6 */
            m_temp_reg_15  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_31);  /* for k=7 */

            /* 0[k] = pi2_tmp[k] - pi2_tmp[31 - k]; */
            m_temp_reg_0  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=0 */
            m_temp_reg_1  = _mm_sub_epi32 (m_temp_reg_20, m_temp_reg_28);   /* for k=1 */
            m_temp_reg_2  = _mm_sub_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=2 */
            m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_21, m_temp_reg_29);   /* for k=3 */

            m_temp_reg_8  = _mm_sub_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=4 */
            m_temp_reg_9  = _mm_sub_epi32 (m_temp_reg_22, m_temp_reg_30);   /* for k=5 */
            m_temp_reg_10 = _mm_sub_epi32 (m_temp_reg_7, m_temp_reg_27);    /* for k=6 */
            m_temp_reg_11 = _mm_sub_epi32 (m_temp_reg_23, m_temp_reg_31);   /* for k=7 */

            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr),    m_temp_reg_0);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+4),  m_temp_reg_1);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+8),  m_temp_reg_2);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+12), m_temp_reg_3);

            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+16), m_temp_reg_8);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+20), m_temp_reg_9);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+24), m_temp_reg_10);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+28), m_temp_reg_11);

/* for k = 8 to 15 */
/* row =0, 1, 2, 3 */
            m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pi4_temp + 8));               /* k = 8-11  */
            m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 8));    /* k = 8-11  */
            m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 8));  /* k = 8-11  */
            m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 8));  /* k = 8-11  */

/* row =0, 1, 2, 3 */
            m_temp_reg_8  = _mm_loadu_si128((__m128i*)(pi4_temp + 12));             /* k = 12-16  */
            m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 12));  /* k = 12-16  */
            m_temp_reg_42 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 12));/* k = 12-16  */
            m_temp_reg_44 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 12));/* k = 12-16  */

            /* K = 8, 9, 10, 11 */
            m_temp_reg_46 = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_47 = _mm_unpacklo_epi32(m_temp_reg_4, m_temp_reg_6);
            m_temp_reg_48 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_49 = _mm_unpackhi_epi32(m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_4 = _mm_unpacklo_epi64(m_temp_reg_46, m_temp_reg_47);
            m_temp_reg_0 = _mm_unpackhi_epi64(m_temp_reg_46, m_temp_reg_47);
            m_temp_reg_5 = _mm_unpacklo_epi64(m_temp_reg_48, m_temp_reg_49);
            m_temp_reg_1 = _mm_unpackhi_epi64(m_temp_reg_48, m_temp_reg_49);

            /* K = 12, 13, 14, 15 */
            m_temp_reg_20 = _mm_unpacklo_epi32(m_temp_reg_8, m_temp_reg_10);
            m_temp_reg_21 = _mm_unpacklo_epi32(m_temp_reg_42, m_temp_reg_44);
            m_temp_reg_24 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_10);
            m_temp_reg_25 = _mm_unpackhi_epi32(m_temp_reg_42, m_temp_reg_44);

            m_temp_reg_6 = _mm_unpacklo_epi64(m_temp_reg_20, m_temp_reg_21);
            m_temp_reg_2 = _mm_unpackhi_epi64(m_temp_reg_20, m_temp_reg_21);
            m_temp_reg_7 = _mm_unpacklo_epi64(m_temp_reg_24, m_temp_reg_25);
            m_temp_reg_3 = _mm_unpackhi_epi64(m_temp_reg_24, m_temp_reg_25);

/* row =0, 1, 2, 3 */
            m_temp_reg_20 = _mm_loadu_si128((__m128i*)(pi4_temp + 16));             /* k = 16-19  */
            m_temp_reg_22 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 16));  /* k = 16-19  */
            m_temp_reg_8  = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 16));/* k = 16-19  */
            m_temp_reg_9  = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 16));/* k = 16-19  */

/* row =0, 1, 2, 3 */
            m_temp_reg_30 = _mm_loadu_si128((__m128i*)(pi4_temp+20));               /* k = 20-23  */
            m_temp_reg_32 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size+20));    /* k = 20-23  */
            m_temp_reg_24 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size+20));  /* k = 20-23  */
            m_temp_reg_25 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size+20));  /* k = 20-23  */

            /* K = 16, 17, 18, 19 */
            m_temp_reg_21 = _mm_unpacklo_epi32(m_temp_reg_20, m_temp_reg_22);
            m_temp_reg_23 = _mm_unpacklo_epi32(m_temp_reg_8, m_temp_reg_9);
            m_temp_reg_41 = _mm_unpackhi_epi32(m_temp_reg_20, m_temp_reg_22);
            m_temp_reg_40 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_9);

            m_temp_reg_11 = _mm_unpacklo_epi64(m_temp_reg_21, m_temp_reg_23);
            m_temp_reg_27 = _mm_unpackhi_epi64(m_temp_reg_21, m_temp_reg_23);
            m_temp_reg_10 = _mm_unpacklo_epi64(m_temp_reg_41, m_temp_reg_40);
            m_temp_reg_26 = _mm_unpackhi_epi64(m_temp_reg_41, m_temp_reg_40);

            /* K = 20, 21, 22, 23 */
            m_temp_reg_20 = _mm_unpacklo_epi32(m_temp_reg_30, m_temp_reg_32);
            m_temp_reg_22 = _mm_unpacklo_epi32(m_temp_reg_24, m_temp_reg_25);
            m_temp_reg_48 = _mm_unpackhi_epi32(m_temp_reg_30, m_temp_reg_32);
            m_temp_reg_49 = _mm_unpackhi_epi32(m_temp_reg_24, m_temp_reg_25);

            m_temp_reg_9 = _mm_unpacklo_epi64(m_temp_reg_20, m_temp_reg_22);
            m_temp_reg_25 = _mm_unpackhi_epi64(m_temp_reg_20, m_temp_reg_22);
            m_temp_reg_8 = _mm_unpacklo_epi64(m_temp_reg_48, m_temp_reg_49);
            m_temp_reg_24 = _mm_unpackhi_epi64(m_temp_reg_48, m_temp_reg_49);

/* for k=8 to 15 */
            /* o[k] = pi4_temp[k] - pi4_temp[31 - k]; */
            m_temp_reg_28  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=8  */
            m_temp_reg_29  = _mm_sub_epi32 (m_temp_reg_0, m_temp_reg_8);     /* for k=9  */
            m_temp_reg_30  = _mm_sub_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=10 */
            m_temp_reg_31  = _mm_sub_epi32 (m_temp_reg_1, m_temp_reg_9);     /* for k=11 */

            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+32), m_temp_reg_28);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+36), m_temp_reg_29);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+40), m_temp_reg_30);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+44), m_temp_reg_31);

            m_temp_reg_28  = _mm_sub_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=12 */
            m_temp_reg_29  = _mm_sub_epi32 (m_temp_reg_2, m_temp_reg_10);    /* for k=13 */
            m_temp_reg_30  = _mm_sub_epi32 (m_temp_reg_7, m_temp_reg_27);    /* for k=14 */
            m_temp_reg_31  = _mm_sub_epi32 (m_temp_reg_3, m_temp_reg_11);    /* for k=15 */

            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+48), m_temp_reg_28);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+52), m_temp_reg_29);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+56), m_temp_reg_30);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+60), m_temp_reg_31);

            /* e[k] = pi4_temp[k] + pi4_temp[31 - k]; */
            m_temp_reg_20  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_24);     /* for k=8  */
            m_temp_reg_21  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_8);      /* for k=9  */
            m_temp_reg_22  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_25);     /* for k=10 */
            m_temp_reg_23  = _mm_add_epi32 (m_temp_reg_1, m_temp_reg_9);      /* for k=11 */

            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_26);      /* for k=12 */
            m_temp_reg_5  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_10);      /* for k=13 */
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_7, m_temp_reg_27);      /* for k=14 */
            m_temp_reg_7  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_11);      /* for k=15 */

/* eo[k] = e[k] - e[15 - k]; */
            m_temp_reg_24  = _mm_sub_epi32 (m_temp_reg_16, m_temp_reg_7);    /* for k=0 */
            m_temp_reg_25  = _mm_sub_epi32 (m_temp_reg_17, m_temp_reg_6);    /* for k=1 */
            m_temp_reg_26  = _mm_sub_epi32 (m_temp_reg_18, m_temp_reg_5);    /* for k=2 */
            m_temp_reg_27  = _mm_sub_epi32 (m_temp_reg_19, m_temp_reg_4);    /* for k=3 */

            m_temp_reg_28  = _mm_sub_epi32 (m_temp_reg_12, m_temp_reg_23);   /* for k=4 */
            m_temp_reg_29  = _mm_sub_epi32 (m_temp_reg_13, m_temp_reg_22);   /* for k=5 */
            m_temp_reg_30  = _mm_sub_epi32 (m_temp_reg_14, m_temp_reg_21);   /* for k=6 */
            m_temp_reg_31  = _mm_sub_epi32 (m_temp_reg_15, m_temp_reg_20);   /* for k=7 */

/* ee[k] = e[k] + e[15 - k]; */
            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_16, m_temp_reg_7);     /* for k=0 */
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_17, m_temp_reg_6);     /* for k=1 */
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_18, m_temp_reg_5);     /* for k=2 */
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_19, m_temp_reg_4);     /* for k=3 */

            m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_23);    /* for k=4 */
            m_temp_reg_9  = _mm_add_epi32 (m_temp_reg_13, m_temp_reg_22);    /* for k=5 */
            m_temp_reg_10 = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_21);    /* for k=6 */
            m_temp_reg_11 = _mm_add_epi32 (m_temp_reg_15, m_temp_reg_20);    /* for k=7 */

/* eeo[k] = ee[k] - ee[7 - k]; */
            m_temp_reg_12  = _mm_sub_epi32 (m_temp_reg_0, m_temp_reg_11);    /* for k=0 */
            m_temp_reg_13  = _mm_sub_epi32 (m_temp_reg_1, m_temp_reg_10);    /* for k=1 */
            m_temp_reg_14  = _mm_sub_epi32 (m_temp_reg_2, m_temp_reg_9);     /* for k=2 */
            m_temp_reg_15  = _mm_sub_epi32 (m_temp_reg_3, m_temp_reg_8);     /* for k=3 */

/* eee[k] = ee[k] + ee[7 - k] */
            m_temp_reg_16  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_11);    /* for k=0 */
            m_temp_reg_17  = _mm_add_epi32 (m_temp_reg_1, m_temp_reg_10);    /* for k=1 */
            m_temp_reg_18  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_9);     /* for k=2 */
            m_temp_reg_19  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_8);     /* for k=3 */

/* eeee[k] = eee[k] + eee[k - 3];*/
            m_temp_reg_20  = _mm_add_epi32 (m_temp_reg_16, m_temp_reg_19);
            m_temp_reg_21  = _mm_add_epi32 (m_temp_reg_17, m_temp_reg_18);

/* eeeo[k] = eee[k] - eee[k - 3];*/
            m_temp_reg_22  = _mm_sub_epi32 (m_temp_reg_16, m_temp_reg_19);
            m_temp_reg_23  = _mm_sub_epi32 (m_temp_reg_17, m_temp_reg_18);

            coeff3_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_16_even[3]);
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const *)g_ai4_ihevc_trans_16_even[4]);

            /* g_ai2_ihevc_trans_32[0][0] * eeee[0] */
            m_temp_reg_0  = _mm_slli_epi32 (m_temp_reg_20, 6);

            /* g_ai2_ihevc_trans_32[0][1] * eeee[1] */
            m_temp_reg_1  = _mm_slli_epi32 (m_temp_reg_21, 6);

            /* g_ai2_ihevc_trans_32[8][0] * eee0[0] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_22, coeff3_4x32b);

            /* g_ai2_ihevc_trans_32[8][1] * eee0[1] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_23, coeff4_4x32b);

            /* g_ai2_ihevc_trans_32[24][0] * eee0[0] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_22, coeff4_4x32b);

            /* g_ai2_ihevc_trans_32[24][1] * eee0[1] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_23, coeff3_4x32b);

            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_7  = _mm_sub_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_9  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_5);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_6, add_4x32b);
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_7, add_4x32b);
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_8, add_4x32b);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_9, add_4x32b);

            m_temp_reg_0 = _mm_srai_epi32(m_temp_reg_0, shift);
            m_temp_reg_1 = _mm_srai_epi32(m_temp_reg_1, shift);
            m_temp_reg_3 = _mm_srai_epi32(m_temp_reg_3, shift);
            m_temp_reg_4 = _mm_srai_epi32(m_temp_reg_4, shift);

            /* convert 32-bit to 16-bit and store results */
            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);
            m_temp_reg_1 = _mm_packs_epi32 (m_temp_reg_1, m_temp_reg_1);
            m_temp_reg_3 = _mm_packs_epi32 (m_temp_reg_3, m_temp_reg_3);
            m_temp_reg_4 = _mm_packs_epi32 (m_temp_reg_4, m_temp_reg_4);

            _mm_storel_epi64((__m128i *)(pi2_dst),m_temp_reg_0);
            _mm_storel_epi64((__m128i *)(pi2_dst+16*dst_strd),m_temp_reg_1);
            _mm_storel_epi64((__m128i *)(pi2_dst+8 *dst_strd),m_temp_reg_3);
            _mm_storel_epi64((__m128i *)(pi2_dst+24*dst_strd),m_temp_reg_4);

/* for(k = 4; k < 32; k += 8) */
            coeff1_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[1]));
            coeff2_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[2]));
            coeff3_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[6]));
            coeff4_4x32b = _mm_loadu_si128 ((__m128i const*) (g_ai4_ihevc_trans_16_even[5]));

/* m_temp_reg_12= eeo[0],m_temp_reg_13= eeo[1], m_temp_reg_14= eeo[2], m_temp_reg_15= eeo[3] */

           /* for k=4 */
           {
               /* g_ai2_ihevc_trans_32[k][0] * eeo[0] */
               m_temp_reg_0  = _mm_mullo_epi32 (coeff1_4x32b,  m_temp_reg_12);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[1] */
               m_temp_reg_1  = _mm_mullo_epi32 (coeff2_4x32b,  m_temp_reg_13);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[2] */
               m_temp_reg_2  = _mm_mullo_epi32 (coeff3_4x32b,  m_temp_reg_14);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[3] */
               m_temp_reg_3  = _mm_mullo_epi32 (coeff4_4x32b,  m_temp_reg_15);

               m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
               m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
               m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
               m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
               m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

               m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+4*dst_strd),m_temp_reg_0);

           }

           /* for k=20 */
           {
               /* g_ai2_ihevc_trans_32[k][0] * eeo[0] */
               m_temp_reg_1  = _mm_mullo_epi32 (coeff3_4x32b,  m_temp_reg_12);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[1] */
               m_temp_reg_2  = _mm_mullo_epi32 (coeff1_4x32b,  m_temp_reg_13);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[2] */
               m_temp_reg_3  = _mm_mullo_epi32 (coeff4_4x32b,  m_temp_reg_14);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[3] */
               m_temp_reg_4  = _mm_mullo_epi32 (coeff2_4x32b,  m_temp_reg_15);

               m_temp_reg_1  = _mm_sub_epi32 (m_temp_reg_1, m_temp_reg_2);
               m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_4);
               m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, m_temp_reg_2);

               m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, add_4x32b);

               m_temp_reg_1  = _mm_srai_epi32(m_temp_reg_1, shift);

               m_temp_reg_1 = _mm_packs_epi32 (m_temp_reg_1, m_temp_reg_1);

               _mm_storel_epi64((__m128i *)(pi2_dst+20*dst_strd),m_temp_reg_1);
           }

           /* for k=28*/
           {
               /* g_ai2_ihevc_trans_32[k][0] * eeo[0] */
               m_temp_reg_2  = _mm_mullo_epi32 (coeff4_4x32b,  m_temp_reg_12);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[1] */
               m_temp_reg_3  = _mm_mullo_epi32 (coeff3_4x32b,  m_temp_reg_13);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[2] */
               m_temp_reg_4  = _mm_mullo_epi32 (coeff2_4x32b,  m_temp_reg_14);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[3] */
               m_temp_reg_5  = _mm_mullo_epi32 (coeff1_4x32b,  m_temp_reg_15);

               m_temp_reg_2  = _mm_sub_epi32 (m_temp_reg_2, m_temp_reg_3);
               m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_5);
               m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);

               m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, add_4x32b);

               m_temp_reg_2 = _mm_srai_epi32(m_temp_reg_2, shift);

               m_temp_reg_2 = _mm_packs_epi32 (m_temp_reg_2, m_temp_reg_2);

               _mm_storel_epi64((__m128i *)(pi2_dst+28*dst_strd),m_temp_reg_2);

           }

           /* for k=12*/
           {
               /* g_ai2_ihevc_trans_32[k][0] * eeo[0] */
               m_temp_reg_3  = _mm_mullo_epi32 (coeff2_4x32b,  m_temp_reg_12);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[1] */
               m_temp_reg_4  = _mm_mullo_epi32 (coeff4_4x32b,  m_temp_reg_13);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[2] */
               m_temp_reg_5  = _mm_mullo_epi32 (coeff1_4x32b,  m_temp_reg_14);
               /* g_ai2_ihevc_trans_32[k][0] * eeo[3] */
               m_temp_reg_6  = _mm_mullo_epi32 (coeff3_4x32b,  m_temp_reg_15);

               m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_3, m_temp_reg_4);
               m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_6);
               m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_3, m_temp_reg_4);

               m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_3, add_4x32b);

               m_temp_reg_3 = _mm_srai_epi32(m_temp_reg_3, shift);

               m_temp_reg_3 = _mm_packs_epi32 (m_temp_reg_3, m_temp_reg_3);

               _mm_storel_epi64((__m128i *)(pi2_dst+12*dst_strd),m_temp_reg_3);

           }
/*  up to here 260+32 cycles with storing */

            m_temp_reg_16 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr));
            m_temp_reg_17 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+4));
            m_temp_reg_18 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+8));
            m_temp_reg_19 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+12));
            m_temp_reg_20 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+16));
            m_temp_reg_21 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+20));
            m_temp_reg_22 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+24));
            m_temp_reg_23 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+28));

            /*for(k = 2; k < 32; k += 4) */

            /* k =2 */
            {
                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_18,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+2*dst_strd),m_temp_reg_0);

            }

            /* k =6 */
            {
                m_temp_reg_9  =_mm_sign_epi32(m_temp_reg_21, minusone_4x32b);
                m_temp_reg_10 =_mm_sign_epi32(m_temp_reg_18, minusone_4x32b);
                m_temp_reg_11 =_mm_sign_epi32(m_temp_reg_16, minusone_4x32b);
                m_temp_reg_12 =_mm_sign_epi32(m_temp_reg_19, minusone_4x32b);
                m_temp_reg_13 =_mm_sign_epi32(m_temp_reg_22, minusone_4x32b);

                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_9,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+6*dst_strd),m_temp_reg_0);

            }

            /* k =10 */
            {
                m_temp_reg_14 =_mm_sign_epi32(m_temp_reg_17, minusone_4x32b);

                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_18,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+10*dst_strd),m_temp_reg_0);
            }

            /* k =14 */
            {
                m_temp_reg_15 =_mm_sign_epi32(m_temp_reg_20, minusone_4x32b);

                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_9,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+14*dst_strd),m_temp_reg_0);
            }

            /* k =18 */
            {
                m_temp_reg_8 =_mm_sign_epi32(m_temp_reg_23, minusone_4x32b);

                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_8,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+18*dst_strd),m_temp_reg_0);
            }

            /* k =22 */
            {
                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+22*dst_strd),m_temp_reg_0);

            }

            /* k =26 */
            {
                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+26*dst_strd),m_temp_reg_0);

            }

            /* k =30 */
            {
                /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_24);
                /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_25);
                /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_26);
                /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_27);
                 /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_28);
                /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_29);
                /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_30);
                /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_31);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

               _mm_storel_epi64((__m128i *)(pi2_dst+30*dst_strd),m_temp_reg_0);

            }

 /* for(k = 1; k < 32; k += 2) */
            {
                __m128i m_temp_reg_32, m_temp_reg_33, m_temp_reg_34, m_temp_reg_35;
                __m128i m_temp_reg_36, m_temp_reg_37, m_temp_reg_38, m_temp_reg_39;

                __m128i m_temp_reg_40, m_temp_reg_41, m_temp_reg_42, m_temp_reg_43;
                __m128i m_temp_reg_44, m_temp_reg_45, m_temp_reg_46;

                m_temp_reg_32 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr));
                m_temp_reg_33 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+4));
                m_temp_reg_34 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+8));
                m_temp_reg_35 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+12));
                m_temp_reg_36 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+16));
                m_temp_reg_37 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+20));
                m_temp_reg_38 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+24));
                m_temp_reg_39 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+28));

                m_temp_reg_40 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+32));
                m_temp_reg_41 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+36));
                m_temp_reg_42 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+40));
                m_temp_reg_43 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+44));
                m_temp_reg_44 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+48));
                m_temp_reg_45 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+52));
                m_temp_reg_46 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_16_ptr+56));

                m_temp_reg_16 = _mm_loadu_si128((__m128i*)(o_temp1_ptr));
                m_temp_reg_17 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+4));
                m_temp_reg_18 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+8));
                m_temp_reg_19 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+12));

                m_temp_reg_20 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+16));
                m_temp_reg_21 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+20));
                m_temp_reg_22 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+24));
                m_temp_reg_23 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+28));

                m_temp_reg_24 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+32));
                m_temp_reg_25 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+36));
                m_temp_reg_26 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+40));
                m_temp_reg_27 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+44));

                m_temp_reg_28 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+48));
                m_temp_reg_29 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+52));
                m_temp_reg_30 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+56));
                m_temp_reg_31 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+60));

                /* for k=1 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+1*dst_strd),m_temp_reg_0);

                }

                /* for k=3 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_40, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_34, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+3*dst_strd),m_temp_reg_0);

                }

                /* for k=5 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_40, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] *///
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+5*dst_strd),m_temp_reg_0);

                }

                /* for k=7 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_38, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+7*dst_strd),m_temp_reg_0);

                }

                /* for k=9 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_40, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_38, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+9*dst_strd),m_temp_reg_0);

                }

                /* for k=11 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+11*dst_strd),m_temp_reg_0);

                }

                /* for k=13 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+13*dst_strd),m_temp_reg_0);

                }
                /* for k=15 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_40, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+15*dst_strd),m_temp_reg_0);

                }
                /* for k=17 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+17*dst_strd),m_temp_reg_0);

                }
                /* for k=19 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_34, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+19*dst_strd),m_temp_reg_0);

                }
                /* for k=21 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_34, minusone_4x32b),  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+21*dst_strd),m_temp_reg_0);

                }
                /* for k=23 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_38, minusone_4x32b),  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+23*dst_strd),m_temp_reg_0);

                }

                /* for k=25 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_43,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_35,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_44, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+25*dst_strd),m_temp_reg_0);

                }
                /* for k=27 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_37,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_46, minusone_4x32b),  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_41,  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+27*dst_strd),m_temp_reg_0);

                }
                /* for k=29 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_45,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_42, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_39,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_36, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_33,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+29*dst_strd),m_temp_reg_0);
                }

                /* for k=31 */
                {
                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_46,  m_temp_reg_16);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_1  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_45, minusone_4x32b),  m_temp_reg_17);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_44,  m_temp_reg_18);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_3  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_43, minusone_4x32b),  m_temp_reg_19);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_42,  m_temp_reg_20);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_5  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_41, minusone_4x32b),  m_temp_reg_21);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_40,  m_temp_reg_22);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_7  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_39, minusone_4x32b),  m_temp_reg_23);

                    /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
                    m_temp_reg_8  = _mm_mullo_epi32 (m_temp_reg_38,  m_temp_reg_24);
                    /* g_ai2_ihevc_trans_32[k][1] * eo[1] */
                    m_temp_reg_9  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_37, minusone_4x32b),  m_temp_reg_25);
                    /* g_ai2_ihevc_trans_32[k][2] * eo[2] */
                    m_temp_reg_10  = _mm_mullo_epi32 (m_temp_reg_36,  m_temp_reg_26);
                    /* g_ai2_ihevc_trans_32[k][3] * eo[3] */
                    m_temp_reg_11  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_35, minusone_4x32b),  m_temp_reg_27);
                    /* g_ai2_ihevc_trans_32[k][4] * eo[4] */
                    m_temp_reg_12  = _mm_mullo_epi32 (m_temp_reg_34,  m_temp_reg_28);
                    /* g_ai2_ihevc_trans_32[k][5] * eo[5] */
                    m_temp_reg_13  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_33, minusone_4x32b),  m_temp_reg_29);
                    /* g_ai2_ihevc_trans_32[k][6] * eo[6] */
                    m_temp_reg_14  = _mm_mullo_epi32 (m_temp_reg_32,  m_temp_reg_30);
                    /* g_ai2_ihevc_trans_32[k][7] * eo[7] */
                    m_temp_reg_15  = _mm_mullo_epi32 (_mm_sign_epi32(m_temp_reg_32, minusone_4x32b),  m_temp_reg_31);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                    m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_9);
                    m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_13);
                    m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_15);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_8, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_10, m_temp_reg_2);
                    m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_4);
                    m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_6);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);
                    m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_4);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_0);

                    m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, add_4x32b);
                    m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

                    m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, m_temp_reg_0);

                    _mm_storel_epi64((__m128i *)(pi2_dst+31*dst_strd),m_temp_reg_0);
                 }
            }
        }
        pi4_temp += 4 * trans_size;
        pi2_dst +=4;
    }
    return u4_blk_sad;
}
