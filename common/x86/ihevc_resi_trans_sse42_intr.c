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
 *  ihevc_resi_trans_x86_intr.c
 *
 * @brief
 *  Contains function definitions for residual and  forward transform
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_resi_trans_4x4_ttype1_sse42()
 *  - ihevc_resi_trans_4x4_sse42()
 *  - ihevc_resi_trans_8x8_sse42()
 *  - ihevc_resi_trans_16x16_sse42()
 *  - ihevc_resi_trans_32x32_sse42()
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
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

#include <immintrin.h>

#define SHIFT_DST_4_1st 1
#define SHIFT_DST_4_2nd 8
#define SHIFT_DCT_4_1st 1
#define SHIFT_DCT_4_2nd 8
#define SHIFT_DCT_8_1st 2
#define SHIFT_DCT_8_2nd 9


/* For 4x4 implementation */
static UWORD8 IHEVCE_SHUFFLEMASKY0Y1_TRNS[16] = { 0x00, 0xFF, 0x04, 0xFF,
                                 0x01, 0xFF, 0x05, 0xFF,
                                 0x02, 0xFF, 0x06, 0xFF,
                                 0x03, 0xFF, 0x07, 0xFF};

static UWORD8 IHEVCE_SHUFFLEMASKY3Y2_TRNS[16] = { 0x0C, 0xFF, 0x08, 0xFF,
                                 0x0D, 0xFF, 0x09, 0xFF,
                                 0x0E, 0xFF, 0x0A, 0xFF,
                                 0x0F, 0xFF, 0x0B, 0xFF};

/* For 16x16 implementation */
static UWORD8 IHEVCE_SHUFFLEMASK_16x16_TRNS_REORDER1[16] = { 0x00, 0xFF, 0x01, 0xFF,
                                            0x03, 0xFF, 0x02, 0xFF,
                                            0x04, 0xFF, 0x05, 0xFF,
                                            0x07, 0xFF, 0x06, 0xFF};

static UWORD8 IHEVCE_SHUFFLEMASK_16x16_TRNS_REORDER2[16] = { 0x08, 0xFF, 0x09, 0xFF,
                                            0x0B, 0xFF, 0x0A, 0xFF,
                                            0x0C, 0xFF, 0x0D, 0xFF,
                                            0x0F, 0xFF, 0x0E, 0xFF};
/* For 32x32 implementation */
static UWORD8 IHEVCE_SHUFFLEMASK_32x32_TRNS_REORDER1[16] = { 0x00, 0xFF, 0x03, 0xFF,
                                            0x01, 0xFF, 0x02, 0xFF,
                                            0x07, 0xFF, 0x04, 0xFF,
                                            0x06, 0xFF, 0x05, 0xFF};

static UWORD8 IHEVCE_SHUFFLEMASK_32x32_TRNS_REORDER2[16] = { 0x08, 0xFF, 0x0B, 0xFF,
                                            0x09, 0xFF, 0x0A, 0xFF,
                                            0x0F, 0xFF, 0x0C, 0xFF,
                                            0x0E, 0xFF, 0x0D, 0xFF};

static UWORD8 IHEVCE_SHUFFLEMASK_32x32_TRNS[16] = { 0x00, 0x01, 0x04, 0x05,
                                   0x02, 0x03, 0x06, 0x07,
                                   0x08, 0x09, 0x0C, 0x0D,
                                   0x0A, 0x0B, 0x0E, 0x0F};

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
 * @param[in] pu1_src
 *  Input 4x4 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *  0 - luma transform, 1 - chroma transform. Not used for 4x4ttyppe1
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

UWORD32 ihevc_resi_trans_4x4_ttype1_sse42(UWORD8 *pu1_src,
                                    UWORD8 *pu1_pred,
                                    WORD32 *pi4_temp,
                                    WORD16 *pi2_dst,
                                    WORD32 src_strd,
                                    WORD32 pred_strd,
                                    WORD32 dst_strd_chr_flag)
{
    WORD32 add;
    UWORD32 sad;
     //   WORD32 chroma_flag;
    WORD32 dst_strd;

    /* all 128 bit registers are named with a suffix mxnb, where m is the */
    /* number of n bits packed in the register                            */
    __m128i src0_4x32b, src1_4x32b, src2_4x32b, src3_4x32b;
    __m128i pred0_4x32b, pred1_4x32b, pred2_4x32b, pred3_4x32b;
    __m128i sad0_4x32b, sad1_4x32b, sad2_4x32b, sad3_4x32b;
    __m128i temp0_4x32b, temp1_4x32b, temp2_4x32b, temp3_4x32b;
    __m128i coeff0_4x32b, coeff1_4x32b, coeff2_4x32b;
    __m128i c0_4x32b, c1_4x32b, c2_4x32b, c3_4x32b;

     //   chroma_flag = dst_strd_chr_flag & 1;
    dst_strd = dst_strd_chr_flag >> 16;

    /* 4 src pels loaded to lower 32 bit, upper 32 is don't care */
    /* 4 times for 4 rows */
    src0_4x32b = _mm_loadl_epi64 ((__m128i *)pu1_src);
    src1_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+src_strd));
    src2_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+2*src_strd));
    src3_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+3*src_strd));
    /* 4 pred pels loaded to lower 32 bit, upper 32 is don't care */
    /* 4 times for 4 rows */
    pred0_4x32b = _mm_loadl_epi64 ((__m128i *)pu1_pred);
    pred1_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+pred_strd));
    pred2_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+2*pred_strd));
    pred3_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+3*pred_strd));

    /* SAD computation (4x4) */
    /* x x src_r1 src_r0 */
    sad0_4x32b = _mm_unpacklo_epi32(src0_4x32b, src1_4x32b);
    /* x x src_r3 src_r2 */
    sad1_4x32b = _mm_unpacklo_epi32(src2_4x32b, src3_4x32b);
    sad2_4x32b = _mm_unpacklo_epi32(pred0_4x32b, pred1_4x32b);
    sad3_4x32b = _mm_unpacklo_epi32(pred2_4x32b, pred3_4x32b);

    /* src_r3 src_r2 src_r1 src_r0 */
    sad0_4x32b = _mm_unpacklo_epi64(sad0_4x32b, sad1_4x32b);
    sad2_4x32b = _mm_unpacklo_epi64(sad2_4x32b, sad3_4x32b);

    sad0_4x32b = _mm_sad_epu8(sad0_4x32b, sad2_4x32b);

    sad2_4x32b = _mm_srli_si128(sad0_4x32b, 8);

    sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);
    sad = _mm_cvtsi128_si32 (sad0_4x32b);
    /* End of SAD (4x4) */

    /***************************    4x4 Transpose  ***************************/
    /* x x x x x x x x b3 a3 b2 a2 b1 a1 b0 a0 */
    temp0_4x32b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
    /* x x x x x x x x d3 c3 d2 c2 d1 c1 d0 c0 */
    temp1_4x32b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);

    temp2_4x32b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);
    temp3_4x32b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);

    /* d3 c3 b3 a3 d2 c2 b2 a2 d1 c1 b1 a1 d0 c0 b0 a0 */
    src0_4x32b = _mm_unpacklo_epi16(temp0_4x32b, temp1_4x32b);

    pred0_4x32b = _mm_unpacklo_epi16(temp2_4x32b, temp3_4x32b);

    /* x x x x x x x x x x x x d1 c1 b1 a1 */
    src1_4x32b = _mm_srli_si128 (src0_4x32b, 4);
    /* x x x x x x x x x x x x d2 c2 b2 a2 */
    src2_4x32b = _mm_srli_si128 (src0_4x32b, 8);
    /* x x x x x x x x x x x x d3 c3 b3 a3 */
    src3_4x32b = _mm_srli_si128 (src0_4x32b, 12);

    pred1_4x32b = _mm_srli_si128 (pred0_4x32b, 4);
    pred2_4x32b = _mm_srli_si128 (pred0_4x32b, 8);
    pred3_4x32b = _mm_srli_si128 (pred0_4x32b, 12);
    /**************************  4x4 Transpose End   *************************/

    /* 8-32 bit conversion */
    src0_4x32b = _mm_cvtepu8_epi32 (src0_4x32b);
    src1_4x32b = _mm_cvtepu8_epi32 (src1_4x32b);
    src2_4x32b = _mm_cvtepu8_epi32 (src2_4x32b);
    src3_4x32b = _mm_cvtepu8_epi32 (src3_4x32b);
    pred0_4x32b = _mm_cvtepu8_epi32 (pred0_4x32b);
    pred1_4x32b = _mm_cvtepu8_epi32 (pred1_4x32b);
    pred2_4x32b = _mm_cvtepu8_epi32 (pred2_4x32b);
    pred3_4x32b = _mm_cvtepu8_epi32 (pred3_4x32b);


    /* Residue + Forward Transform 1st stage */
    {
        __m128i add_4x32b;

        add = 1 << (SHIFT_DST_4_1st - 1);

        /* coeff2_4x32b = 74 74 74 74 */
        coeff2_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_dst_intr_4[2][0]);
        /* coeff0_4x32b = 29 29 29 29 */
        coeff0_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_dst_intr_4[0][0]);
        /* coeff1_4x32b = 55 55 55 55 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_dst_intr_4[1][0]);

        /* residue calculation */
        temp0_4x32b = _mm_sub_epi32 (src0_4x32b, pred0_4x32b);
        temp3_4x32b = _mm_sub_epi32 (src3_4x32b, pred3_4x32b);
        temp1_4x32b = _mm_sub_epi32 (src1_4x32b, pred1_4x32b);
        temp2_4x32b = _mm_sub_epi32 (src2_4x32b, pred2_4x32b);
        /* end of residue calculation */

        src1_4x32b = _mm_add_epi32 (temp0_4x32b, temp1_4x32b); /* r0+r1 */

        /* c0 to c3 calculation */
        c0_4x32b = _mm_add_epi32 (temp0_4x32b, temp3_4x32b); /* r0+r3 */
        c1_4x32b = _mm_add_epi32 (temp1_4x32b, temp3_4x32b); /* r1+r3 */
        c2_4x32b = _mm_sub_epi32 (temp0_4x32b, temp1_4x32b); /* r0-r1 */
        c3_4x32b = _mm_mullo_epi32 (temp2_4x32b, coeff2_4x32b); /* 74*r2 */

        /* add value */
        add_4x32b = _mm_set1_epi32(add);

        src1_4x32b = _mm_sub_epi32 (src1_4x32b, temp3_4x32b); /* r0+r1-r3 */
        temp0_4x32b = _mm_mullo_epi32 (c0_4x32b, coeff0_4x32b); /* 29*c0 */
        temp2_4x32b = _mm_mullo_epi32 (c2_4x32b, coeff0_4x32b); /* 29*c2 */
        temp3_4x32b = _mm_mullo_epi32 (c2_4x32b, coeff1_4x32b); /* 55*c2 */

        src0_4x32b = _mm_mullo_epi32 (c1_4x32b, coeff1_4x32b); /* 55*c1 */
        src2_4x32b = _mm_mullo_epi32 (c0_4x32b, coeff1_4x32b); /* 55*c0 */
        src3_4x32b = _mm_mullo_epi32 (c1_4x32b, coeff0_4x32b); /* 29*c1 */
        src1_4x32b = _mm_mullo_epi32 (src1_4x32b, coeff2_4x32b);/*74*(r0+r1-r3)*/

        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, c3_4x32b); /* 29*c0 + c3 */
        temp2_4x32b = _mm_sub_epi32 (temp2_4x32b, c3_4x32b); /* 29*c2 - c3 */
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, c3_4x32b); /* 55*c2 + c3 */

        src0_4x32b = _mm_add_epi32 (temp0_4x32b, src0_4x32b); /* 29*c0 + 55*c1 + c3 */
        src2_4x32b = _mm_add_epi32 (temp2_4x32b, src2_4x32b); /* 29*c2 + 55*c0 - c3 */
        src3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b); /* 55*c2 - 29*c1 + c3 */

        /* result + add */
        src1_4x32b  = _mm_add_epi32 (src1_4x32b,  add_4x32b);
        src0_4x32b = _mm_add_epi32 (src0_4x32b, add_4x32b);
        src2_4x32b = _mm_add_epi32 (src2_4x32b, add_4x32b);
        src3_4x32b = _mm_add_epi32 (src3_4x32b, add_4x32b);

        /* result >> shift */
        src1_4x32b  = _mm_srai_epi32 (src1_4x32b,  SHIFT_DST_4_1st);
        src0_4x32b = _mm_srai_epi32 (src0_4x32b, SHIFT_DST_4_1st);
        src2_4x32b = _mm_srai_epi32 (src2_4x32b, SHIFT_DST_4_1st);
        src3_4x32b = _mm_srai_epi32 (src3_4x32b, SHIFT_DST_4_1st);
    }

    /* Forward transform 2nd stage */
    {
        __m128i add_4x32b;

        /*************************    4x4 32bit Transpose  ***********************/
        /* b1 a1 b0 a0 */
        temp0_4x32b = _mm_unpacklo_epi32(src0_4x32b, src1_4x32b);
        /* b3 a3 b2 a2 */
        temp1_4x32b = _mm_unpackhi_epi32(src0_4x32b, src1_4x32b);
        /* d1 c1 d0 c0 */
        temp2_4x32b = _mm_unpacklo_epi32(src2_4x32b, src3_4x32b);
        /* d3 c3 d2 c2 */
        temp3_4x32b = _mm_unpackhi_epi32(src2_4x32b, src3_4x32b);

        /* d0 c0 b0 a0 */
        src0_4x32b = _mm_unpacklo_epi64(temp0_4x32b, temp2_4x32b);
        /* d1 c1 b1 a1 */
        src1_4x32b = _mm_unpackhi_epi64(temp0_4x32b, temp2_4x32b);
        /* d2 c2 b2 a2 */
        src2_4x32b = _mm_unpacklo_epi64(temp1_4x32b, temp3_4x32b);
        /* d3 c3 b3 a3 */
        src3_4x32b = _mm_unpackhi_epi64(temp1_4x32b, temp3_4x32b);
        /**************************  4x4 Transpose End   *************************/

        add = 1 << (SHIFT_DST_4_2nd - 1);

        /* c0 to c3 calculation */
        temp1_4x32b = _mm_add_epi32 (src0_4x32b, src1_4x32b); /* r0+r1 */
        c0_4x32b = _mm_add_epi32 (src0_4x32b, src3_4x32b); /* r0+r3 */
        c1_4x32b = _mm_add_epi32 (src1_4x32b, src3_4x32b); /* r1+r3 */
        c2_4x32b = _mm_sub_epi32 (src0_4x32b, src1_4x32b); /* r0-r1 */
        c3_4x32b = _mm_mullo_epi32 (src2_4x32b, coeff2_4x32b); /* 74*r2 */

        /* add value */
        add_4x32b = _mm_set1_epi32(add);

        temp1_4x32b = _mm_sub_epi32 (temp1_4x32b, src3_4x32b); /* r0+r1-r3 */
        temp0_4x32b = _mm_mullo_epi32 (c0_4x32b,  coeff0_4x32b); /* 29*c0 */
        temp2_4x32b = _mm_mullo_epi32 (c2_4x32b,  coeff0_4x32b); /* 29*c2 */
        temp3_4x32b = _mm_mullo_epi32 (c2_4x32b,  coeff1_4x32b); /* 55*c2 */

        src0_4x32b  = _mm_mullo_epi32 (c1_4x32b, coeff1_4x32b); /* 55*c1 */
        src2_4x32b  = _mm_mullo_epi32 (c0_4x32b, coeff1_4x32b); /* 55*c0 */
        src3_4x32b  = _mm_mullo_epi32 (c1_4x32b, coeff0_4x32b); /* 29*c1 */
        temp1_4x32b = _mm_mullo_epi32 (temp1_4x32b, coeff2_4x32b);/*74*(r0+r1-r3)*/

        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, c3_4x32b); /* 29*c0 + c3 */
        temp2_4x32b = _mm_sub_epi32 (temp2_4x32b, c3_4x32b); /* 29*c2 - c3 */
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, c3_4x32b); /* 55*c2 + c3 */

        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, src0_4x32b); /* 29*c0 + 55*c1 + c3 */
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, src2_4x32b); /* 29*c2 + 55*c0 - c3 */
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b); /* 55*c2 - 29*c1 + c3 */

        /* result + add */
        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, add_4x32b);
        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, add_4x32b);
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, add_4x32b);
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, add_4x32b);

        /* result >> shift */
        temp1_4x32b = _mm_srai_epi32 (temp1_4x32b, SHIFT_DST_4_2nd);
        temp0_4x32b = _mm_srai_epi32 (temp0_4x32b, SHIFT_DST_4_2nd);
        temp2_4x32b = _mm_srai_epi32 (temp2_4x32b, SHIFT_DST_4_2nd);
        temp3_4x32b = _mm_srai_epi32 (temp3_4x32b, SHIFT_DST_4_2nd);

        /* 32-16 bit conversion (4x4 DST, upper 64 bits are don't care) */
        temp1_4x32b = _mm_packs_epi32 (temp1_4x32b, temp1_4x32b);
        temp0_4x32b = _mm_packs_epi32 (temp0_4x32b, temp0_4x32b);
        temp2_4x32b = _mm_packs_epi32 (temp2_4x32b, temp2_4x32b);
        temp3_4x32b = _mm_packs_epi32 (temp3_4x32b, temp3_4x32b);

        /* store to temp location */
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ dst_strd), temp1_4x32b);
        _mm_storel_epi64 ((__m128i *)pi2_dst, temp0_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 2*dst_strd), temp2_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 3*dst_strd), temp3_4x32b);
    }

    return sad;
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
 * @param[in] pu1_src
 *  Input 4x4 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *  0 - luma transform, 1 - chroma transform.
 *
 * @returns  Luma : SAD, Chroma : 0
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_resi_trans_4x4_sse42(UWORD8 *pu1_src,
                             UWORD8 *pu1_pred,
                             WORD32 *pi4_temp,
                             WORD16 *pi2_dst,
                             WORD32 src_strd,
                             WORD32 pred_strd,
                             WORD32 dst_strd_chr_flag)
{
    WORD32 add;
    UWORD32 sad = 0;
    WORD32 chroma_flag;
    WORD32 dst_strd;

    /* all 128 bit registers are named with a suffix mxnb, where m is the */
    /* number of n bits packed in the register                            */
    __m128i src0_4x32b, src1_4x32b, src2_4x32b, src3_4x32b;
    __m128i pred0_4x32b, pred1_4x32b, pred2_4x32b, pred3_4x32b;
    __m128i sad0_4x32b, sad2_4x32b;
    __m128i temp0_4x32b, temp1_4x32b, temp2_4x32b, temp3_4x32b;
    __m128i coeff0_4x32b, coeff1_4x32b, coeff2_4x32b;
    __m128i src0_8x8b,src1_8x8b, src0_16x8b;
    __m128i pred0_8x8b,pred1_8x8b, pred0_16x8b;
    __m128i scnd_smask, frst_smask;
    __m128i res_0_1,res_3_2,e0_e1_4x32b,o0_o1_4x32b;
    __m128i e0_e1_row1, e0_e1_row2, e0_e1_row3;
    __m128i o0_o1_row1, o0_o1_row2, o0_o1_row3;

    chroma_flag = dst_strd_chr_flag & 1;
    dst_strd = dst_strd_chr_flag >> 16;

    /* 4 src pels loaded to lower 32 bit, upper 32 is don't care */
    /* 4 times for 4 rows */
    src0_4x32b = _mm_loadl_epi64 ((__m128i *)pu1_src);
    src1_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+src_strd));
    src2_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+2*src_strd));
    src3_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_src+3*src_strd));
    /* 4 pred pels loaded to lower 32 bit, upper 32 is don't care */
    /* 4 times for 4 rows */
    pred0_4x32b = _mm_loadl_epi64 ((__m128i *)pu1_pred);
    pred1_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+pred_strd));
    pred2_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+2*pred_strd));
    pred3_4x32b = _mm_loadl_epi64 ((__m128i *)(pu1_pred+3*pred_strd));
    /* Luma/Chroma Based Update */
    if (chroma_flag)
    {
        WORD32 shuffle_mask = 0x06040200;
        __m128i chroma_shuffle_mask_16x8b;
        chroma_shuffle_mask_16x8b = _mm_cvtsi32_si128 (shuffle_mask);

        src0_4x32b  = _mm_shuffle_epi8 (src0_4x32b, chroma_shuffle_mask_16x8b);
        src1_4x32b  = _mm_shuffle_epi8 (src1_4x32b, chroma_shuffle_mask_16x8b);
        src2_4x32b  = _mm_shuffle_epi8 (src2_4x32b, chroma_shuffle_mask_16x8b);
        src3_4x32b  = _mm_shuffle_epi8 (src3_4x32b, chroma_shuffle_mask_16x8b);

        /* a0 b0 a1 b1 a2 b2 a3 b3 */
        src0_8x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
        /* c0 d0 c1 d1 c2 d2 c3 d3 */
        src1_8x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);
        /* a0 b0 c0 d0 a1 b1 c1 d1... */
        src0_16x8b = _mm_unpacklo_epi16(src0_8x8b, src1_8x8b);

        pred0_4x32b = _mm_shuffle_epi8 (pred0_4x32b, chroma_shuffle_mask_16x8b);
        pred1_4x32b = _mm_shuffle_epi8 (pred1_4x32b, chroma_shuffle_mask_16x8b);
        pred2_4x32b = _mm_shuffle_epi8 (pred2_4x32b, chroma_shuffle_mask_16x8b);
        pred3_4x32b = _mm_shuffle_epi8 (pred3_4x32b, chroma_shuffle_mask_16x8b);

        pred0_8x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);
        pred1_8x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);

        pred0_16x8b = _mm_unpacklo_epi16(pred0_8x8b, pred1_8x8b);
    }
    else
    {
        /* SAD computation (4x4) */
        /* a0 b0 a1 b1 a2 b2 a3 b3 */
        src0_8x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
        /* c0 d0 c1 d1 c2 d2 c3 d3 */
        src1_8x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);
        /* a0 b0 c0 d0 a1 b1 c1 d1... */
        src0_16x8b = _mm_unpacklo_epi16(src0_8x8b, src1_8x8b);
        /* Same as cur */
        pred0_8x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);
        pred1_8x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);

        pred0_16x8b = _mm_unpacklo_epi16(pred0_8x8b, pred1_8x8b);

        sad0_4x32b = _mm_sad_epu8(src0_16x8b, pred0_16x8b);

        sad2_4x32b = _mm_srli_si128(sad0_4x32b, 8);

        sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);
        sad = _mm_cvtsi128_si32 (sad0_4x32b);
        /* End of SAD (4x4) */
    }
    /* End of SAD (4x4) */
    frst_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY0Y1_TRNS[0]);

    scnd_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY3Y2_TRNS[0]);

    /*a0 b0 c0 d0 a1 b1 c1 d1*/
    temp0_4x32b = _mm_shuffle_epi8(src0_16x8b, frst_smask);
    /* pred col1, col 2 */
    temp2_4x32b = _mm_shuffle_epi8(pred0_16x8b, frst_smask);
    /*a3 b3 c3 d3 a2 b2 c2 d2*/
    temp1_4x32b = _mm_shuffle_epi8(src0_16x8b, scnd_smask);
    /* pred col3, col 4*/
    temp3_4x32b = _mm_shuffle_epi8(pred0_16x8b, scnd_smask);

    /* r0 (col 0), r1 (col 1) */
    res_0_1 = _mm_sub_epi16(temp0_4x32b,temp2_4x32b);
    /* r3 (col 3), r2 (col 2) */
    res_3_2 = _mm_sub_epi16(temp1_4x32b,temp3_4x32b);

    /* Residue + Forward Transform 1st stage */
    {
        __m128i add_4x32b;

        add = 1 << (SHIFT_DCT_4_1st - 1);

        /* coeff1_4x32b = 83 83 83 83 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_4_intr[0]);

        /*e & o calculation */
        e0_e1_4x32b = _mm_add_epi16 (res_0_1, res_3_2);
        o0_o1_4x32b = _mm_sub_epi16 (res_0_1, res_3_2);

        /* Get the e0_e1 values of next rows by shifting */
        e0_e1_row1 = _mm_srli_si128(e0_e1_4x32b, 4);
        e0_e1_row2 = _mm_srli_si128(e0_e1_4x32b, 8);
        e0_e1_row3 = _mm_srli_si128(e0_e1_4x32b, 12);

        /* Get the o0_o1 values of next rows by shifting */
        o0_o1_row1 = _mm_srli_si128(o0_o1_4x32b, 4);
        o0_o1_row2 = _mm_srli_si128(o0_o1_4x32b, 8);
        o0_o1_row3 = _mm_srli_si128(o0_o1_4x32b, 12);

        /* add value */
        add_4x32b = _mm_set1_epi32(add);

        /* e0,e1,o0,o1,e0,e1,o0,o1 */
        src0_4x32b = _mm_unpacklo_epi32(e0_e1_4x32b, o0_o1_4x32b);
        src0_4x32b = _mm_unpacklo_epi64(src0_4x32b, src0_4x32b);

        /* e0,e1,o0,o1,e0,e1,o0,o1 */
        src1_4x32b = _mm_unpacklo_epi32(e0_e1_row1, o0_o1_row1);
        src1_4x32b = _mm_unpacklo_epi64(src1_4x32b, src1_4x32b);

        /* e0,e1,o0,o1,e0,e1,o0,o1 */
        src2_4x32b = _mm_unpacklo_epi32(e0_e1_row2, o0_o1_row2);
        src2_4x32b = _mm_unpacklo_epi64(src2_4x32b, src2_4x32b);

        /* e0,e1,o0,o1,e0,e1,o0,o1 */
        src3_4x32b = _mm_unpacklo_epi32(e0_e1_row3, o0_o1_row3);
        src3_4x32b = _mm_unpacklo_epi64(src3_4x32b, src3_4x32b);

        /* A0 B0 C0 D0 */
        src0_4x32b = _mm_madd_epi16(src0_4x32b,coeff1_4x32b);

        /* A1 B1 C1 D1 */
        src1_4x32b = _mm_madd_epi16(src1_4x32b,coeff1_4x32b);

        /* A2 B2 C2 D2 */
        src2_4x32b = _mm_madd_epi16(src2_4x32b,coeff1_4x32b);

        /* A3 B3 C3 D3 */
        src3_4x32b = _mm_madd_epi16(src3_4x32b,coeff1_4x32b);
    }

    /* Forward transform 2nd stage */
    {
        __m128i add_4x32b;
        __m128i e0_4x32b, e1_4x32b, o0_4x32b, o1_4x32b;

        add = 1 << ((SHIFT_DCT_4_2nd + SHIFT_DCT_4_1st) - 1);
        /* coeff0_4x32b = 64 64 64 64 */
        coeff0_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_4_intr[0][0]);
        /* coeff1_4x32b = 83 83 83 83 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_4_intr[1][0]);
        /* coeff2_4x32b = 36 36 36 36 */
        coeff2_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_4_intr[2][0]);
        /*e & o calculation */
        e0_4x32b = _mm_add_epi32 (src0_4x32b, src3_4x32b);
        o0_4x32b = _mm_sub_epi32 (src0_4x32b, src3_4x32b);
        e1_4x32b = _mm_add_epi32 (src1_4x32b, src2_4x32b);
        o1_4x32b = _mm_sub_epi32 (src1_4x32b, src2_4x32b);

        temp0_4x32b = _mm_add_epi32 (e0_4x32b, e1_4x32b);
        temp2_4x32b = _mm_sub_epi32 (e0_4x32b, e1_4x32b);
        temp1_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff1_4x32b); /* 83*o0 */
        temp3_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff2_4x32b); /* 36*o0 */
        src1_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff2_4x32b); /* 36*o1 */
        src3_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff1_4x32b); /* 83*o1 */

        /* add value */
        add_4x32b = _mm_set1_epi32(add);

        temp0_4x32b = _mm_slli_epi32(temp0_4x32b,6);
        temp2_4x32b = _mm_slli_epi32(temp2_4x32b,6);
        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);  /* 83*o0+36*o1 */
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b); /* 36*o0-83*o1 */

        /* result + add */
        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, add_4x32b);
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, add_4x32b);
        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, add_4x32b);
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, add_4x32b);

        /* result >> shift */
        temp0_4x32b = _mm_srai_epi32 (temp0_4x32b, (SHIFT_DCT_4_2nd + SHIFT_DCT_4_1st));
        temp2_4x32b = _mm_srai_epi32 (temp2_4x32b, (SHIFT_DCT_4_2nd + SHIFT_DCT_4_1st));
        temp1_4x32b = _mm_srai_epi32 (temp1_4x32b, (SHIFT_DCT_4_2nd + SHIFT_DCT_4_1st));
        temp3_4x32b = _mm_srai_epi32 (temp3_4x32b, (SHIFT_DCT_4_2nd + SHIFT_DCT_4_1st));

        /* 32-16 bit conversion (4x4 DST, upper 64 bits are don't care) */
        temp0_4x32b = _mm_packs_epi32 (temp0_4x32b, temp0_4x32b);
        temp2_4x32b = _mm_packs_epi32 (temp2_4x32b, temp2_4x32b);
        temp1_4x32b = _mm_packs_epi32 (temp1_4x32b, temp1_4x32b);
        temp3_4x32b = _mm_packs_epi32 (temp3_4x32b, temp3_4x32b);

        /* store to temp location */
        _mm_storel_epi64 ((__m128i *)pi2_dst, temp0_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 2*dst_strd), temp2_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ dst_strd), temp1_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 3*dst_strd), temp3_4x32b);
    }

    return sad;
}



void ihevc_resi_trans_4x4_16bit_sse42(WORD16 *pi2_src,
                          UWORD8 *pu1_pred,
                          WORD16 *pi2_tmp,
                          WORD16 *pi2_dst,
                          WORD32 src_strd,
                          WORD32 pred_strd,
                          WORD32 dst_strd)
{

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
 * @param[in] pu1_src
 *  Input 8x8 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 8x8
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *  0 - luma transform, 1 - chroma transform.
 *
 * @returns  Luma : SAD, Chroma : 0
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_resi_trans_8x8_sse42(UWORD8 *pu1_src,
                             UWORD8 *pu1_pred,
                             WORD32 *pi4_temp,
                             WORD16 *pi2_dst,
                             WORD32 src_strd,
                             WORD32 pred_strd,
                             WORD32 dst_strd_chr_flag)
{
    WORD32 trans_size = TRANS_SIZE_8;
    WORD32 add;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 sad = 0;
    WORD32 chroma_flag;
    WORD32 dst_strd;

    /* all 128 bit registers are named with a suffix mxnb, where m is the */
    /* number of n bits packed in the register                            */
    __m128i src0_4x32b, src1_4x32b, src2_4x32b, src3_4x32b;
    __m128i src4_4x32b, src5_4x32b, src6_4x32b, src7_4x32b;
    __m128i pred0_4x32b, pred1_4x32b, pred2_4x32b, pred3_4x32b;
    __m128i pred0_8x16b, pred1_8x16b, pred2_8x16b, pred3_8x16b;
    __m128i src0_16x8b, src1_16x8b, pred0_16x8b, pred1_16x8b, pred2_16x8b, pred3_16x8b;
    __m128i res_0_1, res_2_3, res_5_4, res_7_6;

    __m128i sad0_4x32b, sad2_4x32b;
    __m128i temp0_16x8b, temp1_16x8b, temp0_8x16b, temp3_8x16b, temp1_8x16b, temp2_8x16b;
    __m128i temp0_4x32b, temp1_4x32b, temp2_4x32b, temp3_4x32b;
    __m128i temp4_4x32b, temp5_4x32b, temp6_4x32b, temp7_4x32b;

    __m128i e0_e1_8x16b, e3_e2_8x16b, o0_o1_8x16b, o3_o2_8x16b;
    __m128i ee0_ee1_8x16b, eo0_eo1_8x16b;
    __m128i o_row_0_1, o_row_2_3;

    __m128i e0_4x32b, e1_4x32b, e2_4x32b, e3_4x32b;
    __m128i o0_4x32b, o1_4x32b, o2_4x32b, o3_4x32b;
    __m128i ee0_4x32b, ee1_4x32b;
    __m128i eo0_4x32b, eo1_4x32b;
    __m128i coeff0_4x32b, coeff1_4x32b, coeff2_4x32b, coeff3_4x32b;
    __m128i coeff4_4x32b, coeff5_4x32b, coeff6_4x32b, coeff7_4x32b;
    __m128i frst_low_smask, frst_hi_smask; //, scnd_low_smask, scnd_hi_smask;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;

    chroma_flag = dst_strd_chr_flag & 1;
    dst_strd = dst_strd_chr_flag >> 16;

    /* Residue + Forward Transform 1st stage */
    {
        /* Luma : 8 src pels loaded to lower 64 bit, Chroma : 16 src pels loaded */ /* 4 times for 1st 4 rows */
        src0_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src1_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src2_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src3_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        /* Luma : 8 pred pels loaded to lower 64 bit, Chroma : 16 pred pels loaded */ /* 4 times for 1st 4 rows */
        pred0_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred1_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred2_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred3_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;

        if (chroma_flag)
        {
            __m128i chroma_shuffle_mask_16x8b;
            chroma_shuffle_mask_16x8b = _mm_set_epi32 (0x0, 0x0, 0x0E0C0A08, 0x06040200);

            src0_4x32b  = _mm_shuffle_epi8 (src0_4x32b, chroma_shuffle_mask_16x8b);
            src1_4x32b  = _mm_shuffle_epi8 (src1_4x32b, chroma_shuffle_mask_16x8b);
            src2_4x32b  = _mm_shuffle_epi8 (src2_4x32b, chroma_shuffle_mask_16x8b);
            src3_4x32b  = _mm_shuffle_epi8 (src3_4x32b, chroma_shuffle_mask_16x8b);

            /* a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 */
            src0_16x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
            /* c0 d0 c1 d1 c2 d2 c3 d3 c4 d4 c5 d5 c6 d6 c7 d7 */
            src1_16x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);

            pred0_4x32b = _mm_shuffle_epi8 (pred0_4x32b, chroma_shuffle_mask_16x8b);
            pred1_4x32b = _mm_shuffle_epi8 (pred1_4x32b, chroma_shuffle_mask_16x8b);
            pred2_4x32b = _mm_shuffle_epi8 (pred2_4x32b, chroma_shuffle_mask_16x8b);
            pred3_4x32b = _mm_shuffle_epi8 (pred3_4x32b, chroma_shuffle_mask_16x8b);

            /* Same as cur */
            pred0_16x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);

            pred1_16x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);
        }
        else
        {
            /* SAD computation (4x8), taking only lower 64 bits of 128 (Luma part) */

            /* a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 */
            src0_16x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
            /* c0 d0 c1 d1 c2 d2 c3 d3 c4 d4 c5 d5 c6 d6 c7 d7 */
            src1_16x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);

            /* Same as cur */
            pred0_16x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);

            pred1_16x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);

            sad0_4x32b = _mm_sad_epu8(src0_16x8b, pred0_16x8b);

            sad2_4x32b = _mm_sad_epu8(src1_16x8b, pred1_16x8b);

            sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);
            sad2_4x32b = _mm_srli_si128(sad0_4x32b, 8);

            sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);

            sad += _mm_cvtsi128_si32 (sad0_4x32b);

            /* End of SAD (4x8) */
        }

        /* End of SAD (4x4) */
        frst_low_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY0Y1_TRNS[0]);

        frst_hi_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASKY3Y2_TRNS[0]);

        /* a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3*/
        temp0_16x8b = _mm_unpacklo_epi16(src0_16x8b, src1_16x8b);

        /* a4 b4 c4 d4 a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7*/
        temp1_16x8b = _mm_unpackhi_epi16(src0_16x8b, src1_16x8b);

        /* Same as cur values population */
        pred2_16x8b = _mm_unpacklo_epi16(pred0_16x8b, pred1_16x8b);

        pred3_16x8b = _mm_unpackhi_epi16(pred0_16x8b, pred1_16x8b);

        /* a0 a1 b0 b1 c0 c1 d0 d1 */
        temp0_8x16b = _mm_shuffle_epi8(temp0_16x8b, frst_low_smask);

        /* a3 a2 b3 b2 c3 c2 d3 d2 */
        temp1_8x16b = _mm_shuffle_epi8(temp0_16x8b, frst_hi_smask);

        /* a4 a5 b4 b5 c4 c5 d4 d5 */
        temp2_8x16b = _mm_shuffle_epi8(temp1_16x8b, frst_low_smask);

        /* a7 a6 b7 b6 c7 c6 d7 d6 */
        temp3_8x16b = _mm_shuffle_epi8(temp1_16x8b, frst_hi_smask);

        /* Same as Cur values */
        pred0_8x16b = _mm_shuffle_epi8(pred2_16x8b, frst_low_smask);

        pred1_8x16b = _mm_shuffle_epi8(pred2_16x8b, frst_hi_smask);

        pred2_8x16b = _mm_shuffle_epi8(pred3_16x8b, frst_low_smask);

        pred3_8x16b = _mm_shuffle_epi8(pred3_16x8b, frst_hi_smask);

        /* residue calculation */
        res_0_1 = _mm_sub_epi16 (temp0_8x16b, pred0_8x16b);
        res_2_3 = _mm_sub_epi16 (temp1_8x16b, pred1_8x16b);
        res_5_4 = _mm_sub_epi16 (temp2_8x16b, pred2_8x16b);
        res_7_6 = _mm_sub_epi16 (temp3_8x16b, pred3_8x16b);

        /* end of residue calculation */

        /*e & o calculation */
        e0_e1_8x16b = _mm_add_epi16 (res_0_1, res_7_6);
        e3_e2_8x16b = _mm_add_epi16 (res_2_3, res_5_4);

        o0_o1_8x16b = _mm_sub_epi16 (res_0_1, res_7_6);
        o3_o2_8x16b = _mm_sub_epi16 (res_2_3, res_5_4);

        /* ee and eo */
        ee0_ee1_8x16b = _mm_adds_epi16 (e0_e1_8x16b, e3_e2_8x16b);
        eo0_eo1_8x16b = _mm_sub_epi16 (e0_e1_8x16b, e3_e2_8x16b);

        /* Even part Calculation and store */

        /* 64 64 64 64 64 64 64 64 */
        coeff0_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[0][0]);
        /* 83, 36, 83, 36, 83, 36, 83, 36 */
        coeff2_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[2][0]);
        /* 64, -64, 64, -64, 64, -64, 64, -64*/
        coeff4_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[4][0]);
        /* 36, -83, 36, -83, 36, -83, 36, -83 */
        coeff6_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[6][0]);

        /* row values */

        temp0_4x32b = _mm_madd_epi16(ee0_ee1_8x16b,coeff0_4x32b);
        temp2_4x32b = _mm_madd_epi16(eo0_eo1_8x16b,coeff2_4x32b);
        temp4_4x32b = _mm_madd_epi16(ee0_ee1_8x16b,coeff4_4x32b);
        temp6_4x32b = _mm_madd_epi16(eo0_eo1_8x16b,coeff6_4x32b);

        /* store to temp location */
        _mm_store_si128 ((__m128i *)pi4_temp, temp0_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 2*trans_size), temp2_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 4*trans_size), temp4_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 6*trans_size), temp6_4x32b);

        /* Odd part calculation */

        /* coeff3_4x32b = 75 75 75 75 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[1][0]);
        /* coeff4_4x32b = 18 18 18 18 */
        coeff3_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[3][0]);
        /* coeff5_4x32b = 89 89 89 89 */
        coeff5_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[5][0]);
        /* coeff6_4x32b = 50 50 50 50 */
        coeff7_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[7][0]);

        o_row_0_1 = _mm_unpacklo_epi32(o0_o1_8x16b,o3_o2_8x16b);

        o_row_2_3 = _mm_unpackhi_epi32(o0_o1_8x16b,o3_o2_8x16b);

        temp0_4x32b = _mm_madd_epi16(o_row_0_1,coeff1_4x32b);

        temp1_4x32b = _mm_madd_epi16(o_row_2_3,coeff1_4x32b);

        temp2_4x32b = _mm_madd_epi16(o_row_0_1,coeff3_4x32b);

        temp3_4x32b = _mm_madd_epi16(o_row_2_3,coeff3_4x32b);

        temp4_4x32b = _mm_madd_epi16(o_row_0_1,coeff5_4x32b);

        temp5_4x32b = _mm_madd_epi16(o_row_2_3,coeff5_4x32b);

        temp6_4x32b = _mm_madd_epi16(o_row_0_1,coeff7_4x32b);

        temp7_4x32b = _mm_madd_epi16(o_row_2_3,coeff7_4x32b);

        src1_4x32b = _mm_hadd_epi32(temp0_4x32b, temp1_4x32b);

        src3_4x32b = _mm_hadd_epi32(temp2_4x32b, temp3_4x32b);

        src5_4x32b = _mm_hadd_epi32(temp4_4x32b, temp5_4x32b);

        src7_4x32b = _mm_hadd_epi32(temp6_4x32b, temp7_4x32b);

        /* store to temp location */
        _mm_store_si128 ((__m128i *) (pi4_temp+   trans_size), src1_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 3*trans_size), src3_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 5*trans_size), src5_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 7*trans_size), src7_4x32b);

        /* point to next 8x4 block for storing */
        pi4_temp = pi4_temp + (trans_size >> 1) ;

        /* Luma : 8 src pels loaded to lower 64 bit, Chroma : 16 src pels loaded */ /* 4 times for 1st 4 rows */
        src0_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src1_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src2_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        src3_4x32b = _mm_loadu_si128 ((__m128i *)pu1_src);
        pu1_src = pu1_src + src_strd;
        /* Luma : 8 pred pels loaded to lower 64 bit, Chroma : 16 pred pels loaded */ /* 4 times for 1st 4 rows */
        pred0_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred1_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred2_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;
        pred3_4x32b = _mm_loadu_si128 ((__m128i *)pu1_pred);
        pu1_pred = pu1_pred + pred_strd;

        if (chroma_flag)
        {
            __m128i chroma_shuffle_mask_16x8b;
            chroma_shuffle_mask_16x8b = _mm_set_epi32 (0x0, 0x0, 0x0E0C0A08, 0x06040200);

            src0_4x32b  = _mm_shuffle_epi8 (src0_4x32b, chroma_shuffle_mask_16x8b);
            src1_4x32b  = _mm_shuffle_epi8 (src1_4x32b, chroma_shuffle_mask_16x8b);
            src2_4x32b  = _mm_shuffle_epi8 (src2_4x32b, chroma_shuffle_mask_16x8b);
            src3_4x32b  = _mm_shuffle_epi8 (src3_4x32b, chroma_shuffle_mask_16x8b);

            /* a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 */
            src0_16x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
            /* c0 d0 c1 d1 c2 d2 c3 d3 c4 d4 c5 d5 c6 d6 c7 d7 */
            src1_16x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);

            pred0_4x32b = _mm_shuffle_epi8 (pred0_4x32b, chroma_shuffle_mask_16x8b);
            pred1_4x32b = _mm_shuffle_epi8 (pred1_4x32b, chroma_shuffle_mask_16x8b);
            pred2_4x32b = _mm_shuffle_epi8 (pred2_4x32b, chroma_shuffle_mask_16x8b);
            pred3_4x32b = _mm_shuffle_epi8 (pred3_4x32b, chroma_shuffle_mask_16x8b);

            /* Same as cur */
            pred0_16x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);

            pred1_16x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);
        }
        else
        {
            /* SAD computation (4x8), taking only lower 64 bits of 128 (Luma part) */

            /* a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 */
            src0_16x8b = _mm_unpacklo_epi8(src0_4x32b, src1_4x32b);
            /* c0 d0 c1 d1 c2 d2 c3 d3 c4 d4 c5 d5 c6 d6 c7 d7 */
            src1_16x8b = _mm_unpacklo_epi8(src2_4x32b, src3_4x32b);

            /* Same as cur */
            pred0_16x8b = _mm_unpacklo_epi8(pred0_4x32b, pred1_4x32b);

            pred1_16x8b = _mm_unpacklo_epi8(pred2_4x32b, pred3_4x32b);

            sad0_4x32b = _mm_sad_epu8(src0_16x8b, pred0_16x8b);

            sad2_4x32b = _mm_sad_epu8(src1_16x8b, pred1_16x8b);

            sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);
            sad2_4x32b = _mm_srli_si128(sad0_4x32b, 8);

            sad0_4x32b = _mm_add_epi16 (sad0_4x32b, sad2_4x32b);
            sad += _mm_cvtsi128_si32 (sad0_4x32b);
            /* End of SAD (4x8) */
        }

        /* a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3*/
        temp0_16x8b = _mm_unpacklo_epi16(src0_16x8b, src1_16x8b);

        /* a4 b4 c4 d4 a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7*/
        temp1_16x8b = _mm_unpackhi_epi16(src0_16x8b, src1_16x8b);

        /* Same as cur values population */
        pred2_16x8b = _mm_unpacklo_epi16(pred0_16x8b, pred1_16x8b);

        pred3_16x8b = _mm_unpackhi_epi16(pred0_16x8b, pred1_16x8b);

        /* a0 a1 b0 b1 c0 c1 d0 d1 */
        temp0_8x16b = _mm_shuffle_epi8(temp0_16x8b, frst_low_smask);

        /* a3 a2 b3 b2 c3 c2 d3 d2 */
        temp1_8x16b = _mm_shuffle_epi8(temp0_16x8b, frst_hi_smask);

        /* a4 a5 b4 b5 c4 c5 d4 d5 */
        temp2_8x16b = _mm_shuffle_epi8(temp1_16x8b, frst_low_smask);

        /* a7 a6 b7 b6 c7 c6 d7 d6 */
        temp3_8x16b = _mm_shuffle_epi8(temp1_16x8b, frst_hi_smask);

        /* Same as Cur values */
        pred0_8x16b = _mm_shuffle_epi8(pred2_16x8b, frst_low_smask);

        pred1_8x16b = _mm_shuffle_epi8(pred2_16x8b, frst_hi_smask);

        pred2_8x16b = _mm_shuffle_epi8(pred3_16x8b, frst_low_smask);

        pred3_8x16b = _mm_shuffle_epi8(pred3_16x8b, frst_hi_smask);

        /* residue calculation */
        res_0_1 = _mm_sub_epi16 (temp0_8x16b, pred0_8x16b);
        res_2_3 = _mm_sub_epi16 (temp1_8x16b, pred1_8x16b);
        res_5_4 = _mm_sub_epi16 (temp2_8x16b, pred2_8x16b);
        res_7_6 = _mm_sub_epi16 (temp3_8x16b, pred3_8x16b);

        /* end of residue calculation */

        /*e & o calculation */
        e0_e1_8x16b = _mm_add_epi16 (res_0_1, res_7_6);
        e3_e2_8x16b = _mm_add_epi16 (res_2_3, res_5_4);

        o0_o1_8x16b = _mm_sub_epi16 (res_0_1, res_7_6);
        o3_o2_8x16b = _mm_sub_epi16 (res_2_3, res_5_4);

        /* ee and eo */
        ee0_ee1_8x16b = _mm_add_epi16 (e0_e1_8x16b, e3_e2_8x16b);
        eo0_eo1_8x16b = _mm_sub_epi16 (e0_e1_8x16b, e3_e2_8x16b);
        /* Even part Calculation and store */

        /* 64 64 64 64 64 64 64 64 */
        coeff0_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[0][0]);
        /* 83, 36, 83, 36, 83, 36, 83, 36 */
        coeff2_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[2][0]);
        /* 64, -64, 64, -64, 64, -64, 64, -64*/
        coeff4_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[4][0]);
        /* 36, -83, 36, -83, 36, -83, 36, -83 */
        coeff6_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[6][0]);

        /* row values */

        temp0_4x32b = _mm_madd_epi16(ee0_ee1_8x16b,coeff0_4x32b);
        temp2_4x32b = _mm_madd_epi16(eo0_eo1_8x16b,coeff2_4x32b);
        temp4_4x32b = _mm_madd_epi16(ee0_ee1_8x16b,coeff4_4x32b);
        temp6_4x32b = _mm_madd_epi16(eo0_eo1_8x16b,coeff6_4x32b);

        /* store to temp location */
        _mm_store_si128 ((__m128i *)pi4_temp, temp0_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 2*trans_size), temp2_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 4*trans_size), temp4_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 6*trans_size), temp6_4x32b);

        /* Odd part calculation */

        /* coeff3_4x32b = 75 75 75 75 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[1][0]);
        /* coeff4_4x32b = 18 18 18 18 */
        coeff3_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[3][0]);
        /* coeff5_4x32b = 89 89 89 89 */
        coeff5_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[5][0]);
        /* coeff6_4x32b = 50 50 50 50 */
        coeff7_4x32b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_8_intr[7][0]);

        o_row_0_1 = _mm_unpacklo_epi32(o0_o1_8x16b,o3_o2_8x16b);

        o_row_2_3 = _mm_unpackhi_epi32(o0_o1_8x16b,o3_o2_8x16b);

        temp0_4x32b = _mm_madd_epi16(o_row_0_1,coeff1_4x32b);

        temp1_4x32b = _mm_madd_epi16(o_row_2_3,coeff1_4x32b);

        temp2_4x32b = _mm_madd_epi16(o_row_0_1,coeff3_4x32b);

        temp3_4x32b = _mm_madd_epi16(o_row_2_3,coeff3_4x32b);

        temp4_4x32b = _mm_madd_epi16(o_row_0_1,coeff5_4x32b);

        temp5_4x32b = _mm_madd_epi16(o_row_2_3,coeff5_4x32b);

        temp6_4x32b = _mm_madd_epi16(o_row_0_1,coeff7_4x32b);

        temp7_4x32b = _mm_madd_epi16(o_row_2_3,coeff7_4x32b);

        src1_4x32b = _mm_hadd_epi32(temp0_4x32b, temp1_4x32b);

        src3_4x32b = _mm_hadd_epi32(temp2_4x32b, temp3_4x32b);

        src5_4x32b = _mm_hadd_epi32(temp4_4x32b, temp5_4x32b);

        src7_4x32b = _mm_hadd_epi32(temp6_4x32b, temp7_4x32b);

        /* store to temp location */
        _mm_store_si128 ((__m128i *) (pi4_temp+   trans_size), src1_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 3*trans_size), src3_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 5*trans_size), src5_4x32b);
        _mm_store_si128 ((__m128i *) (pi4_temp+ 7*trans_size), src7_4x32b);
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    {
        __m128i add_4x32b;

        trans_size = trans_size >> 1;

        add = 1 << ((SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st) - 1);

        /* 8 temp values loaded */ /* 8 times for 1st 4 rows */
        src0_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src4_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src1_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src5_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src2_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src6_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src3_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src7_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;

        /***************************  4x8 16bit Transpose ************************/
        /* b3 a3 b2 a2 b1 a1 b0 a0 */
        temp0_4x32b = _mm_unpacklo_epi32(src0_4x32b, src1_4x32b);
        /* d3 c3 d2 c2 d1 c1 d0 c0 */
        temp1_4x32b = _mm_unpacklo_epi32(src2_4x32b, src3_4x32b);
        /* b7 a7 b6 a6 b5 a5 b4 a4 */
        temp2_4x32b = _mm_unpackhi_epi32(src0_4x32b, src1_4x32b);
        /* d7 c7 d6 c6 d5 c5 d4 c4 */
        temp3_4x32b = _mm_unpackhi_epi32(src2_4x32b, src3_4x32b);

        /* d1 c1 b1 a1 d0 c0 b0 a0 */
        src0_4x32b = _mm_unpacklo_epi64(temp0_4x32b, temp1_4x32b);
        /* d3 c3 b3 a3 d2 c2 b2 a2 */
        src1_4x32b = _mm_unpackhi_epi64(temp0_4x32b, temp1_4x32b);
        /* d5 c5 b5 a5 d4 c4 b4 a4 */
        src2_4x32b = _mm_unpacklo_epi64(temp2_4x32b, temp3_4x32b);
        /* d7 c7 b7 a7 d6 c6 b6 a6 */
        src3_4x32b = _mm_unpackhi_epi64(temp2_4x32b, temp3_4x32b);

        /* b3 a3 b2 a2 b1 a1 b0 a0 */
        temp0_4x32b = _mm_unpacklo_epi32(src4_4x32b, src5_4x32b);
        /* d3 c3 d2 c2 d1 c1 d0 c0 */
        temp1_4x32b = _mm_unpacklo_epi32(src6_4x32b, src7_4x32b);
        /* b7 a7 b6 a6 b5 a5 b4 a4 */
        temp2_4x32b = _mm_unpackhi_epi32(src4_4x32b, src5_4x32b);
        /* d7 c7 d6 c6 d5 c5 d4 c4 */
        temp3_4x32b = _mm_unpackhi_epi32(src6_4x32b, src7_4x32b);

        /* d1 c1 b1 a1 d0 c0 b0 a0 */
        src4_4x32b = _mm_unpacklo_epi64(temp0_4x32b, temp1_4x32b);
        /* d3 c3 b3 a3 d2 c2 b2 a2 */
        src5_4x32b = _mm_unpackhi_epi64(temp0_4x32b, temp1_4x32b);
        /* d5 c5 b5 a5 d4 c4 b4 a4 */
        src6_4x32b = _mm_unpacklo_epi64(temp2_4x32b, temp3_4x32b);
        /* d7 c7 b7 a7 d6 c6 b6 a6 */
        src7_4x32b = _mm_unpackhi_epi64(temp2_4x32b, temp3_4x32b);


        /* coeff1_4x32b = 83 83 83 83 */
        coeff1_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[1][0]);
        /* coeff2_4x32b = 36 36 36 36 */
        coeff2_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[2][0]);

        /* coeff3_4x32b = 75 75 75 75 */
        coeff3_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[3][0]);
        /* coeff4_4x32b = 18 18 18 18 */
        coeff4_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[4][0]);
        /* coeff5_4x32b = 89 89 89 89 */
        coeff5_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[5][0]);
        /* coeff6_4x32b = 50 50 50 50 */
        coeff6_4x32b = _mm_loadu_si128( (__m128i *) &g_ai4_ihevc_trans_8_intr[6][0]);

        /*e & o calculation */
        e0_4x32b = _mm_add_epi32 (src0_4x32b, src7_4x32b);
        o0_4x32b = _mm_sub_epi32 (src0_4x32b, src7_4x32b);
        e1_4x32b = _mm_add_epi32 (src1_4x32b, src6_4x32b);
        o1_4x32b = _mm_sub_epi32 (src1_4x32b, src6_4x32b);
        e2_4x32b = _mm_add_epi32 (src2_4x32b, src5_4x32b);
        o2_4x32b = _mm_sub_epi32 (src2_4x32b, src5_4x32b);
        e3_4x32b = _mm_add_epi32 (src3_4x32b, src4_4x32b);
        o3_4x32b = _mm_sub_epi32 (src3_4x32b, src4_4x32b);

        /* ee and eo */
        ee0_4x32b = _mm_add_epi32 (e0_4x32b, e3_4x32b);
        eo0_4x32b = _mm_sub_epi32 (e0_4x32b, e3_4x32b);
        ee1_4x32b = _mm_add_epi32 (e1_4x32b, e2_4x32b);
        eo1_4x32b = _mm_sub_epi32 (e1_4x32b, e2_4x32b);

        /* Even part */
        temp0_4x32b = _mm_add_epi32 (ee0_4x32b, ee1_4x32b); /* ee0+ee1 */
        temp4_4x32b = _mm_sub_epi32 (ee0_4x32b, ee1_4x32b); /* ee0-ee1 */
        temp2_4x32b = _mm_mullo_epi32 (eo0_4x32b, coeff1_4x32b); /* 83*eo0 */
        temp6_4x32b = _mm_mullo_epi32 (eo0_4x32b, coeff2_4x32b); /* 36*eo0 */
        src2_4x32b  = _mm_mullo_epi32 (eo1_4x32b, coeff2_4x32b); /* 36*eo1 */
        src6_4x32b  = _mm_mullo_epi32 (eo1_4x32b, coeff1_4x32b); /* 83*eo1 */

        /* add value */
        add_4x32b = _mm_set1_epi32(add);

        temp0_4x32b = _mm_slli_epi32 (temp0_4x32b, 6); /* *64 */
        temp4_4x32b = _mm_slli_epi32 (temp4_4x32b, 6); /* *64 */
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, src2_4x32b);
        temp6_4x32b = _mm_sub_epi32 (temp6_4x32b, src6_4x32b);

        /* result + add */
        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, add_4x32b);
        temp4_4x32b = _mm_add_epi32 (temp4_4x32b, add_4x32b);
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, add_4x32b);
        temp6_4x32b = _mm_add_epi32 (temp6_4x32b, add_4x32b);

        /* result >> shift */
        temp0_4x32b = _mm_srai_epi32 (temp0_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp4_4x32b = _mm_srai_epi32 (temp4_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp2_4x32b = _mm_srai_epi32 (temp2_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp6_4x32b = _mm_srai_epi32 (temp6_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));

        /* 32-16 bit conversion */
        temp0_4x32b = _mm_packs_epi32 (temp0_4x32b, temp0_4x32b);
        temp4_4x32b = _mm_packs_epi32 (temp4_4x32b, temp4_4x32b);
        temp2_4x32b = _mm_packs_epi32 (temp2_4x32b, temp2_4x32b);
        temp6_4x32b = _mm_packs_epi32 (temp6_4x32b, temp6_4x32b);

        /* store to final location */
        _mm_storel_epi64 ((__m128i *)pi2_dst, temp0_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 4*dst_strd), temp4_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 2*dst_strd), temp2_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 6*dst_strd), temp6_4x32b);

        /* Odd part */
        temp1_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff5_4x32b); /* 89*o0 */
        src1_4x32b = _mm_mullo_epi32 (o1_4x32b, coeff3_4x32b); /* 75*o1 */
        temp3_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff3_4x32b); /* 75*o0 */
        src3_4x32b = _mm_mullo_epi32 (o1_4x32b, coeff4_4x32b); /* 18*o1 */
        temp5_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff6_4x32b); /* 50*o0 */
        src5_4x32b = _mm_mullo_epi32 (o1_4x32b, coeff5_4x32b); /* 89*o1 */
        temp7_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff4_4x32b); /* 18*o0 */
        src7_4x32b = _mm_mullo_epi32 (o1_4x32b, coeff6_4x32b); /* 50*o1 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_sub_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_sub_epi32 (temp7_4x32b, src7_4x32b);

        src1_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff6_4x32b); /* 50*o2 */
        src3_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff5_4x32b); /* 89*o2 */
        src5_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff4_4x32b); /* 18*o2 */
        src7_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff3_4x32b); /* 75*o2 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_add_epi32 (temp7_4x32b, src7_4x32b);

        src1_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff4_4x32b); /* 18*o3 */
        src3_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff6_4x32b); /* 50*o3 */
        src5_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff3_4x32b); /* 75*o3 */
        src7_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff5_4x32b); /* 89*o3 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_sub_epi32 (temp7_4x32b, src7_4x32b);

        /* result + add */
        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, add_4x32b);
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, add_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, add_4x32b);
        temp7_4x32b = _mm_add_epi32 (temp7_4x32b, add_4x32b);

        /* result >> shift */
        temp1_4x32b = _mm_srai_epi32 (temp1_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp3_4x32b = _mm_srai_epi32 (temp3_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp5_4x32b = _mm_srai_epi32 (temp5_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp7_4x32b = _mm_srai_epi32 (temp7_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));

        /* 32-16 bit conversion */
        temp1_4x32b = _mm_packs_epi32 (temp1_4x32b, temp1_4x32b);
        temp3_4x32b = _mm_packs_epi32 (temp3_4x32b, temp3_4x32b);
        temp5_4x32b = _mm_packs_epi32 (temp5_4x32b, temp5_4x32b);
        temp7_4x32b = _mm_packs_epi32 (temp7_4x32b, temp7_4x32b);

        /* store to final location */
        _mm_storel_epi64 ((__m128i *) (pi2_dst+   dst_strd), temp1_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 3*dst_strd), temp3_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 5*dst_strd), temp5_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 7*dst_strd), temp7_4x32b);

        /* point to next 8x4 block for storing */
        pi2_dst = pi2_dst + trans_size;

        /* 8 temp values loaded */ /* 8 times for 1st 4 rows */
        src0_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src4_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src1_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src5_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src2_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src6_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src3_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;
        src7_4x32b = _mm_loadu_si128 ((__m128i *)pi4_temp);
        pi4_temp = pi4_temp + trans_size;

        /***************************  4x8 16bit Transpose ************************/
        /* b3 a3 b2 a2 b1 a1 b0 a0 */
        temp0_4x32b = _mm_unpacklo_epi32(src0_4x32b, src1_4x32b);
        /* d3 c3 d2 c2 d1 c1 d0 c0 */
        temp1_4x32b = _mm_unpacklo_epi32(src2_4x32b, src3_4x32b);
        /* b7 a7 b6 a6 b5 a5 b4 a4 */
        temp2_4x32b = _mm_unpackhi_epi32(src0_4x32b, src1_4x32b);
        /* d7 c7 d6 c6 d5 c5 d4 c4 */
        temp3_4x32b = _mm_unpackhi_epi32(src2_4x32b, src3_4x32b);

        /* d1 c1 b1 a1 d0 c0 b0 a0 */
        src0_4x32b = _mm_unpacklo_epi64(temp0_4x32b, temp1_4x32b);
        /* d3 c3 b3 a3 d2 c2 b2 a2 */
        src1_4x32b = _mm_unpackhi_epi64(temp0_4x32b, temp1_4x32b);
        /* d5 c5 b5 a5 d4 c4 b4 a4 */
        src2_4x32b = _mm_unpacklo_epi64(temp2_4x32b, temp3_4x32b);
        /* d7 c7 b7 a7 d6 c6 b6 a6 */
        src3_4x32b = _mm_unpackhi_epi64(temp2_4x32b, temp3_4x32b);

        /* b3 a3 b2 a2 b1 a1 b0 a0 */
        temp0_4x32b = _mm_unpacklo_epi32(src4_4x32b, src5_4x32b);
        /* d3 c3 d2 c2 d1 c1 d0 c0 */
        temp1_4x32b = _mm_unpacklo_epi32(src6_4x32b, src7_4x32b);
        /* b7 a7 b6 a6 b5 a5 b4 a4 */
        temp2_4x32b = _mm_unpackhi_epi32(src4_4x32b, src5_4x32b);
        /* d7 c7 d6 c6 d5 c5 d4 c4 */
        temp3_4x32b = _mm_unpackhi_epi32(src6_4x32b, src7_4x32b);

        /* d1 c1 b1 a1 d0 c0 b0 a0 */
        src4_4x32b = _mm_unpacklo_epi64(temp0_4x32b, temp1_4x32b);
        /* d3 c3 b3 a3 d2 c2 b2 a2 */
        src5_4x32b = _mm_unpackhi_epi64(temp0_4x32b, temp1_4x32b);
        /* d5 c5 b5 a5 d4 c4 b4 a4 */
        src6_4x32b = _mm_unpacklo_epi64(temp2_4x32b, temp3_4x32b);
        /* d7 c7 b7 a7 d6 c6 b6 a6 */
        src7_4x32b = _mm_unpackhi_epi64(temp2_4x32b, temp3_4x32b);
        /**************************  4x8 Transpose End   *************************/

        /*e & o calculation */
        e0_4x32b = _mm_add_epi32 (src0_4x32b, src7_4x32b);
        o0_4x32b = _mm_sub_epi32 (src0_4x32b, src7_4x32b);
        e1_4x32b = _mm_add_epi32 (src1_4x32b, src6_4x32b);
        o1_4x32b = _mm_sub_epi32 (src1_4x32b, src6_4x32b);
        e2_4x32b = _mm_add_epi32 (src2_4x32b, src5_4x32b);
        o2_4x32b = _mm_sub_epi32 (src2_4x32b, src5_4x32b);
        e3_4x32b = _mm_add_epi32 (src3_4x32b, src4_4x32b);
        o3_4x32b = _mm_sub_epi32 (src3_4x32b, src4_4x32b);

        /* ee and eo */
        ee0_4x32b = _mm_add_epi32 (e0_4x32b, e3_4x32b);
        eo0_4x32b = _mm_sub_epi32 (e0_4x32b, e3_4x32b);
        ee1_4x32b = _mm_add_epi32 (e1_4x32b, e2_4x32b);
        eo1_4x32b = _mm_sub_epi32 (e1_4x32b, e2_4x32b);

        /* Even part */
        temp0_4x32b = _mm_add_epi32 (ee0_4x32b, ee1_4x32b); /* ee0+ee1 */
        temp4_4x32b = _mm_sub_epi32 (ee0_4x32b, ee1_4x32b); /* ee0-ee1 */
        temp2_4x32b = _mm_mullo_epi32 (eo0_4x32b, coeff1_4x32b); /* 83*eo0 */
        temp6_4x32b = _mm_mullo_epi32 (eo0_4x32b, coeff2_4x32b); /* 36*eo0 */
        src2_4x32b  = _mm_mullo_epi32 (eo1_4x32b, coeff2_4x32b); /* 36*eo1 */
        src6_4x32b  = _mm_mullo_epi32 (eo1_4x32b, coeff1_4x32b); /* 83*eo1 */

        temp0_4x32b = _mm_slli_epi32 (temp0_4x32b, 6); /* *64 */
        temp4_4x32b = _mm_slli_epi32 (temp4_4x32b, 6); /* *64 */
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, src2_4x32b);
        temp6_4x32b = _mm_sub_epi32 (temp6_4x32b, src6_4x32b);

        /* result + add */
        temp0_4x32b = _mm_add_epi32 (temp0_4x32b, add_4x32b);
        temp4_4x32b = _mm_add_epi32 (temp4_4x32b, add_4x32b);
        temp2_4x32b = _mm_add_epi32 (temp2_4x32b, add_4x32b);
        temp6_4x32b = _mm_add_epi32 (temp6_4x32b, add_4x32b);

        /* result >> shift */
        temp0_4x32b = _mm_srai_epi32 (temp0_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp4_4x32b = _mm_srai_epi32 (temp4_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp2_4x32b = _mm_srai_epi32 (temp2_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp6_4x32b = _mm_srai_epi32 (temp6_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));

        /* 32-16 bit conversion */
        temp0_4x32b = _mm_packs_epi32 (temp0_4x32b, temp0_4x32b);
        temp4_4x32b = _mm_packs_epi32 (temp4_4x32b, temp4_4x32b);
        temp2_4x32b = _mm_packs_epi32 (temp2_4x32b, temp2_4x32b);
        temp6_4x32b = _mm_packs_epi32 (temp6_4x32b, temp6_4x32b);

        /* store to final location */
        _mm_storel_epi64 ((__m128i *)pi2_dst, temp0_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 4*dst_strd), temp4_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 2*dst_strd), temp2_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 6*dst_strd), temp6_4x32b);

        /* Odd part */
        temp1_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff5_4x32b); /* 89*o0 */
        src1_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff3_4x32b); /* 75*o1 */
        temp3_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff3_4x32b); /* 75*o0 */
        src3_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff4_4x32b); /* 18*o1 */
        temp5_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff6_4x32b); /* 50*o0 */
        src5_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff5_4x32b); /* 89*o1 */
        temp7_4x32b = _mm_mullo_epi32 (o0_4x32b, coeff4_4x32b); /* 18*o0 */
        src7_4x32b  = _mm_mullo_epi32 (o1_4x32b, coeff6_4x32b); /* 50*o1 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_sub_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_sub_epi32 (temp7_4x32b, src7_4x32b);

        src1_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff6_4x32b); /* 50*o2 */
        src3_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff5_4x32b); /* 89*o2 */
        src5_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff4_4x32b); /* 18*o2 */
        src7_4x32b = _mm_mullo_epi32 (o2_4x32b, coeff3_4x32b); /* 75*o2 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_add_epi32 (temp7_4x32b, src7_4x32b);

        src1_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff4_4x32b); /* 18*o3 */
        src3_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff6_4x32b); /* 50*o3 */
        src5_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff3_4x32b); /* 75*o3 */
        src7_4x32b = _mm_mullo_epi32 (o3_4x32b, coeff5_4x32b); /* 89*o3 */

        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, src1_4x32b);
        temp3_4x32b = _mm_sub_epi32 (temp3_4x32b, src3_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, src5_4x32b);
        temp7_4x32b = _mm_sub_epi32 (temp7_4x32b, src7_4x32b);

        /* result + add */
        temp1_4x32b = _mm_add_epi32 (temp1_4x32b, add_4x32b);
        temp3_4x32b = _mm_add_epi32 (temp3_4x32b, add_4x32b);
        temp5_4x32b = _mm_add_epi32 (temp5_4x32b, add_4x32b);
        temp7_4x32b = _mm_add_epi32 (temp7_4x32b, add_4x32b);

        /* result >> shift */
        temp1_4x32b = _mm_srai_epi32 (temp1_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp3_4x32b = _mm_srai_epi32 (temp3_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp5_4x32b = _mm_srai_epi32 (temp5_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));
        temp7_4x32b = _mm_srai_epi32 (temp7_4x32b, (SHIFT_DCT_8_2nd + SHIFT_DCT_8_1st));

        /* 32-16 bit conversion */
        temp1_4x32b = _mm_packs_epi32 (temp1_4x32b, temp1_4x32b);
        temp3_4x32b = _mm_packs_epi32 (temp3_4x32b, temp3_4x32b);
        temp5_4x32b = _mm_packs_epi32 (temp5_4x32b, temp5_4x32b);
        temp7_4x32b = _mm_packs_epi32 (temp7_4x32b, temp7_4x32b);

        /* store to final location */
        _mm_storel_epi64 ((__m128i *) (pi2_dst+   dst_strd), temp1_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 3*dst_strd), temp3_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 5*dst_strd), temp5_4x32b);
        _mm_storel_epi64 ((__m128i *) (pi2_dst+ 7*dst_strd), temp7_4x32b);
    }

    return sad;
}



void ihevc_resi_trans_8x8_16bit_sse42(WORD16 *pi2_src,
                          UWORD8 *pu1_pred,
                          WORD16 *pi2_tmp,
                          WORD16 *pi2_dst,
                          WORD32 src_strd,
                          WORD32 pred_strd,
                          WORD32 dst_strd)
{

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
 * @param[in] pu1_src
 *  Input 16x16 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 16x16
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *  0 - luma transform, 1 - chroma transform.
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_resi_trans_16x16_sse42(UWORD8 *pu1_src,
                            UWORD8 *pu1_pred,
                               WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 src_strd,
                            WORD32 pred_strd,
                               WORD32 dst_strd_chr_flag)
{
    WORD32 i;
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad = 0;
    WORD32 *o_temp1_ptr;
    WORD32 chroma_flag;
    WORD32 dst_strd;

    WORD32 *g_ai2_ihevc_trans_32_intr_8_ptr;
    WORD32 *g_ai2_ihevc_trans_32_intr_16_ptr;

    __m128i m_temp_reg_0, m_temp_reg_1, m_temp_reg_2, m_temp_reg_3, m_temp_reg_4, m_temp_reg_5, m_temp_reg_6, m_temp_reg_7;
    __m128i m_temp_reg_8, m_temp_reg_9, m_temp_reg_10, m_temp_reg_11, m_temp_reg_12, m_temp_reg_13, m_temp_reg_14,m_temp_reg_15;

        __m128i m_temp_reg_16, m_temp_reg_17;
        __m128i temp0_8x16b, temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b, temp5_8x16b, temp6_8x16b, temp7_8x16b;
        __m128i temp10_8x16b, temp11_8x16b, temp12_8x16b, temp13_8x16b, temp14_8x16b, temp15_8x16b, temp16_8x16b, temp17_8x16b;

        __m128i reg_e0_e1, reg_e3_e2, reg_o0_o1, reg_o3_o2, reg_e4_e5, reg_e7_e6, reg_o4_o5, reg_o7_o6;
        __m128i temp_res0, temp_res1, temp_res2, temp_res3, temp_res4, temp_res5, temp_res6, temp_res7;

        __m128i res_r0_r1, res_r3_r2, res_r4_r5, res_r7_r6, res_r8_r9, res_r11_r10, res_r12_r13, res_r15_r14;
        __m128i frst_low_smask, frst_hi_smask;

           __m128i coeff0_8x16b, coeff1_8x16b, coeff2_8x16b, coeff3_8x16b;
            __m128i coeff4_8x16b, coeff5_8x16b, coeff6_8x16b, coeff7_8x16b, coeff8_8x16b;

            __m128i coeff10_8x16b, coeff11_8x16b, coeff12_8x16b, coeff13_8x16b;
            __m128i coeff14_8x16b, coeff15_8x16b, coeff16_8x16b, coeff17_8x16b;


        __m128i src0_4x32b, src2_4x32b, src4_4x32b, src6_4x32b, src8_4x32b, src10_4x32b, src12_4x32b, src14_4x32b;
        __m128i temp2_4x32b, temp3_4x32b, temp6_4x32b, temp7_4x32b, temp10_4x32b, temp11_4x32b, temp14_4x32b, temp15_4x32b;

        __m128i reg_ee0_ee1, reg_eo0_eo1, reg_ee3_ee2, reg_eo3_eo2;
        __m128i reg_eee0_eee1, reg_eeo0_eeo1;

        __m128i chroma_shuffle_mask_16x8b;

    chroma_flag = dst_strd_chr_flag & 1;
    dst_strd = dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_16;

    /* Residue + Forward Transform 1st stage */
    shift = 3; // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    o_temp1_ptr = pi4_temp + (16 * 16);

    g_ai2_ihevc_trans_32_intr_8_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_8;
    g_ai2_ihevc_trans_32_intr_16_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_16;

    for(i = 0; i < TRANS_SIZE_16; i +=4)
    {
        /* row =0 */
        m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pu1_src));      /* k = 0-8  */
        m_temp_reg_1 = _mm_loadu_si128((__m128i*)(pu1_pred));     /* k = 0-8  */

        /* row =1 */
        m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pu1_src+src_strd));       /* k = 0-8   */
        m_temp_reg_3 = _mm_loadu_si128((__m128i*)(pu1_pred+pred_strd));     /* k = 0-8   */

        /* row =2 */
        m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pu1_src+2*src_strd));       /* k = 0-8   */
        m_temp_reg_5 = _mm_loadu_si128((__m128i*)(pu1_pred+2*pred_strd));     /* k = 0-8   */

        /* row =3 */
        m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pu1_src+3*src_strd));       /* k = 0-8   */
        m_temp_reg_7 = _mm_loadu_si128((__m128i*)(pu1_pred+3*pred_strd));     /* k = 0-8   */

        if (chroma_flag)
        {

            chroma_shuffle_mask_16x8b = _mm_set_epi32 (0x0, 0x0, 0x0E0C0A08, 0x06040200);

            /* row =0 */
            m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pu1_src + 16));      /* k = 0-8  */
            m_temp_reg_11 = _mm_loadu_si128((__m128i*)(pu1_pred + 16));     /* k = 0-8  */

            /* row =1 */
            m_temp_reg_12 = _mm_loadu_si128((__m128i*)(pu1_src+src_strd + 16));       /* k = 0-8   */
            m_temp_reg_13 = _mm_loadu_si128((__m128i*)(pu1_pred+pred_strd + 16));     /* k = 0-8   */

            /* row =2 */
            m_temp_reg_14 = _mm_loadu_si128((__m128i*)(pu1_src+2*src_strd + 16));       /* k = 0-8   */
            m_temp_reg_15 = _mm_loadu_si128((__m128i*)(pu1_pred+2*pred_strd + 16));     /* k = 0-8   */

            /* row =3 */
            m_temp_reg_16 = _mm_loadu_si128((__m128i*)(pu1_src+3*src_strd + 16));       /* k = 0-8   */
            m_temp_reg_17 = _mm_loadu_si128((__m128i*)(pu1_pred+3*pred_strd + 16));     /* k = 0-8   */

            /* CUr values packing into 1 register for row = 0,1,2,3 */
            m_temp_reg_0 = _mm_shuffle_epi8 (m_temp_reg_0, chroma_shuffle_mask_16x8b);
            m_temp_reg_10 = _mm_shuffle_epi8 (m_temp_reg_10, chroma_shuffle_mask_16x8b);

            m_temp_reg_0 = _mm_unpacklo_epi64 (m_temp_reg_0, m_temp_reg_10);

            m_temp_reg_2 = _mm_shuffle_epi8 (m_temp_reg_2, chroma_shuffle_mask_16x8b);
            m_temp_reg_12 = _mm_shuffle_epi8 (m_temp_reg_12, chroma_shuffle_mask_16x8b);

            m_temp_reg_2 = _mm_unpacklo_epi64 (m_temp_reg_2, m_temp_reg_12);

            m_temp_reg_4 = _mm_shuffle_epi8 (m_temp_reg_4, chroma_shuffle_mask_16x8b);
            m_temp_reg_14 = _mm_shuffle_epi8 (m_temp_reg_14, chroma_shuffle_mask_16x8b);

            m_temp_reg_4 = _mm_unpacklo_epi64 (m_temp_reg_4, m_temp_reg_14);

            m_temp_reg_6 = _mm_shuffle_epi8 (m_temp_reg_6, chroma_shuffle_mask_16x8b);
            m_temp_reg_16 = _mm_shuffle_epi8 (m_temp_reg_16, chroma_shuffle_mask_16x8b);

            m_temp_reg_6 = _mm_unpacklo_epi64 (m_temp_reg_6, m_temp_reg_16);

            /* Pred values packing into 1 register for row = 0,1,2,3 */
            m_temp_reg_1 = _mm_shuffle_epi8 (m_temp_reg_1, chroma_shuffle_mask_16x8b);
            m_temp_reg_11 = _mm_shuffle_epi8 (m_temp_reg_11, chroma_shuffle_mask_16x8b);

            m_temp_reg_1 = _mm_unpacklo_epi64 (m_temp_reg_1, m_temp_reg_11);

            m_temp_reg_3 = _mm_shuffle_epi8 (m_temp_reg_3, chroma_shuffle_mask_16x8b);
            m_temp_reg_13 = _mm_shuffle_epi8 (m_temp_reg_13, chroma_shuffle_mask_16x8b);

            m_temp_reg_3 = _mm_unpacklo_epi64 (m_temp_reg_3, m_temp_reg_13);

            m_temp_reg_5 = _mm_shuffle_epi8 (m_temp_reg_5, chroma_shuffle_mask_16x8b);
            m_temp_reg_15 = _mm_shuffle_epi8 (m_temp_reg_15, chroma_shuffle_mask_16x8b);

            m_temp_reg_5 = _mm_unpacklo_epi64 (m_temp_reg_5, m_temp_reg_15);

            m_temp_reg_7 = _mm_shuffle_epi8 (m_temp_reg_7, chroma_shuffle_mask_16x8b);
            m_temp_reg_17 = _mm_shuffle_epi8 (m_temp_reg_17, chroma_shuffle_mask_16x8b);

            m_temp_reg_7 = _mm_unpacklo_epi64 (m_temp_reg_7, m_temp_reg_17);
        }
        else
        {
            /* SAD Computation */
            m_temp_reg_8 = _mm_sad_epu8(m_temp_reg_0, m_temp_reg_1);

            m_temp_reg_9 = _mm_sad_epu8(m_temp_reg_2, m_temp_reg_3);

            m_temp_reg_8 = _mm_add_epi32(m_temp_reg_8, m_temp_reg_9);

            m_temp_reg_9 = _mm_srli_si128(m_temp_reg_8, 8);

            m_temp_reg_8 = _mm_add_epi32(m_temp_reg_8, m_temp_reg_9);

            u4_blk_sad += _mm_cvtsi128_si32 (m_temp_reg_8);

            m_temp_reg_10 = _mm_sad_epu8(m_temp_reg_4, m_temp_reg_5);

            m_temp_reg_11 = _mm_sad_epu8(m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);

            m_temp_reg_11 = _mm_srli_si128(m_temp_reg_10, 8);

            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);

            u4_blk_sad += _mm_cvtsi128_si32 (m_temp_reg_10);
        }

        frst_low_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK_16x16_TRNS_REORDER1[0]);

        frst_hi_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK_16x16_TRNS_REORDER2[0]);

        /* row 0 */
        /* a0 a1 a3 a2 a4 a5 a7 a6 */
        temp0_8x16b = _mm_shuffle_epi8(m_temp_reg_0, frst_low_smask);

        /* a8 a9 a11 a10 a12 a13 a15 a14 */
        temp1_8x16b = _mm_shuffle_epi8(m_temp_reg_0, frst_hi_smask);

        /* row 1 */
        temp2_8x16b = _mm_shuffle_epi8(m_temp_reg_2, frst_low_smask);

        temp3_8x16b = _mm_shuffle_epi8(m_temp_reg_2, frst_hi_smask);

        /* Pred values population same as cur */
        /* pred row 0 */
        temp10_8x16b = _mm_shuffle_epi8(m_temp_reg_1, frst_low_smask);

        temp11_8x16b = _mm_shuffle_epi8(m_temp_reg_1, frst_hi_smask);
        /* pred row 1 */
        temp12_8x16b = _mm_shuffle_epi8(m_temp_reg_3, frst_low_smask);

        temp13_8x16b = _mm_shuffle_epi8(m_temp_reg_3, frst_hi_smask);

        /* residue values */

        /* row 0 */
        /* r0 r1 r3 r2 r4 r5 r7 r6 */
        temp_res0 = _mm_sub_epi16(temp0_8x16b, temp10_8x16b);

        /* r8 r9 r11 r10 r12 r13 r15 r14 */
        temp_res1 = _mm_sub_epi16(temp1_8x16b, temp11_8x16b);

        /* row 1 */
        temp_res2 = _mm_sub_epi16(temp2_8x16b, temp12_8x16b);

        temp_res3 = _mm_sub_epi16(temp3_8x16b, temp13_8x16b);

        /* r0 r1 r0 r1 r3 r2 r3 r2 */
        temp0_8x16b = _mm_unpacklo_epi32(temp_res0, temp_res2);
        /* r4 r5 r4 r5 r7 r6 r7 r6 */
        temp1_8x16b = _mm_unpackhi_epi32(temp_res0, temp_res2);

        /* r8 r9 r8 r9 r11 r10 r11 r10 */
        temp2_8x16b = _mm_unpacklo_epi32(temp_res1, temp_res3);
        /* r12 r13 r12 r13 r15 r14 r15 r14 */
        temp3_8x16b = _mm_unpackhi_epi32(temp_res1, temp_res3);

        /* row 2 */
        temp4_8x16b = _mm_shuffle_epi8(m_temp_reg_4, frst_low_smask);

        temp5_8x16b = _mm_shuffle_epi8(m_temp_reg_4, frst_hi_smask);

        /* row 3 */
        temp6_8x16b = _mm_shuffle_epi8(m_temp_reg_6, frst_low_smask);

        temp7_8x16b = _mm_shuffle_epi8(m_temp_reg_6, frst_hi_smask);

        /* Pred values population same as cur */
        /* row 2 */
        temp14_8x16b = _mm_shuffle_epi8(m_temp_reg_5, frst_low_smask);

        temp15_8x16b = _mm_shuffle_epi8(m_temp_reg_5, frst_hi_smask);

        /* row 3 */
        temp16_8x16b = _mm_shuffle_epi8(m_temp_reg_7, frst_low_smask);

        temp17_8x16b = _mm_shuffle_epi8(m_temp_reg_7, frst_hi_smask);

        /* residue values */
        /* row 2 */
        temp_res4 = _mm_sub_epi16(temp4_8x16b, temp14_8x16b);

        temp_res5 = _mm_sub_epi16(temp5_8x16b, temp15_8x16b);

        /* row 3 */
        temp_res6 = _mm_sub_epi16(temp6_8x16b, temp16_8x16b);

        temp_res7 = _mm_sub_epi16(temp7_8x16b, temp17_8x16b);

        /* r0 r1 r0 r1 r3 r2 r3 r2 */
        temp4_8x16b = _mm_unpacklo_epi32(temp_res4, temp_res6);
        /* r4 r5 r4 r5 r7 r6 r7 r6 */
        temp5_8x16b = _mm_unpackhi_epi32(temp_res4, temp_res6);

        /* r8 r9 r8 r9 r11 r10 r11 r10 */
        temp6_8x16b = _mm_unpacklo_epi32(temp_res5, temp_res7);
        /* r12 r13 r12 r13 r15 r14 r15 r14 */
        temp7_8x16b = _mm_unpackhi_epi32(temp_res5, temp_res7);

        /* Rearranging the residues for further implementation */
        res_r0_r1 = _mm_unpacklo_epi64(temp0_8x16b, temp4_8x16b);

        res_r3_r2 = _mm_unpackhi_epi64(temp0_8x16b, temp4_8x16b);

        res_r4_r5 = _mm_unpacklo_epi64(temp1_8x16b, temp5_8x16b);

        res_r7_r6 = _mm_unpackhi_epi64(temp1_8x16b, temp5_8x16b);

        res_r8_r9 = _mm_unpacklo_epi64(temp2_8x16b, temp6_8x16b);

        res_r11_r10 = _mm_unpackhi_epi64(temp2_8x16b, temp6_8x16b);

        res_r12_r13 = _mm_unpacklo_epi64(temp3_8x16b, temp7_8x16b);

        res_r15_r14 = _mm_unpackhi_epi64(temp3_8x16b, temp7_8x16b);

        /* e[] and o[] calculation */

        reg_e0_e1 = _mm_add_epi16(res_r0_r1, res_r15_r14);

        reg_o0_o1 = _mm_sub_epi16(res_r0_r1, res_r15_r14);


        reg_e3_e2 = _mm_add_epi16(res_r3_r2, res_r12_r13);

        reg_o3_o2 = _mm_sub_epi16(res_r3_r2, res_r12_r13);


        reg_e4_e5 = _mm_add_epi16(res_r4_r5, res_r11_r10);

        reg_o4_o5 = _mm_sub_epi16(res_r4_r5, res_r11_r10);

        reg_e7_e6 = _mm_add_epi16(res_r7_r6, res_r8_r9);

        reg_o7_o6 = _mm_sub_epi16(res_r7_r6, res_r8_r9);

        {
            /* Even part calculations */

            /* ee[] and ee[] calculations */

            reg_ee0_ee1 = _mm_add_epi16(reg_e0_e1, reg_e7_e6);

            reg_eo0_eo1 = _mm_sub_epi16(reg_e0_e1, reg_e7_e6);

            reg_ee3_ee2 = _mm_add_epi16(reg_e3_e2, reg_e4_e5);

            reg_eo3_eo2 = _mm_sub_epi16(reg_e3_e2, reg_e4_e5);

            /* eee[] and eeo[] calculations */

            reg_eee0_eee1 = _mm_add_epi16(reg_ee0_ee1, reg_ee3_ee2);

            reg_eeo0_eeo1 = _mm_sub_epi16(reg_ee0_ee1, reg_ee3_ee2);

            /* coeff's for 0, 8, 4, 12 calculation in corresponding C code */

            coeff0_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[0][0]);

            coeff4_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[3][0]);

            coeff8_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[6][0]);

            coeff12_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[9][0]);

            src0_4x32b = _mm_madd_epi16(reg_eee0_eee1, coeff0_8x16b);

            src4_4x32b = _mm_madd_epi16(reg_eeo0_eeo1, coeff4_8x16b);

            src8_4x32b = _mm_madd_epi16(reg_eee0_eee1, coeff8_8x16b);

            src12_4x32b = _mm_madd_epi16(reg_eeo0_eeo1, coeff12_8x16b);

            /*Store the cols of 0, 4, 8, 12 */
            _mm_store_si128((__m128i *)(pi4_temp),src0_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 4 *trans_size),src4_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 8 *trans_size),src8_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 12 *trans_size),src12_4x32b);

            /* Coeff's for the even non-multiple of 4 cols */

            coeff2_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[1][0]);

            coeff3_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[2][0]);

            coeff6_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[4][0]);

            coeff7_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[5][0]);

            coeff10_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[7][0]);

            coeff11_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[8][0]);

            coeff14_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[10][0]);

            coeff15_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_even[11][0]);

            /* Calculate the 2n and not 4n cols */
            temp2_4x32b = _mm_madd_epi16(reg_eo0_eo1, coeff2_8x16b);

            temp3_4x32b = _mm_madd_epi16(reg_eo3_eo2, coeff3_8x16b);

            temp6_4x32b = _mm_madd_epi16(reg_eo0_eo1, coeff6_8x16b);

            temp7_4x32b = _mm_madd_epi16(reg_eo3_eo2, coeff7_8x16b);

            temp10_4x32b = _mm_madd_epi16(reg_eo0_eo1, coeff10_8x16b);

            temp11_4x32b = _mm_madd_epi16(reg_eo3_eo2, coeff11_8x16b);

            temp14_4x32b = _mm_madd_epi16(reg_eo0_eo1, coeff14_8x16b);

            temp15_4x32b = _mm_madd_epi16(reg_eo3_eo2, coeff15_8x16b);

            src2_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            src6_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src10_4x32b = _mm_add_epi32(temp10_4x32b, temp11_4x32b);

            src14_4x32b = _mm_add_epi32(temp14_4x32b, temp15_4x32b);

            /*Store the cols of 2, 6, 10, 14 */
            _mm_store_si128((__m128i *)(pi4_temp+ 2 * trans_size),src2_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 6 *trans_size),src6_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 10 *trans_size),src10_4x32b);

            _mm_store_si128((__m128i *)(pi4_temp+ 14 *trans_size),src14_4x32b);
        }
        {
            /* Odd part calculations */

            __m128i src1_4x32b, src3_4x32b, src5_4x32b, src7_4x32b, src9_4x32b, src11_4x32b, src13_4x32b, src15_4x32b;

            /* K = 1 and K = 3*/
            {
                coeff0_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[0][0]);

                coeff1_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[1][0]);

                coeff2_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[2][0]);

                coeff3_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[3][0]);

                /* K = 1 madd's */
                m_temp_reg_0 = _mm_madd_epi16(reg_o0_o1, coeff0_8x16b);

                m_temp_reg_1 = _mm_madd_epi16(reg_o3_o2, coeff1_8x16b);

                m_temp_reg_2 = _mm_madd_epi16(reg_o4_o5, coeff2_8x16b);

                m_temp_reg_3 = _mm_madd_epi16(reg_o7_o6, coeff3_8x16b);

                coeff4_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[4][0]);

                coeff5_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[5][0]);

                coeff6_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[6][0]);

                coeff7_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[7][0]);

                /* K = 1 */
                m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
                m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);

                src1_4x32b = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);

                /* For K= 3 */
                m_temp_reg_4 = _mm_madd_epi16(reg_o0_o1, coeff4_8x16b);

                m_temp_reg_5 = _mm_madd_epi16(reg_o3_o2, coeff5_8x16b);

                m_temp_reg_6 = _mm_madd_epi16(reg_o4_o5, coeff6_8x16b);

                m_temp_reg_7 = _mm_madd_epi16(reg_o7_o6, coeff7_8x16b);

                /* K = 3 */
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                src3_4x32b = _mm_add_epi32(m_temp_reg_4, m_temp_reg_6);

                /* Store the values */
                _mm_store_si128((__m128i *)(pi4_temp + trans_size),src1_4x32b);

                _mm_store_si128((__m128i *)(pi4_temp + (3 * trans_size)),src3_4x32b);

                /* K = 5 and K = 7 */

                coeff10_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[8][0]);

                coeff11_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[9][0]);

                coeff12_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[10][0]);

                coeff13_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[11][0]);

                coeff14_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[12][0]);

                coeff15_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[13][0]);

                coeff16_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[14][0]);

                coeff17_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[15][0]);

                /* K = 5 madd's */
                m_temp_reg_10 = _mm_madd_epi16(reg_o0_o1, coeff10_8x16b);

                m_temp_reg_11 = _mm_madd_epi16(reg_o3_o2, coeff11_8x16b);

                m_temp_reg_12 = _mm_madd_epi16(reg_o4_o5, coeff12_8x16b);

                m_temp_reg_13 = _mm_madd_epi16(reg_o7_o6, coeff13_8x16b);

                /* For K= 7*/
                m_temp_reg_14 = _mm_madd_epi16(reg_o0_o1, coeff14_8x16b);

                m_temp_reg_15 = _mm_madd_epi16(reg_o3_o2, coeff15_8x16b);

                m_temp_reg_16 = _mm_madd_epi16(reg_o4_o5, coeff16_8x16b);

                m_temp_reg_17 = _mm_madd_epi16(reg_o7_o6, coeff17_8x16b);

                /* K = 5 */
                m_temp_reg_0 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);

                m_temp_reg_2 = _mm_add_epi32(m_temp_reg_12, m_temp_reg_13);

                src5_4x32b = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);

                /* K = 7 */
                m_temp_reg_4 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);

                m_temp_reg_6 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_17);

                src7_4x32b = _mm_add_epi32(m_temp_reg_4, m_temp_reg_6);

                /* Store the values */
                _mm_store_si128((__m128i *)(pi4_temp + (5 * trans_size)),src5_4x32b);

                _mm_store_si128((__m128i *)(pi4_temp + (7 * trans_size)),src7_4x32b);
            }

            /* K = 9 and K = 11 */
            {
                coeff0_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[16][0]);

                coeff1_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[17][0]);

                coeff2_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[18][0]);

                coeff3_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[19][0]);

                coeff4_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[20][0]);

                coeff5_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[21][0]);

                coeff6_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[22][0]);

                coeff7_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[23][0]);

                /* K = 5 madd's */
                m_temp_reg_0 = _mm_madd_epi16(reg_o0_o1, coeff0_8x16b);

                m_temp_reg_1 = _mm_madd_epi16(reg_o3_o2, coeff1_8x16b);

                m_temp_reg_2 = _mm_madd_epi16(reg_o4_o5, coeff2_8x16b);

                m_temp_reg_3 = _mm_madd_epi16(reg_o7_o6, coeff3_8x16b);

                /* For K= 7 */
                m_temp_reg_4 = _mm_madd_epi16(reg_o0_o1, coeff4_8x16b);

                m_temp_reg_5 = _mm_madd_epi16(reg_o3_o2, coeff5_8x16b);

                m_temp_reg_6 = _mm_madd_epi16(reg_o4_o5, coeff6_8x16b);

                m_temp_reg_7 = _mm_madd_epi16(reg_o7_o6, coeff7_8x16b);

                /* K = 1 */
                m_temp_reg_10 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_1);

                m_temp_reg_12 = _mm_add_epi32(m_temp_reg_2, m_temp_reg_3);

                src9_4x32b = _mm_add_epi32(m_temp_reg_10, m_temp_reg_12);

                /* K = 3 */
                m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
                m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

                src11_4x32b = _mm_add_epi32(m_temp_reg_4, m_temp_reg_6);

                /* Store the values */
                _mm_store_si128((__m128i *)(pi4_temp + (9 * trans_size)),src9_4x32b);

                _mm_store_si128((__m128i *)(pi4_temp + (11 * trans_size)),src11_4x32b);

                /* K = 13 and K = 15 */

                coeff10_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[24][0]);

                coeff11_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[25][0]);

                coeff12_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[26][0]);

                coeff13_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[27][0]);

                coeff14_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[28][0]);

                coeff15_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[29][0]);

                coeff16_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[30][0]);

                coeff17_8x16b = _mm_loadu_si128( (__m128i *) &g_ai2_ihevc_trans_16_intr_odd[31][0]);

                /* K = 5 madd's */
                m_temp_reg_10 = _mm_madd_epi16(reg_o0_o1, coeff10_8x16b);

                m_temp_reg_11 = _mm_madd_epi16(reg_o3_o2, coeff11_8x16b);

                m_temp_reg_12 = _mm_madd_epi16(reg_o4_o5, coeff12_8x16b);

                m_temp_reg_13 = _mm_madd_epi16(reg_o7_o6, coeff13_8x16b);

                /* For K= 7*/
                m_temp_reg_14 = _mm_madd_epi16(reg_o0_o1, coeff14_8x16b);

                m_temp_reg_15 = _mm_madd_epi16(reg_o3_o2, coeff15_8x16b);

                m_temp_reg_16 = _mm_madd_epi16(reg_o4_o5, coeff16_8x16b);

                m_temp_reg_17 = _mm_madd_epi16(reg_o7_o6, coeff17_8x16b);

                /* K = 1 */
                m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);

                m_temp_reg_12 = _mm_add_epi32(m_temp_reg_12, m_temp_reg_13);

                src13_4x32b = _mm_add_epi32(m_temp_reg_10, m_temp_reg_12);

                /* K = 3 */
                m_temp_reg_14 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);

                m_temp_reg_16 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_17);

                src15_4x32b = _mm_add_epi32(m_temp_reg_14, m_temp_reg_16);

                /* Store the values */
                _mm_store_si128((__m128i *)(pi4_temp + (13 * trans_size)),src13_4x32b);

                _mm_store_si128((__m128i *)(pi4_temp + (15 * trans_size)),src15_4x32b);
            }
        }
        pu1_src += 4*src_strd;
        pu1_pred += 4*pred_strd;
        pi4_temp += 4; // Since the stores are at 32 bit */
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = 13; // log2(iHeight) + 6
    add = 1 << (shift - 1);

    for(i = 0; i < TRANS_SIZE_16; i +=4)
    {
        __m128i m_temp_reg_16, m_temp_reg_17, m_temp_reg_18, m_temp_reg_19;
        __m128i m_temp_reg_20, m_temp_reg_21, m_temp_reg_22, m_temp_reg_23;

        __m128i m_temp_reg_24, m_temp_reg_25, m_temp_reg_26, m_temp_reg_27;
        __m128i m_temp_reg_31, m_temp_reg_30, m_temp_reg_29, m_temp_reg_28, m_temp_reg_32;

        /* row =0 */
        m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pi4_temp));      /* k = 0-8  */

        /* row =1 */
        m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size));       /* k = 0-8   */

        /* row =2 */
        m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size));       /* k = 0-8   */

        /* row =3 */
        m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size));       /* k = 0-8   */

        /* row =0 */
        m_temp_reg_8 = _mm_loadu_si128((__m128i*)(pi4_temp+4));                     /* k = 0-8  */
        /* row =1 */
        m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size+4));         /* k = 0-8   */
        /* row =2 */
        m_temp_reg_12 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size+4));       /* k = 0-8   */
        /* row =3 */
        m_temp_reg_14 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size+4));       /* k = 0-8   */

        /* K = 0,1,2,3 */
        m_temp_reg_16 = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_17 = _mm_unpacklo_epi32(m_temp_reg_4, m_temp_reg_6);

        m_temp_reg_18 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_19 = _mm_unpackhi_epi32(m_temp_reg_4, m_temp_reg_6);

        m_temp_reg_4 = _mm_unpacklo_epi64(m_temp_reg_16, m_temp_reg_17);

        m_temp_reg_20 = _mm_unpackhi_epi64(m_temp_reg_16, m_temp_reg_17);

        m_temp_reg_5 = _mm_unpacklo_epi64(m_temp_reg_18, m_temp_reg_19);

        m_temp_reg_21 = _mm_unpackhi_epi64(m_temp_reg_18, m_temp_reg_19);

        /* K = 4,5,6,7 */
        m_temp_reg_0 = _mm_unpacklo_epi32 (m_temp_reg_8, m_temp_reg_10);

        m_temp_reg_1 = _mm_unpacklo_epi32(m_temp_reg_12, m_temp_reg_14);

        m_temp_reg_24 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_10);

        m_temp_reg_25 = _mm_unpackhi_epi32(m_temp_reg_12, m_temp_reg_14);

        m_temp_reg_6 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_1);

        m_temp_reg_22 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_1);

        m_temp_reg_7 = _mm_unpacklo_epi64(m_temp_reg_24, m_temp_reg_25);

        m_temp_reg_23 = _mm_unpackhi_epi64(m_temp_reg_24, m_temp_reg_25);

        /* row =0 */
        m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pi4_temp + 8));      /* k = 0-8  */

        /* row =1 */
        m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size + 8));       /* k = 0-8   */

        /* row =2 */
        m_temp_reg_8 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size + 8));       /* k = 0-8   */

        /* row =3 */
        m_temp_reg_9 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size + 8));       /* k = 0-8   */

        /* row =0 */
        m_temp_reg_10 = _mm_loadu_si128((__m128i*)(pi4_temp+12));                     /* k = 0-8  */
        /* row =1 */
        m_temp_reg_32 = _mm_loadu_si128((__m128i*)(pi4_temp+trans_size+12));         /* k = 0-8   */
        /* row =2 */
        m_temp_reg_24 = _mm_loadu_si128((__m128i*)(pi4_temp+2*trans_size+12));       /* k = 0-8   */
        /* row =3 */
        m_temp_reg_25 = _mm_loadu_si128((__m128i*)(pi4_temp+3*trans_size+12));       /* k = 0-8   */

        /* K = 8,9,10,11 */
        m_temp_reg_1 = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_3 = _mm_unpacklo_epi32(m_temp_reg_8, m_temp_reg_9);

        m_temp_reg_11 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_9 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_9);

        m_temp_reg_31 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_3);

        m_temp_reg_27 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_3);

        m_temp_reg_30 = _mm_unpacklo_epi64(m_temp_reg_11, m_temp_reg_9);

        m_temp_reg_26 = _mm_unpackhi_epi64(m_temp_reg_11, m_temp_reg_9);

        /* K = 8,9,10,11 */
        m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_10, m_temp_reg_32);

        m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_24, m_temp_reg_25);

        m_temp_reg_8 = _mm_unpackhi_epi32(m_temp_reg_10, m_temp_reg_32);

        m_temp_reg_9 = _mm_unpackhi_epi32(m_temp_reg_24, m_temp_reg_25);

        m_temp_reg_29 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_25 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_2);

        m_temp_reg_28 = _mm_unpacklo_epi64(m_temp_reg_8, m_temp_reg_9);

        m_temp_reg_24 = _mm_unpackhi_epi64(m_temp_reg_8, m_temp_reg_9);

        /* for k=0 to 7 */

        /* e[k] = pi4_temp[k] + pi4_temp[31 - k]; */
        m_temp_reg_16  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=0 */
        m_temp_reg_17  = _mm_add_epi32 (m_temp_reg_20, m_temp_reg_28);   /* for k=1 */
        m_temp_reg_18  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=2 */
        m_temp_reg_19  = _mm_add_epi32 (m_temp_reg_21, m_temp_reg_29);   /* for k=3 */

        m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=4 */
        m_temp_reg_13  = _mm_add_epi32 (m_temp_reg_22, m_temp_reg_30);   /* for k=5 */
        m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_7, m_temp_reg_27);    /* for k=6 */
        m_temp_reg_15  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_31);   /* for k=7 */

        /* o[k] = pi4_temp[k] - pi4_temp[31 - k]; */
        m_temp_reg_0  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=0 */
        m_temp_reg_1  = _mm_sub_epi32 (m_temp_reg_20, m_temp_reg_28);   /* for k=1 */
        m_temp_reg_2  = _mm_sub_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=2 */
        m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_21, m_temp_reg_29);   /* for k=3 */

        m_temp_reg_8  = _mm_sub_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=4 */
        m_temp_reg_9  = _mm_sub_epi32 (m_temp_reg_22, m_temp_reg_30);   /* for k=5 */
        m_temp_reg_10  = _mm_sub_epi32 (m_temp_reg_7, m_temp_reg_27);   /* for k=6 */
        m_temp_reg_11  = _mm_sub_epi32 (m_temp_reg_23, m_temp_reg_31);  /* for k=7 */

        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr),    m_temp_reg_0);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+4),  m_temp_reg_1);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+8), m_temp_reg_2);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+12), m_temp_reg_3);

        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+16), m_temp_reg_8);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+20), m_temp_reg_9);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+24), m_temp_reg_10);
        _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+28), m_temp_reg_11);

        /* eo[k] = e[k] - e[15 - k]; */

        m_temp_reg_24  = _mm_sub_epi32 (m_temp_reg_16, m_temp_reg_15);    /* for k=0 */
        m_temp_reg_25  = _mm_sub_epi32 (m_temp_reg_17, m_temp_reg_14);    /* for k=1 */
        m_temp_reg_26  = _mm_sub_epi32 (m_temp_reg_18, m_temp_reg_13);    /* for k=2 */
        m_temp_reg_27  = _mm_sub_epi32 (m_temp_reg_19, m_temp_reg_12);    /* for k=3 */

        /* ee[k] = e[k] + e[15 - k]; */
        m_temp_reg_16  = _mm_add_epi32 (m_temp_reg_16, m_temp_reg_15);    /* for k=0 */
        m_temp_reg_23  = _mm_add_epi32 (m_temp_reg_17, m_temp_reg_14);    /* for k=1 */
        m_temp_reg_17  = _mm_add_epi32 (m_temp_reg_18, m_temp_reg_13);    /* for k=2 */
        m_temp_reg_22  = _mm_add_epi32 (m_temp_reg_19, m_temp_reg_12);    /* for k=3 */

        /* eeo[k] = ee[k] - ee[7 - k]; */
        m_temp_reg_20   = _mm_sub_epi32 (m_temp_reg_16, m_temp_reg_22);    /* for k=0 */
        m_temp_reg_21  = _mm_sub_epi32 (m_temp_reg_23, m_temp_reg_17);     /* for k=1 */

        /* eee[k] = ee[k] + ee[7 - k]; */
        m_temp_reg_18   = _mm_add_epi32 (m_temp_reg_16, m_temp_reg_22);    /* for k=0 */
        m_temp_reg_19  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_17);     /* for k=1 */

        /* seting table values in registers */
        m_temp_reg_14 = _mm_set1_epi32(64);
        m_temp_reg_15 = _mm_set1_epi32(83);
        m_temp_reg_16 = _mm_set1_epi32(36);

        /* g_ai2_ihevc_trans_32[0][0] * eeee[0] */
        m_temp_reg_23  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_18);

        /* g_ai2_ihevc_trans_32[0][1] * eeee[1] */
        m_temp_reg_22  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_19);

        m_temp_reg_14 =_mm_sign_epi32(m_temp_reg_14, _mm_set1_epi32(-1));

        /* g_ai2_ihevc_trans_32[0][1] * eeee[1] */
        m_temp_reg_17  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_19);

        /* g_ai2_ihevc_trans_32[8][0] * eee0[0] */
        m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_20);

        /* g_ai2_ihevc_trans_32[8][1] * eee0[1] */
        m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_21);

        /* g_ai2_ihevc_trans_32[24][0] * eee0[0] */
        m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_20);

        m_temp_reg_15 =_mm_sign_epi32(m_temp_reg_15, _mm_set1_epi32(-1));

        /* g_ai2_ihevc_trans_32[24][1] * eee0[1] */
        m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_21);

        m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_22);
        m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_17);
        m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_5);
        m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

        m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
        m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, _mm_set1_epi32(add));
        m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_3, _mm_set1_epi32(add));
        m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, _mm_set1_epi32(add));

        m_temp_reg_0 = _mm_srai_epi32(m_temp_reg_0, shift);
        m_temp_reg_1 = _mm_srai_epi32(m_temp_reg_1, shift);
        m_temp_reg_3 = _mm_srai_epi32(m_temp_reg_3, shift);
        m_temp_reg_4 = _mm_srai_epi32(m_temp_reg_4, shift);

        m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));
        m_temp_reg_1 = _mm_packs_epi32 (m_temp_reg_1, _mm_set1_epi16(0));
        m_temp_reg_3 = _mm_packs_epi32 (m_temp_reg_3, _mm_set1_epi16(0));
        m_temp_reg_4 = _mm_packs_epi32 (m_temp_reg_4, _mm_set1_epi16(0));

        _mm_storel_epi64((__m128i *)(pi2_dst),m_temp_reg_0);
        _mm_storel_epi64((__m128i *)(pi2_dst+8*dst_strd),m_temp_reg_1);
        _mm_storel_epi64((__m128i *)(pi2_dst+4 *dst_strd),m_temp_reg_3);
        _mm_storel_epi64((__m128i *)(pi2_dst+12*dst_strd),m_temp_reg_4);

        /* for(k = 2; k < 16; k += 4) */

        m_temp_reg_14 = _mm_set1_epi32(89);
        m_temp_reg_15 = _mm_set1_epi32(75);
        m_temp_reg_16 = _mm_set1_epi32(50);
        m_temp_reg_17 = _mm_set1_epi32(18);

        /* for k=2 */
        {
            /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][0] * eo[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][0] * eo[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][0] * eo[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_27);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+2*dst_strd),m_temp_reg_0);
        }

        /* for k=10 */
        {
            /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_24);

            m_temp_reg_14 =_mm_sign_epi32(m_temp_reg_14, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * eo[1] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][0] * eo[2] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][0] * eo[3] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_27);

            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_4);
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, m_temp_reg_2);
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_1, _mm_set1_epi32(add));
            m_temp_reg_1  = _mm_srai_epi32(m_temp_reg_1, shift);

            m_temp_reg_1 = _mm_packs_epi32 (m_temp_reg_1, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+10*dst_strd),m_temp_reg_1);
        }

        /* for k=14*/
        {
            /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_24);

            m_temp_reg_16 =_mm_sign_epi32(m_temp_reg_16, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * eo[1] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_25);

            /* g_ai2_ihevc_trans_32[k][0] * eo[2] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][0] * eo[3] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_27);

            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, _mm_set1_epi32(add));
            m_temp_reg_2 = _mm_srai_epi32(m_temp_reg_2, shift);

            m_temp_reg_2 = _mm_packs_epi32 (m_temp_reg_2, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+14*dst_strd),m_temp_reg_2);
        }

        /* for k=6*/
        {
            /* g_ai2_ihevc_trans_32[k][0] * eo[0] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_24);

            m_temp_reg_17 =_mm_sign_epi32(m_temp_reg_17, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * eo[1] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_25);

            /* g_ai2_ihevc_trans_32[k][0] * eo[2] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][0] * eo[3] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_27);

            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_4);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_6);
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_3, m_temp_reg_4);
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_3, _mm_set1_epi32(add));
            m_temp_reg_3 = _mm_srai_epi32(m_temp_reg_3, shift);

            m_temp_reg_3 = _mm_packs_epi32 (m_temp_reg_3, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+6*dst_strd),m_temp_reg_3);
        }

        /* for(k = 1; k < 16; k += 2) */

        m_temp_reg_16 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr));
        m_temp_reg_17 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+4));
        m_temp_reg_18 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+8));
        m_temp_reg_19 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+12));
        m_temp_reg_20 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+16));
        m_temp_reg_21 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+20));
        m_temp_reg_22 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+24));
        m_temp_reg_23 = _mm_loadu_si128((__m128i*)(g_ai2_ihevc_trans_32_intr_8_ptr+28));

        m_temp_reg_24 = _mm_loadu_si128((__m128i*)(o_temp1_ptr));
        m_temp_reg_25 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+4));
        m_temp_reg_26 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+8));
        m_temp_reg_27 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+12));

        m_temp_reg_28 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+16));
        m_temp_reg_29 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+20));
        m_temp_reg_30 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+24));
        m_temp_reg_31 = _mm_loadu_si128((__m128i*)(o_temp1_ptr+28));

        /* k =1 */
        {
            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_18,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+1*dst_strd),m_temp_reg_0);
        }

        /* k =3 */
        {
            m_temp_reg_9 =_mm_sign_epi32(m_temp_reg_21, _mm_set1_epi32(-1));
            m_temp_reg_10 =_mm_sign_epi32(m_temp_reg_18, _mm_set1_epi32(-1));
            m_temp_reg_11 =_mm_sign_epi32(m_temp_reg_16, _mm_set1_epi32(-1));
            m_temp_reg_12 =_mm_sign_epi32(m_temp_reg_19, _mm_set1_epi32(-1));
            m_temp_reg_13 =_mm_sign_epi32(m_temp_reg_22, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_9,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+3*dst_strd),m_temp_reg_0);
        }

        /* k =5 */

        {
            m_temp_reg_14 =_mm_sign_epi32(m_temp_reg_17, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_18,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+5*dst_strd),m_temp_reg_0);
        }

        /* k =7 */

        {
            m_temp_reg_15 =_mm_sign_epi32(m_temp_reg_20, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_9,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+7*dst_strd),m_temp_reg_0);
        }

        /* k =9 */

        {
            m_temp_reg_8 =_mm_sign_epi32(m_temp_reg_23, _mm_set1_epi32(-1));

            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_8,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+9*dst_strd),m_temp_reg_0);
        }

        /* k =11 */

        {
            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_20,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_14,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+11*dst_strd),m_temp_reg_0);
        }

        /* k =13 */

        {
            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_22,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_12,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_16,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+13*dst_strd),m_temp_reg_0);
        }

        /* k =15 */

        {
            /* g_ai2_ihevc_trans_32[k][0] * o[0] */
            m_temp_reg_0  = _mm_mullo_epi32 (m_temp_reg_23,  m_temp_reg_24);
            /* g_ai2_ihevc_trans_32[k][1] * o[1] */
            m_temp_reg_1  = _mm_mullo_epi32 (m_temp_reg_13,  m_temp_reg_25);
            /* g_ai2_ihevc_trans_32[k][2] * o[2] */
            m_temp_reg_2  = _mm_mullo_epi32 (m_temp_reg_21,  m_temp_reg_26);
            /* g_ai2_ihevc_trans_32[k][3] * o[3] */
            m_temp_reg_3  = _mm_mullo_epi32 (m_temp_reg_15,  m_temp_reg_27);
            /* g_ai2_ihevc_trans_32[k][4] * o[4] */
            m_temp_reg_4  = _mm_mullo_epi32 (m_temp_reg_19,  m_temp_reg_28);
            /* g_ai2_ihevc_trans_32[k][5] * o[5] */
            m_temp_reg_5  = _mm_mullo_epi32 (m_temp_reg_10,  m_temp_reg_29);
            /* g_ai2_ihevc_trans_32[k][6] * o[6] */
            m_temp_reg_6  = _mm_mullo_epi32 (m_temp_reg_17,  m_temp_reg_30);
            /* g_ai2_ihevc_trans_32[k][7] * o[7] */
            m_temp_reg_7  = _mm_mullo_epi32 (m_temp_reg_11,  m_temp_reg_31);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_2, m_temp_reg_3);
            m_temp_reg_4  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_5);
            m_temp_reg_6  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_7);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_6);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_0, _mm_set1_epi32(add));
            m_temp_reg_0  = _mm_srai_epi32(m_temp_reg_0, shift);

            m_temp_reg_0 = _mm_packs_epi32 (m_temp_reg_0, _mm_set1_epi16(0));

            _mm_storel_epi64((__m128i *)(pi2_dst+15*dst_strd),m_temp_reg_0);
        }

        pi4_temp += 4*trans_size;
        pi2_dst +=4;
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
 * @param[in] pu1_src
 *  Input 32x32 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 32x32
 *
 * @param[out] pi2_dst
 *  Output 32x32 coefficients
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_resi_trans_32x32_sse42(UWORD8 *pu1_src,
                            UWORD8 *pu1_pred,
                               WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 src_strd,
                            WORD32 pred_strd,
                               WORD32 dst_strd_chr_flag)
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
    __m128i m_temp_reg_18, m_temp_reg_19, m_temp_reg_20, m_temp_reg_21, m_temp_reg_22, m_temp_reg_23, m_temp_reg_24, m_temp_reg_25;
    __m128i sad0_16x8b, sad1_16x8b, sad2_16x8b, sad3_16x8b;
    __m128i frst_low_smask, frst_hi_smask, minusone_4x32b, add_4x32b;
    __m128i temp_res_0, temp_res_1, temp_res_2, temp_res_3, temp_res_4, temp_res_5, temp_res_6, temp_res_7;
    __m128i res_r0_r3_r1_r2, res_r7_r4_r6_r5,res_r8_r11_r9_r10,res_r15_r12_r14_r13, res_r16_r19_r17_r18, res_r23_r20_r22_r21, res_r24_r27_r25_r26, res_r31_r28_r29_r30;
    __m128i reg_e0_e3_e1_e2, reg_o0_o3_o1_o2, reg_e7_e4_e6_e5, reg_o7_o4_o6_o5, reg_e8_e11_e9_e10, reg_o8_o11_o9_o10, reg_e15_e12_e14_e13, reg_o15_o12_o14_o13;
    __m128i reg_ee0_ee3_ee1_ee2, reg_eo0_eo3_eo1_eo2, reg_ee7_ee4_ee6_ee5, reg_eo7_eo4_eo6_eo5;
    __m128i reg_eee0_eee3_eee1_eee2, reg_eeo0_eeo3_eeo1_eeo2;
    __m128i reg_eeee0_eeee1, reg_eeeo0_eeeo1, reg_eeee0_eeee1_eeeo0_eeeo1;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_32;

    dst_strd = dst_strd_chr_flag >> 16;

    /* Residue + Forward Transform 1st stage */
    shift = 4; // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);
    o_temp = temp_array;
    o_temp1_ptr = temp_array1;

    g_ai2_ihevc_trans_32_intr_8_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_8;
    g_ai2_ihevc_trans_32_intr_16_ptr = (WORD32 *)g_ai2_ihevc_trans_32_intr_16;

/* unrolling outer loop */

    for(i = 0; i < trans_size; i+=2)
    {
        /* row =0 */
        m_temp_reg_0 = _mm_loadu_si128((__m128i*)(pu1_src));      /* k = 0-16  */
        m_temp_reg_1 = _mm_loadu_si128((__m128i*)(pu1_pred));     /* k = 0-16   */

        /* row =1 */
        m_temp_reg_2 = _mm_loadu_si128((__m128i*)(pu1_src+src_strd));       /* k = 0-16   */
        m_temp_reg_3 = _mm_loadu_si128((__m128i*)(pu1_pred+pred_strd));     /* k = 0-16   */

        /* row =0 */
        m_temp_reg_4 = _mm_loadu_si128((__m128i*)(pu1_src+16));      /* k = 0-16  */
        m_temp_reg_5 = _mm_loadu_si128((__m128i*)(pu1_pred+16));     /* k = 0-16   */

        /* row =1 */
        m_temp_reg_6 = _mm_loadu_si128((__m128i*)(pu1_src+src_strd+16));       /* k = 0-16   */
        m_temp_reg_7 = _mm_loadu_si128((__m128i*)(pu1_pred+pred_strd+16));     /* k = 0-16   */

        /* SAD Calculation */
        sad0_16x8b = _mm_sad_epu8(m_temp_reg_0, m_temp_reg_1);

        sad1_16x8b = _mm_sad_epu8(m_temp_reg_2, m_temp_reg_3);

        sad2_16x8b = _mm_sad_epu8(m_temp_reg_4, m_temp_reg_5);

        sad3_16x8b = _mm_sad_epu8(m_temp_reg_6, m_temp_reg_7);

        sad0_16x8b = _mm_add_epi32(sad0_16x8b, sad1_16x8b);

        sad2_16x8b = _mm_add_epi32(sad2_16x8b, sad3_16x8b);

        sad0_16x8b = _mm_add_epi32(sad0_16x8b, sad2_16x8b);

        sad3_16x8b = _mm_srli_si128(sad0_16x8b, 8);

        sad0_16x8b = _mm_add_epi32(sad0_16x8b, sad3_16x8b);

        u4_blk_sad += _mm_cvtsi128_si32 (sad0_16x8b);

        /* Arranging src and pred values */
        frst_low_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK_32x32_TRNS_REORDER1[0]);

        frst_hi_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK_32x32_TRNS_REORDER2[0]);

        /* pu1_src */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_8 = _mm_shuffle_epi8 (m_temp_reg_0, frst_low_smask);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_9 = _mm_shuffle_epi8 (m_temp_reg_0, frst_hi_smask);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_10 = _mm_shuffle_epi8 (m_temp_reg_2, frst_low_smask);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_11 = _mm_shuffle_epi8 (m_temp_reg_2, frst_hi_smask);

        /* a16 a19 a17 a16 a23 a20 a22 a21 */
        m_temp_reg_12 = _mm_shuffle_epi8 (m_temp_reg_4, frst_low_smask);
        /* a24 a27 a25 a26 a31 a28 a29 a30 */
        m_temp_reg_13 = _mm_shuffle_epi8 (m_temp_reg_4, frst_hi_smask);

        /* a16 a19 a17 a16 a23 a20 a22 a21 */
        m_temp_reg_14 = _mm_shuffle_epi8 (m_temp_reg_6, frst_low_smask);
        /* a24 a27 a25 a26 a31 a28 a29 a30 */
        m_temp_reg_15 = _mm_shuffle_epi8 (m_temp_reg_6, frst_hi_smask);

        /* pu1_pred */

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_18 = _mm_shuffle_epi8 (m_temp_reg_1, frst_low_smask);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_19 = _mm_shuffle_epi8 (m_temp_reg_1, frst_hi_smask);

        /* a0 a3 a1 a2 a7 a4 a6 a5 */
        m_temp_reg_20 = _mm_shuffle_epi8 (m_temp_reg_3, frst_low_smask);
        /* a8 a11 a9 a10 a15 a12 a14 a13 */
        m_temp_reg_21 = _mm_shuffle_epi8 (m_temp_reg_3, frst_hi_smask);

        /* a16 a19 a17 a16 a23 a20 a22 a21 */
        m_temp_reg_22 = _mm_shuffle_epi8 (m_temp_reg_5, frst_low_smask);
        /* a24 a27 a25 a26 a31 a28 a29 a30 */
        m_temp_reg_23 = _mm_shuffle_epi8 (m_temp_reg_5, frst_hi_smask);

        /* a16 a19 a17 a16 a23 a20 a22 a21 */
        m_temp_reg_24 = _mm_shuffle_epi8 (m_temp_reg_7, frst_low_smask);
        /* a24 a27 a25 a26 a31 a28 a29 a30 */
        m_temp_reg_25 = _mm_shuffle_epi8 (m_temp_reg_7, frst_hi_smask);

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

        /* row 0 */
        /* r16 r19 r17 r16 r23 r20 r22 r21 */
        temp_res_4 = _mm_sub_epi16(m_temp_reg_12, m_temp_reg_22);
        /* r24 r27 r25 r26 r31 r28 r29 r30 */
        temp_res_5 = _mm_sub_epi16(m_temp_reg_13, m_temp_reg_23);

        /* row 1 */
        /* r16 r19 r17 r16 r23 r20 r22 r21 */
        temp_res_6 = _mm_sub_epi16(m_temp_reg_14, m_temp_reg_24);
        /* r24 r27 r25 r26 r31 r28 r29 r30 */
        temp_res_7 = _mm_sub_epi16(m_temp_reg_15, m_temp_reg_25);

        /* Residue Re-ordering */

        /* r0 r3 r0 r3 r1 r2 r1 r2 */
        res_r0_r3_r1_r2 = _mm_unpacklo_epi32(temp_res_0, temp_res_2);

        res_r7_r4_r6_r5 = _mm_unpackhi_epi32(temp_res_0, temp_res_2);

        res_r8_r11_r9_r10 = _mm_unpacklo_epi32(temp_res_1, temp_res_3);

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

            /* COL 0,8,16,24 calculations */

            frst_low_smask = _mm_loadu_si128((__m128i *) &IHEVCE_SHUFFLEMASK_32x32_TRNS[0]);
            /* eeee[] and eeee[] calculation */

            reg_eeee0_eeee1 = _mm_hadd_epi16(reg_eee0_eee3_eee1_eee2, reg_eee0_eee3_eee1_eee2);

            reg_eeeo0_eeeo1 = _mm_hsub_epi16(reg_eee0_eee3_eee1_eee2, reg_eee0_eee3_eee1_eee2);

            /* (row 0) eeee0 eeee1 (row 1) eeee0 eeee1 (row 0) eeeo0 eeeo1 (row 1) eeeo0 eeeo1 */
            reg_eeee0_eeee1_eeeo0_eeeo1 = _mm_unpacklo_epi64(reg_eeee0_eeee1, reg_eeeo0_eeeo1);

            reg_eeee0_eeee1_eeeo0_eeeo1 = _mm_shuffle_epi8(reg_eeee0_eeee1_eeeo0_eeeo1, frst_low_smask);

            coeff0_8_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[0][0]);

            coeff16_24_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[1][0]);

            src0_8_4x32b = _mm_madd_epi16(reg_eeee0_eeee1_eeeo0_eeeo1, coeff0_8_8x16b);

            src16_24_4x32b = _mm_madd_epi16(reg_eeee0_eeee1_eeeo0_eeeo1, coeff16_24_8x16b);

            src8_4x32b = _mm_srli_si128(src0_8_4x32b, 8);

            src24_4x32b = _mm_srli_si128(src16_24_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp),src0_8_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 16 * trans_size),src16_24_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 8 * trans_size),src8_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 24 * trans_size),src24_4x32b);

            /* COL 4, 12, 20, 28 calculations */

            coeff4_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[2][0]);

            coeff12_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[3][0]);

            coeff20_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[4][0]);

            coeff28_8x16b = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_even[5][0]);

            src4_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff4_8x16b);

            src12_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff12_8x16b);

            src20_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff20_8x16b);

            src28_4x32b = _mm_madd_epi16(reg_eeo0_eeo3_eeo1_eeo2, coeff28_8x16b);

            /*
            src4_4x32b = _mm_shuffle_epi32(src4_4x32b, 216);

            src12_4x32b = _mm_shuffle_epi32(src12_4x32b, 216);

            src20_4x32b = _mm_shuffle_epi32(src20_4x32b, 216);

            src28_4x32b = _mm_shuffle_epi32(src28_4x32b, 216);
            */

            temp0_4x32b = _mm_unpacklo_epi64(src4_4x32b, src12_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src4_4x32b, src12_4x32b);

            src4_12_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

            //src4_12_4x32b = _mm_shuffle_epi32(src4_12_4x32b, 216);

            temp2_4x32b = _mm_unpacklo_epi64(src20_4x32b, src28_4x32b);

            temp3_4x32b = _mm_unpackhi_epi64(src20_4x32b, src28_4x32b);

            src20_28_4x32b = _mm_add_epi32(temp2_4x32b, temp3_4x32b);

            //src20_28_4x32b = _mm_shuffle_epi32(src20_28_4x32b, 216);

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

            src1_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src3_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src1_4x32b, src3_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src1_4x32b, src3_4x32b);


            src1_3_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

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

            src5_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src7_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);


            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

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

            src9_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src11_4x32b = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src9_4x32b, src11_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src9_4x32b, src11_4x32b);


            src9_11_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

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

            src5_4x32b  = _mm_add_epi32(temp10_4x32b, temp11_4x32b);


            temp12_4x32b = _mm_add_epi32(temp4_4x32b, temp5_4x32b);

            temp13_4x32b = _mm_add_epi32(temp6_4x32b, temp7_4x32b);

            src7_4x32b  = _mm_add_epi32(temp12_4x32b, temp13_4x32b);


            temp0_4x32b = _mm_unpacklo_epi64(src5_4x32b, src7_4x32b);

            temp1_4x32b = _mm_unpackhi_epi64(src5_4x32b, src7_4x32b);


            src5_7_4x32b = _mm_add_epi32(temp0_4x32b, temp1_4x32b);

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

            src7_4x32b = _mm_srli_si128(src5_7_4x32b, 8);

            _mm_storel_epi64((__m128i *)(pi4_temp + 29 * trans_size),src5_7_4x32b);

            _mm_storel_epi64((__m128i *)(pi4_temp + 31 * trans_size),src7_4x32b);
        }

        pu1_src += 2 * src_strd;
        pu1_pred += 2 * pred_strd;
        pi4_temp += 2;
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = 15; // log2(iHeight) + 6
    add = 1 << (shift - 1);

    add_4x32b       = _mm_set1_epi32(add);
    minusone_4x32b  = _mm_set1_epi32(-1);

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

            m_temp_reg_4 = _mm_unpacklo_epi64(m_temp_reg_16, m_temp_reg_17);

            m_temp_reg_20 = _mm_unpackhi_epi64(m_temp_reg_16, m_temp_reg_17);

            m_temp_reg_5 = _mm_unpacklo_epi64(m_temp_reg_18, m_temp_reg_19);

            m_temp_reg_21 = _mm_unpackhi_epi64(m_temp_reg_18, m_temp_reg_19);

            /* K = 4,5,6,7 */
            m_temp_reg_0 = _mm_unpacklo_epi32 (m_temp_reg_8, m_temp_reg_10);

            m_temp_reg_1 = _mm_unpacklo_epi32(m_temp_reg_12, m_temp_reg_14);

            m_temp_reg_24 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_10);

            m_temp_reg_25 = _mm_unpackhi_epi32(m_temp_reg_12, m_temp_reg_14);

            m_temp_reg_6 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_1);

            m_temp_reg_22 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_1);

            m_temp_reg_7 = _mm_unpacklo_epi64(m_temp_reg_24, m_temp_reg_25);

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
            m_temp_reg_1 = _mm_unpacklo_epi32(m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_3 = _mm_unpacklo_epi32(m_temp_reg_8, m_temp_reg_9);

            m_temp_reg_11 = _mm_unpackhi_epi32(m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_9 = _mm_unpackhi_epi32(m_temp_reg_8, m_temp_reg_9);

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
            m_temp_reg_16  = _mm_add_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=0 */
            m_temp_reg_17  = _mm_add_epi32 (m_temp_reg_20, m_temp_reg_28);     /* for k=1 */
            m_temp_reg_18  = _mm_add_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=2 */
            m_temp_reg_19  = _mm_add_epi32 (m_temp_reg_21, m_temp_reg_29);     /* for k=3 */

            m_temp_reg_12  = _mm_add_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=4 */
            m_temp_reg_13  = _mm_add_epi32 (m_temp_reg_22, m_temp_reg_30);    /* for k=5 */
            m_temp_reg_14  = _mm_add_epi32 (m_temp_reg_7, m_temp_reg_27);    /* for k=6 */
            m_temp_reg_15  = _mm_add_epi32 (m_temp_reg_23, m_temp_reg_31);    /* for k=7 */

            /* 0[k] = pi2_tmp[k] - pi2_tmp[31 - k]; */
            m_temp_reg_0  = _mm_sub_epi32 (m_temp_reg_4, m_temp_reg_24);    /* for k=0 */
            m_temp_reg_1  = _mm_sub_epi32 (m_temp_reg_20, m_temp_reg_28);     /* for k=1 */
            m_temp_reg_2  = _mm_sub_epi32 (m_temp_reg_5, m_temp_reg_25);    /* for k=2 */
            m_temp_reg_3  = _mm_sub_epi32 (m_temp_reg_21, m_temp_reg_29);     /* for k=3 */

            m_temp_reg_8  = _mm_sub_epi32 (m_temp_reg_6, m_temp_reg_26);    /* for k=4 */
            m_temp_reg_9  = _mm_sub_epi32 (m_temp_reg_22, m_temp_reg_30);    /* for k=5 */
            m_temp_reg_10  = _mm_sub_epi32 (m_temp_reg_7, m_temp_reg_27);    /* for k=6 */
            m_temp_reg_11  = _mm_sub_epi32 (m_temp_reg_23, m_temp_reg_31);    /* for k=7 */

            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr),    m_temp_reg_0);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+4),  m_temp_reg_1);
            _mm_storeu_si128 ((__m128i *)(o_temp1_ptr+8), m_temp_reg_2);
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
            m_temp_reg_20 = _mm_unpacklo_epi32 (m_temp_reg_8, m_temp_reg_10);

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

            m_temp_reg_28  = _mm_sub_epi32 (m_temp_reg_12, m_temp_reg_23);    /* for k=4 */
            m_temp_reg_29  = _mm_sub_epi32 (m_temp_reg_13, m_temp_reg_22);   /* for k=5 */
            m_temp_reg_30  = _mm_sub_epi32 (m_temp_reg_14, m_temp_reg_21);    /* for k=6 */
            m_temp_reg_31  = _mm_sub_epi32 (m_temp_reg_15, m_temp_reg_20);   /* for k=7 */

            /* ee[k] = e[k] + e[15 - k]; */
            m_temp_reg_0  = _mm_add_epi32 (m_temp_reg_16, m_temp_reg_7);    /* for k=0 */
            m_temp_reg_1  = _mm_add_epi32 (m_temp_reg_17, m_temp_reg_6);   /* for k=1 */
            m_temp_reg_2  = _mm_add_epi32 (m_temp_reg_18, m_temp_reg_5);    /* for k=2 */
            m_temp_reg_3  = _mm_add_epi32 (m_temp_reg_19, m_temp_reg_4);   /* for k=3 */

            m_temp_reg_8  = _mm_add_epi32 (m_temp_reg_12, m_temp_reg_23);    /* for k=4 */
            m_temp_reg_9  = _mm_add_epi32 (m_temp_reg_13, m_temp_reg_22);   /* for k=5 */
            m_temp_reg_10  = _mm_add_epi32 (m_temp_reg_14, m_temp_reg_21);    /* for k=6 */
            m_temp_reg_11  = _mm_add_epi32 (m_temp_reg_15, m_temp_reg_20);   /* for k=7 */

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
