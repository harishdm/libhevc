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
 *  ihevc_chroma_iquant_itrans_recon_x86_intr.c
 *
 * @brief
 *  Contains function definitions for inverse  quantization, inverse
 * transform and reconstruction  of chroma interleaved data.
 *
 * @author
 *  100491
 *
 * @par List of Functions:
 *   - ihevc_chroma_iquant_itrans_recon_4x4()
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

#include <stdio.h>
#include <string.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_chroma_iquant_itrans_recon.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"


#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

UWORD8 IHEVCE_UTIL_CHROMA_SHUFFLEMASK[16] = { 0x0, 0x2, 0x4, 0x6,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF};

UWORD8 IHEVCE_UTIL_CHROMA_UN_SHUFFLEMASK[16] = { 0x0, 0xFF, 0x1, 0xFF,
    0x2, 0xFF, 0x3, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF};

UWORD8 IHEVCE_UTIL_CHROMA_BLENDMASK[16] = { 0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF};

 /* All the functions work one component(U or V) of interleaved data depending upon pointers passed to it */
 /* Data visualization */
 /* U V U V U V U V */
 /* U V U V U V U V */
 /* U V U V U V U V */
 /* U V U V U V U V */
 /* If the pointer points to first byte of above stream (U) , functions will operate on U component */
 /* If the pointer points to second byte of above stream (V) , functions will operate on V component */

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs inverse quantization, inverse  transform and
 * reconstruction for 4x4 input block
 *
 * @par Description:
 *  Performs inverse quantization , inverse transform  and adds the
 * prediction data and clips output to 8 bit
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform
 *  1st stage output
 *
 * @param[in] pu1_pred
 *  Prediction 4x4 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu1_dst
 *  Output 4x4 block
 *
 * @param[in] qp_div
 *  Quantization parameter / 6
 *
 * @param[in] qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] zero_cols
 *  Zero columns in pi2_src
 *
 * @param[in] zero_rows
 *  Zero Rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_chroma_iquant_itrans_recon_4x4_sse42(WORD16 *pi2_src,
                                          WORD16 *pi2_tmp,
                                          UWORD8 *pu1_pred,
                                          WORD16 *pi2_dequant_coeff,
                                          UWORD8 *pu1_dst,
                                          WORD32 qp_div,/* qpscaled / 6 */
                                          WORD32 qp_rem,/* qpscaled % 6 */
                                          WORD32 src_strd,
                                          WORD32 pred_strd,
                                          WORD32 dst_strd,
                                          WORD32 zero_cols,
                                          WORD32 zero_rows)
{
    WORD32 shift_iq;
    __m128i shift_in_iquant;
    __m128i m_temp_reg_0;
    __m128i m_temp_reg_1;
    __m128i m_temp_reg_2;
    __m128i m_temp_reg_3;
    __m128i m_temp_reg_10;
    __m128i m_temp_reg_11;
    __m128i m_temp_reg_12;
    __m128i m_temp_reg_13;
    __m128i m_temp_reg_14;
    __m128i m_temp_reg_15;
    __m128i m_temp_reg_20;
    __m128i m_temp_reg_21;
    __m128i m_temp_reg_22;
    __m128i m_temp_reg_23;
    __m128i m_temp_reg_24;
    __m128i m_temp_reg_25;
    __m128i m_temp_reg_30;
    __m128i m_temp_reg_31;
    __m128i m_temp_reg_33;
    __m128i m_temp_reg_34;
    __m128i m_coeff1, m_coeff3;
    __m128i m_rdng_factor;
    __m128i m_count;
    __m128i m_coeff_min;
    __m128i m_coeff_max;
    __m128i smask, un_smask, blend_mask;

    WORD32 shift_select;
    WORD32 coeff_bit_range;
    WORD32 coeff_min;
    WORD32 coeff_max;


    WORD32 i4_shift = IT_SHIFT_STAGE_1;
    WORD32 one = 1;
    WORD32 trans_size = TRANS_SIZE_4;
    __m128i m_add_iq = _mm_cvtsi32_si128(one);
    __m128i m_scale =
        _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));

    smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_SHUFFLEMASK[0]);
    un_smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_UN_SHUFFLEMASK[0]);
    blend_mask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_BLENDMASK[0]);


    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size, bit_depth;

        log2_trans_size = 2;
        bit_depth = 8;
        shift_iq = 5;//bit_depth + log2_trans_size - 5;
    }

    /* Following 3 instructions replicates the value in the */
    /* lower 16 bits of m_add_iq in the entire register */
    m_add_iq = _mm_unpacklo_epi32(m_add_iq, m_add_iq);
    m_add_iq = _mm_unpacklo_epi64(m_add_iq, m_add_iq);

    /* Values of certain variables change wrt this condition */
    if(qp_div > shift_iq)
    {
        WORD32 smqm1 = -(shift_iq - qp_div - 1);

        m_add_iq = _mm_srli_epi16(m_add_iq, smqm1);
        shift_in_iquant = _mm_cvtsi32_si128(-(shift_iq - qp_div));
        shift_select = 0;
        coeff_bit_range = 10;
    }
    else
    {
        WORD32 smqm1 = (shift_iq - qp_div - 1);

        m_add_iq = _mm_slli_epi16(m_add_iq, smqm1);
        shift_in_iquant = _mm_cvtsi32_si128((shift_iq - qp_div));
        shift_select = 1;
        coeff_bit_range = 16;
    }

    coeff_max = (1 << (coeff_bit_range - 1));
    coeff_min = -coeff_max;
    coeff_max -= 1;

    /* IQUANT */
    {
        m_temp_reg_10 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        pi2_dequant_coeff += (trans_size << 1);
        m_temp_reg_12 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        m_temp_reg_10 = _mm_mullo_epi16(m_temp_reg_10, m_scale);
        m_temp_reg_12 = _mm_mullo_epi16(m_temp_reg_12, m_scale);

        m_coeff_min = _mm_cvtsi32_si128(coeff_min);
        m_coeff_max = _mm_cvtsi32_si128(coeff_max);
        m_coeff_min = _mm_unpacklo_epi32(m_coeff_min, m_coeff_min);
        m_coeff_max = _mm_unpacklo_epi32(m_coeff_max, m_coeff_max);
        m_coeff_min = _mm_unpacklo_epi64(m_coeff_min, m_coeff_min);
        m_coeff_max = _mm_unpacklo_epi64(m_coeff_max, m_coeff_max);

        m_temp_reg_0 = _mm_loadl_epi64((__m128i *) pi2_src);
        pi2_src += src_strd;
        m_temp_reg_1 = _mm_loadl_epi64((__m128i *) pi2_src);
        pi2_src += src_strd;
        m_temp_reg_2 = _mm_loadl_epi64((__m128i *) pi2_src);
        pi2_src += src_strd;
        m_temp_reg_3 = _mm_loadl_epi64((__m128i *) pi2_src);

        m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_0);
        m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
        m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_2);
        m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);

        m_temp_reg_0 = _mm_max_epi32(m_temp_reg_0, m_coeff_min);
        m_temp_reg_1 = _mm_max_epi32(m_temp_reg_1, m_coeff_min);
        m_temp_reg_0 = _mm_min_epi32(m_temp_reg_0, m_coeff_max);
        m_temp_reg_1 = _mm_min_epi32(m_temp_reg_1, m_coeff_max);

        m_temp_reg_2 = _mm_max_epi32(m_temp_reg_2, m_coeff_min);
        m_temp_reg_3 = _mm_max_epi32(m_temp_reg_3, m_coeff_min);
        m_temp_reg_2 = _mm_min_epi32(m_temp_reg_2, m_coeff_max);
        m_temp_reg_3 = _mm_min_epi32(m_temp_reg_3, m_coeff_max);

        m_temp_reg_11 = _mm_srli_si128(m_temp_reg_10, 8);
        m_temp_reg_13 = _mm_srli_si128(m_temp_reg_12, 8);
        m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_10);
        m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_12);
        m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_11);
        m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_13);

        m_temp_reg_0 = _mm_mullo_epi32(m_temp_reg_0, m_temp_reg_10);
        m_temp_reg_2 = _mm_mullo_epi32(m_temp_reg_2, m_temp_reg_12);
        m_temp_reg_1 = _mm_mullo_epi32(m_temp_reg_1, m_temp_reg_11);
        m_temp_reg_3 = _mm_mullo_epi32(m_temp_reg_3, m_temp_reg_13);

        m_temp_reg_0 = _mm_add_epi32(m_temp_reg_0, m_add_iq);
        m_temp_reg_1 = _mm_add_epi32(m_temp_reg_1, m_add_iq);
        m_temp_reg_2 = _mm_add_epi32(m_temp_reg_2, m_add_iq);
        m_temp_reg_3 = _mm_add_epi32(m_temp_reg_3, m_add_iq);

        if(shift_select)
        {
            m_temp_reg_0 = _mm_sra_epi32(m_temp_reg_0, shift_in_iquant);
            m_temp_reg_1 = _mm_sra_epi32(m_temp_reg_1, shift_in_iquant);
            m_temp_reg_2 = _mm_sra_epi32(m_temp_reg_2, shift_in_iquant);
            m_temp_reg_3 = _mm_sra_epi32(m_temp_reg_3, shift_in_iquant);
        }
        else
        {
            m_temp_reg_0 = _mm_sll_epi32(m_temp_reg_0, shift_in_iquant);
            m_temp_reg_1 = _mm_sll_epi32(m_temp_reg_1, shift_in_iquant);
            m_temp_reg_2 = _mm_sll_epi32(m_temp_reg_2, shift_in_iquant);
            m_temp_reg_3 = _mm_sll_epi32(m_temp_reg_3, shift_in_iquant);
        }

        m_temp_reg_1 = _mm_packs_epi32(m_temp_reg_0, m_temp_reg_1);
        m_temp_reg_3 = _mm_packs_epi32(m_temp_reg_2, m_temp_reg_3);
        m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_1);
        m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_3);
        m_temp_reg_1 = _mm_srli_si128(m_temp_reg_1, 8);
        m_temp_reg_3 = _mm_srli_si128(m_temp_reg_3, 8);
        m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
        m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);
    }

    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_4_ttype0[0][0]);//36
    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_4_ttype0[2][0]);//83

    /* e */
    {
        m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);
        m_temp_reg_11 = _mm_slli_epi32(m_temp_reg_2, 6);
    }

    /* o */
    {
        m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);//src[1]*36
        m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);//src[3]*83
        m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);//src[1]*83
        m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);//src[3]*36
    }

    m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));

    /* e1 stored in m_temp_reg_31 */
    {
        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_11);
    }

    m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);

    /* e0 stored in m_temp_reg_30 */
    {
        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
    }

    m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);
    m_count = _mm_cvtsi32_si128(i4_shift);

    /* o1 stored in m_temp_reg_33 */
    {
        m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);
    }

    /* e1 + add */
    {
        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
    }

    /* e0 + add */
    {
        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
    }

    /* o0 stored in m_temp_reg_34 */
    {
        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
    }

    /* Stage 1 outputs */
    {
        m_temp_reg_21 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_33);
        m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_33);

        m_temp_reg_20 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_34);
        m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_34);


        m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
        m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
        m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
        m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

        m_temp_reg_20 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
        m_temp_reg_21 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
        m_temp_reg_22 = _mm_srli_si128(m_temp_reg_20, 8);
        m_temp_reg_23 = _mm_srli_si128(m_temp_reg_21, 8);

        m_temp_reg_24 = _mm_unpacklo_epi16(m_temp_reg_20, m_temp_reg_22);
        m_temp_reg_25 = _mm_unpacklo_epi16(m_temp_reg_21, m_temp_reg_23);

        m_temp_reg_20 = _mm_unpacklo_epi32(m_temp_reg_24, m_temp_reg_25);
        m_temp_reg_21 = _mm_unpackhi_epi32(m_temp_reg_24, m_temp_reg_25);
    }

    /* Stage 2 */
    {
        i4_shift = IT_SHIFT_STAGE_2;

        m_temp_reg_22 = _mm_srli_si128(m_temp_reg_20, 8);
        m_temp_reg_23 = _mm_srli_si128(m_temp_reg_21, 8);

        m_temp_reg_20 = _mm_cvtepi16_epi32(m_temp_reg_20);
        m_temp_reg_21 = _mm_cvtepi16_epi32(m_temp_reg_21);

        m_temp_reg_22 = _mm_cvtepi16_epi32(m_temp_reg_22);
        m_temp_reg_23 = _mm_cvtepi16_epi32(m_temp_reg_23);

        /* e */
        {
            m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_20, 6);
        }

        /* o */
        {
            m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_22, m_coeff1);//src[1]*36
            m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_22, m_coeff3);//src[1]*83
            m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_23, m_coeff3);//src[3]*83
            m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_23, m_coeff1);//src[3]*36
        }

        /* e */
        {
            m_temp_reg_11 = _mm_slli_epi32(m_temp_reg_21, 6);
        }

        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));

        /* e1 stored in m_temp_reg_31 */
        {
            m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_11);
        }

        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);

        /* e0 stored in m_temp_reg_30 */
        {
            m_temp_reg_30 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
        }

        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);
        m_count = _mm_cvtsi32_si128(i4_shift);

        /* o1 stored in m_temp_reg_33 */
        {
            m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);
        }

        /* e1 + add */
        {
            m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
        }

        /* e0 + add */
        {
            m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
        }

        /* o0 stored in m_temp_reg_34 */
        {
            m_temp_reg_34 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
        }

        /* Stage 2 outputs */
        {
            m_temp_reg_21 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_33);
            m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_33);
            m_temp_reg_20 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_34);
            m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_34);

            m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
            m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
            m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
            m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

            /* Change the order in which the values are stored */
            m_temp_reg_20 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
            m_temp_reg_21 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
            m_temp_reg_22 = _mm_srli_si128(m_temp_reg_20, 8);
            m_temp_reg_23 = _mm_srli_si128(m_temp_reg_21, 8);

            m_temp_reg_24 = _mm_unpacklo_epi16(m_temp_reg_20, m_temp_reg_22);
            m_temp_reg_25 = _mm_unpacklo_epi16(m_temp_reg_21, m_temp_reg_23);

            m_temp_reg_20 = _mm_unpacklo_epi32(m_temp_reg_24, m_temp_reg_25);
            m_temp_reg_21 = _mm_unpackhi_epi32(m_temp_reg_24, m_temp_reg_25);
        }

        /* Recon and store */
        {
            UWORD32 *pu4_dst = (UWORD32 *) pu1_dst;
            UWORD8  *temp = pu1_dst;

            m_temp_reg_10 = _mm_loadl_epi64((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_11 = _mm_loadl_epi64((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_12 = _mm_loadl_epi64((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_13 = _mm_loadl_epi64((__m128i *) pu1_pred);

            /* Load current recon values, to blend the alternate values at the end*/
            m_temp_reg_30 = _mm_loadl_epi64((__m128i *) temp);
            temp += dst_strd;

            m_temp_reg_31 = _mm_loadl_epi64((__m128i *) temp);
            temp += dst_strd;

            m_temp_reg_33 = _mm_loadl_epi64((__m128i *) temp);
            temp += dst_strd;

            m_temp_reg_34 = _mm_loadl_epi64((__m128i *) temp);

            /* Use shuffle to extract U or V Pred values */
            m_temp_reg_0 = _mm_shuffle_epi8(m_temp_reg_10, smask);
            m_temp_reg_1 = _mm_shuffle_epi8(m_temp_reg_11, smask);
            m_temp_reg_2 = _mm_shuffle_epi8(m_temp_reg_12, smask);
            m_temp_reg_3 = _mm_shuffle_epi8(m_temp_reg_13, smask);

            m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
            m_temp_reg_1 = _mm_cvtepu8_epi16(m_temp_reg_1);
            m_temp_reg_0 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_1);
            m_temp_reg_2 = _mm_cvtepu8_epi16(m_temp_reg_2);
            m_temp_reg_3 = _mm_cvtepu8_epi16(m_temp_reg_3);
            m_temp_reg_1 = _mm_unpacklo_epi64(m_temp_reg_2, m_temp_reg_3);

            m_temp_reg_20 = _mm_add_epi16(m_temp_reg_20, m_temp_reg_0);
            m_temp_reg_21 = _mm_add_epi16(m_temp_reg_21, m_temp_reg_1);

            m_temp_reg_0 = _mm_packus_epi16(m_temp_reg_20, m_temp_reg_21);

            /* Make the Values interleaved using shuffle*/
            m_temp_reg_10 = _mm_shuffle_epi8(m_temp_reg_0, un_smask);
            m_temp_reg_20 = _mm_blendv_epi8(m_temp_reg_10,m_temp_reg_30,blend_mask);

            _mm_storel_epi64((__m128i *)pu4_dst, m_temp_reg_20);
            m_temp_reg_1 = _mm_srli_si128(m_temp_reg_0, 4);

            m_temp_reg_11 = _mm_shuffle_epi8(m_temp_reg_1, un_smask);
            m_temp_reg_21 = _mm_blendv_epi8(m_temp_reg_11,m_temp_reg_31,blend_mask);

            m_temp_reg_2 = _mm_srli_si128(m_temp_reg_0, 8);

            m_temp_reg_12 = _mm_shuffle_epi8(m_temp_reg_2, un_smask);
            m_temp_reg_22 = _mm_blendv_epi8(m_temp_reg_12,m_temp_reg_33,blend_mask);

            m_temp_reg_3 = _mm_srli_si128(m_temp_reg_0, 12);

            m_temp_reg_13 = _mm_shuffle_epi8(m_temp_reg_3, un_smask);
            m_temp_reg_23 = _mm_blendv_epi8(m_temp_reg_13,m_temp_reg_34,blend_mask);

            pu1_dst += dst_strd;
            pu4_dst = (UWORD32 *) (pu1_dst);

            _mm_storel_epi64((__m128i *)pu4_dst, m_temp_reg_21);

            pu1_dst += dst_strd;
            pu4_dst = (UWORD32 *) (pu1_dst);

            _mm_storel_epi64((__m128i *)pu4_dst, m_temp_reg_22);

            pu1_dst += dst_strd;
            pu4_dst = (UWORD32 *) (pu1_dst);

            _mm_storel_epi64((__m128i *)pu4_dst, m_temp_reg_23);
        }
    }
}

