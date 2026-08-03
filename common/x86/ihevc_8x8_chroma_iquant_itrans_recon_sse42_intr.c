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
 *  ihevc_8x8_chroma_iquant_itrans_recon_x86_intr.c
 *
 * @brief
 *  Contains function definitions for inverse  quantization, inverse
 * transform and reconstruction  of chroma interleaved data.
 *
 * @author
 *  100491
 *
 * @par List of Functions:
 *   - ihevc_chroma_iquant_itrans_recon_8x8()
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

UWORD8 IHEVCE_UTIL_CHROMA_8x8_SHUFFLEMASK[16] = { 0x0, 0x2, 0x4, 0x6,
    0x8, 0xA, 0xC, 0xE,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF};

UWORD8 IHEVCE_UTIL_CHROMA_8x8_UN_SHUFFLEMASK[16] = { 0x0, 0xFF, 0x1, 0xFF,
    0x2, 0xFF, 0x3, 0xFF,
    0x4, 0xFF, 0x5, 0xFF,
    0x6, 0xFF, 0x7, 0xFF};

UWORD8 IHEVCE_UTIL_CHROMA_8x8_BLENDMASK[16] = { 0x0, 0xFF, 0x0, 0xFF,
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
 * reconstruction for 8c8 input block
 *
 * @par Description:
 *  Performs inverse quantization , inverse transform  and adds the
 * prediction data and clips output to 8 bit
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 8x8 buffer for storing inverse
 *  transform 1st stage output
 *
 * @param[in] pu1_pred
 *  Prediction 8x8 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu1_dst
 *  Output 8x8 block
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] qp_div
 *  Quantization parameter / 6
 *
 * @param[in] qp_rem
 *  Quantization parameter % 6
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
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
void ihevc_chroma_iquant_itrans_recon_8x8_sse42(WORD16 *pi2_src,
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
    __m128i shift_in_iquant_minus_1;
    __m128i m_temp_reg_0;
    __m128i m_temp_reg_1;
    __m128i m_temp_reg_2;
    __m128i m_temp_reg_3;
    __m128i m_temp_reg_5;
    __m128i m_temp_reg_6;
    __m128i m_temp_reg_7;
    __m128i m_temp_reg_4;
    __m128i m_temp_reg_10;
    __m128i m_temp_reg_11;
    __m128i m_temp_reg_12;
    __m128i m_temp_reg_13;
    __m128i m_temp_reg_14;
    __m128i m_temp_reg_15;
    __m128i m_temp_reg_16;
    __m128i m_temp_reg_17;
    __m128i m_temp_reg_20;
    __m128i m_temp_reg_21;
    __m128i m_temp_reg_22;
    __m128i m_temp_reg_23;
    __m128i m_temp_reg_24;
    __m128i m_temp_reg_25;
    __m128i m_temp_reg_26;
    __m128i m_temp_reg_27;
    __m128i m_temp_reg_30;
    __m128i m_temp_reg_31;
    __m128i m_temp_reg_32;
    __m128i m_temp_reg_33;
    __m128i m_temp_reg_34;
    __m128i m_temp_reg_35;
    __m128i m_temp_reg_36;
    __m128i m_temp_reg_37;
    __m128i m_temp_reg_40;
    __m128i m_temp_reg_41;
    __m128i m_temp_reg_42;
    __m128i m_temp_reg_43;
    __m128i m_temp_reg_44;
    __m128i m_temp_reg_45;
    __m128i m_temp_reg_46;
    __m128i m_temp_reg_47;
    __m128i m_temp_reg_50;
    __m128i m_temp_reg_51;
    __m128i m_temp_reg_52;
    __m128i m_temp_reg_53;
    __m128i m_temp_reg_54;
    __m128i m_temp_reg_55;
    __m128i m_temp_reg_56;
    __m128i m_temp_reg_57;
    __m128i m_temp_reg_60;
    __m128i m_temp_reg_61;
    __m128i m_temp_reg_62;
    __m128i m_temp_reg_63;
    __m128i m_temp_reg_64;
    __m128i m_temp_reg_65;
    __m128i m_temp_reg_66;
    __m128i m_temp_reg_67;
    __m128i m_temp_reg_70;
    __m128i m_temp_reg_71;
    __m128i m_temp_reg_72;
    __m128i m_temp_reg_73;
    __m128i m_temp_reg_74;
    __m128i m_temp_reg_75;
    __m128i m_temp_reg_76;
    __m128i m_temp_reg_77;
    __m128i m_coeff1, m_coeff2, m_coeff3, m_coeff4;
    __m128i smask, un_smask, blend_mask;

    WORD32 shift_select;
    WORD32 i4_shift = IT_SHIFT_STAGE_1;
    WORD32 check_row_stage_1;
    WORD32 check_row_stage_2;
    WORD32 one = 1;
    WORD32 trans_size = TRANS_SIZE_8;

    __m128i m_add_iq = _mm_cvtsi32_si128(one);
    __m128i m_rdng_factor;
    __m128i m_count;
    __m128i m_scale = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));

    /* Check for zero or non_zero: If check = 1, it is non_zero rows */
    check_row_stage_1   = ((zero_rows & 0xF0) != 0xF0) ? 1 : 0;
    check_row_stage_2   = ((zero_cols & 0xF0) != 0xF0) ? 1 : 0;

    smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_8x8_SHUFFLEMASK[0]);
    un_smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_8x8_UN_SHUFFLEMASK[0]);
    blend_mask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_8x8_BLENDMASK[0]);

    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size, bit_depth;

        log2_trans_size = 3;
        bit_depth = 8;
        shift_iq = bit_depth + log2_trans_size - 5;
    }

    /* Following 3 instructions replicates the value in the */
    /* lower 16 bits of m_add_iq in the entire register */
    m_add_iq = _mm_unpacklo_epi32(m_add_iq, m_add_iq);
    m_add_iq = _mm_unpacklo_epi64(m_add_iq, m_add_iq);

    /* Values of certain variables change wrt this condition */
    if(qp_div > shift_iq)
    {
        shift_in_iquant_minus_1 = _mm_cvtsi32_si128(-(shift_iq - qp_div - 1));
        m_add_iq = _mm_srl_epi16(m_add_iq, shift_in_iquant_minus_1);
        shift_in_iquant = _mm_cvtsi32_si128(-(shift_iq - qp_div));
        shift_select = 0;
    }
    else
    {
        shift_in_iquant_minus_1 = _mm_cvtsi32_si128((shift_iq - qp_div - 1));
        m_add_iq = _mm_sll_epi16(m_add_iq, shift_in_iquant_minus_1);
        shift_in_iquant = _mm_cvtsi32_si128((shift_iq - qp_div));
        shift_select = 1;
    }

    /* IQUANT */
    {
        /* row 0 */
        m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_src);
        m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
        m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
        m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
        m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
        pi2_src += src_strd;
        pi2_dequant_coeff += trans_size;

        /* row 1 */
        m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_src);
        m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
        m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
        m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
        m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
        pi2_src += src_strd;
        pi2_dequant_coeff += trans_size;

        /* row 0 */
        m_temp_reg_30 = _mm_cvtepi16_epi32(m_temp_reg_50);
        m_temp_reg_50 = _mm_srli_si128(m_temp_reg_50, 8);
        m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_temp_reg_30);
        m_temp_reg_50 = _mm_cvtepi16_epi32(m_temp_reg_50);
        m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_60, m_temp_reg_50);

        /* row 1 */
        m_temp_reg_31 = _mm_cvtepi16_epi32(m_temp_reg_51);
        m_temp_reg_51 = _mm_srli_si128(m_temp_reg_51, 8);
        m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_1, m_temp_reg_31);
        m_temp_reg_51 = _mm_cvtepi16_epi32(m_temp_reg_51);
        m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_61, m_temp_reg_51);

        /* row 2 */
        m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_src);
        m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
        m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
        m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
        m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
        pi2_src += src_strd;
        pi2_dequant_coeff += trans_size;

        /* row 0 */
        m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_add_iq);
        m_temp_reg_11 = _mm_add_epi32(m_temp_reg_11, m_add_iq);

        /* row 1 */
        m_temp_reg_12 = _mm_add_epi32(m_temp_reg_12, m_add_iq);
        m_temp_reg_13 = _mm_add_epi32(m_temp_reg_13, m_add_iq);

        if(shift_select)
        {
            m_temp_reg_10 = _mm_sra_epi32(m_temp_reg_10, shift_in_iquant);
            m_temp_reg_11 = _mm_sra_epi32(m_temp_reg_11, shift_in_iquant);
            m_temp_reg_12 = _mm_sra_epi32(m_temp_reg_12, shift_in_iquant);
            m_temp_reg_13 = _mm_sra_epi32(m_temp_reg_13, shift_in_iquant);
        }
        else
        {
            m_temp_reg_10 = _mm_sll_epi32(m_temp_reg_10, shift_in_iquant);
            m_temp_reg_11 = _mm_sll_epi32(m_temp_reg_11, shift_in_iquant);
            m_temp_reg_12 = _mm_sll_epi32(m_temp_reg_12, shift_in_iquant);
            m_temp_reg_13 = _mm_sll_epi32(m_temp_reg_13, shift_in_iquant);
        }

        m_temp_reg_70 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
        m_temp_reg_71 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

        /* row 3 */
        m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_src);
        m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
        m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
        m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
        m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
        m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);

        /* row 2 */
        m_temp_reg_30 = _mm_cvtepi16_epi32(m_temp_reg_50);
        m_temp_reg_50 = _mm_srli_si128(m_temp_reg_50, 8);
        m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_temp_reg_30);
        m_temp_reg_50 = _mm_cvtepi16_epi32(m_temp_reg_50);
        m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_60, m_temp_reg_50);

        /* row 3 */
        m_temp_reg_31 = _mm_cvtepi16_epi32(m_temp_reg_51);
        m_temp_reg_51 = _mm_srli_si128(m_temp_reg_51, 8);
        m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_1, m_temp_reg_31);
        m_temp_reg_51 = _mm_cvtepi16_epi32(m_temp_reg_51);
        m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_61, m_temp_reg_51);

        /* row 2 */
        m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_add_iq);
        m_temp_reg_11 = _mm_add_epi32(m_temp_reg_11, m_add_iq);

        /* row 3 */
        m_temp_reg_12 = _mm_add_epi32(m_temp_reg_12, m_add_iq);
        m_temp_reg_13 = _mm_add_epi32(m_temp_reg_13, m_add_iq);

        if(shift_select)
        {
            m_temp_reg_10 = _mm_sra_epi32(m_temp_reg_10, shift_in_iquant);
            m_temp_reg_11 = _mm_sra_epi32(m_temp_reg_11, shift_in_iquant);
            m_temp_reg_12 = _mm_sra_epi32(m_temp_reg_12, shift_in_iquant);
            m_temp_reg_13 = _mm_sra_epi32(m_temp_reg_13, shift_in_iquant);
        }
        else
        {
            m_temp_reg_10 = _mm_sll_epi32(m_temp_reg_10, shift_in_iquant);
            m_temp_reg_11 = _mm_sll_epi32(m_temp_reg_11, shift_in_iquant);
            m_temp_reg_12 = _mm_sll_epi32(m_temp_reg_12, shift_in_iquant);
            m_temp_reg_13 = _mm_sll_epi32(m_temp_reg_13, shift_in_iquant);
        }

        m_temp_reg_72 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
        m_temp_reg_73 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

        if(check_row_stage_1) /* Only First 4 rows of input are non-zero */
        {
            pi2_src += src_strd;
            pi2_dequant_coeff += trans_size;

            /* row 4 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_src += src_strd;
            pi2_dequant_coeff += trans_size;

            /* row 5 */
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
            pi2_src += src_strd;
            pi2_dequant_coeff += trans_size;

            /* row 4 */
            m_temp_reg_30 = _mm_cvtepi16_epi32(m_temp_reg_50);
            m_temp_reg_50 = _mm_srli_si128(m_temp_reg_50, 8);
            m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_temp_reg_30);
            m_temp_reg_50 = _mm_cvtepi16_epi32(m_temp_reg_50);
            m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_60, m_temp_reg_50);

            /* row 5 */
            m_temp_reg_31 = _mm_cvtepi16_epi32(m_temp_reg_51);
            m_temp_reg_51 = _mm_srli_si128(m_temp_reg_51, 8);
            m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_1, m_temp_reg_31);
            m_temp_reg_51 = _mm_cvtepi16_epi32(m_temp_reg_51);
            m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_61, m_temp_reg_51);

            /* row 6 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_src += src_strd;
            pi2_dequant_coeff += trans_size;

            /* row 4 */
            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_add_iq);
            m_temp_reg_11 = _mm_add_epi32(m_temp_reg_11, m_add_iq);

            /* row 5 */
            m_temp_reg_12 = _mm_add_epi32(m_temp_reg_12, m_add_iq);
            m_temp_reg_13 = _mm_add_epi32(m_temp_reg_13, m_add_iq);

            if(shift_select)
            {
                m_temp_reg_10 = _mm_sra_epi32(m_temp_reg_10, shift_in_iquant);
                m_temp_reg_11 = _mm_sra_epi32(m_temp_reg_11, shift_in_iquant);
                m_temp_reg_12 = _mm_sra_epi32(m_temp_reg_12, shift_in_iquant);
                m_temp_reg_13 = _mm_sra_epi32(m_temp_reg_13, shift_in_iquant);
            }
            else
            {
                m_temp_reg_10 = _mm_sll_epi32(m_temp_reg_10, shift_in_iquant);
                m_temp_reg_11 = _mm_sll_epi32(m_temp_reg_11, shift_in_iquant);
                m_temp_reg_12 = _mm_sll_epi32(m_temp_reg_12, shift_in_iquant);
                m_temp_reg_13 = _mm_sll_epi32(m_temp_reg_13, shift_in_iquant);
            }

            m_temp_reg_74 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_75 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

            /* row 7 */
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_dequant_coeff);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);

            /* row 6 */
            m_temp_reg_30 = _mm_cvtepi16_epi32(m_temp_reg_50);
            m_temp_reg_50 = _mm_srli_si128(m_temp_reg_50, 8);
            m_temp_reg_50 = _mm_cvtepi16_epi32(m_temp_reg_50);
            m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_temp_reg_30);
            m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_60, m_temp_reg_50);

            /* row 7 */
            m_temp_reg_31 = _mm_cvtepi16_epi32(m_temp_reg_51);
            m_temp_reg_51 = _mm_srli_si128(m_temp_reg_51, 8);
            m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_1, m_temp_reg_31);
            m_temp_reg_51 = _mm_cvtepi16_epi32(m_temp_reg_51);
            m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_61, m_temp_reg_51);

            /* row 6 */
            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_10, m_add_iq);
            m_temp_reg_11 = _mm_add_epi32(m_temp_reg_11, m_add_iq);

            /* row 7 */
            m_temp_reg_12 = _mm_add_epi32(m_temp_reg_12, m_add_iq);
            m_temp_reg_13 = _mm_add_epi32(m_temp_reg_13, m_add_iq);

            if(shift_select)
            {
                m_temp_reg_10 = _mm_sra_epi32(m_temp_reg_10, shift_in_iquant);
                m_temp_reg_11 = _mm_sra_epi32(m_temp_reg_11, shift_in_iquant);
                m_temp_reg_12 = _mm_sra_epi32(m_temp_reg_12, shift_in_iquant);
                m_temp_reg_13 = _mm_sra_epi32(m_temp_reg_13, shift_in_iquant);
            }
            else
            {
                m_temp_reg_10 = _mm_sll_epi32(m_temp_reg_10, shift_in_iquant);
                m_temp_reg_11 = _mm_sll_epi32(m_temp_reg_11, shift_in_iquant);
                m_temp_reg_12 = _mm_sll_epi32(m_temp_reg_12, shift_in_iquant);
                m_temp_reg_13 = _mm_sll_epi32(m_temp_reg_13, shift_in_iquant);
            }

            m_temp_reg_76 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_77 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);
        }
        else
        {
            /* memset all the co-eff's in bottom 4 rows to zero , Zero_rows Support */
             m_temp_reg_74 = _mm_setzero_si128();
             m_temp_reg_75 = _mm_setzero_si128();
             m_temp_reg_76 = _mm_setzero_si128();
             m_temp_reg_77 = _mm_setzero_si128();
        }
    }

    {
        /* ee0 is present in the registers m_temp_reg_10 and m_temp_reg_11 */
        /* ee1 is present in the registers m_temp_reg_12 and m_temp_reg_13 */
        {
            /* Combining instructions to eliminate them based on zero_rows */

            if(check_row_stage_1) /* Only First 4 rows of input are non-zero */
            {
                m_temp_reg_2 = _mm_cvtepi16_epi32 (m_temp_reg_74);
                m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);
                m_temp_reg_2 = _mm_slli_epi32(m_temp_reg_2, 6);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_74);
                m_temp_reg_3 = _mm_slli_epi32(m_temp_reg_3, 6);
            }
            else
            {
                m_temp_reg_2 = _mm_setzero_si128();
                m_temp_reg_3 = _mm_setzero_si128();
            }

            m_temp_reg_0 = _mm_cvtepi16_epi32 (m_temp_reg_70);

            m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

            m_temp_reg_0 = _mm_slli_epi32(m_temp_reg_0, 6);

            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

            m_temp_reg_1 = _mm_slli_epi32(m_temp_reg_1, 6);

            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_12 = _mm_sub_epi32(m_temp_reg_0, m_temp_reg_2);

            m_temp_reg_0 = _mm_cvtepi16_epi32 (m_temp_reg_72);

            m_temp_reg_11 = _mm_add_epi32(m_temp_reg_1, m_temp_reg_3);
            m_temp_reg_13 = _mm_sub_epi32(m_temp_reg_1, m_temp_reg_3);
        }


        /* eo0 is present in the registers m_temp_reg_14 and m_temp_reg_15 */
        /* eo1 is present in the registers m_temp_reg_16 and m_temp_reg_17 */
        {

            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[2][0]);
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[1][0]);

            /* Combining instructions to eliminate them based on zero_rows */

            if(check_row_stage_1) /* Only First 4 rows of input are non-zero */
            {
                m_temp_reg_2  = _mm_cvtepi16_epi32 (m_temp_reg_76);
                m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);
                m_temp_reg_3  = _mm_cvtepi16_epi32(m_temp_reg_76);
                m_temp_reg_6  = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
                m_temp_reg_2  = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                m_temp_reg_7  = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);
                m_temp_reg_3  = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);
            }
            else
            {
                m_temp_reg_6  = _mm_setzero_si128();
                m_temp_reg_7  = _mm_setzero_si128();
            }

            m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);

            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_72);

            m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);

            m_temp_reg_0 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);

            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[2][0]);
            m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[3][0]);

            m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);

            m_temp_reg_1 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);


            m_temp_reg_16 = _mm_sub_epi32(m_temp_reg_4, m_temp_reg_6);
            m_temp_reg_14 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);

            /* Loading coeff for computing o0, o1, o2 and o3 in the next block */
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[0][0]);
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[1][0]);

            m_temp_reg_60 = _mm_cvtepi16_epi32 (m_temp_reg_71);
            m_temp_reg_62 = _mm_cvtepi16_epi32 (m_temp_reg_73);

            m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
            m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);

            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_71);
            m_temp_reg_63 = _mm_cvtepi16_epi32(m_temp_reg_73);

            m_temp_reg_17 = _mm_sub_epi32(m_temp_reg_5, m_temp_reg_7);
            m_temp_reg_15 = _mm_add_epi32(m_temp_reg_1, m_temp_reg_3);
        }

        /* e */
        {
            /* e0 stored in m_temp_reg_40 and m_temp_reg_41 */
            /* e1 stored in m_temp_reg_42 and m_temp_reg_43 */
            /* e3 stored in m_temp_reg_46 and m_temp_reg_47 */
            /* e2 stored in m_temp_reg_44 and m_temp_reg_45 */
            m_temp_reg_42 = _mm_add_epi32(m_temp_reg_12, m_temp_reg_16);
            m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_16);

            m_temp_reg_40 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_14);
            m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_14);

            m_temp_reg_43 = _mm_add_epi32(m_temp_reg_13, m_temp_reg_17);
            m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_13, m_temp_reg_17);

            m_temp_reg_41 = _mm_add_epi32(m_temp_reg_11, m_temp_reg_15);
            m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_11, m_temp_reg_15);

        }

        /* o */
        {
            if(check_row_stage_1)
            {
                m_temp_reg_64 = _mm_cvtepi16_epi32 (m_temp_reg_75);
                m_temp_reg_66 = _mm_cvtepi16_epi32 (m_temp_reg_77);

                m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);

                m_temp_reg_65 = _mm_cvtepi16_epi32(m_temp_reg_75);
                m_temp_reg_67 = _mm_cvtepi16_epi32(m_temp_reg_77);

                m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_64, m_coeff3);
                m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_65, m_coeff3);
                m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_66, m_coeff4);
                m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_67, m_coeff4);

            }

            else
            {
                m_temp_reg_24 = _mm_setzero_si128();
                m_temp_reg_25 = _mm_setzero_si128();
                m_temp_reg_26 = _mm_setzero_si128();
                m_temp_reg_27 = _mm_setzero_si128();
            }

            /* o0 stored in m_temp_reg_30 and m_temp_reg_31 */
            {
                m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_60, m_coeff1);
                m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_62, m_coeff2);
                m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_61, m_coeff1);
                m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_63, m_coeff2);

                m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                m_count = _mm_cvtsi32_si128(i4_shift);
                m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                m_temp_reg_30 = _mm_add_epi32(m_temp_reg_20, m_temp_reg_22);
                m_temp_reg_31 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_23);
                m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
            }

            /* Column 0 of destination computed here */
            /* It is stored in m_temp_reg_50 */
            /* Column 7 of destination computed here */
            /* It is stored in m_temp_reg_57 */
            {
                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_31);
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_31);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);


                m_temp_reg_50 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                m_temp_reg_57 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
            }

            /* o1 stored in m_temp_reg_32 and m_temp_reg_33 */
            {
                if(check_row_stage_1)
                {
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_64, m_coeff1);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_65, m_coeff1);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_66, m_coeff3);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_67, m_coeff3);
                }

                m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_60, m_coeff2);
                m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_62, m_coeff4);
                m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_61, m_coeff2);
                m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_63, m_coeff4);

                m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_32, m_temp_reg_24);
                m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_33, m_temp_reg_25);
                m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_32, m_temp_reg_26);
                m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_33, m_temp_reg_27);
            }

            /* Column 1 of destination computed here */
            /* It is stored in m_temp_reg_51 */
            /* Column 6 of destination computed here */
            /* It is stored in m_temp_reg_56 */
            {
                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_32);
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_32);

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_33);
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_33);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                m_temp_reg_51 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                m_temp_reg_56 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
            }

            /* o2 stored in m_temp_reg_34 and m_temp_reg_35 */
            {
                m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_60, m_coeff3);
                m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_62, m_coeff1);
                m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_61, m_coeff3);
                m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_63, m_coeff1);

                if(check_row_stage_1)
                {
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_64, m_coeff4);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_65, m_coeff4);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_66, m_coeff2);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_67, m_coeff2);
                }

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_24);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_25);
                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_26);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_27);
            }

            /* Column 2 of destination computed here */
            /* It is stored in m_temp_reg_52 */
            /* Column 5 of destination computed here */
            /* It is stored in m_temp_reg_55 */
            {
                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_34);
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_34);

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_35);
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_35);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                m_temp_reg_52 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                m_temp_reg_55 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
            }


            /* o3 stored in m_temp_reg_36 and m_temp_reg_37 */
            {
                m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_60, m_coeff4);
                m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_62, m_coeff3);
                m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_61, m_coeff4);
                m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_63, m_coeff3);

                if(check_row_stage_1)
                {
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_64, m_coeff2);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_65, m_coeff2);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_66, m_coeff1);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_67, m_coeff1);
                }

                m_temp_reg_36 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                m_temp_reg_37 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                m_temp_reg_36 = _mm_add_epi32(m_temp_reg_36, m_temp_reg_24);
                m_temp_reg_37 = _mm_add_epi32(m_temp_reg_37, m_temp_reg_25);
                m_temp_reg_36 = _mm_sub_epi32(m_temp_reg_36, m_temp_reg_26);
                m_temp_reg_37 = _mm_sub_epi32(m_temp_reg_37, m_temp_reg_27);
            }

            /* Column 3 of destination computed here */
            /* It is stored in m_temp_reg_53 */
            /* Column 4 of destination computed here */
            /* It is stored in m_temp_reg_54 */
            {
                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_36);
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_36);

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_37);
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_37);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                m_temp_reg_53 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                m_temp_reg_54 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
            }
        }

        /* Transpose of the destination 8x8 matrix done here */
        /* and ultimately stored in registers m_temp_reg_50 to m_temp_reg_57 */
        /* respectively */
        {
            m_temp_reg_10 = _mm_unpacklo_epi16(m_temp_reg_50, m_temp_reg_51);
            m_temp_reg_11 = _mm_unpacklo_epi16(m_temp_reg_52, m_temp_reg_53);
            m_temp_reg_14 = _mm_unpackhi_epi16(m_temp_reg_50, m_temp_reg_51);
            m_temp_reg_15 = _mm_unpackhi_epi16(m_temp_reg_52, m_temp_reg_53);
            m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_1 = _mm_unpackhi_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_14, m_temp_reg_15);
            m_temp_reg_3 = _mm_unpackhi_epi32(m_temp_reg_14, m_temp_reg_15);

            m_temp_reg_12 = _mm_unpacklo_epi16(m_temp_reg_54, m_temp_reg_55);
            m_temp_reg_13 = _mm_unpacklo_epi16(m_temp_reg_56, m_temp_reg_57);
            m_temp_reg_16 = _mm_unpackhi_epi16(m_temp_reg_54, m_temp_reg_55);
            m_temp_reg_17 = _mm_unpackhi_epi16(m_temp_reg_56, m_temp_reg_57);
            m_temp_reg_4 = _mm_unpacklo_epi32(m_temp_reg_12, m_temp_reg_13);
            m_temp_reg_5 = _mm_unpackhi_epi32(m_temp_reg_12, m_temp_reg_13);
            m_temp_reg_6 = _mm_unpacklo_epi32(m_temp_reg_16, m_temp_reg_17);
            m_temp_reg_7 = _mm_unpackhi_epi32(m_temp_reg_16, m_temp_reg_17);

            m_temp_reg_50 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_4);
            m_temp_reg_51 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_4);
            m_temp_reg_52 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_5);
            m_temp_reg_53 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_5);

            m_temp_reg_54 = _mm_unpacklo_epi64(m_temp_reg_2, m_temp_reg_6);
            m_temp_reg_55 = _mm_unpackhi_epi64(m_temp_reg_2, m_temp_reg_6);
            m_temp_reg_56 = _mm_unpacklo_epi64(m_temp_reg_3, m_temp_reg_7);
            m_temp_reg_57 = _mm_unpackhi_epi64(m_temp_reg_3, m_temp_reg_7);
        }
    }

    /* Stage 2 */
    {
        i4_shift = IT_SHIFT_STAGE_2;

        if(check_row_stage_2)
        {
            /* ee0 is present in the registers m_temp_reg_10 and m_temp_reg_11 */
            /* ee1 is present in the registers m_temp_reg_12 and m_temp_reg_13 */
            {
                m_temp_reg_0 = _mm_cvtepi16_epi32 (m_temp_reg_50);
                m_temp_reg_2 = _mm_cvtepi16_epi32 (m_temp_reg_54);

                m_temp_reg_0 = _mm_slli_epi32(m_temp_reg_0, 6);
                m_temp_reg_2 = _mm_slli_epi32(m_temp_reg_2, 6);

                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_50, 8);
                m_temp_reg_3 = _mm_srli_si128(m_temp_reg_54, 8);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);

                m_temp_reg_1 = _mm_slli_epi32(m_temp_reg_1, 6);
                m_temp_reg_3 = _mm_slli_epi32(m_temp_reg_3, 6);

                m_temp_reg_10 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_12 = _mm_sub_epi32(m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_56);
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_52);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[1][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[2][0]);

                m_temp_reg_11 = _mm_add_epi32(m_temp_reg_1, m_temp_reg_3);
                m_temp_reg_13 = _mm_sub_epi32(m_temp_reg_1, m_temp_reg_3);
            }


            /* eo0 is present in the registers m_temp_reg_14 and m_temp_reg_15 */
            /* eo1 is present in the registers m_temp_reg_16 and m_temp_reg_17 */
            {
                m_temp_reg_66 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
                m_temp_reg_64 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                m_temp_reg_62 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                m_temp_reg_60 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);

                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_52, 8);
                m_temp_reg_3 = _mm_srli_si128(m_temp_reg_56, 8);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);

                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[3][0]);

                m_temp_reg_65 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
                m_temp_reg_67 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);
                m_temp_reg_61 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                m_temp_reg_63 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);

                m_temp_reg_16 = _mm_sub_epi32(m_temp_reg_64, m_temp_reg_66);
                m_temp_reg_14 = _mm_add_epi32(m_temp_reg_60, m_temp_reg_62);

                /* Loading coeff for computing o0, o1, o2 and o3 in the next block */
                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[1][0]);

                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_51);
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_53);
                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_51, 8);
                m_temp_reg_3 = _mm_srli_si128(m_temp_reg_53, 8);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);

                m_temp_reg_17 = _mm_sub_epi32(m_temp_reg_65, m_temp_reg_67);
                m_temp_reg_15 = _mm_add_epi32(m_temp_reg_61, m_temp_reg_63);
            }

            /* e */
            {
                /* e0 stored in m_temp_reg_40 and m_temp_reg_41 */
                /* e1 stored in m_temp_reg_42 and m_temp_reg_43 */
                /* e3 stored in m_temp_reg_46 and m_temp_reg_47 */
                /* e2 stored in m_temp_reg_44 and m_temp_reg_45 */
                m_temp_reg_42 = _mm_add_epi32(m_temp_reg_12, m_temp_reg_16);
                m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_16);

                m_temp_reg_40 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_14);
                m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_14);

                m_temp_reg_43 = _mm_add_epi32(m_temp_reg_13, m_temp_reg_17);
                m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_13, m_temp_reg_17);

                m_temp_reg_41 = _mm_add_epi32(m_temp_reg_11, m_temp_reg_15);
                m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_11, m_temp_reg_15);

            }

            /* o */
            {
                m_temp_reg_4 = _mm_cvtepi16_epi32(m_temp_reg_55);
                m_temp_reg_5 = _mm_srli_si128(m_temp_reg_55, 8);
                m_temp_reg_6 = _mm_cvtepi16_epi32(m_temp_reg_57);
                m_temp_reg_7 = _mm_srli_si128(m_temp_reg_57, 8);
                m_temp_reg_5 = _mm_cvtepi16_epi32(m_temp_reg_5);
                m_temp_reg_7 = _mm_cvtepi16_epi32(m_temp_reg_7);

                /* o0 stored in m_temp_reg_30 and m_temp_reg_31 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_4, m_coeff3);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_5, m_coeff3);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_6, m_coeff4);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_7, m_coeff4);

                    m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                    m_count = _mm_cvtsi32_si128(i4_shift);
                    m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                    m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_23);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                }

                /* Column 0 of destination computed here */
                /* It is stored in m_temp_reg_50 */
                /* Column 7 of destination computed here */
                /* It is stored in m_temp_reg_57 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_31);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_31);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);


                    m_temp_reg_50 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_57 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }

                /* o1 stored in m_temp_reg_32 and m_temp_reg_33 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff4);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff4);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_4, m_coeff1);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_5, m_coeff1);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_6, m_coeff3);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_7, m_coeff3);

                    m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                    m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_32, m_temp_reg_24);
                    m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_33, m_temp_reg_25);
                    m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_32, m_temp_reg_26);
                    m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_33, m_temp_reg_27);
                }

                /* Column 1 of destination computed here */
                /* It is stored in m_temp_reg_51 */
                /* Column 6 of destination computed here */
                /* It is stored in m_temp_reg_56 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_32);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_32);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_33);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_33);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_51 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_56 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }

                /* o2 stored in m_temp_reg_34 and m_temp_reg_35 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff3);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_4, m_coeff4);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_5, m_coeff4);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_6, m_coeff2);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_7, m_coeff2);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_24);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_25);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_26);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_27);
                }

                /* Column 2 of destination computed here */
                /* It is stored in m_temp_reg_52 */
                /* Column 5 of destination computed here */
                /* It is stored in m_temp_reg_55 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_34);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_34);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_35);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_35);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_52 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_55 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }


                /* o3 stored in m_temp_reg_36 and m_temp_reg_37 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff4);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff3);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff4);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_4, m_coeff2);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_5, m_coeff2);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_6, m_coeff1);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_7, m_coeff1);

                    m_temp_reg_36 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_37 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                    m_temp_reg_36 = _mm_add_epi32(m_temp_reg_36, m_temp_reg_24);
                    m_temp_reg_37 = _mm_add_epi32(m_temp_reg_37, m_temp_reg_25);
                    m_temp_reg_36 = _mm_sub_epi32(m_temp_reg_36, m_temp_reg_26);
                    m_temp_reg_37 = _mm_sub_epi32(m_temp_reg_37, m_temp_reg_27);
                }

                /* Column 3 of destination computed here */
                /* It is stored in m_temp_reg_53 */
                /* Column 4 of destination computed here */
                /* It is stored in m_temp_reg_54 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_36);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_36);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_37);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_37);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_53 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_54 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }
            }
         }
         else
         {
             /* Optimized Chen's multiplication if bottom rows are "0" */

            /* ee0 is present in the registers m_temp_reg_10 and m_temp_reg_11 */
            /* ee1 is present in the registers m_temp_reg_12 and m_temp_reg_13 */
            {
                m_temp_reg_0 = _mm_cvtepi16_epi32 (m_temp_reg_50);
                m_temp_reg_0 = _mm_slli_epi32(m_temp_reg_0, 6);

                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_50, 8);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
                m_temp_reg_1 = _mm_slli_epi32(m_temp_reg_1, 6);

                m_temp_reg_10 = m_temp_reg_0;

                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_52);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[1][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[2][0]);

                m_temp_reg_11 = m_temp_reg_1;

            }


            /* eo0 is present in the registers m_temp_reg_14 and m_temp_reg_15 */
            /* eo1 is present in the registers m_temp_reg_16 and m_temp_reg_17 */
            {

                m_temp_reg_64 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);

                m_temp_reg_60 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);

                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_52, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);


                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[3][0]);

                m_temp_reg_65 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);

                m_temp_reg_61 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);


                m_temp_reg_16 = m_temp_reg_64;
                m_temp_reg_14 = m_temp_reg_60;

                /* Loading coeff for computing o0, o1, o2 and o3 in the next block */
                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[1][0]);

                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_51);
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_53);
                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_51, 8);
                m_temp_reg_3 = _mm_srli_si128(m_temp_reg_53, 8);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);



                /* e */
                {
                    /* e0 stored in m_temp_reg_40 and m_temp_reg_41 */
                    /* e1 stored in m_temp_reg_42 and m_temp_reg_43 */
                    /* e3 stored in m_temp_reg_46 and m_temp_reg_47 */
                    /* e2 stored in m_temp_reg_44 and m_temp_reg_45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_16);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_16);

                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_14);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_14);

                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_11, m_temp_reg_65);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_11, m_temp_reg_65);

                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_11, m_temp_reg_61);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_11, m_temp_reg_61);

                }

            }
            {

                /* o0 stored in m_temp_reg_30 and m_temp_reg_31 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);

                    m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                    m_count = _mm_cvtsi32_si128(i4_shift);
                    m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                    m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_23);

                }

                /* Column 0 of destination computed here */
                /* It is stored in m_temp_reg_50 */
                /* Column 7 of destination computed here */
                /* It is stored in m_temp_reg_57 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_31);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_31);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);


                    m_temp_reg_50 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_57 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }

                /* o1 stored in m_temp_reg_32 and m_temp_reg_33 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff4);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff4);

                    m_temp_reg_32 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_33 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);

                }

                /* Column 1 of destination computed here */
                /* It is stored in m_temp_reg_51 */
                /* Column 6 of destination computed here */
                /* It is stored in m_temp_reg_56 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_32);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_32);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_33);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_33);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_51 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_56 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }

                /* o2 stored in m_temp_reg_34 and m_temp_reg_35 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff3);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                }

                /* Column 2 of destination computed here */
                /* It is stored in m_temp_reg_52 */
                /* Column 5 of destination computed here */
                /* It is stored in m_temp_reg_55 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_34);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_34);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_35);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_35);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_52 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_55 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }


                /* o3 stored in m_temp_reg_36 and m_temp_reg_37 */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_0, m_coeff4);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_2, m_coeff3);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_1, m_coeff4);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);

                    m_temp_reg_36 = _mm_sub_epi32(m_temp_reg_20, m_temp_reg_22);
                    m_temp_reg_37 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_23);
                }

                /* Column 3 of destination computed here */
                /* It is stored in m_temp_reg_53 */
                /* Column 4 of destination computed here */
                /* It is stored in m_temp_reg_54 */
                {
                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_36);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_36);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_37);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_37);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_20, m_rdng_factor);
                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_21, m_rdng_factor);
                    m_temp_reg_22 = _mm_add_epi32(m_temp_reg_22, m_rdng_factor);
                    m_temp_reg_23 = _mm_add_epi32(m_temp_reg_23, m_rdng_factor);

                    m_temp_reg_20 = _mm_sra_epi32(m_temp_reg_20, m_count);
                    m_temp_reg_21 = _mm_sra_epi32(m_temp_reg_21, m_count);
                    m_temp_reg_22 = _mm_sra_epi32(m_temp_reg_22, m_count);
                    m_temp_reg_23 = _mm_sra_epi32(m_temp_reg_23, m_count);

                    m_temp_reg_53 = _mm_packs_epi32(m_temp_reg_20, m_temp_reg_21);
                    m_temp_reg_54 = _mm_packs_epi32(m_temp_reg_22, m_temp_reg_23);
                }
            }
        }

        /* Transpose of the destination 8x8 matrix done here */
        /* and ultimately stored in registers m_temp_reg_50 to m_temp_reg_57 */
        /* respectively */
        {
            m_temp_reg_10 = _mm_unpacklo_epi16(m_temp_reg_50, m_temp_reg_51);
            m_temp_reg_11 = _mm_unpacklo_epi16(m_temp_reg_52, m_temp_reg_53);
            m_temp_reg_14 = _mm_unpackhi_epi16(m_temp_reg_50, m_temp_reg_51);
            m_temp_reg_15 = _mm_unpackhi_epi16(m_temp_reg_52, m_temp_reg_53);
            m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_1 = _mm_unpackhi_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_14, m_temp_reg_15);
            m_temp_reg_3 = _mm_unpackhi_epi32(m_temp_reg_14, m_temp_reg_15);

            m_temp_reg_12 = _mm_unpacklo_epi16(m_temp_reg_54, m_temp_reg_55);
            m_temp_reg_13 = _mm_unpacklo_epi16(m_temp_reg_56, m_temp_reg_57);
            m_temp_reg_16 = _mm_unpackhi_epi16(m_temp_reg_54, m_temp_reg_55);
            m_temp_reg_17 = _mm_unpackhi_epi16(m_temp_reg_56, m_temp_reg_57);
            m_temp_reg_4 = _mm_unpacklo_epi32(m_temp_reg_12, m_temp_reg_13);
            m_temp_reg_5 = _mm_unpackhi_epi32(m_temp_reg_12, m_temp_reg_13);
            m_temp_reg_6 = _mm_unpacklo_epi32(m_temp_reg_16, m_temp_reg_17);
            m_temp_reg_7 = _mm_unpackhi_epi32(m_temp_reg_16, m_temp_reg_17);

            m_temp_reg_10 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_4);
            m_temp_reg_11 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_4);
            m_temp_reg_12 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_5);
            m_temp_reg_13 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_5);

            m_temp_reg_14 = _mm_unpacklo_epi64(m_temp_reg_2, m_temp_reg_6);
            m_temp_reg_15 = _mm_unpackhi_epi64(m_temp_reg_2, m_temp_reg_6);
            m_temp_reg_16 = _mm_unpacklo_epi64(m_temp_reg_3, m_temp_reg_7);
            m_temp_reg_17 = _mm_unpackhi_epi64(m_temp_reg_3, m_temp_reg_7);
        }

        /* Recon and store */
        {
            UWORD8 *temp = pu1_dst;

            m_temp_reg_0 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_1 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_2 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_3 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_4 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_5 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_6 = _mm_loadu_si128((__m128i *) pu1_pred);
            pu1_pred += pred_strd;
            m_temp_reg_7 = _mm_loadu_si128((__m128i *) pu1_pred);

            /* Load the recon values */
            m_temp_reg_20 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_21 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_22 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_23 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_24 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_25 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_26 = _mm_loadu_si128((__m128i *) temp);
            temp += dst_strd;
            m_temp_reg_27 = _mm_loadu_si128((__m128i *) temp);

            /* Use shuffle to extract U or V Pred values */
            m_temp_reg_30 = _mm_shuffle_epi8(m_temp_reg_0, smask);
            m_temp_reg_31 = _mm_shuffle_epi8(m_temp_reg_1, smask);
            m_temp_reg_32 = _mm_shuffle_epi8(m_temp_reg_2, smask);
            m_temp_reg_33 = _mm_shuffle_epi8(m_temp_reg_3, smask);
            m_temp_reg_34 = _mm_shuffle_epi8(m_temp_reg_4, smask);
            m_temp_reg_35 = _mm_shuffle_epi8(m_temp_reg_5, smask);
            m_temp_reg_36 = _mm_shuffle_epi8(m_temp_reg_6, smask);
            m_temp_reg_37 = _mm_shuffle_epi8(m_temp_reg_7, smask);

            m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_30);
            m_temp_reg_1 = _mm_cvtepu8_epi16(m_temp_reg_31);
            m_temp_reg_2 = _mm_cvtepu8_epi16(m_temp_reg_32);
            m_temp_reg_3 = _mm_cvtepu8_epi16(m_temp_reg_33);
            m_temp_reg_4 = _mm_cvtepu8_epi16(m_temp_reg_34);
            m_temp_reg_5 = _mm_cvtepu8_epi16(m_temp_reg_35);
            m_temp_reg_6 = _mm_cvtepu8_epi16(m_temp_reg_36);
            m_temp_reg_7 = _mm_cvtepu8_epi16(m_temp_reg_37);

            m_temp_reg_50 = _mm_add_epi16(m_temp_reg_10, m_temp_reg_0);
            m_temp_reg_51 = _mm_add_epi16(m_temp_reg_11, m_temp_reg_1);
            m_temp_reg_52 = _mm_add_epi16(m_temp_reg_12, m_temp_reg_2);
            m_temp_reg_53 = _mm_add_epi16(m_temp_reg_13, m_temp_reg_3);
            m_temp_reg_54 = _mm_add_epi16(m_temp_reg_14, m_temp_reg_4);
            m_temp_reg_55 = _mm_add_epi16(m_temp_reg_15, m_temp_reg_5);
            m_temp_reg_56 = _mm_add_epi16(m_temp_reg_16, m_temp_reg_6);
            m_temp_reg_57 = _mm_add_epi16(m_temp_reg_17, m_temp_reg_7);

            m_temp_reg_50 = _mm_packus_epi16(m_temp_reg_50, m_temp_reg_50);
            m_temp_reg_51 = _mm_packus_epi16(m_temp_reg_51, m_temp_reg_51);
            m_temp_reg_52 = _mm_packus_epi16(m_temp_reg_52, m_temp_reg_52);
            m_temp_reg_53 = _mm_packus_epi16(m_temp_reg_53, m_temp_reg_53);
            m_temp_reg_54 = _mm_packus_epi16(m_temp_reg_54, m_temp_reg_54);
            m_temp_reg_55 = _mm_packus_epi16(m_temp_reg_55, m_temp_reg_55);
            m_temp_reg_56 = _mm_packus_epi16(m_temp_reg_56, m_temp_reg_56);
            m_temp_reg_57 = _mm_packus_epi16(m_temp_reg_57, m_temp_reg_57);

            /* Make the recon Values interleaved using shuffle*/
            m_temp_reg_10 = _mm_shuffle_epi8(m_temp_reg_50, un_smask);
            m_temp_reg_11 = _mm_shuffle_epi8(m_temp_reg_51, un_smask);
            m_temp_reg_12 = _mm_shuffle_epi8(m_temp_reg_52, un_smask);
            m_temp_reg_13 = _mm_shuffle_epi8(m_temp_reg_53, un_smask);
            m_temp_reg_14 = _mm_shuffle_epi8(m_temp_reg_54, un_smask);
            m_temp_reg_15 = _mm_shuffle_epi8(m_temp_reg_55, un_smask);
            m_temp_reg_16 = _mm_shuffle_epi8(m_temp_reg_56, un_smask);
            m_temp_reg_17 = _mm_shuffle_epi8(m_temp_reg_57, un_smask);

            m_temp_reg_50 = _mm_blendv_epi8(m_temp_reg_10,m_temp_reg_20,blend_mask);
            m_temp_reg_51 = _mm_blendv_epi8(m_temp_reg_11,m_temp_reg_21,blend_mask);
            m_temp_reg_52 = _mm_blendv_epi8(m_temp_reg_12,m_temp_reg_22,blend_mask);
            m_temp_reg_53 = _mm_blendv_epi8(m_temp_reg_13,m_temp_reg_23,blend_mask);
            m_temp_reg_54 = _mm_blendv_epi8(m_temp_reg_14,m_temp_reg_24,blend_mask);
            m_temp_reg_55 = _mm_blendv_epi8(m_temp_reg_15,m_temp_reg_25,blend_mask);
            m_temp_reg_56 = _mm_blendv_epi8(m_temp_reg_16,m_temp_reg_26,blend_mask);
            m_temp_reg_57 = _mm_blendv_epi8(m_temp_reg_17,m_temp_reg_27,blend_mask);

            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_50);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_51);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_52);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_53);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_54);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_55);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_56);
            pu1_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_57);
            pu1_dst += dst_strd;

        }
    }
}

