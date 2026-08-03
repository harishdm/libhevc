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
 *  ihevc_16x16_chroma_iquant_itrans_recon_x86_intr.c
 *
 * @brief
 *  Contains function definitions for inverse  quantization, inverse
 * transform and reconstruction  of chroma interleaved data.
 *
 * @author
 *  100491
 *
 * @par List of Functions:
 *   - ihevc_chroma_iquant_itrans_recon_16x16()
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

/* All the functions work one component(U or V) of interleaved data depending upon pointers passed to it */
/* Data visualization */
/* U V U V U V U V */
/* U V U V U V U V */
/* U V U V U V U V */
/* U V U V U V U V */
/* If the pointer points to first byte of above stream (U) , functions will operate on U component */
/* If the pointer points to second byte of above stream (V) , functions will operate on V component */


#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

static UWORD8 IHEVCE_UTIL_CHROMA_16x16_SHUFFLEMASK[16] = { 0x0, 0x2, 0x4, 0x6,
    0x8, 0xA, 0xC, 0xE,
    0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF};

static UWORD8 IHEVCE_UTIL_CHROMA_16x16_UN_SHUFFLEMASK[16] = { 0x0, 0xFF, 0x1, 0xFF,
    0x2, 0xFF, 0x3, 0xFF,
    0x4, 0xFF, 0x5, 0xFF,
    0x6, 0xFF, 0x7, 0xFF};

static UWORD8 IHEVCE_UTIL_CHROMA_16x16_BLENDMASK[16] = { 0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF,
    0x0, 0xFF, 0x0, 0xFF};

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs inverse quantization, inverse  transform and
 * reconstruction for 16x16 input block
 *
 * @par Description:
 *  Performs inverse quantization , inverse transform  and adds the
 * prediction data and clips output to 8 bit
 *
 * @param[in] pi2_src
 *  Input 16x16 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 16x16 buffer for storing inverse transform
 *  1st stage output
 *
 * @param[in] pu1_pred
 *  Prediction 16x16 block
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pu1_dst
 *  Output 16x16 block
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


void ihevc_chroma_iquant_itrans_recon_16x16_sse42(WORD16 *pi2_src,
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
    __m128i m_temp_reg_60;
    __m128i m_temp_reg_61;
    __m128i m_temp_reg_70;
    __m128i m_temp_reg_71;
    __m128i m_temp_reg_72;
    __m128i m_temp_reg_73;
    __m128i m_temp_reg_74;
    __m128i m_temp_reg_75;
    __m128i m_temp_reg_76;
    __m128i m_temp_reg_77;
    __m128i m_rdng_factor;
    __m128i m_count;
    __m128i m_coeff1, m_coeff2, m_coeff3, m_coeff4;
    __m128i m_coeff5, m_coeff6, m_coeff7, m_coeff8;
    __m128i smask, un_smask, blend_mask;

    WORD32 shift_select;
    WORD32 i;

    WORD32  zero_last8_cols_stg1;
    WORD32  zero_last8_rows_stg1;
    WORD32  zero_last12_rows_stg1;
    UWORD8  *temp = (UWORD8 *)(pi2_tmp + 1024);

    WORD32  zero_last12_rows_stg2;
    WORD32  zero_last8_rows_stg2;

    WORD32  loop = 0;

    WORD32 i4_shift = IT_SHIFT_STAGE_1;
    WORD32 one = 1;
    WORD32 trans_size = TRANS_SIZE_16;
    __m128i m_add_iq = _mm_cvtsi32_si128(one);
    __m128i m_scale = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));

    /* Inverse Quantization constants */
    {
        WORD32 log2_trans_size, bit_depth;

        log2_trans_size = 4;
        bit_depth = 8;
        shift_iq = bit_depth + log2_trans_size - 5;
    }

    /* Following 3 instructions replicates the value in the */
    /* lower 16 bits of m_add_iq in the entire register */
    m_add_iq = _mm_unpacklo_epi32(m_add_iq, m_add_iq);
    m_add_iq = _mm_unpacklo_epi64(m_add_iq, m_add_iq);

    smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_16x16_SHUFFLEMASK[0]);
    un_smask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_16x16_UN_SHUFFLEMASK[0]);
    blend_mask = _mm_loadu_si128((__m128i *) &IHEVCE_UTIL_CHROMA_16x16_BLENDMASK[0]);

    /* Last 8 cols of 16x16 block are skipped based on the below flag */
    zero_last8_cols_stg1 = ((zero_cols & 0xFF00) == 0xFF00) ? 1 : 0;
    zero_last8_rows_stg1 = ((zero_rows & 0xFF00) == 0xFF00) ? 1 : 0;
    zero_last12_rows_stg1 = ((zero_rows & 0xFFF0) == 0xFFF0) ? 1 : 0;

    zero_last12_rows_stg2 = ((zero_cols & 0xFFF0) == 0xFFF0) ? 1 : 0;
    zero_last8_rows_stg2 = zero_last8_cols_stg1;

    if(zero_last8_cols_stg1)
    {
        loop = 1;
    }
    else
        loop = 2;

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


    /* i = 0 => lower 8 samples */
    /* i = 1 => higher 8 samples */
    for(i = 0; i < loop; i++)
    {
        /* IQUANT for samples used in eo, eeo and eee */
        {
            WORD32 sample_half_index = i << 3;
            WORD16 *pi2_tmp_src = pi2_src + sample_half_index;
            WORD16 *pi2_tmp_dequant = pi2_dequant_coeff + sample_half_index;

            /* row 0 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

            /* row 1 */
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            /* Optimization check based on zero_rows */
            if(!zero_last12_rows_stg1)
            {
                /* row 2 */
                m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                pi2_tmp_src += (src_strd << 1);
                pi2_tmp_dequant += (trans_size <<  1);

                /* row 3 */
                m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                pi2_tmp_src += (src_strd << 1);
                pi2_tmp_dequant += (trans_size <<  1);

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

                /* Optimization check based on zero_rows */
                if(!zero_last8_rows_stg1)
                {
                    /* row 4 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

                    /* row 5 */
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                    m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                    m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    /* row 6 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

                    /* row 7 */
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
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
                    /*Set the corresponding output registers of quant to zero, if zero_rows */
                    m_temp_reg_74 = _mm_setzero_si128();
                    m_temp_reg_75 = _mm_setzero_si128();
                    m_temp_reg_76 = _mm_setzero_si128();
                    m_temp_reg_77 = _mm_setzero_si128();
                }

            }
            else
            {
                /*Set the corresponding output registers of quant to zero, if zero_rows */
                m_temp_reg_72 = _mm_setzero_si128();
                m_temp_reg_73 = _mm_setzero_si128();
                m_temp_reg_74 = _mm_setzero_si128();
                m_temp_reg_75 = _mm_setzero_si128();
                m_temp_reg_76 = _mm_setzero_si128();
                m_temp_reg_77 = _mm_setzero_si128();
            }

        }

        {
            WORD16 *pi2_scratch = (i) ? (pi2_tmp + 8 * trans_size) : pi2_tmp;

            /* If last 12 rows are zero */
            if(zero_last12_rows_stg1)
            {

                /* eeo */
                /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
                /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
                {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36


                    m_temp_reg_20 = _mm_setzero_si128();
                    m_temp_reg_22 = _mm_setzero_si128();
                    m_temp_reg_21 = _mm_setzero_si128();
                    m_temp_reg_23 = _mm_setzero_si128();

                }

                /* eee */
                /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
                /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
                {
                    /* Loading coeff and src for use in next block */
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0

                    m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);

                    m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                    m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);

                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                    m_temp_reg_24 = m_temp_reg_10;
                    m_temp_reg_26 = m_temp_reg_10;
                    m_temp_reg_25 = m_temp_reg_14;
                    m_temp_reg_27 = m_temp_reg_14;
                }

                /* eo */

                /* eo0[0-3] */
                {
                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);

                    m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);

                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);

                    /* e[0][0-3] stored in pi2_tmp[0][0-7] */
                    /* e[7][0-3] stored in pi2_tmp[0][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pi2_tmp[1][0-7] */
                    /* e[7][4-7] stored in pi2_tmp[1][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pi2_tmp[2][0-7] */
                    /* e[6][0-3] stored in pi2_tmp[2][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pi2_tmp[3][0-7] */
                    /* e[6][4-7] stored in pi2_tmp[3][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);

                    /* e[2][0-3] stored in pi2_tmp[4][0-7] */
                    /* e[5][0-3] stored in pi2_tmp[4][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);

                    /* e[2][4-7] stored in pi2_tmp[5][0-7] */
                    /* e[5][4-7] stored in pi2_tmp[5][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);

                    /* e[3][0-3] stored in pi2_tmp[6][0-7] */
                    /* e[4][0-3] stored in pi2_tmp[6][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);

                    /* e[3][4-7] stored in pi2_tmp[7][0-7] */
                    /* e[4][4-7] stored in pi2_tmp[7][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }
            }
            /* If last 8 rows are zero */
            else if(zero_last8_rows_stg1)
            {
                /* eeo */
                /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
                /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
                {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36

                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 4

                    m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                    m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                    m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_72);
                    m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                    m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);

                    m_temp_reg_20 = m_temp_reg_10;
                    m_temp_reg_22 = m_temp_reg_12;

                    m_temp_reg_21 = m_temp_reg_14;
                    m_temp_reg_23 = m_temp_reg_16;

                }

                /* eee */
                /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
                /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
                {
                    /* Loading coeff and src for use in next block */
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0

                    m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);

                    m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                    m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);

                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                    m_temp_reg_24 = m_temp_reg_10;
                    m_temp_reg_26 = m_temp_reg_10;
                    m_temp_reg_25 = m_temp_reg_14;
                    m_temp_reg_27 = m_temp_reg_14;
                }

                /* eo0[0-3] */
                {
                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);//        : zero_last12_rows_stg1

                    m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);//         : zero_last12_rows_stg1

                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);//     : zero_last12_rows_stg1

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);//        : zero_last12_rows_stg1

                    /* e[0][0-3] stored in pi2_tmp[0][0-7] */
                    /* e[7][0-3] stored in pi2_tmp[0][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);//     : zero_last12_rows_stg1

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pi2_tmp[1][0-7] */
                    /* e[7][4-7] stored in pi2_tmp[1][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);//     : zero_last12_rows_stg1

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pi2_tmp[2][0-7] */
                    /* e[6][0-3] stored in pi2_tmp[2][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);//     : zero_last12_rows_stg1

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pi2_tmp[3][0-7] */
                    /* e[6][4-7] stored in pi2_tmp[3][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);//     : zero_last12_rows_stg1

                    /* e[2][0-3] stored in pi2_tmp[4][0-7] */
                    /* e[5][0-3] stored in pi2_tmp[4][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);//     : zero_last12_rows_stg1

                    /* e[2][4-7] stored in pi2_tmp[5][0-7] */
                    /* e[5][4-7] stored in pi2_tmp[5][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);//     : zero_last12_rows_stg1

                    /* e[3][0-3] stored in pi2_tmp[6][0-7] */
                    /* e[4][0-3] stored in pi2_tmp[6][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);//     : zero_last12_rows_stg1

                    /* e[3][4-7] stored in pi2_tmp[7][0-7] */
                    /* e[4][4-7] stored in pi2_tmp[7][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);//      : zero_last12_rows_stg1
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);//      : zero_last12_rows_stg1

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }
            }
            /* If all the rows are non-zero */
            else
            {

                /* eeo */
                /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
                /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
                {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36

                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 4
                    m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 12

                    m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                    m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                    m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                    m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);

                    m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                    m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);

                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_72);
                    m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_76);

                    m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                    m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);
                    m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
                    m_temp_reg_17 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);

                    m_temp_reg_20 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);

                    m_temp_reg_21 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
                    m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_17);
                }

                /* eee */
                /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
                /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
                {
                    /* Loading coeff and src for use in next block */
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0
                    m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 8

                    m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);
                    m_temp_reg_11 = _mm_slli_epi32(m_temp_reg_2, 6);

                    m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                    m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);

                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);
                    m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_74);

                    m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);
                    m_temp_reg_15 = _mm_slli_epi32(m_temp_reg_3, 6);

                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                    m_temp_reg_24 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_26 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_11);

                    m_temp_reg_25 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
                    m_temp_reg_27 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_15);
                }
                /* eo0[0-3] */
                {
                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_75);
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_77);

                    m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                    m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                    m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);


                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);
                    m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_75);
                    m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);

                    /* e[0][0-3] stored in pi2_tmp[0][0-7] */
                    /* e[7][0-3] stored in pi2_tmp[0][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pi2_tmp[1][0-7] */
                    /* e[7][4-7] stored in pi2_tmp[1][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pi2_tmp[2][0-7] */
                    /* e[6][0-3] stored in pi2_tmp[2][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pi2_tmp[3][0-7] */
                    /* e[6][4-7] stored in pi2_tmp[3][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                    /* e[2][0-3] stored in pi2_tmp[4][0-7] */
                    /* e[5][0-3] stored in pi2_tmp[4][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                    /* e[2][4-7] stored in pi2_tmp[5][0-7] */
                    /* e[5][4-7] stored in pi2_tmp[5][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                    /* e[3][0-3] stored in pi2_tmp[6][0-7] */
                    /* e[4][0-3] stored in pi2_tmp[6][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1);

                    /* e[3][4-7] stored in pi2_tmp[7][0-7] */
                    /* e[4][4-7] stored in pi2_tmp[7][8-15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 8;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 8;
                }
            }
        }

        /* IQUANT for samples used in o */
        {
            WORD32 sample_half_index = i << 3;
            WORD16 *pi2_tmp_src = pi2_src + sample_half_index + src_strd;
            WORD16 *pi2_tmp_dequant = pi2_dequant_coeff + sample_half_index + trans_size;

            /* row 0 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

            /* row 1 */
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            /* Optimization check based on zero_rows */
            if(!zero_last12_rows_stg1)
            {
                /* row 2 */
                m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                pi2_tmp_src += (src_strd << 1);
                pi2_tmp_dequant += (trans_size <<  1);

                /* row 3 */
                m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                pi2_tmp_src += (src_strd << 1);
                pi2_tmp_dequant += (trans_size <<  1);

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

                /* Optimization check based on zero_rows */
                if(!zero_last8_rows_stg1)
                {
                    /* row 4 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

                    /* row 5 */
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                    m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                    m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    /* row 6 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

                    /* row 7 */
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
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
                    /*Set the corresponding output registers of quant to zero, if zero_rows */
                    m_temp_reg_74 = _mm_setzero_si128();
                    m_temp_reg_75 = _mm_setzero_si128();
                    m_temp_reg_76 = _mm_setzero_si128();
                    m_temp_reg_77 = _mm_setzero_si128();
                }
            }
            else
            {
                /*Set the corresponding output registers of quant to zero, if zero_rows */
                m_temp_reg_72 = _mm_setzero_si128();
                m_temp_reg_73 = _mm_setzero_si128();
                m_temp_reg_74 = _mm_setzero_si128();
                m_temp_reg_75 = _mm_setzero_si128();
                m_temp_reg_76 = _mm_setzero_si128();
                m_temp_reg_77 = _mm_setzero_si128();
            }

        }

        /* o & stage 1 out */
        {
            WORD32 j;
            WORD16 *pi2_src_scratch = (i) ? (pi2_tmp + 8 * trans_size) : pi2_tmp;
            WORD16 *pi2_dst_scratch = (i) ? (pi2_tmp + 8 * trans_size) : pi2_tmp;
            WORD32 out_stride = (trans_size << 1);
            WORD32 in_stride = trans_size << 1;

            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[0][0]);//90
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[1][0]);//87
            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[2][0]);//80
            m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[3][0]);//70
            m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[4][0]);//57
            m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[5][0]);//43
            m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[6][0]);//25
            m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[7][0]);//9

            if(zero_last12_rows_stg1)
            {
                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);
                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }
                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }
                }
            }
            else if (zero_last8_rows_stg1)
            {
                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                        m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                        m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);

                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }
                }

            }
            else
            {
                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                        m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                        m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                        m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);
                        m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                        m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);
                        m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7
                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 9
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_75);//row 11
                    m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 13
                    m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);//row 15

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff7);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);
                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff6);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff8);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff5);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch -= out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 8;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += 8;
                    }
                }
            }
        }

        /* Transpose */
        {
            WORD16 *pi2_src_scratch = (i) ? (pi2_tmp + 8 * trans_size) : pi2_tmp;
            WORD16 *pi2_dst_scratch = ((i) ? (pi2_tmp + 8 * trans_size) : pi2_tmp);
            WORD32 out_stride = (trans_size << 1);
            WORD32 in_stride = (trans_size << 1);
            WORD32 j;

            for(j = 0; j < 2; j++)
            {
                m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//b, a
                pi2_src_scratch += in_stride;
                m_temp_reg_31 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//d, c
                pi2_src_scratch += in_stride;
                m_temp_reg_32 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//f, e
                pi2_src_scratch += in_stride;
                m_temp_reg_33 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//h, g
                pi2_src_scratch += 8;
                m_temp_reg_34 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//j, i
                pi2_src_scratch -= in_stride;
                m_temp_reg_35 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//l, k
                pi2_src_scratch -= in_stride;
                m_temp_reg_36 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//n, m
                pi2_src_scratch -= in_stride;
                m_temp_reg_37 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//p, o
                pi2_src_scratch += 8;

                m_temp_reg_40 = _mm_unpacklo_epi16(m_temp_reg_30, m_temp_reg_31);//ca3ca2ca1ca0
                m_temp_reg_41 = _mm_unpackhi_epi16(m_temp_reg_31, m_temp_reg_30);//bd3bd2bd1bd0

                m_temp_reg_42 = _mm_unpacklo_epi16(m_temp_reg_32, m_temp_reg_33);//ge3ge2ge1ge0
                m_temp_reg_43 = _mm_unpackhi_epi16(m_temp_reg_33, m_temp_reg_32);//fh3fh2fh1fh0

                m_temp_reg_44 = _mm_unpacklo_epi16(m_temp_reg_34, m_temp_reg_35);//ki3ki2ki1ki0
                m_temp_reg_45 = _mm_unpackhi_epi16(m_temp_reg_35, m_temp_reg_34);//jl3jl2jl1jl0

                m_temp_reg_46 = _mm_unpacklo_epi16(m_temp_reg_36, m_temp_reg_37);//om3om2om1om0
                m_temp_reg_47 = _mm_unpackhi_epi16(m_temp_reg_37, m_temp_reg_36);//np3np2np1np0


                m_temp_reg_30 = _mm_unpacklo_epi32(m_temp_reg_40, m_temp_reg_42);//ge1ca1ge0ca0
                m_temp_reg_31 = _mm_unpackhi_epi32(m_temp_reg_40, m_temp_reg_42);//ge3ca3ge2ca2

                m_temp_reg_32 = _mm_unpacklo_epi32(m_temp_reg_44, m_temp_reg_46);//om1ki1om0ki0
                m_temp_reg_33 = _mm_unpackhi_epi32(m_temp_reg_44, m_temp_reg_46);//om3ki3om2ki2

                m_temp_reg_34 = _mm_unpacklo_epi32(m_temp_reg_43, m_temp_reg_41);//bd1fh1bd0fh0
                m_temp_reg_35 = _mm_unpackhi_epi32(m_temp_reg_43, m_temp_reg_41);//bd3fh3bd2fh2

                m_temp_reg_36 = _mm_unpacklo_epi32(m_temp_reg_47, m_temp_reg_45);//jl1np1jl0np0
                m_temp_reg_37 = _mm_unpackhi_epi32(m_temp_reg_47, m_temp_reg_45);//jl3np3jl2np2


                m_temp_reg_40 = _mm_unpacklo_epi64(m_temp_reg_30, m_temp_reg_32);//omkigeca0
                m_temp_reg_41 = _mm_unpackhi_epi64(m_temp_reg_30, m_temp_reg_32);//omkigeca1

                m_temp_reg_42 = _mm_unpacklo_epi64(m_temp_reg_31, m_temp_reg_33);//omkigeca2
                m_temp_reg_43 = _mm_unpackhi_epi64(m_temp_reg_31, m_temp_reg_33);//omkigeca3

                m_temp_reg_44 = _mm_unpacklo_epi64(m_temp_reg_36, m_temp_reg_34);//bdfhjlnp0
                m_temp_reg_45 = _mm_unpackhi_epi64(m_temp_reg_36, m_temp_reg_34);//bdfhjlnp1

                m_temp_reg_46 = _mm_unpacklo_epi64(m_temp_reg_37, m_temp_reg_35);//bdfhjlnp2
                m_temp_reg_47 = _mm_unpackhi_epi64(m_temp_reg_37, m_temp_reg_35);//bdfhjlnp3

                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
                pi2_dst_scratch += out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_44);
                pi2_dst_scratch += out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_41);
                pi2_dst_scratch += out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_45);
                pi2_dst_scratch += 8;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_42);
                pi2_dst_scratch -= out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_46);
                pi2_dst_scratch -= out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_43);
                pi2_dst_scratch -= out_stride;
                _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_47);
                pi2_dst_scratch += 8;
            }
        }
    }

    if(zero_last8_cols_stg1)
    {
        WORD16 *pi2_dst_scratch = (pi2_tmp + 8 * trans_size);
        WORD32 out_stride = (trans_size << 1);
        WORD32 j;

        m_temp_reg_40 = _mm_setzero_si128();
        for(j = 0; j < 2; j++)
        {
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch += out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch += out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch += out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch += 8;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch -= out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch -= out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch -= out_stride;
            _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_40);
            pi2_dst_scratch += 8;
        }
    }

    /*for(i = 0; i < 256; i++)
    {
        pi2_tmp[i] = 9;
    }*/


    /* Stage 2 */
    for(i = 0; i < 2; i++)
    {
        WORD16 *pi2_src_temp = (i) ? (pi2_tmp + 2 * trans_size) : (WORD16 *)(pi2_tmp);
        WORD32 stride = (trans_size);

        i4_shift = IT_SHIFT_STAGE_2;

        if(zero_last12_rows_stg2)
        {
            /* eeo */
            /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//0

                pi2_src_temp += (stride * 9);

                if(!i)
                {
                    pi2_src_temp += (stride * 6 + 8);
                }
                else
                {
                    pi2_src_temp += (stride * 2 + 8);
                }

                pi2_src_temp -= (stride * 9);

                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//2

                m_temp_reg_20 = _mm_setzero_si128();
                m_temp_reg_22 = _mm_setzero_si128();

                m_temp_reg_21 = _mm_setzero_si128();
                m_temp_reg_23 = _mm_setzero_si128();
            }

            /* eee */
            /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
            /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
            {
                /* Loading coeff and src for use in next block */
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0

                m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);

                m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                m_temp_reg_24 = m_temp_reg_10;
                m_temp_reg_26 = m_temp_reg_10;

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);

                m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);

                m_temp_reg_25 = m_temp_reg_14;
                m_temp_reg_27 = m_temp_reg_14;
            }

            /* eo */
            {
                UWORD8 *pu1_scratch = temp;
                WORD32 out_stride = dst_strd;

                m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    /* e[0][0-3] stored in pu1_dst[0] */
                    /* e[7][0-3] stored in pu1_dst[1] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)(pu1_scratch), m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pu1_dst[2] */
                    /* e[7][4-7] stored in pu1_dst[3] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pu1_dst[4] */
                    /* e[6][0-3] stored in pu1_dst[5] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pu1_dst[6]*/
                    /* e[6][4-7] stored in pu1_dst[7] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);

                    /* e[2][0-3] stored in pu1_dst[8]*/
                    /* e[5][0-3] stored in pu1_dst[9] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);

                    /* e[2][4-7] stored in pu1_dst[10]*/
                    /* e[5][4-7] stored in pu1_dst[11] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);

                    /* e[3][0-3] stored in pu1_dst[12]*/
                    /* e[4][0-3] stored in pu1_dst[13] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);

                    /* e[3][4-7] stored in pu1_dst[14]*/
                    /* e[4][4-7] stored in pu1_dst[15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }
            }
        }
        else if(zero_last8_rows_stg2)
        {
            /* eeo */
            /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//0
                pi2_src_temp += (stride);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_temp);//4
                pi2_src_temp += (stride * 8);

                if(!i)
                {
                    pi2_src_temp += (stride * 6 + 8);
                }
                else
                {
                    pi2_src_temp += (stride * 2 + 8);
                }

                pi2_src_temp -= (stride * 8);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_temp);//6
                pi2_src_temp -= (stride);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//2


                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 4

                m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);

                m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_72);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);

                m_temp_reg_20 = m_temp_reg_10;
                m_temp_reg_22 = m_temp_reg_12;

                m_temp_reg_21 = m_temp_reg_14;
                m_temp_reg_23 = m_temp_reg_16;
            }

            /* eee */
            /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
            /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
            {
                /* Loading coeff and src for use in next block */
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0

                m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);

                m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                m_temp_reg_24 = m_temp_reg_10;
                m_temp_reg_26 = m_temp_reg_10;

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);

                m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);

                m_temp_reg_25 = m_temp_reg_14;
                m_temp_reg_27 = m_temp_reg_14;
            }

            /* eo */
            {
                UWORD8 *pu1_scratch = temp;
                WORD32 out_stride = dst_strd;

                m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    /* e[0][0-3] stored in pu1_dst[0] */
                    /* e[7][0-3] stored in pu1_dst[1] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)(pu1_scratch), m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pu1_dst[2] */
                    /* e[7][4-7] stored in pu1_dst[3] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pu1_dst[4] */
                    /* e[6][0-3] stored in pu1_dst[5] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pu1_dst[6]*/
                    /* e[6][4-7] stored in pu1_dst[7] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                    /* e[2][0-3] stored in pu1_dst[8]*/
                    /* e[5][0-3] stored in pu1_dst[9] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);

                    /* e[2][4-7] stored in pu1_dst[10]*/
                    /* e[5][4-7] stored in pu1_dst[11] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);

                    /* e[3][0-3] stored in pu1_dst[12]*/
                    /* e[4][0-3] stored in pu1_dst[13] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);

                    /* e[3][4-7] stored in pu1_dst[14]*/
                    /* e[4][4-7] stored in pu1_dst[15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }
            }
        }

        else
        {
            /* eeo */
            /* eeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
            /* eeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
            {
                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//0
                pi2_src_temp += (stride);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_temp);//4
                pi2_src_temp += (stride * 7);
                m_temp_reg_74 = _mm_loadu_si128((__m128i *) pi2_src_temp);//8
                pi2_src_temp += (stride);
                m_temp_reg_76 = _mm_loadu_si128((__m128i *) pi2_src_temp);//12
                if(!i)
                {
                    pi2_src_temp += (stride * 6 + 8);
                }
                else
                {
                    pi2_src_temp += (stride * 2 + 8);
                }
                m_temp_reg_77 = _mm_loadu_si128((__m128i *) pi2_src_temp);//14
                pi2_src_temp -= (stride);
                m_temp_reg_75 = _mm_loadu_si128((__m128i *) pi2_src_temp);//10
                pi2_src_temp -= (stride * 7);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_temp);//6
                pi2_src_temp -= (stride);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//2


                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 4
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 12

                m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);

                m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_72);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_76);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
                m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);
                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
                m_temp_reg_17 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);

                /* Loading coeff and src for use in next block */
                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 0
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 8

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_17);
            }

            /* eee */
            /* eee[0] stored in m_temp_reg_24 and m_temp_reg_25 */
            /* eee[1] stored in m_temp_reg_26 and m_temp_reg_27 */
            {
                m_temp_reg_10 = _mm_slli_epi32(m_temp_reg_0, 6);
                m_temp_reg_11 = _mm_slli_epi32(m_temp_reg_2, 6);

                m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_74);

                m_temp_reg_14 = _mm_slli_epi32(m_temp_reg_1, 6);
                m_temp_reg_15 = _mm_slli_epi32(m_temp_reg_3, 6);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                m_temp_reg_24 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);
                m_temp_reg_26 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_11);

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);
                m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_75);
                m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_77);
                m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);

                m_temp_reg_25 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);
                m_temp_reg_27 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_15);
            }

            /* eo */
            {
                UWORD8 *pu1_scratch = temp;
                WORD32 out_stride = dst_strd;

                m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);
                m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_75);
                m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_40 = _mm_add_epi32(m_temp_reg_24, m_temp_reg_20);
                    m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_24, m_temp_reg_20);

                    /* e[0][0-3] stored in pu1_dst[0] */
                    /* e[7][0-3] stored in pu1_dst[1] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)(pu1_scratch), m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                    /* ee[0] and ee[3] stored in m_temp_reg_40-41 & m_temp_reg_46-47 */
                    m_temp_reg_41 = _mm_add_epi32(m_temp_reg_25, m_temp_reg_21);
                    m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_25, m_temp_reg_21);

                    /* e[0][4-7] stored in pu1_dst[2] */
                    /* e[7][4-7] stored in pu1_dst[3] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_42 = _mm_add_epi32(m_temp_reg_26, m_temp_reg_22);
                    m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_26, m_temp_reg_22);

                    /* e[1][0-3] stored in pu1_dst[4] */
                    /* e[6][0-3] stored in pu1_dst[5] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);

                    /* ee[1] and ee[2] stored in m_temp_reg_4-43 & m_temp_reg_44-45 */
                    m_temp_reg_43 = _mm_add_epi32(m_temp_reg_27, m_temp_reg_23);
                    m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_27, m_temp_reg_23);

                    /* e[1][4-7] stored in pu1_dst[6]*/
                    /* e[6][4-7] stored in pu1_dst[7] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                    /* e[2][0-3] stored in pu1_dst[8]*/
                    /* e[5][0-3] stored in pu1_dst[9] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                    /* e[2][4-7] stored in pu1_dst[10]*/
                    /* e[5][4-7] stored in pu1_dst[11] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                    /* e[3][0-3] stored in pu1_dst[12]*/
                    /* e[4][0-3] stored in pu1_dst[13] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }

                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1);

                    /* e[3][4-7] stored in pu1_dst[14]*/
                    /* e[4][4-7] stored in pu1_dst[15] */
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_34);
                    pu1_scratch += out_stride;
                    _mm_storeu_si128((__m128i *)pu1_scratch, m_temp_reg_35);
                    pu1_scratch += out_stride;
                }
            }
        }

        if(zero_last12_rows_stg2)
        {
            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                UWORD8 *pu1_src_scratch = temp;
                WORD16 *pi2_dst_scratch = (i) ? (pi2_tmp + 8) : (pi2_tmp);
                WORD32 out_stride = (trans_size);
                WORD32 in_stride = (dst_strd) * 4;

                pi2_src_temp = pi2_tmp + (stride * 4) + i * (stride * 2);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[0][0]);//90
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[1][0]);//87
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[4][0]);//57

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//1

                pi2_src_temp += (stride * 9);

                if(0 == i)
                {
                    pi2_src_temp -= (stride * 2 - 8);
                }
                else
                {
                    pi2_src_temp -= (stride * 6 - 8);
                }
                pi2_src_temp -= (stride * 9);

                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//3


                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);
                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[2][0]);//80
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[3][0]);//70
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[7][0]);//9

                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[5][0]);//43
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[6][0]);//25

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }
                }
            }
        }
        else if (zero_last8_rows_stg2)
        {
            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                UWORD8 *pu1_src_scratch = temp;
                WORD16 *pi2_dst_scratch = (i) ? (pi2_tmp + 8) : (pi2_tmp);
                WORD32 out_stride = (trans_size);
                WORD32 in_stride = (dst_strd) * 4;

                pi2_src_temp = pi2_tmp + (stride * 4) + i * (stride * 2);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[0][0]);//90
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[1][0]);//87
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[2][0]);//80
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[3][0]);//70
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[4][0]);//57
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[5][0]);//43
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[6][0]);//25
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[7][0]);//9

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//1
                pi2_src_temp += (stride);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_temp);//5
                pi2_src_temp += (stride * 8);

                if(0 == i)
                {
                    pi2_src_temp -= (stride * 2 - 8);
                }
                else
                {
                    pi2_src_temp -= (stride * 6 - 8);
                }

                pi2_src_temp -= (stride * 8);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_temp);//7
                pi2_src_temp -= (stride);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//3


                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                        m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                        m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);
                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }
                }
            }
        }
        else
        {
            /* o & stage 2 pre-transposed out */
            {
                WORD32 j;
                UWORD8 *pu1_src_scratch = temp;
                WORD16 *pi2_dst_scratch = (i) ? (pi2_tmp + 8) : (pi2_tmp);
                WORD32 out_stride = (trans_size);
                WORD32 in_stride = (dst_strd) * 4;

                pi2_src_temp = pi2_tmp + (stride * 4) + i * (stride * 2);

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[0][0]);//90
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[1][0]);//87
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[2][0]);//80
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[3][0]);//70
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[4][0]);//57
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[5][0]);//43
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[6][0]);//25
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_odd[7][0]);//9

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_temp);//1
                pi2_src_temp += (stride);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_temp);//5
                pi2_src_temp += (stride * 7);
                m_temp_reg_74 = _mm_loadu_si128((__m128i *) pi2_src_temp);//9
                pi2_src_temp += (stride);
                m_temp_reg_76 = _mm_loadu_si128((__m128i *) pi2_src_temp);//13
                if(0 == i)
                {
                    pi2_src_temp -= (stride * 2 - 8);
                }
                else
                {
                    pi2_src_temp -= (stride * 6 - 8);
                }
                m_temp_reg_77 = _mm_loadu_si128((__m128i *) pi2_src_temp);//15
                pi2_src_temp -= (stride);
                m_temp_reg_75 = _mm_loadu_si128((__m128i *) pi2_src_temp);//11
                pi2_src_temp -= (stride * 7);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_temp);//7
                pi2_src_temp -= (stride);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_temp);//3


                for(j = 0; j < 2; j++)
                {
                    if(j)
                    {
                        m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                        m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                        m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                        m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                        m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);
                        m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                        m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);
                        m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);
                    }

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7
                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 9
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_75);//row 11
                    m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 13
                    m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);//row 15

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff7);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff8);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_rdng_factor = _mm_cvtsi32_si128((1 << (i4_shift - 1)));
                        m_count = _mm_cvtsi32_si128(i4_shift);
                        m_rdng_factor = _mm_unpacklo_epi32(m_rdng_factor, m_rdng_factor);
                        m_rdng_factor = _mm_unpacklo_epi64(m_rdng_factor, m_rdng_factor);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o1[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff7);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff6);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff5);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }

                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff6);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff8);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += ((!i) * out_stride + 8);
                    }

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff5);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch -= (in_stride);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += out_stride;
                    }

                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pu1_src_scratch);
                        pu1_src_scratch += (dst_strd);

                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);
                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_22);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);
                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_24);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);
                        m_temp_reg_31 =_mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 =_mm_add_epi32(m_temp_reg_30, m_temp_reg_26);
                        m_temp_reg_31 =_mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 =_mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_rdng_factor);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_rdng_factor);
                        m_temp_reg_31 = _mm_sra_epi32(m_temp_reg_31, m_count);
                        m_temp_reg_30 = _mm_sra_epi32(m_temp_reg_30, m_count);

                        m_temp_reg_30 = _mm_packs_epi32(m_temp_reg_30, m_temp_reg_31);

                        _mm_storeu_si128((__m128i *) pi2_dst_scratch, m_temp_reg_30);
                        pi2_dst_scratch += (i * out_stride + 8);
                    }
                }
            }
        }
    }

    /* Transpose */
    {
        WORD16 *pi2_src_scratch;
        UWORD8 *pu1_pred_temp = pu1_pred;
        WORD32 out_stride = dst_strd;
        WORD32 in_stride = trans_size;
        WORD32 j;

        for(i = 0; i < 2; i++)
        {
            pi2_src_scratch = (i) ? (pi2_tmp + 8): pi2_tmp;

            for(j = 0; j < 2; j++)
            {
                m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//b, a
                pi2_src_scratch += in_stride;
                m_temp_reg_31 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//d, c
                pi2_src_scratch += ((!i) * in_stride + 8);
                m_temp_reg_32 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//f, e
                pi2_src_scratch += (in_stride);
                m_temp_reg_33 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//h, g
                pi2_src_scratch += (i * in_stride + 8);
                m_temp_reg_34 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//j, i
                pi2_src_scratch += in_stride;
                m_temp_reg_35 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//l, k
                pi2_src_scratch += ((!i) * in_stride + 8);
                m_temp_reg_36 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//n, m
                pi2_src_scratch += in_stride;
                m_temp_reg_37 = _mm_loadu_si128((__m128i *) pi2_src_scratch);//p, o
                pi2_src_scratch += (i * in_stride + 8);

                m_temp_reg_40 = _mm_unpacklo_epi16(m_temp_reg_30, m_temp_reg_31);//ca3ca2ca1ca0
                m_temp_reg_41 = _mm_unpackhi_epi16(m_temp_reg_31, m_temp_reg_30);//bd3bd2bd1bd0

                m_temp_reg_42 = _mm_unpacklo_epi16(m_temp_reg_32, m_temp_reg_33);//ge3ge2ge1ge0
                m_temp_reg_43 = _mm_unpackhi_epi16(m_temp_reg_33, m_temp_reg_32);//fh3fh2fh1fh0

                m_temp_reg_44 = _mm_unpacklo_epi16(m_temp_reg_34, m_temp_reg_35);//ki3ki2ki1ki0
                m_temp_reg_45 = _mm_unpackhi_epi16(m_temp_reg_35, m_temp_reg_34);//jl3jl2jl1jl0

                m_temp_reg_46 = _mm_unpacklo_epi16(m_temp_reg_36, m_temp_reg_37);//om3om2om1om0
                m_temp_reg_47 = _mm_unpackhi_epi16(m_temp_reg_37, m_temp_reg_36);//np3np2np1np0


                m_temp_reg_30 = _mm_unpacklo_epi32(m_temp_reg_40, m_temp_reg_42);//ge1ca1ge0ca0
                m_temp_reg_31 = _mm_unpackhi_epi32(m_temp_reg_40, m_temp_reg_42);//ge3ca3ge2ca2

                m_temp_reg_32 = _mm_unpacklo_epi32(m_temp_reg_44, m_temp_reg_46);//om1ki1om0ki0
                m_temp_reg_33 = _mm_unpackhi_epi32(m_temp_reg_44, m_temp_reg_46);//om3ki3om2ki2

                m_temp_reg_34 = _mm_unpacklo_epi32(m_temp_reg_43, m_temp_reg_41);//bd1fh1bd0fh0
                m_temp_reg_35 = _mm_unpackhi_epi32(m_temp_reg_43, m_temp_reg_41);//bd3fh3bd2fh2

                m_temp_reg_36 = _mm_unpacklo_epi32(m_temp_reg_47, m_temp_reg_45);//jl1np1jl0np0
                m_temp_reg_37 = _mm_unpackhi_epi32(m_temp_reg_47, m_temp_reg_45);//jl3np3jl2np2


                m_temp_reg_40 = _mm_unpacklo_epi64(m_temp_reg_30, m_temp_reg_32);//omkigeca0

                /* Loads increased for pred, U and V */
                m_temp_reg_50 = _mm_loadu_si128((__m128i *)pu1_pred_temp);
                m_temp_reg_51 = _mm_loadu_si128((__m128i *)(pu1_pred_temp + 16));

                m_temp_reg_70 = _mm_loadu_si128((__m128i *)pu1_dst);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *)(pu1_dst + 16));

                /* Use shuffle to extract U or V Pred values */
                m_temp_reg_10 = _mm_shuffle_epi8(m_temp_reg_50, smask);
                m_temp_reg_11 = _mm_shuffle_epi8(m_temp_reg_51, smask);

                /* m_temp_reg_20 : Has the 16 U values */
                m_temp_reg_20 = _mm_unpacklo_epi64(m_temp_reg_10, m_temp_reg_11);

                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_40, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_unpacklo_epi64(m_temp_reg_36, m_temp_reg_34);//bdfhjlnp0
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_44, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                /* Make the recon Values interleaved using shuffle*/
                m_temp_reg_10 = _mm_shuffle_epi8(m_temp_reg_20, un_smask);
                m_temp_reg_11 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_11 = _mm_shuffle_epi8(m_temp_reg_11, un_smask);

                m_temp_reg_20 = _mm_blendv_epi8(m_temp_reg_10,m_temp_reg_70,blend_mask);
                m_temp_reg_21 = _mm_blendv_epi8(m_temp_reg_11,m_temp_reg_71,blend_mask);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_20);
                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_21);

                pu1_dst += out_stride;
                pu1_pred_temp += pred_strd;

                m_temp_reg_41 = _mm_unpackhi_epi64(m_temp_reg_30, m_temp_reg_32);//omkigeca1

                /* Loads increased for pred, U and V */
                m_temp_reg_60 = _mm_loadu_si128((__m128i *)pu1_pred_temp);
                m_temp_reg_61 = _mm_loadu_si128((__m128i *)(pu1_pred_temp + 16));

                m_temp_reg_70 = _mm_loadu_si128((__m128i *)pu1_dst);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *)(pu1_dst + 16));

                /* Use shuffle to extract U or V Pred values */
                m_temp_reg_12 = _mm_shuffle_epi8(m_temp_reg_60, smask);
                m_temp_reg_13 = _mm_shuffle_epi8(m_temp_reg_61, smask);

                /* m_temp_reg_22 : Has the 16 U values */
                m_temp_reg_22 = _mm_unpacklo_epi64(m_temp_reg_12, m_temp_reg_13);

                m_temp_reg_1 = _mm_cvtepu8_epi16(m_temp_reg_22);
                m_temp_reg_41 = _mm_add_epi16(m_temp_reg_41, m_temp_reg_1);
                m_temp_reg_1 = _mm_srli_si128(m_temp_reg_22, 8);
                m_temp_reg_1 = _mm_cvtepu8_epi16(m_temp_reg_1);
                m_temp_reg_45 = _mm_unpackhi_epi64(m_temp_reg_36, m_temp_reg_34);//bdfhjlnp1
                m_temp_reg_45 = _mm_add_epi16(m_temp_reg_45, m_temp_reg_1);
                m_temp_reg_22 = _mm_packus_epi16(m_temp_reg_41, m_temp_reg_45);

                /* Make the recon Values interleaved using shuffle*/
                m_temp_reg_12 = _mm_shuffle_epi8(m_temp_reg_22, un_smask);
                m_temp_reg_13 = _mm_srli_si128(m_temp_reg_22, 8);
                m_temp_reg_13 = _mm_shuffle_epi8(m_temp_reg_13, un_smask);

                m_temp_reg_22 = _mm_blendv_epi8(m_temp_reg_12,m_temp_reg_70,blend_mask);
                m_temp_reg_23 = _mm_blendv_epi8(m_temp_reg_13,m_temp_reg_71,blend_mask);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_22);
                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_23);

                pu1_dst += out_stride;
                pu1_pred_temp += pred_strd;

                m_temp_reg_42 = _mm_unpacklo_epi64(m_temp_reg_31, m_temp_reg_33);//omkigeca2

                /* Loads increased for pred, U and V */
                m_temp_reg_50 = _mm_loadu_si128((__m128i *)pu1_pred_temp);
                m_temp_reg_51 = _mm_loadu_si128((__m128i *)(pu1_pred_temp + 16));

                m_temp_reg_70 = _mm_loadu_si128((__m128i *)pu1_dst);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *)(pu1_dst + 16));

                /* Use shuffle to extract U or V Pred values */
                m_temp_reg_14 = _mm_shuffle_epi8(m_temp_reg_50, smask);
                m_temp_reg_15 = _mm_shuffle_epi8(m_temp_reg_51, smask);

                /* m_temp_reg_24 : Has the 16 U values */
                m_temp_reg_24 = _mm_unpacklo_epi64(m_temp_reg_14, m_temp_reg_15);

                m_temp_reg_2 = _mm_cvtepu8_epi16(m_temp_reg_24);
                m_temp_reg_42 = _mm_add_epi16(m_temp_reg_42, m_temp_reg_2);
                m_temp_reg_2 = _mm_srli_si128(m_temp_reg_24, 8);
                m_temp_reg_2 = _mm_cvtepu8_epi16(m_temp_reg_2);
                m_temp_reg_46 = _mm_unpacklo_epi64(m_temp_reg_37, m_temp_reg_35);//bdfhjlnp2
                m_temp_reg_46 = _mm_add_epi16(m_temp_reg_46, m_temp_reg_2);
                m_temp_reg_24 = _mm_packus_epi16(m_temp_reg_42, m_temp_reg_46);

                /* Make the recon Values interleaved using shuffle*/
                m_temp_reg_14 = _mm_shuffle_epi8(m_temp_reg_24, un_smask);
                m_temp_reg_15 = _mm_srli_si128(m_temp_reg_24, 8);
                m_temp_reg_15 = _mm_shuffle_epi8(m_temp_reg_15, un_smask);

                m_temp_reg_24 = _mm_blendv_epi8(m_temp_reg_14,m_temp_reg_70,blend_mask);
                m_temp_reg_25 = _mm_blendv_epi8(m_temp_reg_15,m_temp_reg_71,blend_mask);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_24);
                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_25);

                pu1_dst += out_stride;
                pu1_pred_temp += pred_strd;

                m_temp_reg_43 = _mm_unpackhi_epi64(m_temp_reg_31, m_temp_reg_33);//omkigeca3

                /* Loads increased for pred, U and V */
                m_temp_reg_60 = _mm_loadu_si128((__m128i *)pu1_pred_temp);
                m_temp_reg_61 = _mm_loadu_si128((__m128i *)(pu1_pred_temp + 16));

                m_temp_reg_70 = _mm_loadu_si128((__m128i *)pu1_dst);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *)(pu1_dst + 16));

                /* Use shuffle to extract U or V Pred values */
                m_temp_reg_16 = _mm_shuffle_epi8(m_temp_reg_60, smask);
                m_temp_reg_17 = _mm_shuffle_epi8(m_temp_reg_61, smask);

                /* m_temp_reg_26 : Has the 16 U values */
                m_temp_reg_26 = _mm_unpacklo_epi64(m_temp_reg_16, m_temp_reg_17);

                m_temp_reg_3 = _mm_cvtepu8_epi16(m_temp_reg_26);
                m_temp_reg_43 = _mm_add_epi16(m_temp_reg_43, m_temp_reg_3);
                m_temp_reg_3 = _mm_srli_si128(m_temp_reg_26, 8);
                m_temp_reg_3 = _mm_cvtepu8_epi16(m_temp_reg_3);
                m_temp_reg_47 = _mm_unpackhi_epi64(m_temp_reg_37, m_temp_reg_35);//bdfhjlnp3
                m_temp_reg_47 = _mm_add_epi16(m_temp_reg_47, m_temp_reg_3);
                m_temp_reg_26 = _mm_packus_epi16(m_temp_reg_43, m_temp_reg_47);

                /* Make the recon Values interleaved using shuffle*/
                m_temp_reg_16 = _mm_shuffle_epi8(m_temp_reg_26, un_smask);
                m_temp_reg_17 = _mm_srli_si128(m_temp_reg_26, 8);
                m_temp_reg_17 = _mm_shuffle_epi8(m_temp_reg_17, un_smask);

                m_temp_reg_26 = _mm_blendv_epi8(m_temp_reg_16,m_temp_reg_70,blend_mask);
                m_temp_reg_27 = _mm_blendv_epi8(m_temp_reg_17,m_temp_reg_71,blend_mask);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_26);
                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_27);

                pu1_dst += out_stride;
                pu1_pred_temp += pred_strd;
            }
        }
    }
}

