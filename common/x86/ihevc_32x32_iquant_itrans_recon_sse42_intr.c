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
 *  ihevc_iquant_itrans_recon.c
 *
 * @brief
 *  Contains function definitions for inverse  quantization, inverse
 * transform and reconstruction
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_iquant_itrans_recon_4x4_ttype1()
 *  - ihevc_iquant_itrans_recon_4x4()
 *  - ihevc_iquant_itrans_recon_8x8()
 *  - ihevc_iquant_itrans_recon_16x16()
 *  - ihevc_iquant_itrans_recon_32x32()
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
#include "ihevc_iquant_itrans_recon.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"


//#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

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
 *  Temporary 16x16 buffer for storing inverse
 *  transform 1st stage output
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
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
void ihevc_iquant_itrans_recon_32x32_sse42(WORD16 *pi2_src,
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
    /* Inverse Transform */

    WORD32 j;
    WORD32 add;
    WORD32 shift;
    WORD16 *pi2_tmp_orig;
    WORD32 shift_iq;

    WORD32 temp_array[1024];
    WORD16 temp1_array[1024];
    WORD32 *o_temp_ptr;
    WORD16 *temp_ptr;

    __m128i shift_in_iquant;
    __m128i shift_in_iquant_minus_1;
    __m128i m_temp_reg_0;
    __m128i m_temp_reg_1;
    __m128i m_temp_reg_2;
    __m128i m_temp_reg_3;
    __m128i m_temp_reg_4;
    __m128i m_temp_reg_5;
    __m128i m_temp_reg_6;
    __m128i m_temp_reg_7;
    __m128i m_temp_reg_10;
    __m128i m_temp_reg_11;
    __m128i m_temp_reg_12;
    __m128i m_temp_reg_13;
    __m128i m_temp_reg_14;
    __m128i m_temp_reg_15;
    __m128i m_temp_reg_16;
    __m128i m_temp_reg_17;
    __m128i m_temp_reg_18;
    __m128i m_temp_reg_19;
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

    __m128i m_temp_reg_80;
    __m128i m_temp_reg_81;
    __m128i m_temp_reg_82;
    __m128i m_temp_reg_83;
    __m128i m_temp_reg_84;
    __m128i m_temp_reg_85;
    __m128i m_temp_reg_86;
    __m128i m_temp_reg_87;

    __m128i m_temp_reg_90;
    __m128i m_temp_reg_91;
    __m128i m_temp_reg_92;
    __m128i m_temp_reg_93;
    __m128i m_temp_reg_94;
    __m128i m_temp_reg_95;
    __m128i m_temp_reg_96;
    __m128i m_temp_reg_97;

    __m128i m_rdng_factor;
    __m128i m_count;
    __m128i m_coeff1, m_coeff2, m_coeff3, m_coeff4;
    __m128i m_coeff5, m_coeff6, m_coeff7, m_coeff8;

    __m128i m_coeff9, m_coeff10, m_coeff11, m_coeff12;
    __m128i m_coeff13, m_coeff14, m_coeff15;


    __m128i temp1, temp2, temp3, temp4;
    __m128i temp5, temp6, temp7, temp8;

    WORD32 shift_select;
    WORD32 i;


    WORD32  zero_last24_cols_stg1;
    WORD32  zero_last24_rows_stg1;
    WORD32  zero_last28_rows_stg1;

    WORD32  zero_last28_rows_stg2;
    WORD32  zero_last24_rows_stg2;

    WORD32  trans_size_stg1;
//  WORD32  trans_size_stg2;

    WORD32 i4_shift = IT_SHIFT_STAGE_1;
    WORD32 one = 1;
    WORD32 trans_size = TRANS_SIZE_32;
    __m128i m_add_iq = _mm_cvtsi32_si128(one);
    __m128i m_scale = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));


    /* Last 8 cols of 16x16 block are skipped based on the below flag :  */
    zero_last24_cols_stg1 = ((zero_cols & 0xFFFFFF00) == 0xFFFFFF00) ? 1 : 0;
    zero_last24_rows_stg1 = ((zero_rows & 0xFFFFFF00) == 0xFFFFFF00) ? 1 : 0;
    zero_last28_rows_stg1 = ((zero_rows & 0xFFFFFFF0) == 0xFFFFFFF0) ? 1 : 0;

    zero_last28_rows_stg2 = ((zero_cols & 0xFFFFFFF0) == 0xFFFFFFF0) ? 1 : 0;
    zero_last24_rows_stg2 = zero_last24_cols_stg1;

    if((zero_last28_rows_stg2) || (zero_last24_cols_stg1))
    {
        trans_size_stg1 = 8;

    }
    else
    {
        trans_size_stg1 = 32;
    }


    pi2_tmp_orig = pi2_tmp;
    o_temp_ptr  = temp_array;
    temp_ptr = temp1_array;
     /* Inverse Quantization constants */
        {
            WORD32 log2_trans_size, bit_depth;

            log2_trans_size = 5;
            bit_depth = 8 + 0;
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

    for(i = 0; i < trans_size_stg1; i +=8)
        {

            /* IQUANT for samples used in eo, eeo and eee */
            {

            WORD16 *pi2_tmp_src = pi2_src;
            WORD16 *pi2_tmp_dequant = pi2_dequant_coeff;

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

            if(!zero_last28_rows_stg1)
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

                if(!zero_last24_rows_stg1)
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

            /* row 6 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            /* row 2 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            m_temp_reg_80 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_81 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

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

            /* row 4 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            m_temp_reg_82 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_83 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

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

            /* row 6 */
            m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
            m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
            m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
            m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            m_temp_reg_84 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_85 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

            /* row 7 */
            m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
            m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
            m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
            m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
            m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
            pi2_tmp_src += (src_strd << 1);
            pi2_tmp_dequant += (trans_size <<  1);

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

            m_temp_reg_86 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
            m_temp_reg_87 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);
                }
                else
                {
                    m_temp_reg_74 = _mm_setzero_si128();
                    m_temp_reg_75 = _mm_setzero_si128();

                    m_temp_reg_76 = _mm_setzero_si128();
                    m_temp_reg_77 = _mm_setzero_si128();

                    m_temp_reg_80 = _mm_setzero_si128();
                    m_temp_reg_81 = _mm_setzero_si128();

                    m_temp_reg_82 = _mm_setzero_si128();
                    m_temp_reg_83 = _mm_setzero_si128();

                    m_temp_reg_84 = _mm_setzero_si128();
                    m_temp_reg_85 = _mm_setzero_si128();

                    m_temp_reg_86 = _mm_setzero_si128();
                    m_temp_reg_87 = _mm_setzero_si128();

                }

            }
            else
            {
                m_temp_reg_72 = _mm_setzero_si128();
                m_temp_reg_73 = _mm_setzero_si128();

                m_temp_reg_74 = _mm_setzero_si128();
                m_temp_reg_75 = _mm_setzero_si128();

                m_temp_reg_76 = _mm_setzero_si128();
                m_temp_reg_77 = _mm_setzero_si128();

                m_temp_reg_80 = _mm_setzero_si128();
                m_temp_reg_81 = _mm_setzero_si128();

                m_temp_reg_82 = _mm_setzero_si128();
                m_temp_reg_83 = _mm_setzero_si128();

                m_temp_reg_84 = _mm_setzero_si128();
                m_temp_reg_85 = _mm_setzero_si128();

                m_temp_reg_86 = _mm_setzero_si128();
                m_temp_reg_87 = _mm_setzero_si128();
            }
            }
/*
0*trans_size = m_temp_reg_70
  2*trans_size = m_temp_reg_71
  4*trans_size = m_temp_reg_72
  6*trans_size = m_temp_reg_73
  8*trans_size = m_temp_reg_74
  10*trans_size = m_temp_reg_75
  12*trans_size = m_temp_reg_76
  14*trans_size = m_temp_reg_77*/

/*16*trans_size = m_temp_reg_80
  18*trans_size = m_temp_reg_81
  20*trans_size = m_temp_reg_82
  22*trans_size = m_temp_reg_83
  24*trans_size = m_temp_reg_84
  26*trans_size = m_temp_reg_85
  28*trans_size = m_temp_reg_86
30*trans_size = m_temp_reg_87
*/
        if(zero_last28_rows_stg1)
        {
            /* eeo */
        /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
        /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */
        {
            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

            m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

            m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

/* eeeo[0]= m_temp_reg_20  */
/* eeeo[1]= m_temp_reg_21  */
/* eeee[0]= m_temp_reg_22  */
/* eeee[1]= m_temp_reg_23  */

            /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_40 = m_temp_reg_14;

            /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_43 = m_temp_reg_14;

            /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_42 = m_temp_reg_16;

            /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_41 = m_temp_reg_16;

            m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

            m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

            m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);


/* eeeo[0]= m_temp_reg_20  */
/* eeeo[1]= m_temp_reg_21  */
/* eeee[0]= m_temp_reg_22  */
/* eeee[1]= m_temp_reg_23  */

            /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_44 = m_temp_reg_14;

            /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_47 = m_temp_reg_14;

            /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_46 = m_temp_reg_16;

            /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_45 = m_temp_reg_16;


            }
            /* eo */
            {
                WORD32 *pi2_scratch = o_temp_ptr;

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);

                m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);

                /* eo0[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

                /* eo0[4-7] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }
                /* eo1[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

            /* eo1[4-7] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

            /* eo2[0-3] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

            /* eo2[4-7] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

                /**************************************************************************/


            /* eo3[0-3] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }

            /* eo3[4-7] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }


                /* eo4[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

            }
                /* eo4[4-7] */
        {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /***********************************************************************/

                /* eo5[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo5[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }

                /* eo6[0-3] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo6[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


                /* eo7[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo7[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }

            }
        }
        else if(zero_last24_rows_stg1)
        {
            {
                /* eeo */
                /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
                /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                /* eeeo[0]= m_temp_reg_20  */
                /* eeeo[1]= m_temp_reg_21  */
                /* eeee[0]= m_temp_reg_22  */
                /* eeee[1]= m_temp_reg_23  */

                /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_40 = m_temp_reg_14;

                /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_43 = m_temp_reg_14;

                /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_42 = m_temp_reg_16;

                /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_41 = m_temp_reg_16;

                /* for row 4 to 7 */

                m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);


                /* eeeo[0]= m_temp_reg_20  */
                /* eeeo[1]= m_temp_reg_21  */
                /* eeee[0]= m_temp_reg_22  */
                /* eeee[1]= m_temp_reg_23  */

                /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_44 = m_temp_reg_14;

                /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_47 = m_temp_reg_14;

                /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_46 = m_temp_reg_16;

                /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_45 = m_temp_reg_16;


                /* eeo[]
                /* for(k = 0; k < 4; k++) */

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_72);

                m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);

                m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_72);

                m_temp_reg_33 = _mm_setzero_si128();

                /* eeo */
                {
                    /* eeo0[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                        m_temp_reg_90 = m_temp_reg_34;
                        m_temp_reg_97 = m_temp_reg_35;
                    }
                    /* eeo0[4-7] */
            {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);

                        m_temp_reg_91 = m_temp_reg_34;
                        m_temp_reg_96 = m_temp_reg_35;

                    }

                    /* eeo1[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                        /* e[1][0-3] stored in pi2_tmp[2][0-7] */
                        /* e[6][0-3] stored in pi2_tmp[2][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);

                        m_temp_reg_92 = m_temp_reg_34;
                        m_temp_reg_95 = m_temp_reg_35;

                    }

                    /* eo1[4-7] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);

                        /* e[1][4-7] stored in pi2_tmp[3][0-7] */
                        /* e[6][4-7] stored in pi2_tmp[3][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);

                        m_temp_reg_93 = m_temp_reg_34;
                        m_temp_reg_94 = m_temp_reg_35;


            }

                    /* eo2[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);

                        /* e[2][0-3] stored in pi2_tmp[4][0-7] */
                        /* e[5][0-3] stored in pi2_tmp[4][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);

                        temp1 = m_temp_reg_34;
                        temp7 = m_temp_reg_35;

                    }

                    /* eo2[4-7] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);

                        /* e[2][4-7] stored in pi2_tmp[5][0-7] */
                        /* e[5][4-7] stored in pi2_tmp[5][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);

                        temp2 = m_temp_reg_34;
                        temp6 = m_temp_reg_35;

                    }

                    /* eo3[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);

                        /* e[3][0-3] stored in pi2_tmp[6][0-7] */
                        /* e[4][0-3] stored in pi2_tmp[6][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);

                        temp3 = m_temp_reg_34;
                        temp5 = m_temp_reg_35;

            }


            /* eo3[4-7] */
            {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);

                        /* e[3][4-7] stored in pi2_tmp[7][0-7] */
                        /* e[4][4-7] stored in pi2_tmp[7][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_47, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_47, m_temp_reg_30);

                        temp4 = m_temp_reg_34;
                        temp8 = m_temp_reg_35;


                    }
                    /* All values of ee[] array in pi2_temp */

                    /* for(k = 0; k < 8; k++) */


                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                    m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);

                    m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);

                }
            }
            /* eo */
            {
                WORD32 *pi2_scratch = o_temp_ptr;

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_90, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_90, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_91, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_91, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
                /* eo1[0-3] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_92, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_92, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_93, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_93, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(temp1, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp1, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


                /* eo2[4-7] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(temp2, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp2, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /**************************************************************************/


                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(temp3, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp3, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(temp4, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp4, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


                /* eo4[0-3] */
            {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);

                    m_temp_reg_34 = _mm_add_epi32(temp5, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp5, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
                /* eo4[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);

                    m_temp_reg_34 = _mm_add_epi32(temp8, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp8, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /***********************************************************************/

                /* eo5[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                    m_temp_reg_34 = _mm_add_epi32(temp7, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp7, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


            /* eo5[4-7] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);

                m_temp_reg_34 = _mm_add_epi32(temp6, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(temp6, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }

            /* eo6[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_95, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_95, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


            /* eo6[4-7] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_94, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_94, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


            /* eo7[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_97, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_97, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }


            /* eo7[4-7] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_96, m_temp_reg_30);
                m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_96, m_temp_reg_30);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                pi2_scratch += 4;
                _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                pi2_scratch += 4;

            }
            }

         }
        else
        {
            {
                /* eeo */
                /* eeeo[0] stored in m_temp_reg_20 and m_temp_reg_21 */
                /* eeeo[1] stored in m_temp_reg_22 and m_temp_reg_23 */

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_74); /* row 8 */
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_84); /* row 24 */

                m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_80);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);
                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                m_temp_reg_17 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);  /* eeeo[0] */
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);  /* eeeo[1] */

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);  /* eeee[0] */
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_17);  /* eeee[1] */

                /* eeeo[0]= m_temp_reg_20  */
                /* eeeo[1]= m_temp_reg_21  */
                /* eeee[0]= m_temp_reg_22  */
                /* eeee[1]= m_temp_reg_23  */

                /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_40 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[0] */

                /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_43 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[1] */

                /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_42 = _mm_sub_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[1] */

                /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_41 = _mm_add_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[0] */

                /* for row 4 to 7 */

                m_temp_reg_74 = _mm_srli_si128(m_temp_reg_74, 8);
                m_temp_reg_84 = _mm_srli_si128(m_temp_reg_84, 8);

                m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_74); /* row 8 */
                m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_84); /* row 24 */

                m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
                m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
                m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
                m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);

                m_temp_reg_70 = _mm_srli_si128(m_temp_reg_70, 8);
                m_temp_reg_80 = _mm_srli_si128(m_temp_reg_80, 8);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);
                m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_80);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);
                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                m_temp_reg_17 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);

                m_temp_reg_20 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);  /* eeeo[0] */
                m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);  /* eeeo[1] */

                m_temp_reg_21 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);  /* eeee[0] */
                m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_17);  /* eeee[1] */

                /* eeeo[0]= m_temp_reg_20  */
                /* eeeo[1]= m_temp_reg_21  */
                /* eeee[0]= m_temp_reg_22  */
                /* eeee[1]= m_temp_reg_23  */

                /* eee[0] = eeee[0] + eeeo[0]; */
                m_temp_reg_44 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[0] */

                /* eee[3] = eeee[0] - eeeo[0]; */
                m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[1] */

                /* eee[2] = eeee[1] - eeeo[1]; */
                m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[1] */

                /* eee[1] = eeee[1] + eeeo[1];*/
                m_temp_reg_45 = _mm_add_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[0] */


                /* eeo[]
                /* for(k = 0; k < 4; k++) */

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_72);
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_76);
                m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_82);
                m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_86);

                m_temp_reg_72 = _mm_srli_si128(m_temp_reg_72, 8);
                m_temp_reg_76 = _mm_srli_si128(m_temp_reg_76, 8);
                m_temp_reg_82 = _mm_srli_si128(m_temp_reg_82, 8);
                m_temp_reg_86 = _mm_srli_si128(m_temp_reg_86, 8);

                /* eeo */
                {
                    /* eeo0[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                        m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_72);
                        m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_76);
                        m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_82);
                        m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_86);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_40, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_40, m_temp_reg_30);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                        m_temp_reg_90 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                        m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);
                    }
                    /* eeo0[4-7] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_44, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_44, m_temp_reg_30);
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                        m_temp_reg_91 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                        m_temp_reg_96 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    }

                    /* eeo1[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);



                        /* e[1][0-3] stored in pi2_tmp[2][0-7] */
                        /* e[6][0-3] stored in pi2_tmp[2][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_41, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_41, m_temp_reg_30);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                        m_temp_reg_92 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                        m_temp_reg_95 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    }

                    /* eo1[4-7] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);


                        /* e[1][4-7] stored in pi2_tmp[3][0-7] */
                        /* e[6][4-7] stored in pi2_tmp[3][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_45, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_45, m_temp_reg_30);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                        m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                        m_temp_reg_94 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);


                    }

                    /* eo2[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                        /* e[2][0-3] stored in pi2_tmp[4][0-7] */
                        /* e[5][0-3] stored in pi2_tmp[4][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_42, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_42, m_temp_reg_30);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                        temp1 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                        temp7 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    }

                    /* eo2[4-7] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                        /* e[2][4-7] stored in pi2_tmp[5][0-7] */
                        /* e[5][4-7] stored in pi2_tmp[5][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_46, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_46, m_temp_reg_30);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                        temp2 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                        temp6 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    }

                    /* eo3[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                        m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                        /* e[3][0-3] stored in pi2_tmp[6][0-7] */
                        /* e[4][0-3] stored in pi2_tmp[6][8-15] */
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_43, m_temp_reg_30);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_43, m_temp_reg_30);
                        m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                        m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);
                        m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                        m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);
                        temp3 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                        temp5 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

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
                        temp4 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                        temp8 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);


                    }
                    /* All values of ee[] array in pi2_temp */

                    /* for(k = 0; k < 8; k++) */


                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                    m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_73);
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_75);
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_77);

                    m_temp_reg_18 = _mm_cvtepi16_epi32(m_temp_reg_81);
                    m_temp_reg_19 = _mm_cvtepi16_epi32(m_temp_reg_83);
                    m_temp_reg_20 = _mm_cvtepi16_epi32(m_temp_reg_85);
                    m_temp_reg_21 = _mm_cvtepi16_epi32(m_temp_reg_87);

                    m_temp_reg_71 = _mm_srli_si128(m_temp_reg_71, 8);
                    m_temp_reg_73 = _mm_srli_si128(m_temp_reg_73, 8);
                    m_temp_reg_75 = _mm_srli_si128(m_temp_reg_75, 8);
                    m_temp_reg_77 = _mm_srli_si128(m_temp_reg_77, 8);

                    m_temp_reg_81 = _mm_srli_si128(m_temp_reg_81, 8);
                    m_temp_reg_83 = _mm_srli_si128(m_temp_reg_83, 8);
                    m_temp_reg_85 = _mm_srli_si128(m_temp_reg_85, 8);
                    m_temp_reg_87 = _mm_srli_si128(m_temp_reg_87, 8);

                }
            }
            /* eo */
            {
                WORD32 *pi2_scratch = o_temp_ptr;

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff5);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff6);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff7);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff8);

                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_71);
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_73);
                    m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_75);
                    m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);
                    m_temp_reg_22 = _mm_cvtepi16_epi32(m_temp_reg_81);
                    m_temp_reg_23 = _mm_cvtepi16_epi32(m_temp_reg_83);
                    m_temp_reg_24 = _mm_cvtepi16_epi32(m_temp_reg_85);
                    m_temp_reg_25 = _mm_cvtepi16_epi32(m_temp_reg_87);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_90, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_90, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);


                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /* eo0[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff5);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff6);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff7);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_91, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_91, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
                /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff3);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff1);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff4);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff7);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_92, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_92, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo1[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff8);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff6);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff3);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff1);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff4);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff7);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_93, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_93, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff7);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff5);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff1);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(temp1, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp1, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo2[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff7);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff5);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff1);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff6);

                    m_temp_reg_34 = _mm_add_epi32(temp2, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp2, m_temp_reg_30);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
        /**************************************************************************/


                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff1);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff7);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff3);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(temp3, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp3, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo3[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff8);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff1);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff7);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff3);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff5);

                    m_temp_reg_34 = _mm_add_epi32(temp4, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp4, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo4[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff8);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff2);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff6);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff4);

                    m_temp_reg_34 = _mm_add_epi32(temp5, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp5, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
                /* eo4[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff7);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff8);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff2);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff6);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff4);

                    m_temp_reg_34 = _mm_add_epi32(temp8, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp8, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

        /***********************************************************************/

                /* eo5[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff2);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff4);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff8);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff3);

                    m_temp_reg_34 = _mm_add_epi32(temp7, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp7, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo5[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff5);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff7);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff2);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff4);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff8);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff3);

                    m_temp_reg_34 = _mm_add_epi32(temp6, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(temp6, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }

                /* eo6[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff6);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff8);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff5);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff2);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_95, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_95, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo6[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff6);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff8);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff5);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff2);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_94, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_94, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo7[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff4);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff3);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff2);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff1);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_97, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_97, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }


                /* eo7[4-7] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_16, m_coeff6);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_17, m_coeff5);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_22, m_coeff4);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_23, m_coeff3);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_24, m_coeff2);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_25, m_coeff1);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_96, m_temp_reg_30);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_96, m_temp_reg_30);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_31);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_31);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_32);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_33);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_5);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_35 = _mm_sub_epi32(m_temp_reg_35, m_temp_reg_6);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);
                    m_temp_reg_35 = _mm_add_epi32(m_temp_reg_35, m_temp_reg_7);

                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_34);
                    pi2_scratch += 4;
                    _mm_storeu_si128((__m128i *)pi2_scratch, m_temp_reg_35);
                    pi2_scratch += 4;

                }
            }

         }
        /*  All e[] are done */
        /****************************/


        /* IQUANT for samples used in O */
        {

            WORD16 *pi2_tmp_src = pi2_src + src_strd;
            WORD16 *pi2_tmp_dequant = pi2_dequant_coeff + trans_size;

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

            if(!zero_last28_rows_stg1)
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

                if(!zero_last24_rows_stg1)
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

                    /* row 6 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                    m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                    m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    /* row 2 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    m_temp_reg_80 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_81 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

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

                    /* row 4 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    m_temp_reg_82 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_83 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

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

                    /* row 6 */
                    m_temp_reg_60 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_50 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    m_temp_reg_60 = _mm_srli_si128(m_temp_reg_60, 8);
                    m_temp_reg_50 = _mm_mullo_epi16(m_temp_reg_50, m_scale);
                    m_temp_reg_60 = _mm_cvtepi16_epi32(m_temp_reg_60);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    m_temp_reg_84 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_85 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);

                    /* row 7 */
                    m_temp_reg_61 = _mm_loadu_si128((__m128i *) pi2_tmp_src);
                    m_temp_reg_51 = _mm_loadu_si128((__m128i *) pi2_tmp_dequant);
                    m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    m_temp_reg_61 = _mm_srli_si128(m_temp_reg_61, 8);
                    m_temp_reg_51 = _mm_mullo_epi16(m_temp_reg_51, m_scale);
                    m_temp_reg_61 = _mm_cvtepi16_epi32(m_temp_reg_61);
                    pi2_tmp_src += (src_strd << 1);
                    pi2_tmp_dequant += (trans_size <<  1);

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

                    m_temp_reg_86 = _mm_packs_epi32(m_temp_reg_10, m_temp_reg_11);
                    m_temp_reg_87 = _mm_packs_epi32(m_temp_reg_12, m_temp_reg_13);
                }
                else
                {
                    m_temp_reg_74 = _mm_setzero_si128();
                    m_temp_reg_75 = _mm_setzero_si128();

                    m_temp_reg_76 = _mm_setzero_si128();
                    m_temp_reg_77 = _mm_setzero_si128();

                    m_temp_reg_80 = _mm_setzero_si128();
                    m_temp_reg_81 = _mm_setzero_si128();

                    m_temp_reg_82 = _mm_setzero_si128();
                    m_temp_reg_83 = _mm_setzero_si128();

                    m_temp_reg_84 = _mm_setzero_si128();
                    m_temp_reg_85 = _mm_setzero_si128();

                    m_temp_reg_86 = _mm_setzero_si128();
                    m_temp_reg_87 = _mm_setzero_si128();
                }
            }
            else
            {
                m_temp_reg_72 = _mm_setzero_si128();
                m_temp_reg_73 = _mm_setzero_si128();

                m_temp_reg_74 = _mm_setzero_si128();
                m_temp_reg_75 = _mm_setzero_si128();

                m_temp_reg_76 = _mm_setzero_si128();
                m_temp_reg_77 = _mm_setzero_si128();

                m_temp_reg_80 = _mm_setzero_si128();
                m_temp_reg_81 = _mm_setzero_si128();

                m_temp_reg_82 = _mm_setzero_si128();
                m_temp_reg_83 = _mm_setzero_si128();

                m_temp_reg_84 = _mm_setzero_si128();
                m_temp_reg_85 = _mm_setzero_si128();

                m_temp_reg_86 = _mm_setzero_si128();
                m_temp_reg_87 = _mm_setzero_si128();
            }
        }
/*1*trans_size = m_temp_reg_70
3*trans_size = m_temp_reg_71
5*trans_size = m_temp_reg_72
7*trans_size = m_temp_reg_73
9*trans_size = m_temp_reg_74
11*trans_size = m_temp_reg_75
13*trans_size = m_temp_reg_76
15*trans_size = m_temp_reg_77*/

/*17*trans_size = m_temp_reg_80
19*trans_size = m_temp_reg_81
21*trans_size = m_temp_reg_82
23*trans_size = m_temp_reg_83
25*trans_size = m_temp_reg_84
27*trans_size = m_temp_reg_85
29*trans_size = m_temp_reg_86
31*trans_size = m_temp_reg_87*/

        if(zero_last28_rows_stg1)
        {
            /* o & stage 1 out */
            {
                WORD32 j;
                WORD32 *pi2_src_scratch = o_temp_ptr;
                WORD16 *pi2_dst_scratch = temp_ptr;
                WORD32 out_stride = (trans_size << 1);
                WORD32 in_stride = trans_size >> 1;

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);

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
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

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
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );

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

                     /* o2[0-3] */
                     {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );

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

                     /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);

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
                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );

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

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                    /* o6[0-3] */
                     {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                     /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch += 8;

                    }
                     /* o8[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;
                    }
                      /* o9[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;
                    }

                        /* o10[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;
                    }

                     /* o11[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;

                    }

                   /* o12[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;

                   }


                    /* o13[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;
                    }

                      /* o14[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch -= in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch -= out_stride;

                      }

                      /* o15[0-3] */
                      {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch +=8;
                   }
                }
            }
        }
        else if(zero_last24_rows_stg1)
        {
            /* o & stage 1 out */
            {
                WORD32 j;
                WORD32 *pi2_src_scratch = o_temp_ptr;
                WORD16 *pi2_dst_scratch = temp_ptr;
                WORD32 out_stride = (trans_size << 1);
                WORD32 in_stride = trans_size >> 1;

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);

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
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

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
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff10);

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

                    /* o2[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff12);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff14);

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

                     /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff14);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7 );

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
                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff9  );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1  );

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

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4  );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6  );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                    /* o6[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff13);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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
                    /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff11);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

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
                        pi2_dst_scratch += 8;

                    }
                     /* o8[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff10);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4 );

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
                        pi2_dst_scratch -= out_stride;
                    }
                      /* o9[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff15);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2 );

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
                        pi2_dst_scratch -= out_stride;
                    }

                        /* o10[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff11);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff9 );

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
                        pi2_dst_scratch -= out_stride;
                    }

                     /* o11[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff15);

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
                        pi2_dst_scratch -= out_stride;

                    }

                   /* o12[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8 );

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
                        pi2_dst_scratch -= out_stride;

                   }


                    /* o13[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1 );

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
                        pi2_dst_scratch -= out_stride;
                    }

                      /* o14[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5 );

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
                        pi2_dst_scratch -= out_stride;

                      }

                      /* o15[0-3] */
                      {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff13);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff12);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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
                        pi2_dst_scratch +=8;
                   }
                }
            }
        }
        else
        {
            /* o & stage 1 out */
            {
                WORD32 j;
                WORD32 *pi2_src_scratch = o_temp_ptr;
                WORD16 *pi2_dst_scratch = temp_ptr;
                WORD32 out_stride = (trans_size << 1);
                WORD32 in_stride = trans_size >> 1;

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);

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

                        m_temp_reg_80 = _mm_srli_si128(m_temp_reg_80, 8);
                        m_temp_reg_81 = _mm_srli_si128(m_temp_reg_81, 8);
                        m_temp_reg_82 = _mm_srli_si128(m_temp_reg_82, 8);
                        m_temp_reg_83 = _mm_srli_si128(m_temp_reg_83, 8);
                        m_temp_reg_84 = _mm_srli_si128(m_temp_reg_84, 8);
                        m_temp_reg_85 = _mm_srli_si128(m_temp_reg_85, 8);
                        m_temp_reg_86 = _mm_srli_si128(m_temp_reg_86, 8);
                        m_temp_reg_87 = _mm_srli_si128(m_temp_reg_87, 8);
                    }


                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                    m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                    m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7
                    m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 9
                    m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_75);//row 11
                    m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 13
                    m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);//row 15

                    temp1 = _mm_cvtepi16_epi32(m_temp_reg_80);//row 17
                    temp2 = _mm_cvtepi16_epi32(m_temp_reg_81);//row 19
                    temp3 = _mm_cvtepi16_epi32(m_temp_reg_82);//row 21
                    temp4 = _mm_cvtepi16_epi32(m_temp_reg_83);//row 23
                    temp5 = _mm_cvtepi16_epi32(m_temp_reg_84);//row 25
                    temp6 = _mm_cvtepi16_epi32(m_temp_reg_85);//row 27
                    temp7 = _mm_cvtepi16_epi32(m_temp_reg_86);//row 29
                    temp8 = _mm_cvtepi16_epi32(m_temp_reg_87);//row 31


                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff6);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff7);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff8);
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff9);
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff10);
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff11);
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff12);
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff13);
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff14);
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff15);

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

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff10);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff13);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff15);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff12);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff9 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff6 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff3 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff1 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff2 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff5 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff8 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff11);
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff14);

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                     /* o2[0-3] */
                     {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff12);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff14);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff9 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff5 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff10);
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff15);
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff11);
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff6 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff1 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff3 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff8 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff13);

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                     /* o3[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff14);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff13);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff11);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff4 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff2 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff9 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff15);
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff8 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff1 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff5 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff12);

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                    /* o4[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff9  );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1  );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8  );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff14 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff5  );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3  );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff12);
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff10);
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff1 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff7 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff15);
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff6 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff2 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff11);

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

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                    /* o5[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4  );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6  );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff14 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3  );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff7  );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff13 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff2 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff8 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff12);
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff1 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff9 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff11);
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff1 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff10);

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                    /* o6[0-3] */
                     {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff13);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff11);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff14);
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff4 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff8 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff10);
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff2 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff15);
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff3 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff9 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += in_stride;

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                     /* o7[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff11);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff13);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff15);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff1 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff14);
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff2 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff12);
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff4 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff10);
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff6 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff8 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch += 8;

                        }
                     /* o8[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff10);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff12);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff14);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff15 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff1  );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff13 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff3  );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff11 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff5  );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff9  );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff7  );

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

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;
                    }
                      /* o9[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff15);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff10);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff14);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff1 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff11);
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff7 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff5 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff13);
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff1 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff12);
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff6 );

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


                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;
                    }

                        /* o10[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff11);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff9 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff12);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff8 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff13);
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff7 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff3 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff14);
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff6 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff4 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff15);
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff5 );

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

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;
                    }

                     /* o11[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff15);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff10);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff12);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff3  );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff5  );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff14 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff8  );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff1  );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff9  );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff13 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff4  );

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);


                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;

                    }

                   /* o12[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff15);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff9 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff11 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff13 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff6  );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff1  );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff7  );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff14 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff10 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff3  );

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;

                   }


                    /* o13[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff11);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff15);
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff10);

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff5  );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff1  );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff4  );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff9  );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff14 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff12 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff7  );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff2  );

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;
                    }

                      /* o14[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5 );
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2 );
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1 );
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff6 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff9  );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff12 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff15 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff13 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff10 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff7  );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff4  );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff1  );

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

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);




                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch -= out_stride;

                      }

                      /* o15[0-3] */
                      {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff13);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff12);
                        m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff11);
                        m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff10);
                        m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff9 );
                        m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff8 );

                        m_temp_reg_40 = _mm_mullo_epi32(temp1, m_coeff7 );
                        m_temp_reg_41 = _mm_mullo_epi32(temp2, m_coeff6 );
                        m_temp_reg_42 = _mm_mullo_epi32(temp3, m_coeff5 );
                        m_temp_reg_43 = _mm_mullo_epi32(temp4, m_coeff4 );
                        m_temp_reg_44 = _mm_mullo_epi32(temp5, m_coeff3 );
                        m_temp_reg_45 = _mm_mullo_epi32(temp6, m_coeff2 );
                        m_temp_reg_46 = _mm_mullo_epi32(temp7, m_coeff1 );
                        m_temp_reg_47 = _mm_mullo_epi32(temp8, m_coeff1 );

                        m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                        pi2_src_scratch += 4;

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


                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                        pi2_dst_scratch +=8;
                   }
                }
            }
        }
        /* Transpose */
        {
            WORD16 *pi2_src_scratch = temp_ptr;
            WORD16 *pi2_dst_scratch = pi2_tmp;
            WORD32 in_stride = (trans_size << 1);

            for(j = 0; j < 2; j++)
            {
                m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_31 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_32 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_33 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_34 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_35 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_36 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_37 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += 8;

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_74 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_75 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_76 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch -= in_stride;
                m_temp_reg_77 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += 8;


                m_temp_reg_40 = _mm_unpacklo_epi16(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_41 = _mm_unpackhi_epi16(m_temp_reg_31, m_temp_reg_30);

                m_temp_reg_42 = _mm_unpacklo_epi16(m_temp_reg_32, m_temp_reg_33);
                m_temp_reg_43 = _mm_unpackhi_epi16(m_temp_reg_33, m_temp_reg_32);

                m_temp_reg_44 = _mm_unpacklo_epi16(m_temp_reg_34, m_temp_reg_35);
                m_temp_reg_45 = _mm_unpackhi_epi16(m_temp_reg_35, m_temp_reg_34);

                m_temp_reg_46 = _mm_unpacklo_epi16(m_temp_reg_36, m_temp_reg_37);
                m_temp_reg_47 = _mm_unpackhi_epi16(m_temp_reg_37, m_temp_reg_36);

                m_temp_reg_80 = _mm_unpacklo_epi16(m_temp_reg_70, m_temp_reg_71);
                m_temp_reg_81 = _mm_unpackhi_epi16(m_temp_reg_71, m_temp_reg_70);

                m_temp_reg_82 = _mm_unpacklo_epi16(m_temp_reg_72, m_temp_reg_73);
                m_temp_reg_83 = _mm_unpackhi_epi16(m_temp_reg_73, m_temp_reg_72);

                m_temp_reg_84 = _mm_unpacklo_epi16(m_temp_reg_74, m_temp_reg_75);
                m_temp_reg_85 = _mm_unpackhi_epi16(m_temp_reg_75, m_temp_reg_74);

                m_temp_reg_86 = _mm_unpacklo_epi16(m_temp_reg_76, m_temp_reg_77);
                m_temp_reg_87 = _mm_unpackhi_epi16(m_temp_reg_77, m_temp_reg_76);

                /****************/

                m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_40, m_temp_reg_42);
                m_temp_reg_1 = _mm_unpackhi_epi32(m_temp_reg_40, m_temp_reg_42);

                m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_44, m_temp_reg_46);
                m_temp_reg_3 = _mm_unpackhi_epi32(m_temp_reg_44, m_temp_reg_46);

                m_temp_reg_4 = _mm_unpacklo_epi32(m_temp_reg_80, m_temp_reg_82);
                m_temp_reg_5 = _mm_unpackhi_epi32(m_temp_reg_80, m_temp_reg_82);

                m_temp_reg_6 = _mm_unpacklo_epi32(m_temp_reg_84, m_temp_reg_86);
                m_temp_reg_7 = _mm_unpackhi_epi32(m_temp_reg_84, m_temp_reg_86);

                m_temp_reg_90 = _mm_unpacklo_epi32(m_temp_reg_43, m_temp_reg_41);
                m_temp_reg_91 = _mm_unpackhi_epi32(m_temp_reg_43, m_temp_reg_41);

                m_temp_reg_92 = _mm_unpacklo_epi32(m_temp_reg_47, m_temp_reg_45);
                m_temp_reg_93 = _mm_unpackhi_epi32(m_temp_reg_47, m_temp_reg_45);

                m_temp_reg_94 = _mm_unpacklo_epi32(m_temp_reg_83, m_temp_reg_81);
                m_temp_reg_95 = _mm_unpackhi_epi32(m_temp_reg_83, m_temp_reg_81);

                m_temp_reg_96 = _mm_unpacklo_epi32(m_temp_reg_87, m_temp_reg_85);
                m_temp_reg_97 = _mm_unpackhi_epi32(m_temp_reg_87, m_temp_reg_85);

                /******************/

                m_temp_reg_30 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_2);
                m_temp_reg_31 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_2);

                m_temp_reg_32 = _mm_unpacklo_epi64(m_temp_reg_92, m_temp_reg_90);
                m_temp_reg_33 = _mm_unpackhi_epi64(m_temp_reg_92, m_temp_reg_90);

                m_temp_reg_34 = _mm_unpacklo_epi64(m_temp_reg_4, m_temp_reg_6);
                m_temp_reg_35 = _mm_unpackhi_epi64(m_temp_reg_4, m_temp_reg_6);

                m_temp_reg_36 = _mm_unpacklo_epi64(m_temp_reg_96, m_temp_reg_94);
                m_temp_reg_37 = _mm_unpackhi_epi64(m_temp_reg_96, m_temp_reg_94);

                m_temp_reg_80 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_3);
                m_temp_reg_81 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_3);

                m_temp_reg_82 = _mm_unpacklo_epi64(m_temp_reg_93, m_temp_reg_91);
                m_temp_reg_83 = _mm_unpackhi_epi64(m_temp_reg_93, m_temp_reg_91);

                m_temp_reg_84 = _mm_unpacklo_epi64(m_temp_reg_5, m_temp_reg_7);
                m_temp_reg_85 = _mm_unpackhi_epi64(m_temp_reg_5, m_temp_reg_7);

                m_temp_reg_86 = _mm_unpacklo_epi64(m_temp_reg_97, m_temp_reg_95);
                m_temp_reg_87 = _mm_unpackhi_epi64(m_temp_reg_97, m_temp_reg_95);

                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size), m_temp_reg_30);
                _mm_storeu_si128((__m128i *)(pi2_dst_scratch+0*trans_size+8), m_temp_reg_34);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size+16), m_temp_reg_36);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size+24), m_temp_reg_32);

                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size), m_temp_reg_31);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+8), m_temp_reg_35);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+16), m_temp_reg_37);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+24), m_temp_reg_33);

                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size), m_temp_reg_80);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+8), m_temp_reg_84);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+16), m_temp_reg_86);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+24), m_temp_reg_82);

                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size), m_temp_reg_81);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+8), m_temp_reg_85);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+16), m_temp_reg_87);
                _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+24), m_temp_reg_83);

                pi2_dst_scratch += 4*trans_size;
            }
        }
        pi2_src +=8;
        pi2_dequant_coeff +=8;
        pi2_tmp += 8*trans_size;
        zero_cols = zero_cols >> 1;
    }

    if(trans_size_stg1 != TRANS_SIZE_32)
    {
        m_temp_reg_10 = _mm_setzero_si128();

        for(i = trans_size_stg1; i < 32 ; i += 8)
            {
            WORD16 *pi2_dst_scratch = pi2_tmp;

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+0*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+1*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+2*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+3*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+4*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+4*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+4*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+4*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+5*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+5*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+5*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+5*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+6*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+6*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+6*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+6*trans_size+24), m_temp_reg_10);

            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+7*trans_size), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+7*trans_size+8), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+7*trans_size+16), m_temp_reg_10);
            _mm_storeu_si128((__m128i *) (pi2_dst_scratch+7*trans_size+24), m_temp_reg_10);

            pi2_tmp += 8*trans_size;
        }
            }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */
        shift = IT_SHIFT_STAGE_2;
        add = 1 << (shift - 1);

        for(j = 0; j < trans_size; j +=4)
        {
            i4_shift = IT_SHIFT_STAGE_2;

            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            if(zero_last28_rows_stg2)
            {
            {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                    m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50


                    m_temp_reg_10 = _mm_loadu_si128((__m128i *) & pi2_tmp[2*trans_size]);

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_10);

                    /* eo0[0-3] */
                    {
                        m_temp_reg_90 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                    }
                    /* eo1[0-3] */
                    {
                        m_temp_reg_91 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                    }
                    /* eo2[0-3] */
                    {
                        m_temp_reg_92 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    }

                    /* eo3[0-3] */
                    {
                        m_temp_reg_93 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    }
                    /* eo4[0-3] */
                    {
                        m_temp_reg_94 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                    }

                    /* eo5[0-3] */
                    {
                        m_temp_reg_95 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                    }

                    /* eo6[0-3] */
            {
                        m_temp_reg_96 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
            }
                    /* eo7[0-3] */
            {
                        m_temp_reg_97 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                    }
            }

                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) &pi2_tmp[0 * trans_size]);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                /* e[]*/

                temp1 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_90);  /* ee[0] */
                temp2 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_90);  /* ee[15] */

                temp3 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_91);  /* ee[1] */
                temp4 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_91);  /* ee[14] */

                temp5 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_92);  /* ee[2] */
                temp6 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_92);  /* ee[13] */

                temp7 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_93);  /* ee[3] */
                temp8 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_93);  /* ee[12] */

                m_temp_reg_90 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_94);  /* ee[4] */
                m_temp_reg_91 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_94);  /* ee[11] */

                m_temp_reg_92 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_95);  /* ee[5] */
                m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_95);  /* ee[10] */

                m_temp_reg_94 = _mm_add_epi32(m_temp_reg_16, m_temp_reg_96);  /* ee[6] */
                m_temp_reg_95 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_96);  /* ee[9] */

                m_temp_reg_96 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_97);  /* ee[7] */
                m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_14, m_temp_reg_97);  /* ee[8] */

                /*o[k]*/
                {

                    WORD16 *pi2_dst_scratch = temp_ptr;
                    WORD32 out_stride = 8;


                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                    m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                    m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                    m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                    m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                    m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                    m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                    m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                    m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);


                    m_temp_reg_70 = _mm_loadu_si128((__m128i *) & pi2_tmp[trans_size]);
                    m_temp_reg_71 = _mm_loadu_si128((__m128i *) &pi2_tmp[3 * trans_size]);

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3

                    /* o0[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                        m_temp_reg_31 = _mm_sub_epi32(temp1, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp1, m_temp_reg_20);

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
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );

                        m_temp_reg_31 = _mm_sub_epi32(temp3, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp3, m_temp_reg_20);

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

                     /* o2[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );

                        m_temp_reg_31 = _mm_sub_epi32(temp5, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp5, m_temp_reg_20);

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

                     /* o3[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);

                        m_temp_reg_31 = _mm_sub_epi32(temp7, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp7, m_temp_reg_20);

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
                    /* o4[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_90, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_90, m_temp_reg_20);

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

                    /* o5[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_92, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_92, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                    /* o6[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_94, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_94, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                     /* o7[0-3] */
                {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_96, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_96, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch += 8;

                    }
                     /* o8[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_97, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_97, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                      /* o9[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_95, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_95, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                        /* o10[0-3] */
                {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_93, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_93, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                     /* o11[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_91, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_91, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                   /* o12[0-3] */
                 {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );

                        m_temp_reg_31 = _mm_sub_epi32(temp8, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp8, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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


                    /* o13[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );

                        m_temp_reg_31 = _mm_sub_epi32(temp6, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp6, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                      /* o14[0-3] */
                    {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);

                        m_temp_reg_31 = _mm_sub_epi32(temp4, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp4, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                      /* o15[0-3] */
                {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);

                        m_temp_reg_31 = _mm_sub_epi32(temp2, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp2, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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
                        pi2_dst_scratch +=8;
                   }
                }

                    }
            else if(zero_last24_rows_stg2)
            {
                /* eo */
                {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                    m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                    m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                    m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                    m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50


                    m_temp_reg_10 = _mm_loadu_si128((__m128i *) & pi2_tmp[2*trans_size]);
                    m_temp_reg_11 = _mm_loadu_si128((__m128i *) &pi2_tmp[6 * trans_size]);

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_10);
                    m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_11);

                    /* eo0[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);

                        m_temp_reg_90 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);

                    }

                                /* eo1[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);

                        m_temp_reg_91 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);

                    }
                    /* eo2[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);

                        m_temp_reg_92 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);
                    }

                    /* eo3[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);

                        m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    }
                    /* eo4[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);

                        m_temp_reg_94 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    }
                    /* eo5[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);

                        m_temp_reg_95 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    }
                    /* eo6[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);

                        m_temp_reg_96 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    }
                    /* eo7[0-3] */
                    {
                        m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                        m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);

                        m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    }

                }

                /* eeo */
                {
                    m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
                    m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
                    m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
                    m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

                    m_temp_reg_72 = _mm_loadu_si128((__m128i *) & pi2_tmp[4*trans_size]);

                    m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_72);

                    /* eeo0[0-3] */
                    {
                        temp1 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);

                    }

                    /* eeo1[0-3] */
                    {
                        temp2 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);

                    }

                    /* eo2[0-3] */
                    {
                        temp3 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);

                    }


                    /* eo3[0-3] */
                {
                        temp4 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);

                    }

                }

                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) &pi2_tmp[0 * trans_size]);

                m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);

                m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
                m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);

                m_temp_reg_70 = _mm_add_epi32(m_temp_reg_14, temp1);  /* ee[0] */
                m_temp_reg_71 = _mm_sub_epi32(m_temp_reg_14, temp1);  /* ee[7] */

                m_temp_reg_72 = _mm_add_epi32(m_temp_reg_16, temp2);  /* ee[1] */
                m_temp_reg_73 = _mm_sub_epi32(m_temp_reg_16, temp2);  /* ee[6] */

                m_temp_reg_74 = _mm_add_epi32(m_temp_reg_16, temp3);  /* ee[2] */
                m_temp_reg_75 = _mm_sub_epi32(m_temp_reg_16, temp3);  /* ee[5] */

                m_temp_reg_76 = _mm_add_epi32(m_temp_reg_14, temp4);  /* ee[3] */
                m_temp_reg_77 = _mm_sub_epi32(m_temp_reg_14, temp4);  /* ee[4] */

                /* e[]*/

                temp1 = _mm_add_epi32(m_temp_reg_70, m_temp_reg_90);  /* ee[0] */
                temp2 = _mm_sub_epi32(m_temp_reg_70, m_temp_reg_90);  /* ee[15] */

                temp3 = _mm_add_epi32(m_temp_reg_72, m_temp_reg_91);  /* ee[1] */
                temp4 = _mm_sub_epi32(m_temp_reg_72, m_temp_reg_91);  /* ee[14] */

                temp5 = _mm_add_epi32(m_temp_reg_74, m_temp_reg_92);  /* ee[2] */
                temp6 = _mm_sub_epi32(m_temp_reg_74, m_temp_reg_92);  /* ee[13] */

                temp7 = _mm_add_epi32(m_temp_reg_76, m_temp_reg_93);  /* ee[3] */
                temp8 = _mm_sub_epi32(m_temp_reg_76, m_temp_reg_93);  /* ee[12] */

                m_temp_reg_90 = _mm_add_epi32(m_temp_reg_77, m_temp_reg_94);  /* ee[4] */
                m_temp_reg_91 = _mm_sub_epi32(m_temp_reg_77, m_temp_reg_94);  /* ee[11] */

                m_temp_reg_92 = _mm_add_epi32(m_temp_reg_75, m_temp_reg_95);  /* ee[5] */
                m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_75, m_temp_reg_95);  /* ee[10] */

                m_temp_reg_94 = _mm_add_epi32(m_temp_reg_73, m_temp_reg_96);  /* ee[6] */
                m_temp_reg_95 = _mm_sub_epi32(m_temp_reg_73, m_temp_reg_96);  /* ee[9] */

                m_temp_reg_96 = _mm_add_epi32(m_temp_reg_71, m_temp_reg_97);  /* ee[7] */
                m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_71, m_temp_reg_97);  /* ee[8] */

            /*o[k] */
            {

                WORD16 *pi2_dst_scratch = temp_ptr;
                WORD32 out_stride = 8;


                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);


                m_temp_reg_70 = _mm_loadu_si128((__m128i *) & pi2_tmp[trans_size]);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) &pi2_tmp[3 * trans_size]);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) &pi2_tmp[5 * trans_size]);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) &pi2_tmp[7 * trans_size]);

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7

                /* o0[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                    m_temp_reg_31 = _mm_sub_epi32(temp1, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp1, m_temp_reg_20);

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
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff10);

                    m_temp_reg_31 = _mm_sub_epi32(temp3, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp3, m_temp_reg_20);

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

                 /* o2[0-3] */
                 {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff12);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff14);

                    m_temp_reg_31 = _mm_sub_epi32(temp5, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp5, m_temp_reg_20);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                 /* o3[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff14);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7 );

                    m_temp_reg_31 = _mm_sub_epi32(temp7, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp7, m_temp_reg_20);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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
                /* o4[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff9  );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1  );

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_90, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_90, m_temp_reg_20);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                /* o5[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4  );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6  );

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_92, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_92, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                /* o6[0-3] */
                 {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff13);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_94, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_94, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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
                 /* o7[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff11);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_96, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_96, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

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
                    pi2_dst_scratch += 8;

                }
                 /* o8[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff10);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4 );

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_97, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_97, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

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
                  /* o9[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff15);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2 );

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_95, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_95, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

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

                    /* o10[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff11);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff9 );

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_93, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_93, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

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

                 /* o11[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff15);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_91, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_91, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

               /* o12[0-3] */
                  {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8 );

                    m_temp_reg_31 = _mm_sub_epi32(temp8, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp8, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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


                /* o13[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1 );

                    m_temp_reg_31 = _mm_sub_epi32(temp6, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp6, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                  /* o14[0-3] */
            {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8 );
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5 );

                        m_temp_reg_31 = _mm_sub_epi32(temp4, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp4, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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

                  /* o15[0-3] */
                  {
                        m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                        m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);
                        m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff13);
                        m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff12);

                        m_temp_reg_31 = _mm_sub_epi32(temp2, m_temp_reg_20);
                        m_temp_reg_30 = _mm_add_epi32(temp2, m_temp_reg_20);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                        m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                        m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                        m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                        m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

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
                        pi2_dst_scratch +=8;
            }
        }
        }
            else
        {
            /* eo */
             {
                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[0][0]);//89
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[1][0]);//75
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[2][0]);//18
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[3][0]);//50
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[4][0]);//89
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[5][0]);//75
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[6][0]);//18
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_8[7][0]);//50


                m_temp_reg_10 = _mm_loadu_si128((__m128i *) & pi2_tmp[2*trans_size]);
                m_temp_reg_11 = _mm_loadu_si128((__m128i *) &pi2_tmp[6 * trans_size]);
                m_temp_reg_12 = _mm_loadu_si128((__m128i *) &pi2_tmp[10 * trans_size]);
                m_temp_reg_13 = _mm_loadu_si128((__m128i *) &pi2_tmp[14 * trans_size]);
                m_temp_reg_18 = _mm_loadu_si128((__m128i *) &pi2_tmp[18 * trans_size]);
                m_temp_reg_19 = _mm_loadu_si128((__m128i *) &pi2_tmp[22 * trans_size]);
                m_temp_reg_20 = _mm_loadu_si128((__m128i *) &pi2_tmp[26 * trans_size]);
                m_temp_reg_21 = _mm_loadu_si128((__m128i *) &pi2_tmp[30 * trans_size]);

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_10);
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_11);
                m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_12);
                m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_13);

                m_temp_reg_18 = _mm_cvtepi16_epi32(m_temp_reg_18);
                m_temp_reg_19 = _mm_cvtepi16_epi32(m_temp_reg_19);
                m_temp_reg_20 = _mm_cvtepi16_epi32(m_temp_reg_20);
                m_temp_reg_21 = _mm_cvtepi16_epi32(m_temp_reg_21);

                /* eo0[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff5);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff6);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff7);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff8);

                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_90 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);

                }

                            /* eo1[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff3);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff1);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff4);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff7);


                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_91 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);


                }


                /* eo2[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff7);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff5);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff1);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff6);


                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_92 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);

                }



                /* eo3[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff1);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff7);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff3);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff5);


                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);


                }




                /* eo4[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff8);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff2);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff6);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff4);

                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_94 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);

                }

                /* eo5[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff2);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff4);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff8);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff3);


                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_95 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);

                }


                /* eo6[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff6);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff8);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff5);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff2);


                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_96 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_7);

                }


                /* eo7[0-3] */
                {
                    m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8);
                    m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7);
                    m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6);
                    m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5);

                    m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_18, m_coeff4);
                    m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_19, m_coeff3);
                    m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_20, m_coeff2);
                    m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_21, m_coeff1);


                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_4);
                    m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_5);
                    m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_6);
                    m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_7);

                }

            }

        /* eeo */
        {
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[1][0]);//89
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[2][0]);//75
            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[5][0]);//18
            m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[6][0]);//50

            m_temp_reg_72 = _mm_loadu_si128((__m128i *) & pi2_tmp[4*trans_size]);
            m_temp_reg_76 = _mm_loadu_si128((__m128i *) &pi2_tmp[12 * trans_size]);
            m_temp_reg_82 = _mm_loadu_si128((__m128i *) &pi2_tmp[20 * trans_size]);
            m_temp_reg_86 = _mm_loadu_si128((__m128i *) &pi2_tmp[28 * trans_size]);

            m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_72);
            m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_76);
            m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_82);
            m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_86);




            /* eeo0[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2);
                m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4);
                m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);

                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);
                temp1 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);

            }

            /* eeo1[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3);
                m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1);
                m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_32);
                temp2 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);

            }

            /* eo2[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3);
                m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);

                temp3 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_33);

            }


            /* eo3[0-3] */
            {
                m_temp_reg_30 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3);
                m_temp_reg_31 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4);
                m_temp_reg_32 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                m_temp_reg_33 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1);

                m_temp_reg_34 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_34 = _mm_add_epi32(m_temp_reg_34, m_temp_reg_32);

                temp4 = _mm_sub_epi32(m_temp_reg_34, m_temp_reg_33);

            }

    }

            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[3][0]);//83
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[4][0]);//36
            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_16_even[0][0]);//64

            m_temp_reg_74 = _mm_loadu_si128((__m128i *) & pi2_tmp[8*trans_size]);
            m_temp_reg_84 = _mm_loadu_si128((__m128i *) &pi2_tmp[24 * trans_size]);

            m_temp_reg_0 = _mm_cvtepi16_epi32(m_temp_reg_74); /* row 8 */
            m_temp_reg_2 = _mm_cvtepi16_epi32(m_temp_reg_84); /* row 24 */

            m_temp_reg_10 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
            m_temp_reg_11 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
            m_temp_reg_12 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
            m_temp_reg_13 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);

            m_temp_reg_70 = _mm_loadu_si128((__m128i *) &pi2_tmp[0 * trans_size]);
            m_temp_reg_80 = _mm_loadu_si128((__m128i *) &pi2_tmp[16 * trans_size]);

            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_70);
            m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_80);

            m_temp_reg_14 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
            m_temp_reg_15 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);
            m_temp_reg_16 = _mm_mullo_epi32(m_temp_reg_1, m_coeff3);
            m_temp_reg_17 = _mm_mullo_epi32(m_temp_reg_3, m_coeff3);

            m_temp_reg_20 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_11);  /* eeeo[0] */
            m_temp_reg_22 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_13);  /* eeeo[1] */

            m_temp_reg_21 = _mm_add_epi32(m_temp_reg_14, m_temp_reg_15);  /* eeee[0] */
            m_temp_reg_23 = _mm_sub_epi32(m_temp_reg_16, m_temp_reg_17);  /* eeee[1] */

/* eeeo[0]= m_temp_reg_20  */
/* eeeo[1]= m_temp_reg_21  */
/* eeee[0]= m_temp_reg_22  */
/* eeee[1]= m_temp_reg_23  */

            /* eee[0] = eeee[0] + eeeo[0]; */
            m_temp_reg_40 = _mm_add_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[0] */

            /* eee[3] = eeee[0] - eeeo[0]; */
            m_temp_reg_43 = _mm_sub_epi32(m_temp_reg_21, m_temp_reg_20);  /* eeeo[1] */

            /* eee[2] = eeee[1] - eeeo[1]; */
            m_temp_reg_42 = _mm_sub_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[1] */

            /* eee[1] = eeee[1] + eeeo[1];*/
            m_temp_reg_41 = _mm_add_epi32(m_temp_reg_23, m_temp_reg_22);  /* eeee[0] */

            m_temp_reg_70 = _mm_add_epi32(m_temp_reg_40, temp1);  /* ee[0] */
            m_temp_reg_71 = _mm_sub_epi32(m_temp_reg_40, temp1);  /* ee[7] */

            m_temp_reg_72 = _mm_add_epi32(m_temp_reg_41, temp2);  /* ee[1] */
            m_temp_reg_73 = _mm_sub_epi32(m_temp_reg_41, temp2);  /* ee[6] */

            m_temp_reg_74 = _mm_add_epi32(m_temp_reg_42, temp3);  /* ee[2] */
            m_temp_reg_75 = _mm_sub_epi32(m_temp_reg_42, temp3);  /* ee[5] */

            m_temp_reg_76 = _mm_add_epi32(m_temp_reg_43, temp4);  /* ee[3] */
            m_temp_reg_77 = _mm_sub_epi32(m_temp_reg_43, temp4);  /* ee[4] */

/* e[]*/

            temp1 = _mm_add_epi32(m_temp_reg_70, m_temp_reg_90);  /* ee[0] */
            temp2 = _mm_sub_epi32(m_temp_reg_70, m_temp_reg_90);  /* ee[15] */

            temp3 = _mm_add_epi32(m_temp_reg_72, m_temp_reg_91);  /* ee[1] */
            temp4 = _mm_sub_epi32(m_temp_reg_72, m_temp_reg_91);  /* ee[14] */

            temp5 = _mm_add_epi32(m_temp_reg_74, m_temp_reg_92);  /* ee[2] */
            temp6 = _mm_sub_epi32(m_temp_reg_74, m_temp_reg_92);  /* ee[13] */

            temp7 = _mm_add_epi32(m_temp_reg_76, m_temp_reg_93);  /* ee[3] */
            temp8 = _mm_sub_epi32(m_temp_reg_76, m_temp_reg_93);  /* ee[12] */

            m_temp_reg_90 = _mm_add_epi32(m_temp_reg_77, m_temp_reg_94);  /* ee[4] */
            m_temp_reg_91 = _mm_sub_epi32(m_temp_reg_77, m_temp_reg_94);  /* ee[11] */

            m_temp_reg_92 = _mm_add_epi32(m_temp_reg_75, m_temp_reg_95);  /* ee[5] */
            m_temp_reg_93 = _mm_sub_epi32(m_temp_reg_75, m_temp_reg_95);  /* ee[10] */

            m_temp_reg_94 = _mm_add_epi32(m_temp_reg_73, m_temp_reg_96);  /* ee[6] */
            m_temp_reg_95 = _mm_sub_epi32(m_temp_reg_73, m_temp_reg_96);  /* ee[9] */

            m_temp_reg_96 = _mm_add_epi32(m_temp_reg_71, m_temp_reg_97);  /* ee[7] */
            m_temp_reg_97 = _mm_sub_epi32(m_temp_reg_71, m_temp_reg_97);  /* ee[8] */

/*o[k] */
            {

                WORD16 *pi2_dst_scratch = temp_ptr;
                WORD32 out_stride = 8;


                m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[0][0]);
                m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[1][0]);
                m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[2][0]);
                m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[3][0]);
                m_coeff5 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[4][0]);
                m_coeff6 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[5][0]);
                m_coeff7 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[6][0]);
                m_coeff8 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[7][0]);

                m_coeff9 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[8][0]);
                m_coeff10 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[9][0]);
                m_coeff11 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[10][0]);
                m_coeff12 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[11][0]);
                m_coeff13 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[12][0]);
                m_coeff14 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[13][0]);
                m_coeff15 = _mm_loadu_si128((__m128i *) &g_ai2_ihevc_trans_32_intr_16[14][0]);


                m_temp_reg_70 = _mm_loadu_si128((__m128i *) & pi2_tmp[trans_size]);
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) &pi2_tmp[3 * trans_size]);
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) &pi2_tmp[5 * trans_size]);
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) &pi2_tmp[7 * trans_size]);
                m_temp_reg_74 = _mm_loadu_si128((__m128i *) &pi2_tmp[9 * trans_size]);
                m_temp_reg_75 = _mm_loadu_si128((__m128i *) &pi2_tmp[11 * trans_size]);
                m_temp_reg_76 = _mm_loadu_si128((__m128i *) &pi2_tmp[13 * trans_size]);
                m_temp_reg_77 = _mm_loadu_si128((__m128i *) &pi2_tmp[15 * trans_size]);

                m_temp_reg_80 = _mm_loadu_si128((__m128i *) &pi2_tmp[17 * trans_size]);
                m_temp_reg_81 = _mm_loadu_si128((__m128i *) &pi2_tmp[19 * trans_size]);
                m_temp_reg_82 = _mm_loadu_si128((__m128i *) & pi2_tmp[21 * trans_size]);
                m_temp_reg_83 = _mm_loadu_si128((__m128i *) &pi2_tmp[23 * trans_size]);
                m_temp_reg_84 = _mm_loadu_si128((__m128i *) &pi2_tmp[25 * trans_size]);
                m_temp_reg_85 = _mm_loadu_si128((__m128i *) &pi2_tmp[27 * trans_size]);
                m_temp_reg_86 = _mm_loadu_si128((__m128i *) &pi2_tmp[29 * trans_size]);
                m_temp_reg_87 = _mm_loadu_si128((__m128i *) &pi2_tmp[31 * trans_size]);

                m_temp_reg_10 = _mm_cvtepi16_epi32(m_temp_reg_70);//row 1
                m_temp_reg_11 = _mm_cvtepi16_epi32(m_temp_reg_71);//row 3
                m_temp_reg_12 = _mm_cvtepi16_epi32(m_temp_reg_72);//row 5
                m_temp_reg_13 = _mm_cvtepi16_epi32(m_temp_reg_73);//row 7
                m_temp_reg_14 = _mm_cvtepi16_epi32(m_temp_reg_74);//row 9
                m_temp_reg_15 = _mm_cvtepi16_epi32(m_temp_reg_75);//row 11
                m_temp_reg_16 = _mm_cvtepi16_epi32(m_temp_reg_76);//row 13
                m_temp_reg_17 = _mm_cvtepi16_epi32(m_temp_reg_77);//row 15

                m_temp_reg_70 = _mm_cvtepi16_epi32(m_temp_reg_80);//row 17
                m_temp_reg_71 = _mm_cvtepi16_epi32(m_temp_reg_81);//row 19
                m_temp_reg_72 = _mm_cvtepi16_epi32(m_temp_reg_82);//row 21
                m_temp_reg_73 = _mm_cvtepi16_epi32(m_temp_reg_83);//row 23
                m_temp_reg_74 = _mm_cvtepi16_epi32(m_temp_reg_84);//row 25
                m_temp_reg_75 = _mm_cvtepi16_epi32(m_temp_reg_85);//row 27
                m_temp_reg_76 = _mm_cvtepi16_epi32(m_temp_reg_86);//row 29
                m_temp_reg_77 = _mm_cvtepi16_epi32(m_temp_reg_87);//row 31

                /* o0[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff2);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff3);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff4);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff5);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff6);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff7);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff8);
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff9);
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff10);
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff11);
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff12);
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff13);
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff14);
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff15);

                    m_temp_reg_31 = _mm_sub_epi32(temp1, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp1, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff1 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff4 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff7 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff10);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff13);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff15);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff12);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff9 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff6 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff3 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff1 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff2 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff5 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff8 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff11);
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff14);


                    m_temp_reg_31 = _mm_sub_epi32(temp3, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp3, m_temp_reg_20);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                 /* o2[0-3] */
                 {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff2 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff7 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff12);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff14);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff9 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff4 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff5 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff10);
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff15);
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff11);
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff6 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff1 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff3 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff8 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff13);


                    m_temp_reg_31 = _mm_sub_epi32(temp5, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp5, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                 /* o3[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff3 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff10);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff14);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff7 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff6 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff13);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff11);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff4 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff2 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff9 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff15);
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff8 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff1 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff5 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff12);



                    m_temp_reg_31 = _mm_sub_epi32(temp7, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp7, m_temp_reg_20);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                /* o4[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff4  );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff13 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff9  );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1  );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff8  );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff14 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff5  );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff3  );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff12);
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff10);
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff1 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff7 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff15);
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff6 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff2 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff11);


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_90, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_90, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                /* o5[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff5  );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff15 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff4  );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff6  );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff14 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff3  );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff7  );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff13 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff2 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff8 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff12);
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff1 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff9 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff11);
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff1 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff10);


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_92, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_92, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                /* o6[0-3] */
                 {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff6 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff12);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff13);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff5 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff7 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff11);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff14);
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff4 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff8 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff10);
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff2 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff15);
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff3 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff9 );



                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_94, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_94, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                 /* o7[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff7 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff9 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff5 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff11);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff3 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff13);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff1 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff15);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff1 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff14);
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff2 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff12);
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff4 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff10);
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff6 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff8 );



                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_96, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_96, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                    pi2_dst_scratch += 8;

                    }
                 /* o8[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff8 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff6 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff10);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff4 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff12);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff2 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff14);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff1 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff15 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff1  );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff13 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff3  );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff11 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff5  );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff9  );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff7  );


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_97, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_97, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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
                  /* o9[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff9 );
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff3 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff15);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff2 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff10);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff8 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff4 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff14);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff1 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff11);
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff7 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff5 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff13);
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff1 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff12);
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff6 );


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_95, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_95, m_temp_reg_20);

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


                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                    /* o10[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff10);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff1 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff11);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff9 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff1 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff12);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff8 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff2 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff13);
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff7 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff3 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff14);
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff6 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff4 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff15);
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff5 );


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_93, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_93, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                 /* o11[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff11);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff2 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff6 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff15);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff7 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff10);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff12);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff3  );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff5  );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff14 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff8  );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff1  );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff9  );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff13 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff4  );


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_91, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_91, m_temp_reg_20);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_21);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_21);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_22);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_22);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_23);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_23);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_24);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_24);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_25);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_25);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);


                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

               /* o12[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff12);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff5 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff1 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff8 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff15);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff9 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff2 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff4 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff11 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff13 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff6  );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff1  );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff7  );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff14 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff10 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff3  );


                    m_temp_reg_31 = _mm_sub_epi32(temp8, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp8, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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


                /* o13[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff13);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff8 );
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff3 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff1 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff6 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff11);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff15);
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff10);

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff5  );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff1  );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff4  );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff9  );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff14 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff12 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff7  );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff2  );


                    m_temp_reg_31 = _mm_sub_epi32(temp6, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp6, m_temp_reg_20);

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

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_26);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_26);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_27);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_27);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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

                  /* o14[0-3] */
                {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff14);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff11);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff8 );
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff5 );
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff2 );
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff1 );
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff3 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff6 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff9  );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff12 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff15 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff13 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff10 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff7  );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff4  );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff1  );

                    m_temp_reg_31 = _mm_sub_epi32(temp4, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp4, m_temp_reg_20);

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



                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_47);

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

                  /* o15[0-3] */
                  {
                    m_temp_reg_20 = _mm_mullo_epi32(m_temp_reg_10, m_coeff15);
                    m_temp_reg_21 = _mm_mullo_epi32(m_temp_reg_11, m_coeff14);
                    m_temp_reg_22 = _mm_mullo_epi32(m_temp_reg_12, m_coeff13);
                    m_temp_reg_23 = _mm_mullo_epi32(m_temp_reg_13, m_coeff12);
                    m_temp_reg_24 = _mm_mullo_epi32(m_temp_reg_14, m_coeff11);
                    m_temp_reg_25 = _mm_mullo_epi32(m_temp_reg_15, m_coeff10);
                    m_temp_reg_26 = _mm_mullo_epi32(m_temp_reg_16, m_coeff9 );
                    m_temp_reg_27 = _mm_mullo_epi32(m_temp_reg_17, m_coeff8 );

                    m_temp_reg_40 = _mm_mullo_epi32(m_temp_reg_70, m_coeff7 );
                    m_temp_reg_41 = _mm_mullo_epi32(m_temp_reg_71, m_coeff6 );
                    m_temp_reg_42 = _mm_mullo_epi32(m_temp_reg_72, m_coeff5 );
                    m_temp_reg_43 = _mm_mullo_epi32(m_temp_reg_73, m_coeff4 );
                    m_temp_reg_44 = _mm_mullo_epi32(m_temp_reg_74, m_coeff3 );
                    m_temp_reg_45 = _mm_mullo_epi32(m_temp_reg_75, m_coeff2 );
                    m_temp_reg_46 = _mm_mullo_epi32(m_temp_reg_76, m_coeff1 );
                    m_temp_reg_47 = _mm_mullo_epi32(m_temp_reg_77, m_coeff1 );


                    m_temp_reg_31 = _mm_sub_epi32(temp2, m_temp_reg_20);
                    m_temp_reg_30 = _mm_add_epi32(temp2, m_temp_reg_20);

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


                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_40);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_40);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_41);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_41);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_42);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_42);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_43);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_43);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_44);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_44);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_45);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_45);

                    m_temp_reg_31 = _mm_sub_epi32(m_temp_reg_31, m_temp_reg_46);
                    m_temp_reg_30 = _mm_add_epi32(m_temp_reg_30, m_temp_reg_46);

                    m_temp_reg_31 = _mm_add_epi32(m_temp_reg_31, m_temp_reg_47);
                    m_temp_reg_30 = _mm_sub_epi32(m_temp_reg_30, m_temp_reg_47);

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
                    pi2_dst_scratch +=8;
               }
           }
            }

           /* Transpose */
        {

            WORD16 *pi2_src_scratch = temp_ptr;
            WORD32 out_stride = dst_strd;
            WORD32 in_stride = 8;

                m_temp_reg_30 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_31 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_32 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_33 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_34 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_35 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_36 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_37 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += 8;

                m_temp_reg_70 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_71 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_72 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_73 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_74 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_75 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_76 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += in_stride;
                m_temp_reg_77 = _mm_loadu_si128((__m128i *) pi2_src_scratch);
                pi2_src_scratch += 8;


                m_temp_reg_40 = _mm_unpacklo_epi16(m_temp_reg_30, m_temp_reg_31);
                m_temp_reg_41 = _mm_unpackhi_epi16(m_temp_reg_31, m_temp_reg_30);

                m_temp_reg_42 = _mm_unpacklo_epi16(m_temp_reg_32, m_temp_reg_33);
                m_temp_reg_43 = _mm_unpackhi_epi16(m_temp_reg_33, m_temp_reg_32);

                m_temp_reg_44 = _mm_unpacklo_epi16(m_temp_reg_34, m_temp_reg_35);
                m_temp_reg_45 = _mm_unpackhi_epi16(m_temp_reg_35, m_temp_reg_34);

                m_temp_reg_46 = _mm_unpacklo_epi16(m_temp_reg_36, m_temp_reg_37);
                m_temp_reg_47 = _mm_unpackhi_epi16(m_temp_reg_37, m_temp_reg_36);

                m_temp_reg_80 = _mm_unpacklo_epi16(m_temp_reg_70, m_temp_reg_71);
                m_temp_reg_81 = _mm_unpackhi_epi16(m_temp_reg_71, m_temp_reg_70);

                m_temp_reg_82 = _mm_unpacklo_epi16(m_temp_reg_72, m_temp_reg_73);
                m_temp_reg_83 = _mm_unpackhi_epi16(m_temp_reg_73, m_temp_reg_72);

                m_temp_reg_84 = _mm_unpacklo_epi16(m_temp_reg_74, m_temp_reg_75);
                m_temp_reg_85 = _mm_unpackhi_epi16(m_temp_reg_75, m_temp_reg_74);

                m_temp_reg_86 = _mm_unpacklo_epi16(m_temp_reg_76, m_temp_reg_77);
                m_temp_reg_87 = _mm_unpackhi_epi16(m_temp_reg_77, m_temp_reg_76);


                m_temp_reg_0 = _mm_unpacklo_epi32(m_temp_reg_40, m_temp_reg_42);
                m_temp_reg_1 = _mm_unpackhi_epi32(m_temp_reg_40, m_temp_reg_42);

                m_temp_reg_2 = _mm_unpacklo_epi32(m_temp_reg_44, m_temp_reg_46);
                m_temp_reg_3 = _mm_unpackhi_epi32(m_temp_reg_44, m_temp_reg_46);

                m_temp_reg_4 = _mm_unpacklo_epi32(m_temp_reg_80, m_temp_reg_82);
                m_temp_reg_5 = _mm_unpackhi_epi32(m_temp_reg_80, m_temp_reg_82);

                m_temp_reg_6 = _mm_unpacklo_epi32(m_temp_reg_84, m_temp_reg_86);
                m_temp_reg_7 = _mm_unpackhi_epi32(m_temp_reg_84, m_temp_reg_86);

                m_temp_reg_90 = _mm_unpacklo_epi32(m_temp_reg_43, m_temp_reg_41);
                m_temp_reg_91 = _mm_unpackhi_epi32(m_temp_reg_43, m_temp_reg_41);

                m_temp_reg_92 = _mm_unpacklo_epi32(m_temp_reg_47, m_temp_reg_45);
                m_temp_reg_93 = _mm_unpackhi_epi32(m_temp_reg_47, m_temp_reg_45);

                m_temp_reg_94 = _mm_unpacklo_epi32(m_temp_reg_83, m_temp_reg_81);
                m_temp_reg_95 = _mm_unpackhi_epi32(m_temp_reg_83, m_temp_reg_81);

                m_temp_reg_96 = _mm_unpacklo_epi32(m_temp_reg_87, m_temp_reg_85);
                m_temp_reg_97 = _mm_unpackhi_epi32(m_temp_reg_87, m_temp_reg_85);


                m_temp_reg_30 = _mm_unpacklo_epi64(m_temp_reg_0, m_temp_reg_2);       // row0 = 0-7
                m_temp_reg_31 = _mm_unpackhi_epi64(m_temp_reg_0, m_temp_reg_2);       // row1 = 0-7

                m_temp_reg_32 = _mm_unpacklo_epi64(m_temp_reg_92, m_temp_reg_90);     // row0=24-31
                m_temp_reg_33 = _mm_unpackhi_epi64(m_temp_reg_92, m_temp_reg_90);     // row1=24-31

                m_temp_reg_34 = _mm_unpacklo_epi64(m_temp_reg_4, m_temp_reg_6);       // row0=8-15
                m_temp_reg_35 = _mm_unpackhi_epi64(m_temp_reg_4, m_temp_reg_6);       // row1=8-15

                m_temp_reg_36 = _mm_unpacklo_epi64(m_temp_reg_96, m_temp_reg_94);     // row0=16-23
                m_temp_reg_37 = _mm_unpackhi_epi64(m_temp_reg_96, m_temp_reg_94);     // row1=16-23

                m_temp_reg_80 = _mm_unpacklo_epi64(m_temp_reg_1, m_temp_reg_3);      // row2 =0-7
                m_temp_reg_81 = _mm_unpackhi_epi64(m_temp_reg_1, m_temp_reg_3);      // row3 =0-7

                m_temp_reg_82 = _mm_unpacklo_epi64(m_temp_reg_93, m_temp_reg_91);    // row2=24-31
                m_temp_reg_83 = _mm_unpackhi_epi64(m_temp_reg_93, m_temp_reg_91);    // row3=24-31

                m_temp_reg_84 = _mm_unpacklo_epi64(m_temp_reg_5, m_temp_reg_7);      // row2=8-15
                m_temp_reg_85 = _mm_unpackhi_epi64(m_temp_reg_5, m_temp_reg_7);      // row3=8-15

                m_temp_reg_86 = _mm_unpacklo_epi64(m_temp_reg_97, m_temp_reg_95);    // row2=16-23
                m_temp_reg_87 = _mm_unpackhi_epi64(m_temp_reg_97, m_temp_reg_95);    // row3=16-23

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)pu1_pred);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_30, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_34, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_20);

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)(pu1_pred+16));
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_36, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_32, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_20);
                pu1_dst += out_stride;
                pu1_pred += pred_strd;


                m_temp_reg_20 = _mm_loadu_si128((__m128i *)pu1_pred);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_31, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_35, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_20);

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)(pu1_pred+16));
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_37, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_33, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_20);
                pu1_dst += out_stride;
                pu1_pred += pred_strd;

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)pu1_pred);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_80, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_84, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_20);

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)(pu1_pred+16));
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_86, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_82, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_20);
                pu1_dst += out_stride;
                pu1_pred += pred_strd;


                m_temp_reg_20 = _mm_loadu_si128((__m128i *)pu1_pred);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_81, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_85, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)pu1_dst, m_temp_reg_20);

                m_temp_reg_20 = _mm_loadu_si128((__m128i *)(pu1_pred+16));
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_20);
                m_temp_reg_40 = _mm_add_epi16(m_temp_reg_87, m_temp_reg_0);
                m_temp_reg_0 = _mm_srli_si128(m_temp_reg_20, 8);
                m_temp_reg_0 = _mm_cvtepu8_epi16(m_temp_reg_0);
                m_temp_reg_44 = _mm_add_epi16(m_temp_reg_83, m_temp_reg_0);
                m_temp_reg_20 = _mm_packus_epi16(m_temp_reg_40, m_temp_reg_44);

                _mm_storeu_si128((__m128i *)(pu1_dst+16), m_temp_reg_20);
                pu1_dst += out_stride;
                pu1_pred += pred_strd;

          }
            pi2_tmp +=4;
        }
}

