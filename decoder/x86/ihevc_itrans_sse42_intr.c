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
 *  ihevc_itrans.c
 *
 * @brief
 *  Contains function definitions for single stage  inverse transform
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_itrans_4x4_ttype1()
 *  - ihevc_itrans_4x4()
 *  - ihevc_itrans_8x8()
 *  - ihevc_itrans_16x16()
 *  - ihevc_itrans_32x32()
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
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  Inverse transform for 8x8 input
 * block
 *
 * @par Description:
 *  Performs single stage 8x8 inverse transform by utilizing  the symmetry of
 * transformation matrix and reducing number  of multiplications wherever
 * possible but keeping the  number of operations(addition,multiplication and
 * shift)  same
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[out] pi2_dst
 *  Output 8x8 block
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
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
#if ITRANS_8X8==X86INTR
void ihevc_itrans_8x8_sse42(WORD16 *pi2_src,
                      WORD16 *pi2_dst,
                      WORD32 src_strd,
                      WORD32 dst_strd,
                      WORD32 i4_shift,
                      WORD32 zero_cols)
{
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
    __m128i m_src1, m_src2, m_src3, m_src4;
    __m128i m_coeff1, m_coeff2, m_coeff3, m_coeff4;

    __m128i m_rdng_factor;
    __m128i m_count;

    {
        /* ee0 is present in the registers m_temp_reg_10 and m_temp_reg_11 */
        /* ee1 is present in the registers m_temp_reg_12 and m_temp_reg_13 */
        {
            m_src1 = _mm_loadu_si128((__m128i *) pi2_src);
            m_src2 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd << 2)));
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[0][0]);

            m_temp_reg_0 = _mm_cvtepi16_epi32 (m_src1);
            m_temp_reg_2 = _mm_cvtepi16_epi32 (m_src2);
            m_temp_reg_0 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
            m_temp_reg_2 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
            m_temp_reg_1 = _mm_srli_si128(m_src1, 8);
            m_temp_reg_3 = _mm_srli_si128(m_src2, 8);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
            m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);
            m_temp_reg_1 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
            m_temp_reg_3 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);

            /* Loading source and coeff for computing eo0 and eo1 in the next block */
            m_src1 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd << 1)));
            m_src2 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd * 6)));
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[1][0]);
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_even_8[2][0]);

            m_temp_reg_10 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_12 = _mm_sub_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_11 = _mm_add_epi32(m_temp_reg_1, m_temp_reg_3);
            m_temp_reg_13 = _mm_sub_epi32(m_temp_reg_1, m_temp_reg_3);
        }


        /* eo0 is present in the registers m_temp_reg_14 and m_temp_reg_15 */
        /* eo1 is present in the registers m_temp_reg_16 and m_temp_reg_17 */
        {
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_src1);
            m_temp_reg_2 = _mm_cvtepi16_epi32(m_src2);
            m_temp_reg_1 = _mm_srli_si128(m_src1, 8);
            m_temp_reg_3 = _mm_srli_si128(m_src2, 8);
            m_temp_reg_4 = _mm_mullo_epi32(m_temp_reg_0, m_coeff2);
            m_temp_reg_0 = _mm_mullo_epi32(m_temp_reg_0, m_coeff1);
            m_temp_reg_6 = _mm_mullo_epi32(m_temp_reg_2, m_coeff1);
            m_temp_reg_2 = _mm_mullo_epi32(m_temp_reg_2, m_coeff2);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
            m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);
            m_temp_reg_5 = _mm_mullo_epi32(m_temp_reg_1, m_coeff2);
            m_temp_reg_1 = _mm_mullo_epi32(m_temp_reg_1, m_coeff1);
            m_temp_reg_7 = _mm_mullo_epi32(m_temp_reg_3, m_coeff1);
            m_temp_reg_3 = _mm_mullo_epi32(m_temp_reg_3, m_coeff2);

            /* Loading source and coeff for computing o0, o1, o2 and o3 in the next block */
            m_src1 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd)));
            m_src2 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd * 3)));
            m_src3 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd * 5)));
            m_src4 = _mm_loadu_si128((__m128i *) (pi2_src + (src_strd * 7)));
            m_coeff1 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[0][0]);
            m_coeff2 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[1][0]);
            m_coeff3 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[2][0]);
            m_coeff4 = _mm_loadu_si128((__m128i *) &g_ai4_ihevc_trans_intr_odd_8[3][0]);

            m_temp_reg_14 = _mm_add_epi32(m_temp_reg_0, m_temp_reg_2);
            m_temp_reg_15 = _mm_add_epi32(m_temp_reg_1, m_temp_reg_3);
            m_temp_reg_16 = _mm_sub_epi32(m_temp_reg_4, m_temp_reg_6);
            m_temp_reg_17 = _mm_sub_epi32(m_temp_reg_5, m_temp_reg_7);
        }

        /* e */
        {
            /* e0 stored in m_temp_reg_40 and m_temp_reg_41 */
            m_temp_reg_40 = _mm_add_epi32(m_temp_reg_10, m_temp_reg_14);
            m_temp_reg_41 = _mm_add_epi32(m_temp_reg_11, m_temp_reg_15);

            /* e3 stored in m_temp_reg_46 and m_temp_reg_47 */
            m_temp_reg_46 = _mm_sub_epi32(m_temp_reg_10, m_temp_reg_14);
            m_temp_reg_47 = _mm_sub_epi32(m_temp_reg_11, m_temp_reg_15);

            /* e1 stored in m_temp_reg_42 and m_temp_reg_43 */
            m_temp_reg_42 = _mm_add_epi32(m_temp_reg_12, m_temp_reg_16);
            m_temp_reg_43 = _mm_add_epi32(m_temp_reg_13, m_temp_reg_17);

            /* e2 stored in m_temp_reg_44 and m_temp_reg_45 */
            m_temp_reg_44 = _mm_sub_epi32(m_temp_reg_12, m_temp_reg_16);
            m_temp_reg_45 = _mm_sub_epi32(m_temp_reg_13, m_temp_reg_17);
        }

        /* o */
        {
            m_temp_reg_0 = _mm_cvtepi16_epi32(m_src1);
            m_temp_reg_2 = _mm_cvtepi16_epi32(m_src2);
            m_temp_reg_1 = _mm_srli_si128(m_src1, 8);
            m_temp_reg_3 = _mm_srli_si128(m_src2, 8);
            m_temp_reg_1 = _mm_cvtepi16_epi32(m_temp_reg_1);
            m_temp_reg_3 = _mm_cvtepi16_epi32(m_temp_reg_3);
            m_temp_reg_4 = _mm_cvtepi16_epi32(m_src3);
            m_temp_reg_6 = _mm_cvtepi16_epi32(m_src4);
            m_temp_reg_5 = _mm_srli_si128(m_src3, 8);
            m_temp_reg_7 = _mm_srli_si128(m_src4, 8);
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
            /* It is stored in m_temp_reg_10 */
            /* Column 7 of destination computed here */
            /* It is stored in m_temp_reg_17 */
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
            /* It is stored in m_temp_reg_11 */
            /* Column 6 of destination computed here */
            /* It is stored in m_temp_reg_16 */
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
            /* It is stored in m_temp_reg_12 */
            /* Column 5 of destination computed here */
            /* It is stored in m_temp_reg_15 */
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
            /* It is stored in m_temp_reg_13 */
            /* Column 4 of destination computed here */
            /* It is stored in m_temp_reg_14 */
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
        /* and ultimately stored in registers m_temp_reg_0 to m_temp_reg_7 */
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

        /* Store */
        {
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_10);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_11);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_12);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_13);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_14);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_15);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_16);
            pi2_dst += dst_strd;
            _mm_storeu_si128((__m128i *)pi2_dst, m_temp_reg_17);
        }
    }
}
#endif
