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
*  ihevc_hbd_quant_iquant_ssd_x86_intr.c
*
* @brief
*  Contains function definitions for quantization, followed by Inverse
*  quantization to find transform domain SSD
*
* @author
*  100647
*
* @par List of Functions:
*   - ihevc_hbd_quant_iquant_ssd_flat_scale_mat()
*
* @remarks
*  May not exactly match with the C code. The transform domain SSD is done at 32
*  bit precision without clipping the inv. quant o/p to 16 bit (as opposed to C)
*
* TO DO : zero_row, zero_col opt. need to be added
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
#include "ihevc_quant_iquant_ssd.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"

#if MSVC
#include <intrin.h>
#endif

#include <immintrin.h>

/*****************************************************************************/
/* Globals                                                                   */
/*****************************************************************************/

/**
*******************************************************************************
*
* @brief
*  This function performs quantization(using flat scale matrix), followed by
*  inverse quantization to find transform domain SSD
*
* @par Description:
*  Performs quantization on coeffs
*
* @param[in] pi2_coeffs
*  4x4 Coeffs
*
* @param[in] pi2_quant_coeff
*  Scaling Matrix
*
* @param[out] pi2_dst
*  Output 4x4 coefficients
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
* @param[in] dst_strd
*  Output Stride
*
* @param[out] csbf
*  coded sub block flag
*
* @param[in] csbf_strd
*  coded sub block flag
*
* @param[out] zero_col
*  zero column flag
*
* @param[out] zero_row
*  zero column flag
*
* @returns  cbf
* coded block flag
*
* @remarks
*  None
*
*******************************************************************************
*/
WORD32 ihevc_hbd_quant_iquant_ssd_flat_scale_mat_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 i, j;
    WORD32 log2_size;
    WORD32 cbf = 0;

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = i4_bit_depth;
    WORD32 q_bits, transform_shift, temp;
    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    WORD32 shift_iq;
    WORD32 shift_in_iquant;
    WORD32 shift_select;

    /* Initialize cost to zero */
    WORD32 ssd_cost = 0;

    /* scale_q can be opt. like scale_iq */
    __m128i m_scale_q   = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_scale_iq  = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));
    __m128i m_zero      = _mm_set1_epi32(0);
    __m128i m_one       = _mm_set1_epi32(1);
    __m128i m_ssd_acc   = _mm_set1_epi32(0);

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_add_q;
    __m128i m_add_iq = _mm_set1_epi32(1);

    /* Quant initialization */
    GETRANGE(log2_size, trans_size);
    log2_size -= 1;
    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_add_q = _mm_set1_epi32(temp);

    /* IQuant initialization */
    shift_iq = bit_depth + log2_size - 5;
    /* Values of certain variables change wrt this condition */
    if(shift_iq > qp_div)
    {
        WORD32  shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        m_add_iq = _mm_slli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        WORD32 shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        m_add_iq = _mm_srli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */
            /*****************************************************************/
            /*********************       Quantization    *********************/
            /*****************************************************************/

            /*  Convert coeff from 16 to 32 bits    */
            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);
            m_src_temp1 = _mm_cvtepi16_epi32(m_src_temp1);
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);
            m_src_temp3 = _mm_cvtepi16_epi32(m_src_temp3);

            m_src_temp4 = _mm_abs_epi32(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi32(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi32(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi32(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi32(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi32(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi32(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi32(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi32(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi32(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi32(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi32(m_src_temp11, 1);

            m_sign0 = _mm_add_epi32(m_src_temp8,  m_one);   /*  sign(row0)  */
            m_sign1 = _mm_add_epi32(m_src_temp9 , m_one);   /*  sign(row1)  */
            m_sign2 = _mm_add_epi32(m_src_temp10, m_one);   /*  sign(row2)  */
            m_sign3 = _mm_add_epi32(m_src_temp11, m_one);   /*  sign(row3)  */

            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/
            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale_q);  /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/
            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_q);          /*  col =0, 1, 2, 3 */

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/
            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);          /*  col =0, 1, 2, 3 */

            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/
            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);         /* col= [0][1][2][3]*/
            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/
            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2);         /* col= [0][1][2][3]*/
            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);         /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_sign0 = _mm_cmpeq_epi32 (m_zero, m_src_temp8);
            m_sign1 = _mm_cmpeq_epi32 (m_zero, m_src_temp9);

            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            /********************* Inverse Quantization  *********************/
            m_src_temp12 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((0)*trans_size)));     /* deq row =0*/
            m_src_temp12 = _mm_mullo_epi16(m_src_temp12, m_scale_iq);
            m_src_temp13 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((1)*trans_size)));     /* deq row =1*/
            m_src_temp13 = _mm_mullo_epi16(m_src_temp13, m_scale_iq);
            m_src_temp14 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((2)*trans_size)));     /* deq row =2*/
            m_src_temp14 = _mm_mullo_epi16(m_src_temp14, m_scale_iq);
            m_src_temp15 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((3)*trans_size)));     /* deq row= 3*/
            m_src_temp15 = _mm_mullo_epi16(m_src_temp15, m_scale_iq);
            /********************* Inverse Quantization  *********************/

            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((0)*dst_q_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((1)*dst_q_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((2)*dst_q_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((3)*dst_q_strd)),m_src_temp11);

            if(!(_mm_test_all_ones (m_sign0))||!(_mm_test_all_ones (m_sign1)))
            {
                *(csbf + block_col) = 1;
            }

            if(*(csbf + block_col) == 1)
            {
                /* zero_col update *//* temp_zero_col = ~zero_col */
                temp_zero_col = (temp_zero_col)
                    | (0xF << block_col * 4);
                // zero col can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 colums of 4x4 block
                // even if any 4x4 csbf is set

                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

            block_col++;
            /*****************************************************************/
            /******************** End of Quantization    *********************/
            /*****************************************************************/
            /*****************************************************************/
            /********************* Inverse Quantization  *********************/
            /*****************************************************************/
            m_src_temp12 = _mm_cvtepi16_epi32(m_src_temp12);
            m_src_temp13 = _mm_cvtepi16_epi32(m_src_temp13);
            m_src_temp14 = _mm_cvtepi16_epi32(m_src_temp14);
            m_src_temp15 = _mm_cvtepi16_epi32(m_src_temp15);

            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp8);
            m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp10);
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp9);
            m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp11);

            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_src_temp12);
            m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_src_temp13);
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_src_temp14);
            m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_src_temp15);

            m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_iq);
            m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_iq);
            m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_iq);
            m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_iq);

            if(shift_select)
            {
                m_src_temp4 = _mm_srai_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_srai_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_srai_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_srai_epi32(m_src_temp7, shift_in_iquant);
            }
            else
            {
                m_src_temp4 = _mm_slli_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_slli_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_slli_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_slli_epi32(m_src_temp7, shift_in_iquant);
            }

            /*****************************************************************/
            /******************* End of Inverse Quantization *****************/
            /*****************************************************************/

            /*****************************************************************/
            /***************** SSD Computation & Accumulation ****************/
            /*****************************************************************/
            /* trans_coeff - inv.quant */
            m_src_temp0 = _mm_sub_epi32(m_src_temp0, m_src_temp4);
            m_src_temp1 = _mm_sub_epi32(m_src_temp1, m_src_temp5);
            m_src_temp2 = _mm_sub_epi32(m_src_temp2, m_src_temp6);
            m_src_temp3 = _mm_sub_epi32(m_src_temp3, m_src_temp7);
            /* SD */
            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_src_temp0);
            m_src_temp1 = _mm_mullo_epi32(m_src_temp1, m_src_temp1);
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_src_temp2);
            m_src_temp3 = _mm_mullo_epi32(m_src_temp3, m_src_temp3);

            /********************* Inverse Quantization  *********************/
            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */
            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((0)*dst_iq_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((1)*dst_iq_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((2)*dst_iq_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((3)*dst_iq_strd)),m_src_temp11);
            /********************* Inverse Quantization  *********************/

            /* SSD */
            m_src_temp0 = _mm_add_epi32(m_src_temp0, m_src_temp1);
            m_src_temp2 = _mm_add_epi32(m_src_temp2, m_src_temp3);

            m_src_temp0 = _mm_add_epi32(m_src_temp0, m_src_temp2);
            /* a3+a2 a1+a0 a3+a2 a1+a0 */
            m_src_temp0 = _mm_hadd_epi32(m_src_temp0, m_src_temp0);
            /* a3+a2+a1+a0 a3+a2+a1+a0 */
            m_src_temp0 = _mm_hadd_epi32(m_src_temp0, m_src_temp0);

            /* SSD Accumulation */
            m_ssd_acc = _mm_add_epi32 (m_ssd_acc, m_src_temp0);
        }

        block_col    = 0;
        block_row   += 4;
        csbf        += csbf_strd;

        pi2_coeffs  += 4*src_strd;
        pi2_q_dst   += 4*dst_q_strd;
        pi2_iq_dst  += 4*dst_iq_strd;
        pi2_quant_coeff     += 4*trans_size;
        pi2_dequant_coeff   += 4*trans_size;
    }

    ssd_cost = _mm_cvtsi128_si32(m_ssd_acc);

    *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing

    /* Store the cost */
    *pi8_cost = ssd_cost;

    return cbf;
}

WORD32 ihevc_hbd_quant_iquant_flat_scale_mat_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 i, j;
    WORD32 log2_size;
    WORD32 cbf = 0;

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = i4_bit_depth;
    WORD32 q_bits, transform_shift, temp;
    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    WORD32 shift_iq;
    WORD32 shift_in_iquant;
    WORD32 shift_select;

    /* scale_q can be opt. like scale_iq */
    __m128i m_scale_q   = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_scale_iq  = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));
    __m128i m_zero      = _mm_set1_epi32(0);
    __m128i m_one       = _mm_set1_epi32(1);

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_add_q;
    __m128i m_add_iq = _mm_set1_epi32(1);

    GETRANGE(log2_size, trans_size);
    log2_size -= 1;
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));
    m_add_q = _mm_set1_epi32(temp);

    shift_iq = bit_depth + log2_size - 5;

    if(shift_iq > qp_div)
    {
        WORD32  shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        m_add_iq = _mm_slli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        WORD32 shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        m_add_iq = _mm_srli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */
            /*****************************************************************/
            /*********************       Quantization    *********************/
            /*****************************************************************/

            /*  Convert coeff from 16 to 32 bits    */
            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);
            m_src_temp1 = _mm_cvtepi16_epi32(m_src_temp1);
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);
            m_src_temp3 = _mm_cvtepi16_epi32(m_src_temp3);

            m_src_temp4 = _mm_abs_epi32(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi32(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi32(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi32(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi32(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi32(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi32(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi32(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi32(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi32(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi32(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi32(m_src_temp11, 1);

            m_sign0 = _mm_add_epi32(m_src_temp8,  m_one);   /*  sign(row0)  */
            m_sign1 = _mm_add_epi32(m_src_temp9 , m_one);   /*  sign(row1)  */
            m_sign2 = _mm_add_epi32(m_src_temp10, m_one);   /*  sign(row2)  */
            m_sign3 = _mm_add_epi32(m_src_temp11, m_one);   /*  sign(row3)  */

            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/
            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale_q);  /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/
            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_q);          /*  col =0, 1, 2, 3 */

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/
            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);          /*  col =0, 1, 2, 3 */

            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/
            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);         /* col= [0][1][2][3]*/
            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/
            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2);         /* col= [0][1][2][3]*/
            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);         /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_sign0 = _mm_cmpeq_epi32 (m_zero, m_src_temp8);
            m_sign1 = _mm_cmpeq_epi32 (m_zero, m_src_temp9);

            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            /********************* Inverse Quantization  *********************/
            m_src_temp12 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((0)*trans_size)));     /* deq row =0*/
            m_src_temp12 = _mm_mullo_epi16(m_src_temp12, m_scale_iq);
            m_src_temp13 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((1)*trans_size)));     /* deq row =1*/
            m_src_temp13 = _mm_mullo_epi16(m_src_temp13, m_scale_iq);
            m_src_temp14 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((2)*trans_size)));     /* deq row =2*/
            m_src_temp14 = _mm_mullo_epi16(m_src_temp14, m_scale_iq);
            m_src_temp15 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((3)*trans_size)));     /* deq row= 3*/
            m_src_temp15 = _mm_mullo_epi16(m_src_temp15, m_scale_iq);
            /********************* Inverse Quantization  *********************/

            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((0)*dst_q_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((1)*dst_q_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((2)*dst_q_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((3)*dst_q_strd)),m_src_temp11);

            if(!(_mm_test_all_ones (m_sign0))||!(_mm_test_all_ones (m_sign1)))
            {
                *(csbf + block_col) = 1;
            }

            if(*(csbf + block_col) == 1)
            {
                /* zero_col update *//* temp_zero_col = ~zero_col */
                temp_zero_col = (temp_zero_col)
                    | (0xF << block_col * 4);
                // zero col can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 colums of 4x4 block
                // even if any 4x4 csbf is set

                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

            block_col++;
            /*****************************************************************/
            /******************** End of Quantization    *********************/
            /*****************************************************************/
            /*****************************************************************/
            /********************* Inverse Quantization  *********************/
            /*****************************************************************/
            m_src_temp12 = _mm_cvtepi16_epi32(m_src_temp12);
            m_src_temp13 = _mm_cvtepi16_epi32(m_src_temp13);
            m_src_temp14 = _mm_cvtepi16_epi32(m_src_temp14);
            m_src_temp15 = _mm_cvtepi16_epi32(m_src_temp15);

            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp8);
            m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp10);
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp9);
            m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp11);

            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_src_temp12);
            m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_src_temp13);
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_src_temp14);
            m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_src_temp15);

            m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_iq);
            m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_iq);
            m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_iq);
            m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_iq);

            if(shift_select)
            {
                m_src_temp4 = _mm_srai_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_srai_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_srai_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_srai_epi32(m_src_temp7, shift_in_iquant);
            }
            else
            {
                m_src_temp4 = _mm_slli_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_slli_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_slli_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_slli_epi32(m_src_temp7, shift_in_iquant);
            }

            /*****************************************************************/
            /******************* End of Inverse Quantization *****************/
            /*****************************************************************/

            /********************* Inverse Quantization  *********************/
            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */
            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((0)*dst_iq_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((1)*dst_iq_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((2)*dst_iq_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((3)*dst_iq_strd)),m_src_temp11);
            /********************* Inverse Quantization  *********************/
        }

        block_col    = 0;
        block_row   += 4;
        csbf        += csbf_strd;

        pi2_coeffs  += 4*src_strd;
        pi2_q_dst   += 4*dst_q_strd;
        pi2_iq_dst  += 4*dst_iq_strd;
        pi2_quant_coeff     += 4*trans_size;
        pi2_dequant_coeff   += 4*trans_size;
    }

    *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing

    return cbf;
}

WORD32 ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 i, j;
    WORD32 log2_size;
    WORD32 cbf = 0;

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = i4_bit_depth;
    WORD32 q_bits, transform_shift, temp;
    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    WORD32 shift_iq;
    WORD32 shift_in_iquant;
    WORD32 shift_select;

    /* Initialize cost to zero */
    WORD32 ssd_cost = 0;

    /* scale_q can be opt. like scale_iq */
    __m128i m_scale_q   = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_scale_iq  = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));
    __m128i m_zero      = _mm_set1_epi32(0);
    __m128i m_one       = _mm_set1_epi32(1);
    //  __m128i m_one_alt   = _mm_set1_epi8(1);
    __m128i m_ssd_acc   = _mm_set1_epi32(0);

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13,m_src_temp14,m_src_temp15;
    __m128i m_src_temp16, m_src_temp17,m_src_temp18,m_src_temp19;
    __m128i m_src_temp20;//m_src_temp21,m_src_temp22;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_add_q,m_add_alt_q;
    __m128i m_add_iq = _mm_set1_epi32(1);

    /* Quant initialization */
    GETRANGE(log2_size, trans_size);
    log2_size -= 1;
    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_add_q      = _mm_set1_epi32(temp);
    m_src_temp20 = _mm_set1_epi32(1<<(q_bits+1));

    temp = ((1 << QUANT_ROUND_FACTOR_Q)>>1);
    temp = ((temp) << (q_bits - QUANT_ROUND_FACTOR_Q));
    m_add_alt_q  = _mm_set1_epi32(temp);

    /* IQuant initialization */
    shift_iq = bit_depth + log2_size - 5;
    /* Values of certain variables change wrt this condition */
    if(shift_iq > qp_div)
    {
        WORD32  shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        m_add_iq = _mm_slli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        WORD32 shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        m_add_iq = _mm_srli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */
            /*****************************************************************/
            /*********************       Quantization    *********************/
            /*****************************************************************/

            /*  Convert coeff from 16 to 32 bits    */
            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);
            m_src_temp1 = _mm_cvtepi16_epi32(m_src_temp1);
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);
            m_src_temp3 = _mm_cvtepi16_epi32(m_src_temp3);

            m_src_temp4 = _mm_abs_epi32(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi32(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi32(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi32(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi32(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi32(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi32(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi32(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi32(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi32(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi32(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi32(m_src_temp11, 1);

            m_sign0 = _mm_add_epi32(m_src_temp8,  m_one);   /*  sign(row0)  */
            m_sign1 = _mm_add_epi32(m_src_temp9 , m_one);   /*  sign(row1)  */
            m_sign2 = _mm_add_epi32(m_src_temp10, m_one);   /*  sign(row2)  */
            m_sign3 = _mm_add_epi32(m_src_temp11, m_one);   /*  sign(row3)  */


            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/
            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale_q);  /*  col =0, 1, 2, 3 */


            /**************************************************************************************/
            /* Here we apply half rounding for all coeffs(x) whose quant coeff(z) > 1, i.e.       */
            /* if the two rounding factors are a and b, and quant factor is y, then we first      */
            /* calculate z = (x+a)/y. If (z>1), then z = (x+b)/y.                                 */
            /* This is equivalent to seeing if z = (x+a) > y ? {(x+a)/y : (x+b)/y}                */
            /**************************************************************************************/

            /*row =0*/ m_src_temp12 = _mm_add_epi32(m_src_temp4, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =1*/ m_src_temp13 = _mm_add_epi32(m_src_temp5, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =2*/ m_src_temp14 = _mm_add_epi32(m_src_temp6, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =3*/ m_src_temp15 = _mm_add_epi32(m_src_temp7, m_add_alt_q);      /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/
            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_q);          /*  col =0, 1, 2, 3 */

            m_src_temp16 = _mm_cmpgt_epi32(m_src_temp4,m_src_temp20);
            m_src_temp17 = _mm_cmpgt_epi32(m_src_temp5,m_src_temp20);
            m_src_temp18 = _mm_cmpgt_epi32(m_src_temp6,m_src_temp20);
            m_src_temp19 = _mm_cmpgt_epi32(m_src_temp7,m_src_temp20);

            m_src_temp4  = _mm_blendv_epi8(m_src_temp4,m_src_temp12,m_src_temp16);
            m_src_temp5  = _mm_blendv_epi8(m_src_temp5,m_src_temp13,m_src_temp17);
            m_src_temp6  = _mm_blendv_epi8(m_src_temp6,m_src_temp14,m_src_temp18);
            m_src_temp7  = _mm_blendv_epi8(m_src_temp7,m_src_temp15,m_src_temp19);

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/
            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);          /*  col =0, 1, 2, 3 */

            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/
            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);         /* col= [0][1][2][3]*/
            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/
            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2);         /* col= [0][1][2][3]*/
            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);         /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_sign0 = _mm_cmpeq_epi32 (m_zero, m_src_temp8);
            m_sign1 = _mm_cmpeq_epi32 (m_zero, m_src_temp9);

            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            /********************* Inverse Quantization  *********************/
            m_src_temp12 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((0)*trans_size)));     /* deq row =0*/
            m_src_temp12 = _mm_mullo_epi16(m_src_temp12, m_scale_iq);
            m_src_temp13 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((1)*trans_size)));     /* deq row =1*/
            m_src_temp13 = _mm_mullo_epi16(m_src_temp13, m_scale_iq);
            m_src_temp14 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((2)*trans_size)));     /* deq row =2*/
            m_src_temp14 = _mm_mullo_epi16(m_src_temp14, m_scale_iq);
            m_src_temp15 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((3)*trans_size)));     /* deq row= 3*/
            m_src_temp15 = _mm_mullo_epi16(m_src_temp15, m_scale_iq);
            /********************* Inverse Quantization  *********************/

            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((0)*dst_q_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((1)*dst_q_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((2)*dst_q_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((3)*dst_q_strd)),m_src_temp11);

            if(!(_mm_test_all_ones (m_sign0))||!(_mm_test_all_ones (m_sign1)))
            {
                *(csbf + block_col) = 1;
            }

            if(*(csbf + block_col) == 1)
            {
                /* zero_col update *//* temp_zero_col = ~zero_col */
                temp_zero_col = (temp_zero_col)
                    | (0xF << block_col * 4);
                // zero col can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 colums of 4x4 block
                // even if any 4x4 csbf is set

                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

            block_col++;
            /*****************************************************************/
            /******************** End of Quantization    *********************/
            /*****************************************************************/
            /*****************************************************************/
            /********************* Inverse Quantization  *********************/
            /*****************************************************************/
            m_src_temp12 = _mm_cvtepi16_epi32(m_src_temp12);
            m_src_temp13 = _mm_cvtepi16_epi32(m_src_temp13);
            m_src_temp14 = _mm_cvtepi16_epi32(m_src_temp14);
            m_src_temp15 = _mm_cvtepi16_epi32(m_src_temp15);

            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp8);
            m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp10);
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp9);
            m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp11);

            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_src_temp12);
            m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_src_temp13);
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_src_temp14);
            m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_src_temp15);

            m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_iq);
            m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_iq);
            m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_iq);
            m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_iq);

            if(shift_select)
            {
                m_src_temp4 = _mm_srai_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_srai_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_srai_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_srai_epi32(m_src_temp7, shift_in_iquant);
            }
            else
            {
                m_src_temp4 = _mm_slli_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_slli_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_slli_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_slli_epi32(m_src_temp7, shift_in_iquant);
            }

            /*****************************************************************/
            /******************* End of Inverse Quantization *****************/
            /*****************************************************************/

            /*****************************************************************/
            /***************** SSD Computation & Accumulation ****************/
            /*****************************************************************/
            /* trans_coeff - inv.quant */
            m_src_temp0 = _mm_sub_epi32(m_src_temp0, m_src_temp4);
            m_src_temp1 = _mm_sub_epi32(m_src_temp1, m_src_temp5);
            m_src_temp2 = _mm_sub_epi32(m_src_temp2, m_src_temp6);
            m_src_temp3 = _mm_sub_epi32(m_src_temp3, m_src_temp7);
            /* SD */
            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_src_temp0);
            m_src_temp1 = _mm_mullo_epi32(m_src_temp1, m_src_temp1);
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_src_temp2);
            m_src_temp3 = _mm_mullo_epi32(m_src_temp3, m_src_temp3);

            /********************* Inverse Quantization  *********************/
            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */
            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((0)*dst_iq_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((1)*dst_iq_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((2)*dst_iq_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((3)*dst_iq_strd)),m_src_temp11);
            /********************* Inverse Quantization  *********************/

            /* SSD */
            if ((i == 16) && (j == 8))
                j = 8;
            m_src_temp0 = _mm_add_epi32(m_src_temp0, m_src_temp1);
            m_src_temp2 = _mm_add_epi32(m_src_temp2, m_src_temp3);

            m_src_temp0 = _mm_add_epi32(m_src_temp0, m_src_temp2);
            /* a3+a2 a1+a0 a3+a2 a1+a0 */
            m_src_temp0 = _mm_hadd_epi32(m_src_temp0, m_src_temp0);
            /* a3+a2+a1+a0 a3+a2+a1+a0 */
            m_src_temp0 = _mm_hadd_epi32(m_src_temp0, m_src_temp0);

            /* SSD Accumulation */
            m_ssd_acc = _mm_add_epi32 (m_ssd_acc, m_src_temp0);
        }

        block_col    = 0;
        block_row   += 4;
        csbf        += csbf_strd;

        pi2_coeffs  += 4*src_strd;
        pi2_q_dst   += 4*dst_q_strd;
        pi2_iq_dst  += 4*dst_iq_strd;
        pi2_quant_coeff     += 4*trans_size;
        pi2_dequant_coeff   += 4*trans_size;
    }

    ssd_cost = _mm_cvtsi128_si32(m_ssd_acc);

    *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing

    /* Store the cost */
    *pi8_cost = ssd_cost;

    return cbf;
}

WORD32 ihevc_hbd_quant_iquant_flat_scale_mat_rdoq_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 i, j;
    WORD32 log2_size;
    WORD32 cbf = 0;

     /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 quant_multiplier = 4 ;
    WORD16 bit_depth = i4_bit_depth;
    WORD32 q_bits, transform_shift, temp;
    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    WORD32 shift_iq;
    WORD32 shift_in_iquant;
    WORD32 shift_select;

    __m128i m_scale_q   = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_scale_iq  = _mm_loadu_si128((__m128i *) &(g_ihevc_iquant_intr_scales[qp_rem][0]));
    __m128i m_zero      = _mm_set1_epi32(0);
    __m128i m_one       = _mm_set1_epi32(1);

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13,m_src_temp14,m_src_temp15;
    __m128i m_src_temp16, m_src_temp17,m_src_temp18,m_src_temp19;
    __m128i m_src_temp20;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_add_q,m_add_alt_q;
    __m128i m_add_iq = _mm_set1_epi32(1);

    GETRANGE(log2_size, trans_size);
    log2_size -= 1;
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));
    m_add_q      = _mm_set1_epi32(temp);
    m_src_temp20 = _mm_set1_epi32(1<<(q_bits+1));
    temp = ((1 << QUANT_ROUND_FACTOR_Q)>>1);
    temp = ((temp) << (q_bits - QUANT_ROUND_FACTOR_Q));
    m_add_alt_q  = _mm_set1_epi32(temp);

    shift_iq = bit_depth + log2_size - 5;

    if(shift_iq > qp_div)
    {
        WORD32  shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        m_add_iq = _mm_slli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        WORD32 shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        m_add_iq = _mm_srli_epi32(m_add_iq, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadl_epi64((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */
            /*****************************************************************/
            /*********************       Quantization    *********************/
            /*****************************************************************/

            /*  Convert coeff from 16 to 32 bits    */
            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);
            m_src_temp1 = _mm_cvtepi16_epi32(m_src_temp1);
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);
            m_src_temp3 = _mm_cvtepi16_epi32(m_src_temp3);

            m_src_temp4 = _mm_abs_epi32(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi32(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi32(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi32(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi32(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi32(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi32(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi32(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi32(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi32(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi32(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi32(m_src_temp11, 1);

            m_sign0 = _mm_add_epi32(m_src_temp8,  m_one);   /*  sign(row0)  */
            m_sign1 = _mm_add_epi32(m_src_temp9 , m_one);   /*  sign(row1)  */
            m_sign2 = _mm_add_epi32(m_src_temp10, m_one);   /*  sign(row2)  */
            m_sign3 = _mm_add_epi32(m_src_temp11, m_one);   /*  sign(row3)  */


            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/
            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale_q);  /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale_q);  /*  col =0, 1, 2, 3 */


            /**************************************************************************************/
            /* Here we apply half rounding for all coeffs(x) whose quant coeff(z) > 1, i.e.       */
            /* if the two rounding factors are a and b, and quant factor is y, then we first      */
            /* calculate z = (x+a)/y. If (z>1), then z = (x+b)/y.                                 */
            /* This is equivalent to seeing if z = (x+a) > y ? {(x+a)/y : (x+b)/y}                */
            /**************************************************************************************/

            /*row =0*/ m_src_temp12 = _mm_add_epi32(m_src_temp4, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =1*/ m_src_temp13 = _mm_add_epi32(m_src_temp5, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =2*/ m_src_temp14 = _mm_add_epi32(m_src_temp6, m_add_alt_q);      /*  col =0, 1, 2, 3 */
            /*row =3*/ m_src_temp15 = _mm_add_epi32(m_src_temp7, m_add_alt_q);      /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/
            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_q);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_q);          /*  col =0, 1, 2, 3 */

            m_src_temp16 = _mm_cmpgt_epi32(m_src_temp4,m_src_temp20);
            m_src_temp17 = _mm_cmpgt_epi32(m_src_temp5,m_src_temp20);
            m_src_temp18 = _mm_cmpgt_epi32(m_src_temp6,m_src_temp20);
            m_src_temp19 = _mm_cmpgt_epi32(m_src_temp7,m_src_temp20);

            m_src_temp4  = _mm_blendv_epi8(m_src_temp4,m_src_temp12,m_src_temp16);
            m_src_temp5  = _mm_blendv_epi8(m_src_temp5,m_src_temp13,m_src_temp17);
            m_src_temp6  = _mm_blendv_epi8(m_src_temp6,m_src_temp14,m_src_temp18);
            m_src_temp7  = _mm_blendv_epi8(m_src_temp7,m_src_temp15,m_src_temp19);

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/
            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);          /*  col =0, 1, 2, 3 */
            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);          /*  col =0, 1, 2, 3 */

            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/
            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);         /* col= [0][1][2][3]*/
            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/
            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2);         /* col= [0][1][2][3]*/
            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);         /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_sign0 = _mm_cmpeq_epi32 (m_zero, m_src_temp8);
            m_sign1 = _mm_cmpeq_epi32 (m_zero, m_src_temp9);

            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            /********************* Inverse Quantization  *********************/
            m_src_temp12 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((0)*trans_size)));     /* deq row =0*/
            m_src_temp12 = _mm_mullo_epi16(m_src_temp12, m_scale_iq);
            m_src_temp13 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((1)*trans_size)));     /* deq row =1*/
            m_src_temp13 = _mm_mullo_epi16(m_src_temp13, m_scale_iq);
            m_src_temp14 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((2)*trans_size)));     /* deq row =2*/
            m_src_temp14 = _mm_mullo_epi16(m_src_temp14, m_scale_iq);
            m_src_temp15 = _mm_loadl_epi64((__m128i *) (pi2_dequant_coeff+j+((3)*trans_size)));     /* deq row= 3*/
            m_src_temp15 = _mm_mullo_epi16(m_src_temp15, m_scale_iq);
            /********************* Inverse Quantization  *********************/

            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((0)*dst_q_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((1)*dst_q_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((2)*dst_q_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_q_dst+j+((3)*dst_q_strd)),m_src_temp11);

            if(!(_mm_test_all_ones (m_sign0))||!(_mm_test_all_ones (m_sign1)))
            {
                *(csbf + block_col) = 1;
            }

            if(*(csbf + block_col) == 1)
            {
                /* zero_col update *//* temp_zero_col = ~zero_col */
                temp_zero_col = (temp_zero_col)
                    | (0xF << block_col * 4);
                // zero col can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 colums of 4x4 block
                // even if any 4x4 csbf is set

                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

            block_col++;
            /*****************************************************************/
            /******************** End of Quantization    *********************/
            /*****************************************************************/
            /*****************************************************************/
            /********************* Inverse Quantization  *********************/
            /*****************************************************************/
            m_src_temp12 = _mm_cvtepi16_epi32(m_src_temp12);
            m_src_temp13 = _mm_cvtepi16_epi32(m_src_temp13);
            m_src_temp14 = _mm_cvtepi16_epi32(m_src_temp14);
            m_src_temp15 = _mm_cvtepi16_epi32(m_src_temp15);

            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp8);
            m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp10);
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp9);
            m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp11);

            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_src_temp12);
            m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_src_temp13);
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_src_temp14);
            m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_src_temp15);

            m_src_temp4 = _mm_add_epi32(m_src_temp4, m_add_iq);
            m_src_temp5 = _mm_add_epi32(m_src_temp5, m_add_iq);
            m_src_temp6 = _mm_add_epi32(m_src_temp6, m_add_iq);
            m_src_temp7 = _mm_add_epi32(m_src_temp7, m_add_iq);

            if(shift_select)
            {
                m_src_temp4 = _mm_srai_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_srai_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_srai_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_srai_epi32(m_src_temp7, shift_in_iquant);
            }
            else
            {
                m_src_temp4 = _mm_slli_epi32(m_src_temp4, shift_in_iquant);
                m_src_temp5 = _mm_slli_epi32(m_src_temp5, shift_in_iquant);
                m_src_temp6 = _mm_slli_epi32(m_src_temp6, shift_in_iquant);
                m_src_temp7 = _mm_slli_epi32(m_src_temp7, shift_in_iquant);
            }

            /*****************************************************************/
            /******************* End of Inverse Quantization *****************/
            /*****************************************************************/

            /********************* Inverse Quantization  *********************/
            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/
            m_src_temp8 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp9 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */
            m_src_temp10 = _mm_srli_si128(m_src_temp8, 8);  /*  row =1 [0][1][2][3] */
            m_src_temp11 = _mm_srli_si128(m_src_temp9, 8);  /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((0)*dst_iq_strd)),m_src_temp8);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((1)*dst_iq_strd)),m_src_temp10);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((2)*dst_iq_strd)),m_src_temp9);
            _mm_storel_epi64 ((__m128i *)(pi2_iq_dst+j+((3)*dst_iq_strd)),m_src_temp11);
            /********************* Inverse Quantization  *********************/
            if ((i == 16) && (j == 8))
                j = 8;
        }

        block_col    = 0;
        block_row   += 4;
        csbf        += csbf_strd;

        pi2_coeffs  += 4*src_strd;
        pi2_q_dst   += 4*dst_q_strd;
        pi2_iq_dst  += 4*dst_iq_strd;
        pi2_quant_coeff     += 4*trans_size;
        pi2_dequant_coeff   += 4*trans_size;
    }

    *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing

    return cbf;
}

/**
*******************************************************************************
*
* @brief
*  This function performs quantization(using flat scale matrix), followed by
*  inverse quantization to find transform domain SSD; when we perform RDOQ.
*  In case the quantized value turns out to be grater than 1, we then requantize
*  use half rounding.
*
* @par Description:
*  Performs quantization on coeffs
*
* @param[in] pi2_coeffs
*  4x4 Coeffs
*
* @param[in] pi2_quant_coeff
*  Scaling Matrix
*
* @param[out] pi2_dst
*  Output 4x4 coefficients
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
* @param[in] dst_strd
*  Output Stride
*
* @param[out] csbf
*  coded sub block flag
*
* @param[in] csbf_strd
*  coded sub block flag
*
* @param[out] zero_col
*  zero column flag
*
* @param[out] zero_row
*  zero column flag
*
* @returns  cbf
* coded block flag
*
* @remarks
*  None
*
*******************************************************************************
*/

WORD32 ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 q_bits, log2_size, quant_multiplier;
    WORD32 shift_in_iquant_minus_1;
    WORD32 shift_iq, transform_shift;
    WORD32 shift_in_iquant, shift_select, row, col;
    WORD32 cbf = 0;
    WORD16 *pi2_coeffs_temp;
    WORD16 *pi2_q_dst_temp;
    WORD16 *pi2_iq_dst_temp;
    WORD32 *pi4_quant_round_factor_0_1_temp;
    WORD32 *pi4_quant_round_factor_1_2_temp;
    WORD32 block_row = 0;
    WORD32 block_col = 0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    __m128i zero_32x4b, quant_coeff, qadd1_32x4b, two_16x8b;
    __m128i one_16x8b, iquant_coeff, add_iq_32x4b, ssd_acc;
    __m128i sign1_16x8b, sign2_16x8b;
    __m128i src1_16x8b, src2_16x8b;
    __m128i src01_16x8b, src02_16x8b, src03_16x8b, src04_16x8b;
    __m128i temp1_16x8b, temp2_16x8b, temp3_16x8b, temp4_16x8b;
    __m128i temp5_16x8b, temp6_16x8b;
    __m128i res01_32x4b, res02_32x4b, res03_32x4b,res04_32x4b;
    __m128i res11_32x4b, res12_32x4b, res13_32x4b,res14_32x4b;
    __m128i res21_32x4b, res22_32x4b, res23_32x4b,res24_32x4b;
    __m128i res31_32x4b, res32_32x4b, res33_32x4b,res34_32x4b;
    __m128i des01_16x8b, des02_16x8b, des11_16x8b, des12_16x8b;
    __m128i des21_16x8b, des22_16x8b, des31_16x8b, des32_16x8b;
    __m128i dst0_16x8b,dst1_16x8b;
    __m128i qadd20_32x4b, qadd21_32x4b, qadd22_32x4b, qadd23_32x4b;
    __m128i qadd30_32x4b, qadd31_32x4b, qadd32_32x4b, qadd33_32x4b;
    __m128i mul1_32x4b, mul2_32x4b, mul3_32x4b, mul4_32x4b;

    /*Assumption*/
    //ASSERT (0 == g_ihevc_iquant_scales[qp_rem] % 16);

    /*Initialisation*/
    pi2_coeffs_temp = pi2_coeffs;
    pi2_q_dst_temp = pi2_q_dst;
    pi2_iq_dst_temp = pi2_iq_dst;
    pi4_quant_round_factor_0_1_temp = pi4_quant_round_factor_0_1;
    pi4_quant_round_factor_1_2_temp = pi4_quant_round_factor_1_2;

    /* Quant initialization */
    GETRANGE(log2_size, trans_size);
    log2_size -= 1;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - i4_bit_depth - log2_size;
    quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;

    /* IQuant initialization */
    shift_iq = i4_bit_depth + log2_size - 5;

    zero_32x4b = _mm_set1_epi32 (0x0);
    quant_coeff = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    qadd1_32x4b = _mm_set1_epi32((1 << QUANT_ROUND_FACTOR_Q)/2);
    qadd1_32x4b = _mm_slli_epi32 (qadd1_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
    two_16x8b = _mm_set1_epi16(0x2);
    one_16x8b = _mm_set1_epi16(0x1);

    iquant_coeff = _mm_set1_epi32(g_ihevc_iquant_scales[qp_rem]<<4);
    add_iq_32x4b = _mm_set1_epi32(0x1);
    ssd_acc = _mm_set1_epi32(0x0);

    /* Values of certain variables change wrt this condition */
    if((shift_iq - qp_div - 1)>=0)
    {
        shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        add_iq_32x4b = _mm_slli_epi32(add_iq_32x4b, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        add_iq_32x4b = _mm_srai_epi32(add_iq_32x4b, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    if(trans_size > 7)
    {
        for(row = 0; row < trans_size; row += 4)
        {
            pi2_coeffs = pi2_coeffs_temp + row * src_strd;
            pi2_q_dst  = pi2_q_dst_temp + row * dst_q_strd;
            pi2_iq_dst = pi2_iq_dst_temp + row * dst_iq_strd;
            pi4_quant_round_factor_0_1 = pi4_quant_round_factor_0_1_temp + trans_size * row;
            pi4_quant_round_factor_1_2 = pi4_quant_round_factor_1_2_temp + trans_size * row;

            for(col = trans_size; col >= 8; col -= 8)
            {
                *(csbf + block_col) = 0;
                *(csbf + block_col + 1) = 0;

                /***********************LOOP UNROLLING-0*****************/
                src01_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 0 * src_strd));
                src02_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 1 * src_strd));

                /**************QUANTISATION *****************************/
                sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src01_16x8b);
                sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src02_16x8b);

                sign1_16x8b  = _mm_slli_epi16 (sign1_16x8b, 0x1);
                sign2_16x8b  = _mm_slli_epi16 (sign2_16x8b, 0x1);

                sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
                sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

                src1_16x8b = _mm_abs_epi16(src01_16x8b);
                src2_16x8b = _mm_abs_epi16(src02_16x8b);

                temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
                temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
                temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

                mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
                mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
                mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
                mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

                /**************QUANTISATION : qadd = 0******************/
                res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
                res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
                res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
                res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

                des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
                des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

                /**************QUANTISATION ****************************/
                /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
                res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
                res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
                res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
                res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

                res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
                res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
                res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
                res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

                des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
                des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_1_2************/
                qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2));
                qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 4));
                qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 1 * trans_size));
                qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 1 * trans_size + 4));

                qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
                res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
                res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
                res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

                res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
                res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
                res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
                res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

                des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
                des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_0_1************/
                qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1));
                qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 4));
                qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 1 * trans_size));
                qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 1 * trans_size + 4));

                qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
                res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
                res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
                res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

                res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
                res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
                res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
                res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

                des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
                des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

                /****************QUANTISATION ***************************/
                /****************COMPARE des01, des02 with 1 & 2*********/
                temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
                temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
                temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
                temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

                temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
                temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

                dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
                dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 0 * dst_q_strd), dst0_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 1 * dst_q_strd), dst1_16x8b);

                /****************INVERSE QUANTISATION********************/
                temp5_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

                temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
                mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
                mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
                mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

                /*****************SD Computation & Accumulation**********/
                temp5_16x8b = _mm_srli_si128 (src01_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (src02_16x8b, 0x8);

                /***************INVERSE QUANTISATION********************/
                mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
                mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
                mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
                mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

                if(shift_select)
                {
                    mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
                }
                else
                {
                    mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
                }

                des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
                des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 0 * dst_iq_strd), des11_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 1 * dst_iq_strd), des12_16x8b);

                /*************SD Computation & Accumulation************/
                temp1_16x8b = _mm_cvtepi16_epi32(src01_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(src02_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                temp1_16x8b = _mm_sub_epi32(temp1_16x8b, mul1_32x4b);
                temp2_16x8b = _mm_sub_epi32(temp2_16x8b, mul2_32x4b);
                temp3_16x8b = _mm_sub_epi32(temp3_16x8b, mul3_32x4b);
                temp4_16x8b = _mm_sub_epi32(temp4_16x8b, mul4_32x4b);

                temp1_16x8b = _mm_mullo_epi32(temp1_16x8b, temp1_16x8b);
                temp2_16x8b = _mm_mullo_epi32(temp2_16x8b, temp2_16x8b);
                temp3_16x8b = _mm_mullo_epi32(temp3_16x8b, temp3_16x8b);
                temp4_16x8b = _mm_mullo_epi32(temp4_16x8b, temp4_16x8b);

                temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp2_16x8b);
                temp3_16x8b = _mm_add_epi32(temp3_16x8b, temp4_16x8b);
                temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp3_16x8b);

                temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);
                temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);

                /**************** SSD Accumulation *********************/
                ssd_acc = _mm_add_epi32 (ssd_acc, temp1_16x8b);

                /***************ENDING OF LOOP UNROLLING-0***************/

                /***********************CSBF*****************************/
                temp1_16x8b = _mm_unpacklo_epi64 (dst0_16x8b, dst1_16x8b);
                temp2_16x8b = _mm_unpackhi_epi64 (dst0_16x8b, dst1_16x8b);

                temp1_16x8b = _mm_cmpeq_epi16 (temp1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_cmpeq_epi16 (temp2_16x8b, zero_32x4b);

                *(csbf + block_col) = !(_mm_test_all_ones (temp1_16x8b));
                *(csbf + block_col + 1) = !(_mm_test_all_ones (temp2_16x8b));
                /***********************CSBF END*************************/

                /********************LOOP UNROLLING-1********************/
                src01_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 2 * src_strd));
                src02_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 3 * src_strd));

                /**************QUANTISATION *****************************/
                sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src01_16x8b);
                sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src02_16x8b);

                sign1_16x8b  = _mm_srli_epi16 (sign1_16x8b, 0x1);
                sign2_16x8b  = _mm_srli_epi16 (sign2_16x8b, 0x1);

                sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
                sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

                src1_16x8b = _mm_abs_epi16(src01_16x8b);
                src2_16x8b = _mm_abs_epi16(src02_16x8b);

                temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
                temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
                temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

                mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
                mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
                mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
                mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

                /**************QUANTISATION : qadd = 0******************/
                res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
                res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
                res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
                res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

                des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
                des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

                /**************QUANTISATION ****************************/
                /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
                res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
                res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
                res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
                res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

                res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
                res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
                res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
                res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

                des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
                des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_1_2************/
                qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 2 * trans_size));
                qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 2 * trans_size + 4));
                qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 3 * trans_size));
                qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 3 * trans_size + 4));

                qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
                res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
                res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
                res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

                res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
                res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
                res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
                res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

                des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
                des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_0_1************/
                qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 2 * trans_size));
                qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 2 * trans_size + 4));
                qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 3 * trans_size));
                qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 3 * trans_size + 4));

                qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
                res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
                res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
                res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

                res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
                res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
                res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
                res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

                des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
                des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

                /****************QUANTISATION ***************************/
                /****************COMPARE des01, des02 with 1 & 2*********/
                temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
                temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
                temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
                temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

                temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
                temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

                dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
                dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

                /******************* QUANTISATION ***********************/
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 2 * dst_q_strd), dst0_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 3 * dst_q_strd), dst1_16x8b);

                /******************INVERSE QUANTISATION******************/
                temp5_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

                temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
                mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
                mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
                mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

                /***************SD Computation & Accumulation*************/
                temp5_16x8b = _mm_srli_si128 (src01_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (src02_16x8b, 0x8);

                /*****************INVERSE QUANTISATION********************/
                mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
                mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
                mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
                mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

                if(shift_select)
                {
                    mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
                }
                else
                {
                    mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
                }

                des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
                des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 2 * dst_iq_strd), des11_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 3 * dst_iq_strd), des12_16x8b);

                /***************SD Computation & Accumulation**************/
                temp1_16x8b = _mm_cvtepi16_epi32(src01_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(src02_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                temp1_16x8b = _mm_sub_epi32(temp1_16x8b, mul1_32x4b);
                temp2_16x8b = _mm_sub_epi32(temp2_16x8b, mul2_32x4b);
                temp3_16x8b = _mm_sub_epi32(temp3_16x8b, mul3_32x4b);
                temp4_16x8b = _mm_sub_epi32(temp4_16x8b, mul4_32x4b);

                temp1_16x8b = _mm_mullo_epi32(temp1_16x8b, temp1_16x8b);
                temp2_16x8b = _mm_mullo_epi32(temp2_16x8b, temp2_16x8b);
                temp3_16x8b = _mm_mullo_epi32(temp3_16x8b, temp3_16x8b);
                temp4_16x8b = _mm_mullo_epi32(temp4_16x8b, temp4_16x8b);

                temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp2_16x8b);
                temp3_16x8b = _mm_add_epi32(temp3_16x8b, temp4_16x8b);
                temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp3_16x8b);

                temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);
                temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);

                /******************* SSD Accumulation *******************/
                ssd_acc = _mm_add_epi32 (ssd_acc, temp1_16x8b);
                /***************ENDING OF LOOP UNROLLING-0***************/

                /************************CSBF****************************/
                temp1_16x8b = _mm_unpacklo_epi64 (dst0_16x8b, dst1_16x8b);
                temp2_16x8b = _mm_unpackhi_epi64 (dst0_16x8b, dst1_16x8b);

                temp1_16x8b = _mm_cmpeq_epi16 (temp1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_cmpeq_epi16 (temp2_16x8b, zero_32x4b);

                *(csbf + block_col) =*(csbf + block_col) || (!(_mm_test_all_ones (temp1_16x8b)));
                *(csbf + block_col + 1) = *(csbf + block_col + 1) || (!(_mm_test_all_ones (temp2_16x8b)));

                temp_zero_col = (temp_zero_col) | ((0xF << block_col * 4) * csbf[block_col]);
                temp_zero_col = (temp_zero_col) | ((0xF << (block_col + 1) * 4) * csbf[block_col + 1]);
                temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col]);
                temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col + 1]);

                cbf = cbf || (*(csbf + block_col));
                cbf = cbf || (*(csbf + block_col + 1));
                /************************CSBF END************************/

                block_col  += 2;
                pi2_coeffs += 8;                        /*pointer update*/
                pi2_q_dst  += 8;                        /*pointer update*/
                pi2_iq_dst += 8;                        /*pointer update*/
                pi4_quant_round_factor_0_1 += 8;        /*pointer update*/
                pi4_quant_round_factor_1_2 += 8;        /*pointer update*/

            }

            block_col  = 0;
            block_row += 4;
            csbf      += csbf_strd; /*pointer update*/
        }
    }
    if(trans_size == 4)
    {
        *(csbf + block_col) = 0;

        src01_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 0 * src_strd));
        src02_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 1 * src_strd));
        src03_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 2 * src_strd));
        src04_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 3 * src_strd));

        /******************* QUANTISATION ***********************/
        src1_16x8b = _mm_unpacklo_epi64 (src01_16x8b, src02_16x8b);
        src2_16x8b = _mm_unpacklo_epi64 (src03_16x8b, src04_16x8b);

        sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src1_16x8b);
        sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src2_16x8b);

        sign1_16x8b  = _mm_slli_epi16 (sign1_16x8b, 0x1);
        sign2_16x8b  = _mm_slli_epi16 (sign2_16x8b, 0x1);

        sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
        sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

        src1_16x8b = _mm_abs_epi16(src1_16x8b);
        src2_16x8b = _mm_abs_epi16(src2_16x8b);

        temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
        temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
        temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
        temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

        mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
        mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
        mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
        mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

        /**************QUANTISATION : qadd = 0******************/
        res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
        res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
        res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
        res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

        des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
        des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

        /**************QUANTISATION ****************************/
        /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
        res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
        res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
        res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
        res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

        res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
        res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
        res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
        res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

        des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
        des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

        /****************QUANTISATION ***************************/
        /**********qadd = *pi4_quant_round_factor_1_2************/
        qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2));
        qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 4));
        qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 8));
        qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 12));

        qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

        res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
        res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
        res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
        res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

        res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
        res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
        res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
        res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

        des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
        des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

        /****************QUANTISATION ***************************/
        /**********qadd = *pi4_quant_round_factor_0_1************/
        qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1));
        qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 4));
        qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 8));
        qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 12));

        qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

        res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
        res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
        res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
        res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

        res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
        res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
        res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
        res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

        des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
        des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

        /****************QUANTISATION ***************************/
        /****************COMPARE des01, des02 with 1 & 2*********/
        temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
        temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
        temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
        temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

        temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
        temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

        des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
        des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
        des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
        des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

        dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
        dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

        temp2_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
        temp4_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 0 * dst_q_strd), dst0_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 2 * dst_q_strd), dst1_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 1 * dst_q_strd), temp2_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 3 * dst_q_strd), temp4_16x8b);

        /*****************INVERSE QUANTISATION********************/
        temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
        temp2_16x8b = _mm_cvtepi16_epi32(temp2_16x8b);
        temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
        temp4_16x8b = _mm_cvtepi16_epi32(temp4_16x8b);

        mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
        mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
        mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
        mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

        mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
        mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
        mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
        mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

        if(shift_select)
        {
            mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
            mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
            mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
            mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
        }
        else
        {
            mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
            mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
            mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
            mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
        }

        des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
        des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

        temp1_16x8b = _mm_srli_si128 (des11_16x8b, 0x8);
        temp2_16x8b = _mm_srli_si128 (des12_16x8b, 0x8);

        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 0 * dst_iq_strd), des11_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 2 * dst_iq_strd), des12_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 1 * dst_iq_strd), temp1_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 3 * dst_iq_strd), temp2_16x8b);

        /**************SD Computation & Accumulation**************/
        temp1_16x8b = _mm_cvtepi16_epi32(src01_16x8b);
        temp2_16x8b = _mm_cvtepi16_epi32(src02_16x8b);
        temp3_16x8b = _mm_cvtepi16_epi32(src03_16x8b);
        temp4_16x8b = _mm_cvtepi16_epi32(src04_16x8b);

        temp1_16x8b = _mm_sub_epi32(temp1_16x8b, mul1_32x4b);
        temp2_16x8b = _mm_sub_epi32(temp2_16x8b, mul2_32x4b);
        temp3_16x8b = _mm_sub_epi32(temp3_16x8b, mul3_32x4b);
        temp4_16x8b = _mm_sub_epi32(temp4_16x8b, mul4_32x4b);

        temp1_16x8b = _mm_mullo_epi32(temp1_16x8b, temp1_16x8b);
        temp2_16x8b = _mm_mullo_epi32(temp2_16x8b, temp2_16x8b);
        temp3_16x8b = _mm_mullo_epi32(temp3_16x8b, temp3_16x8b);
        temp4_16x8b = _mm_mullo_epi32(temp4_16x8b, temp4_16x8b);

        temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp2_16x8b);
        temp3_16x8b = _mm_add_epi32(temp3_16x8b, temp4_16x8b);
        temp1_16x8b = _mm_add_epi32(temp1_16x8b, temp3_16x8b);

        temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);
        temp1_16x8b = _mm_hadd_epi32(temp1_16x8b, temp1_16x8b);

        /**************** SSD Accumulation *******************/
        ssd_acc = _mm_add_epi32 (ssd_acc, temp1_16x8b);

        /***********************CSBF**************************/
        temp1_16x8b = _mm_cmpeq_epi16 (dst0_16x8b, zero_32x4b);
        temp2_16x8b = _mm_cmpeq_epi16 (dst1_16x8b, zero_32x4b);

        *(csbf + block_col) = !((_mm_test_all_ones (temp1_16x8b)) &&
            (_mm_test_all_ones (temp2_16x8b)));

        temp_zero_col = (temp_zero_col) | ((0xF << block_col * 4) * csbf[block_col]);
        temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col]);

        cbf = cbf || (*(csbf + block_col));
        /***********************CSBF END**********************/
    }

    *pi8_cost = _mm_cvtsi128_si32(ssd_acc);
    *zero_col = ~temp_zero_col;     //final zero_col storing
    *zero_row = ~temp_zero_row;     //final zero_row storing

    return cbf;
}

WORD32 ihevc_hbd_q_iq_flat_scale_mat_var_rnd_fact_sse42
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    )
{
    WORD32 q_bits, log2_size, quant_multiplier;
    WORD32 shift_in_iquant_minus_1;
    WORD32 shift_iq, transform_shift;
    WORD32 shift_in_iquant, shift_select, row, col;
    WORD32 cbf = 0;
    WORD16 *pi2_coeffs_temp;
    WORD16 *pi2_q_dst_temp;
    WORD16 *pi2_iq_dst_temp;
    WORD32 *pi4_quant_round_factor_0_1_temp;
    WORD32 *pi4_quant_round_factor_1_2_temp;
    WORD32 block_row = 0;
    WORD32 block_col = 0;
    WORD32 temp_zero_col = 0;
    WORD32 temp_zero_row = 0;

    __m128i zero_32x4b, quant_coeff, qadd1_32x4b, two_16x8b;
    __m128i one_16x8b, iquant_coeff, add_iq_32x4b;
    __m128i sign1_16x8b, sign2_16x8b;
    __m128i src1_16x8b, src2_16x8b;
    __m128i src01_16x8b, src02_16x8b, src03_16x8b, src04_16x8b;
    __m128i temp1_16x8b, temp2_16x8b, temp3_16x8b, temp4_16x8b;
    __m128i temp5_16x8b, temp6_16x8b;
    __m128i res01_32x4b, res02_32x4b, res03_32x4b,res04_32x4b;
    __m128i res11_32x4b, res12_32x4b, res13_32x4b,res14_32x4b;
    __m128i res21_32x4b, res22_32x4b, res23_32x4b,res24_32x4b;
    __m128i res31_32x4b, res32_32x4b, res33_32x4b,res34_32x4b;
    __m128i des01_16x8b, des02_16x8b, des11_16x8b, des12_16x8b;
    __m128i des21_16x8b, des22_16x8b, des31_16x8b, des32_16x8b;
    __m128i dst0_16x8b,dst1_16x8b;
    __m128i qadd20_32x4b, qadd21_32x4b, qadd22_32x4b, qadd23_32x4b;
    __m128i qadd30_32x4b, qadd31_32x4b, qadd32_32x4b, qadd33_32x4b;
    __m128i mul1_32x4b, mul2_32x4b, mul3_32x4b, mul4_32x4b;

    pi2_coeffs_temp = pi2_coeffs;
    pi2_q_dst_temp = pi2_q_dst;
    pi2_iq_dst_temp = pi2_iq_dst;
    pi4_quant_round_factor_0_1_temp = pi4_quant_round_factor_0_1;
    pi4_quant_round_factor_1_2_temp = pi4_quant_round_factor_1_2;

    GETRANGE(log2_size, trans_size);
    log2_size -= 1;
    transform_shift = MAX_TR_DYNAMIC_RANGE - i4_bit_depth - log2_size;
    quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;

    shift_iq = i4_bit_depth + log2_size - 5;
    zero_32x4b = _mm_set1_epi32 (0x0);
    quant_coeff = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    qadd1_32x4b = _mm_set1_epi32((1 << QUANT_ROUND_FACTOR_Q)/2);
    qadd1_32x4b = _mm_slli_epi32 (qadd1_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
    two_16x8b = _mm_set1_epi16(0x2);
    one_16x8b = _mm_set1_epi16(0x1);

    iquant_coeff = _mm_set1_epi32(g_ihevc_iquant_scales[qp_rem]<<4);
    add_iq_32x4b = _mm_set1_epi32(0x1);

    if((shift_iq - qp_div - 1)>=0)
    {
        shift_in_iquant_minus_1 = (shift_iq - qp_div - 1);
        add_iq_32x4b = _mm_slli_epi32(add_iq_32x4b, shift_in_iquant_minus_1);

        shift_in_iquant = (shift_iq - qp_div);
        shift_select    = 1;
    }
    else
    {
        shift_in_iquant_minus_1 = (-(shift_iq - qp_div - 1));
        add_iq_32x4b = _mm_srai_epi32(add_iq_32x4b, shift_in_iquant_minus_1);

        shift_in_iquant = (-(shift_iq - qp_div));
        shift_select    = 0;
    }

    if(trans_size > 7)
    {
        for(row = 0; row < trans_size; row += 4)
        {
            pi2_coeffs = pi2_coeffs_temp + row * src_strd;
            pi2_q_dst  = pi2_q_dst_temp + row * dst_q_strd;
            pi2_iq_dst = pi2_iq_dst_temp + row * dst_iq_strd;
            pi4_quant_round_factor_0_1 = pi4_quant_round_factor_0_1_temp + trans_size * row;
            pi4_quant_round_factor_1_2 = pi4_quant_round_factor_1_2_temp + trans_size * row;

            for(col = trans_size; col >= 8; col -= 8)
            {
                *(csbf + block_col) = 0;
                *(csbf + block_col + 1) = 0;

                /***********************LOOP UNROLLING-0*****************/
                src01_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 0 * src_strd));
                src02_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 1 * src_strd));

                /**************QUANTISATION *****************************/
                sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src01_16x8b);
                sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src02_16x8b);

                sign1_16x8b  = _mm_slli_epi16 (sign1_16x8b, 0x1);
                sign2_16x8b  = _mm_slli_epi16 (sign2_16x8b, 0x1);

                sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
                sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

                src1_16x8b = _mm_abs_epi16(src01_16x8b);
                src2_16x8b = _mm_abs_epi16(src02_16x8b);

                temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
                temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
                temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

                mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
                mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
                mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
                mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

                /**************QUANTISATION : qadd = 0******************/
                res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
                res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
                res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
                res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

                des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
                des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

                /**************QUANTISATION ****************************/
                /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
                res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
                res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
                res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
                res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

                res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
                res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
                res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
                res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

                des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
                des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_1_2************/
                qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2));
                qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 4));
                qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 1 * trans_size));
                qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 1 * trans_size + 4));

                qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
                res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
                res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
                res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

                res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
                res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
                res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
                res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

                des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
                des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_0_1************/
                qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1));
                qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 4));
                qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 1 * trans_size));
                qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 1 * trans_size + 4));

                qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
                res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
                res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
                res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

                res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
                res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
                res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
                res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

                des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
                des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

                /****************QUANTISATION ***************************/
                /****************COMPARE des01, des02 with 1 & 2*********/
                temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
                temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
                temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
                temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

                temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
                temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

                dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
                dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 0 * dst_q_strd), dst0_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 1 * dst_q_strd), dst1_16x8b);

                /****************INVERSE QUANTISATION********************/
                temp5_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

                temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
                mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
                mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
                mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

                /***************INVERSE QUANTISATION********************/
                mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
                mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
                mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
                mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

                if(shift_select)
                {
                    mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
                }
                else
                {
                    mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
                }

                des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
                des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 0 * dst_iq_strd), des11_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 1 * dst_iq_strd), des12_16x8b);

                /***************ENDING OF LOOP UNROLLING-0***************/

                /***********************CSBF*****************************/
                temp1_16x8b = _mm_unpacklo_epi64 (dst0_16x8b, dst1_16x8b);
                temp2_16x8b = _mm_unpackhi_epi64 (dst0_16x8b, dst1_16x8b);

                temp1_16x8b = _mm_cmpeq_epi16 (temp1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_cmpeq_epi16 (temp2_16x8b, zero_32x4b);

                *(csbf + block_col) = !(_mm_test_all_ones (temp1_16x8b));
                *(csbf + block_col + 1) = !(_mm_test_all_ones (temp2_16x8b));
                /***********************CSBF END*************************/

                /********************LOOP UNROLLING-1********************/
                src01_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 2 * src_strd));
                src02_16x8b = _mm_loadu_si128 ((__m128i const* )(pi2_coeffs + 3 * src_strd));

                /**************QUANTISATION *****************************/
                sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src01_16x8b);
                sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src02_16x8b);

                sign1_16x8b  = _mm_srli_epi16 (sign1_16x8b, 0x1);
                sign2_16x8b  = _mm_srli_epi16 (sign2_16x8b, 0x1);

                sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
                sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

                src1_16x8b = _mm_abs_epi16(src01_16x8b);
                src2_16x8b = _mm_abs_epi16(src02_16x8b);

                temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
                temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
                temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

                mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
                mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
                mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
                mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

                /**************QUANTISATION : qadd = 0******************/
                res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
                res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
                res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
                res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

                des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
                des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

                /**************QUANTISATION ****************************/
                /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
                res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
                res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
                res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
                res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

                res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
                res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
                res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
                res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

                des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
                des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_1_2************/
                qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 2 * trans_size));
                qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 2 * trans_size + 4));
                qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 3 * trans_size));
                qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 3 * trans_size + 4));

                qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
                res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
                res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
                res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

                res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
                res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
                res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
                res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

                des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
                des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

                /****************QUANTISATION ***************************/
                /**********qadd = *pi4_quant_round_factor_0_1************/
                qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 2 * trans_size));
                qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 2 * trans_size + 4));
                qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 3 * trans_size));
                qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 3 * trans_size + 4));

                qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
                qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

                res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
                res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
                res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
                res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

                res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
                res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
                res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
                res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

                des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
                des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

                /****************QUANTISATION ***************************/
                /****************COMPARE des01, des02 with 1 & 2*********/
                temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
                temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
                temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
                temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

                temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
                temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
                des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
                des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

                dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
                dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

                /******************* QUANTISATION ***********************/
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 2 * dst_q_strd), dst0_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_q_dst + 3 * dst_q_strd), dst1_16x8b);

                /******************INVERSE QUANTISATION******************/
                temp5_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
                temp6_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

                temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
                temp2_16x8b = _mm_cvtepi16_epi32(temp5_16x8b);
                temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
                temp4_16x8b = _mm_cvtepi16_epi32(temp6_16x8b);

                mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
                mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
                mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
                mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

                /*****************INVERSE QUANTISATION********************/
                mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
                mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
                mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
                mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

                if(shift_select)
                {
                    mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
                }
                else
                {
                    mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
                    mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
                    mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
                    mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
                }

                des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
                des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 2 * dst_iq_strd), des11_16x8b);
                _mm_storeu_si128 ((__m128i*)(pi2_iq_dst + 3 * dst_iq_strd), des12_16x8b);

                /***************ENDING OF LOOP UNROLLING-0***************/

                /************************CSBF****************************/
                temp1_16x8b = _mm_unpacklo_epi64 (dst0_16x8b, dst1_16x8b);
                temp2_16x8b = _mm_unpackhi_epi64 (dst0_16x8b, dst1_16x8b);

                temp1_16x8b = _mm_cmpeq_epi16 (temp1_16x8b, zero_32x4b);
                temp2_16x8b = _mm_cmpeq_epi16 (temp2_16x8b, zero_32x4b);

                *(csbf + block_col) =*(csbf + block_col) || (!(_mm_test_all_ones (temp1_16x8b)));
                *(csbf + block_col + 1) = *(csbf + block_col + 1) || (!(_mm_test_all_ones (temp2_16x8b)));

                temp_zero_col = (temp_zero_col) | ((0xF << block_col * 4) * csbf[block_col]);
                temp_zero_col = (temp_zero_col) | ((0xF << (block_col + 1) * 4) * csbf[block_col + 1]);
                temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col]);
                temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col + 1]);

                cbf = cbf || (*(csbf + block_col));
                cbf = cbf || (*(csbf + block_col + 1));
                /************************CSBF END************************/

                block_col  += 2;
                pi2_coeffs += 8;                        /*pointer update*/
                pi2_q_dst  += 8;                        /*pointer update*/
                pi2_iq_dst += 8;                        /*pointer update*/
                pi4_quant_round_factor_0_1 += 8;        /*pointer update*/
                pi4_quant_round_factor_1_2 += 8;        /*pointer update*/

            }

            block_col  = 0;
            block_row += 4;
            csbf      += csbf_strd; /*pointer update*/
        }
    }
    if(trans_size == 4)
    {
        *(csbf + block_col) = 0;

        src01_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 0 * src_strd));
        src02_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 1 * src_strd));
        src03_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 2 * src_strd));
        src04_16x8b = _mm_loadl_epi64 ((__m128i const* )(pi2_coeffs + 3 * src_strd));

        /******************* QUANTISATION ***********************/
        src1_16x8b = _mm_unpacklo_epi64 (src01_16x8b, src02_16x8b);
        src2_16x8b = _mm_unpacklo_epi64 (src03_16x8b, src04_16x8b);

        sign1_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src1_16x8b);
        sign2_16x8b  = _mm_cmpgt_epi16 (zero_32x4b, src2_16x8b);

        sign1_16x8b  = _mm_slli_epi16 (sign1_16x8b, 0x1);
        sign2_16x8b  = _mm_slli_epi16 (sign2_16x8b, 0x1);

        sign1_16x8b  = _mm_add_epi16 (sign1_16x8b, one_16x8b);
        sign2_16x8b  = _mm_add_epi16 (sign2_16x8b, one_16x8b);

        src1_16x8b = _mm_abs_epi16(src1_16x8b);
        src2_16x8b = _mm_abs_epi16(src2_16x8b);

        temp1_16x8b = _mm_unpacklo_epi16 (src1_16x8b, zero_32x4b);
        temp2_16x8b = _mm_unpackhi_epi16 (src1_16x8b, zero_32x4b);
        temp3_16x8b = _mm_unpacklo_epi16 (src2_16x8b, zero_32x4b);
        temp4_16x8b = _mm_unpackhi_epi16 (src2_16x8b, zero_32x4b);

        mul1_32x4b = _mm_madd_epi16 (temp1_16x8b, quant_coeff);
        mul2_32x4b = _mm_madd_epi16 (temp2_16x8b, quant_coeff);
        mul3_32x4b = _mm_madd_epi16 (temp3_16x8b, quant_coeff);
        mul4_32x4b = _mm_madd_epi16 (temp4_16x8b, quant_coeff);

        /**************QUANTISATION : qadd = 0******************/
        res01_32x4b = _mm_srli_epi32 (mul1_32x4b, q_bits);
        res02_32x4b = _mm_srli_epi32 (mul2_32x4b, q_bits);
        res03_32x4b = _mm_srli_epi32 (mul3_32x4b, q_bits);
        res04_32x4b = _mm_srli_epi32 (mul4_32x4b, q_bits);

        des01_16x8b = _mm_packs_epi32 (res01_32x4b, res02_32x4b);
        des02_16x8b = _mm_packs_epi32 (res03_32x4b, res04_32x4b);

        /**************QUANTISATION ****************************/
        /********qadd = (1 << QUANT_ROUND_FACTOR_Q)/2)**********/
        res11_32x4b = _mm_add_epi32 (mul1_32x4b, qadd1_32x4b);
        res12_32x4b = _mm_add_epi32 (mul2_32x4b, qadd1_32x4b);
        res13_32x4b = _mm_add_epi32 (mul3_32x4b, qadd1_32x4b);
        res14_32x4b = _mm_add_epi32 (mul4_32x4b, qadd1_32x4b);

        res11_32x4b = _mm_srli_epi32 (res11_32x4b, q_bits);
        res12_32x4b = _mm_srli_epi32 (res12_32x4b, q_bits);
        res13_32x4b = _mm_srli_epi32 (res13_32x4b, q_bits);
        res14_32x4b = _mm_srli_epi32 (res14_32x4b, q_bits);

        des11_16x8b = _mm_packs_epi32 (res11_32x4b, res12_32x4b);
        des12_16x8b = _mm_packs_epi32 (res13_32x4b, res14_32x4b);

        /****************QUANTISATION ***************************/
        /**********qadd = *pi4_quant_round_factor_1_2************/
        qadd20_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2));
        qadd21_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 4));
        qadd22_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 8));
        qadd23_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_1_2 + 12));

        qadd20_32x4b = _mm_slli_epi32 (qadd20_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd21_32x4b = _mm_slli_epi32 (qadd21_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd22_32x4b = _mm_slli_epi32 (qadd22_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd23_32x4b = _mm_slli_epi32 (qadd23_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

        res21_32x4b = _mm_add_epi32 (mul1_32x4b, qadd20_32x4b);
        res22_32x4b = _mm_add_epi32 (mul2_32x4b, qadd21_32x4b);
        res23_32x4b = _mm_add_epi32 (mul3_32x4b, qadd22_32x4b);
        res24_32x4b = _mm_add_epi32 (mul4_32x4b, qadd23_32x4b);

        res21_32x4b = _mm_srli_epi32 (res21_32x4b, q_bits);
        res22_32x4b = _mm_srli_epi32 (res22_32x4b, q_bits);
        res23_32x4b = _mm_srli_epi32 (res23_32x4b, q_bits);
        res24_32x4b = _mm_srli_epi32 (res24_32x4b, q_bits);

        des21_16x8b = _mm_packs_epi32 (res21_32x4b, res22_32x4b);
        des22_16x8b = _mm_packs_epi32 (res23_32x4b, res24_32x4b);

        /****************QUANTISATION ***************************/
        /**********qadd = *pi4_quant_round_factor_0_1************/
        qadd30_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1));
        qadd31_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 4));
        qadd32_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 8));
        qadd33_32x4b = _mm_loadu_si128 ((__m128i const* )(pi4_quant_round_factor_0_1 + 12));

        qadd30_32x4b = _mm_slli_epi32 (qadd30_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd31_32x4b = _mm_slli_epi32 (qadd31_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd32_32x4b = _mm_slli_epi32 (qadd32_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));
        qadd33_32x4b = _mm_slli_epi32 (qadd33_32x4b, (q_bits - QUANT_ROUND_FACTOR_Q));

        res31_32x4b = _mm_add_epi32 (mul1_32x4b, qadd30_32x4b);
        res32_32x4b = _mm_add_epi32 (mul2_32x4b, qadd31_32x4b);
        res33_32x4b = _mm_add_epi32 (mul3_32x4b, qadd32_32x4b);
        res34_32x4b = _mm_add_epi32 (mul4_32x4b, qadd33_32x4b);

        res31_32x4b = _mm_srli_epi32 (res31_32x4b, q_bits);
        res32_32x4b = _mm_srli_epi32 (res32_32x4b, q_bits);
        res33_32x4b = _mm_srli_epi32 (res33_32x4b, q_bits);
        res34_32x4b = _mm_srli_epi32 (res34_32x4b, q_bits);

        des31_16x8b = _mm_packs_epi32 (res31_32x4b, res32_32x4b);
        des32_16x8b = _mm_packs_epi32 (res33_32x4b, res34_32x4b);

        /****************QUANTISATION ***************************/
        /****************COMPARE des01, des02 with 1 & 2*********/
        temp1_16x8b = _mm_cmplt_epi16 (des01_16x8b, two_16x8b);
        temp2_16x8b = _mm_cmplt_epi16 (des01_16x8b, one_16x8b);
        temp3_16x8b = _mm_cmplt_epi16 (des02_16x8b, two_16x8b);
        temp4_16x8b = _mm_cmplt_epi16 (des02_16x8b, one_16x8b);

        temp5_16x8b = _mm_andnot_si128 (temp2_16x8b, temp1_16x8b);
        temp6_16x8b = _mm_andnot_si128 (temp4_16x8b, temp3_16x8b);

        des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des31_16x8b, temp2_16x8b);
        des11_16x8b = _mm_blendv_epi8 ( des11_16x8b, des21_16x8b, temp5_16x8b);
        des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des32_16x8b, temp4_16x8b);
        des12_16x8b = _mm_blendv_epi8 ( des12_16x8b, des22_16x8b, temp6_16x8b);

        dst0_16x8b = _mm_sign_epi16 (des11_16x8b, sign1_16x8b);
        dst1_16x8b = _mm_sign_epi16 (des12_16x8b, sign2_16x8b);

        temp2_16x8b = _mm_srli_si128 (dst0_16x8b, 0x8);
        temp4_16x8b = _mm_srli_si128 (dst1_16x8b, 0x8);

        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 0 * dst_q_strd), dst0_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 2 * dst_q_strd), dst1_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 1 * dst_q_strd), temp2_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_q_dst + 3 * dst_q_strd), temp4_16x8b);

        /*****************INVERSE QUANTISATION********************/
        temp1_16x8b = _mm_cvtepi16_epi32(dst0_16x8b);
        temp2_16x8b = _mm_cvtepi16_epi32(temp2_16x8b);
        temp3_16x8b = _mm_cvtepi16_epi32(dst1_16x8b);
        temp4_16x8b = _mm_cvtepi16_epi32(temp4_16x8b);

        mul1_32x4b = _mm_mullo_epi32 (temp1_16x8b, iquant_coeff);
        mul2_32x4b = _mm_mullo_epi32 (temp2_16x8b, iquant_coeff);
        mul3_32x4b = _mm_mullo_epi32 (temp3_16x8b, iquant_coeff);
        mul4_32x4b = _mm_mullo_epi32 (temp4_16x8b, iquant_coeff);

        mul1_32x4b = _mm_add_epi32 (add_iq_32x4b, mul1_32x4b);
        mul2_32x4b = _mm_add_epi32 (add_iq_32x4b, mul2_32x4b);
        mul3_32x4b = _mm_add_epi32 (add_iq_32x4b, mul3_32x4b);
        mul4_32x4b = _mm_add_epi32 (add_iq_32x4b, mul4_32x4b);

        if(shift_select)
        {
            mul1_32x4b = _mm_srai_epi32(mul1_32x4b, shift_in_iquant);
            mul2_32x4b = _mm_srai_epi32(mul2_32x4b, shift_in_iquant);
            mul3_32x4b = _mm_srai_epi32(mul3_32x4b, shift_in_iquant);
            mul4_32x4b = _mm_srai_epi32(mul4_32x4b, shift_in_iquant);
        }
        else
        {
            mul1_32x4b = _mm_slli_epi32(mul1_32x4b, shift_in_iquant);
            mul2_32x4b = _mm_slli_epi32(mul2_32x4b, shift_in_iquant);
            mul3_32x4b = _mm_slli_epi32(mul3_32x4b, shift_in_iquant);
            mul4_32x4b = _mm_slli_epi32(mul4_32x4b, shift_in_iquant);
        }

        des11_16x8b = _mm_packs_epi32(mul1_32x4b, mul2_32x4b);
        des12_16x8b = _mm_packs_epi32(mul3_32x4b, mul4_32x4b);

        temp1_16x8b = _mm_srli_si128 (des11_16x8b, 0x8);
        temp2_16x8b = _mm_srli_si128 (des12_16x8b, 0x8);

        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 0 * dst_iq_strd), des11_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 2 * dst_iq_strd), des12_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 1 * dst_iq_strd), temp1_16x8b);
        _mm_storel_epi64 ((__m128i*)(pi2_iq_dst + 3 * dst_iq_strd), temp2_16x8b);

        /***********************CSBF**************************/
        temp1_16x8b = _mm_cmpeq_epi16 (dst0_16x8b, zero_32x4b);
        temp2_16x8b = _mm_cmpeq_epi16 (dst1_16x8b, zero_32x4b);

        *(csbf + block_col) = !((_mm_test_all_ones (temp1_16x8b)) &&
            (_mm_test_all_ones (temp2_16x8b)));

        temp_zero_col = (temp_zero_col) | ((0xF << block_col * 4) * csbf[block_col]);
        temp_zero_row = (temp_zero_row) | ((0xF << block_row) * csbf[block_col]);

        cbf = cbf || (*(csbf + block_col));
        /***********************CSBF END**********************/
    }

    *zero_col = ~temp_zero_col;     //final zero_col storing
    *zero_row = ~temp_zero_row;     //final zero_row storing

    return cbf;
}
