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
 *  ihevc_quant.c
 *
 * @brief
 *  Contains function definitions for quantization
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *   - ihevc_quant_4x4_ttype1()
 *   - ihevc_quant_4x4()
 *   - ihevc_quant_8x8()
 *   - ihevc_quant_16x16()
 *   - ihevc_quant_32x32()
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
#include "ihevc_quant.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#define ZERO_ROW        1

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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
 * @returns  cbf
 * coded block flag
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


WORD32 ihevc_quant_4x4_ttype1_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15, m_src_temp16, m_src_temp17;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign, m_sign1;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 2;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier ;

    m_temp = _mm_cvtsi32_si128(q_add);
    m_temp = _mm_slli_epi64(m_temp, (q_bits - QUANT_ROUND_FACTOR_Q));
    m_temp = _mm_unpacklo_epi64(m_temp, m_temp);

    /*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((0)*trans_size)));    /* quant_coeff */
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/

            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((1)*trans_size)));    /* quant_coeff */
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/

            m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((2)*trans_size)));    /* quant_coeff */
            m_src_temp5 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/

            m_src_temp6 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((3)*trans_size)));    /* quant_coeff */
            m_src_temp7 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row=3 */

            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);              /* pi2_quant_coeff [0][1][2][3] row =0*/
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);              /* pi2_quant_coeff [0][1][2][3] row =1*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);              /* pi2_quant_coeff [0][1][2][3] row =2*/
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);              /* pi2_quant_coeff [0][1][2][3] row =3*/

            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_scale);        /* row =0 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_scale);        /* row =1 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);        /* row =2 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);        /* row =3 quant_coeff for col =0, 1, 2, 3 */

            m_src_temp1= _mm_unpacklo_epi64(m_src_temp1, m_src_temp3); /* row =0, 1*/ /*32-bit resolution*/
            m_src_temp3 = _mm_unpacklo_epi64(m_src_temp5, m_src_temp7); /* row =2, 3*/

            m_src_temp5 = _mm_abs_epi16(m_src_temp1);                   /* 16bit resolution*/
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);

/* row =0, 1, 2, 3*/
            m_sign = _mm_cmpgt_epi16(m_zero, m_src_temp1);
            m_sign1 = _mm_cmpgt_epi16(m_zero, m_src_temp3);

            m_sign = _mm_slli_epi16(m_sign, 1);
            m_sign1 = _mm_slli_epi16(m_sign1, 1);

            m_sign = _mm_add_epi16(m_sign, m_one);
            m_sign1 = _mm_add_epi16(m_sign1, m_one);    /* sign value */

/* converting 16 bit to 64-bit for abs(inp)*/
/* for row =0, 1*/
            m_src_temp8 = _mm_cvtepi16_epi64(m_src_temp5);      /*col =0,1*/
            m_src_temp9 = _mm_srli_si128(m_src_temp5, 4);
            m_src_temp9 = _mm_cvtepi16_epi64(m_src_temp9);      /*col =2,3*/

            m_src_temp10 = _mm_srli_si128(m_src_temp5, 8);
            m_src_temp10 = _mm_cvtepi16_epi64(m_src_temp10);    /*col =0,1*/
            m_src_temp11 = _mm_srli_si128(m_src_temp5, 12);
            m_src_temp11 = _mm_cvtepi16_epi64(m_src_temp11);    /*col =2,3*/

/* for row =2, 3*/
            m_src_temp12 = _mm_cvtepi16_epi64(m_src_temp7);     /*col =0,1*/
            m_src_temp13 = _mm_srli_si128(m_src_temp7, 4);
            m_src_temp13 = _mm_cvtepi16_epi64(m_src_temp13);    /*col =2,3*/

            m_src_temp14 = _mm_srli_si128(m_src_temp7, 8);
            m_src_temp14 = _mm_cvtepi16_epi64(m_src_temp14);    /*col =0,1*/
            m_src_temp15 = _mm_srli_si128(m_src_temp7, 12);
            m_src_temp15 = _mm_cvtepi16_epi64(m_src_temp15);    /*col =2,3*/

/* converting 32 bit to 64-bit for quant_coeff[]*/
/* for row =0, 1*/
            m_src_temp5 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =0,1*/
            m_src_temp0 = _mm_srli_si128(m_src_temp0, 4);
            m_src_temp0 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =2,3*/

            m_src_temp7 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =0,1*/
            m_src_temp2 = _mm_srli_si128(m_src_temp2, 4);
            m_src_temp2 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =2,3*/

/* for row =2, 3*/
            m_src_temp16 = _mm_cvtepi32_epi64(m_src_temp4);     /*col =0,1*/
            m_src_temp4 = _mm_srli_si128(m_src_temp4, 4);
            m_src_temp4 = _mm_cvtepi32_epi64(m_src_temp4);      /*col =2,3*/

            m_src_temp17 = _mm_cvtepi32_epi64(m_src_temp6);     /*col =0,1*/
            m_src_temp6 = _mm_srli_si128(m_src_temp6, 4);
            m_src_temp6 = _mm_cvtepi32_epi64(m_src_temp6);      /*col =2,3*/

/* tmp = tmp * (quant_coeff); */

/*row =0*/  m_src_temp8 = _mm_mul_epi32(m_src_temp8, m_src_temp5);      /*  col =0, 1 */
            m_src_temp9 = _mm_mul_epi32(m_src_temp9, m_src_temp0);      /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_mul_epi32(m_src_temp10, m_src_temp7);    /*  col =0, 1 */
            m_src_temp11 = _mm_mul_epi32(m_src_temp11, m_src_temp2);    /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_mul_epi32(m_src_temp12, m_src_temp16);   /*  col =0, 1 */
            m_src_temp13 = _mm_mul_epi32(m_src_temp13, m_src_temp4);    /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_mul_epi32(m_src_temp14, m_src_temp17);   /*  col =0, 1 */
            m_src_temp15 = _mm_mul_epi32(m_src_temp15, m_src_temp6);    /*  col =2, 3 */

/* tmp = tmp + (((LWORD64)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */

/*row =0*/  m_src_temp8 = _mm_add_epi64(m_src_temp8, m_temp);           /*  col =0, 1 */
            m_src_temp9 = _mm_add_epi64(m_src_temp9, m_temp);           /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_add_epi64(m_src_temp10, m_temp);         /*  col =0, 1 */
            m_src_temp11 = _mm_add_epi64(m_src_temp11, m_temp);         /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_add_epi64(m_src_temp12, m_temp);         /*  col =0, 1 */
            m_src_temp13 = _mm_add_epi64(m_src_temp13, m_temp);         /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_add_epi64(m_src_temp14, m_temp);         /*  col =0, 1 */
            m_src_temp15 = _mm_add_epi64(m_src_temp15, m_temp);         /*  col =2, 3 */

/* tmp = tmp >> q_bits;    */
/*row =0*/  m_src_temp8 = _mm_srli_epi64(m_src_temp8, q_bits);          /*  col =0, 1 */
            m_src_temp9 = _mm_srli_epi64(m_src_temp9, q_bits);          /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_srli_epi64(m_src_temp10, q_bits);        /*  col =0, 1 */
            m_src_temp11 = _mm_srli_epi64(m_src_temp11, q_bits);        /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_srli_epi64(m_src_temp12, q_bits);            /*  col =0, 1 */
            m_src_temp13 = _mm_srli_epi64(m_src_temp13, q_bits);            /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_srli_epi64(m_src_temp14, q_bits);        /*  col =0, 1 */
            m_src_temp15 = _mm_srli_epi64(m_src_temp15, q_bits);        /*  col =2, 3 */

/* unpack */

            m_src_temp8 = _mm_shuffle_epi32(m_src_temp8, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp9 = _mm_shuffle_epi32(m_src_temp9, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp10 = _mm_shuffle_epi32(m_src_temp10, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp11 = _mm_shuffle_epi32(m_src_temp11, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp12 = _mm_shuffle_epi32(m_src_temp12, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp13 = _mm_shuffle_epi32(m_src_temp13, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp14 = _mm_shuffle_epi32(m_src_temp14, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp15 = _mm_shuffle_epi32(m_src_temp15, _MM_SHUFFLE(0, 0, 2, 0) );


/*32-bit result*/
/*row=0*/   m_src_temp0 = _mm_unpacklo_epi64 (m_src_temp8,  m_src_temp9);/*col = 0, 1, 2, 3*/
/*row=1*/   m_src_temp1 = _mm_unpacklo_epi64 (m_src_temp10, m_src_temp11);/*col = 0, 1, 2, 3*/
/*row=2*/   m_src_temp2 = _mm_unpacklo_epi64 (m_src_temp12, m_src_temp13);/*col = 0, 1, 2, 3*/
/*row=3*/   m_src_temp3 = _mm_unpacklo_epi64 (m_src_temp14, m_src_temp15);/*col = 0, 1, 2, 3*/

/* sign value coverting 16-bit to 32-bit*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_sign);  /* sign col= [0][1][2][3]*/
            m_sign = _mm_srli_si128(m_sign, 8);
            m_sign = _mm_cvtepi16_epi32(m_sign);       /* sign col= [0][1][2][3]*/

            m_src_temp5 = _mm_cvtepi16_epi32(m_sign1);  /* sign col= [0][1][2][3]*/
            m_sign1 = _mm_srli_si128(m_sign1, 8);
            m_sign1 = _mm_cvtepi16_epi32(m_sign1);     /* sign col= [0][1][2][3]*/

/* tmp = tmp * sign;  */
/*row=0*/   m_src_temp0 = _mm_sign_epi32(m_src_temp0, m_src_temp4); /* col= [0][1][2][3]*/
/*row=1*/   m_src_temp1 = _mm_sign_epi32(m_src_temp1, m_sign);      /* col= [0][1][2][3]*/
/*row=2*/   m_src_temp2 = _mm_sign_epi32(m_src_temp2, m_src_temp5); /* col= [0][1][2][3]*/
/*row=3*/   m_src_temp3 = _mm_sign_epi32(m_src_temp3, m_sign1);     /* col= [0][1][2][3]*/

/* out = (WORD16) CLIP_S16(tmp);    */
            m_src_temp0 = _mm_packs_epi32(m_src_temp0, m_src_temp1);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp2, m_src_temp3);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp8 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp9 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp8))||!(_mm_test_all_ones (m_src_temp9)))
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
#if ZERO_ROW
                 /* zero row update */ /* temp_zero_row = ~zero_row */
                 temp_zero_row = (temp_zero_row) | (0xF << block_row);
                 // zero row can be optimized further. Now clearing the
                 // entire 4 bits corresponding to 4 rows of 4x4 block
                 // even if any 4x4 csbf is set
#endif
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
#if ZERO_ROW
        block_row += 4;
#endif
        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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
 * @returns  cbf
 * coded block flag
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

WORD32 ihevc_quant_flat_scale_mat_4x4_ttype1_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5, m_src_temp6, m_src_temp7, m_src_temp8;
    __m128i m_src_temp9, m_src_temp10, m_src_temp11, m_src_temp12;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    //__m128i m_sign;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;
    WORD32 temp;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col, temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 2;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_temp = _mm_set1_epi32(temp);

    m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((0)*src_strd)));     /* inp row =0*/
    m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((1)*src_strd)));     /* inp row =1*/
    m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((2)*src_strd)));     /* inp row =2*/
    m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((3)*src_strd)));     /* inp row= 3 */


    m_src_temp5 = _mm_abs_epi16(m_src_temp1);   /*  abs(row0)   */
    m_src_temp6 = _mm_abs_epi16(m_src_temp2);   /*  abs(row1)   */
    m_src_temp7 = _mm_abs_epi16(m_src_temp3);   /*  abs(row2)   */
    m_src_temp8 = _mm_abs_epi16(m_src_temp4);   /*  abs(row3)   */

    m_src_temp9  = _mm_cmpgt_epi16(m_zero, m_src_temp1);    /*  sign(row0)  */
    m_src_temp10 = _mm_cmpgt_epi16(m_zero, m_src_temp2);    /*  sign(row1)  */
    m_src_temp11 = _mm_cmpgt_epi16(m_zero, m_src_temp3);    /*  sign(row2)  */
    m_src_temp12 = _mm_cmpgt_epi16(m_zero, m_src_temp4);    /*  sign(row3)  */

    m_src_temp9  = _mm_slli_epi16(m_src_temp9, 1);
    m_src_temp10 = _mm_slli_epi16(m_src_temp10, 1);
    m_src_temp11 = _mm_slli_epi16(m_src_temp11, 1);
    m_src_temp12 = _mm_slli_epi16(m_src_temp12, 1);

    m_src_temp9  = _mm_add_epi16(m_src_temp9, m_one);   /*  sign(row0)  */
    m_src_temp10 = _mm_add_epi16(m_src_temp10, m_one);  /*  sign(row1)  */
    m_src_temp11 = _mm_add_epi16(m_src_temp11, m_one);  /*  sign(row2)  */
    m_src_temp12 = _mm_add_epi16(m_src_temp12, m_one);  /*  sign(row3)  */

    /*  Convert sign from 16 to 32 bits */
    m_sign0 = _mm_cvtepi16_epi32(m_src_temp9 );
    m_sign1 = _mm_cvtepi16_epi32(m_src_temp10);
    m_sign2 = _mm_cvtepi16_epi32(m_src_temp11);
    m_sign3 = _mm_cvtepi16_epi32(m_src_temp12);

/*******************************************/
/* converting 16 bit to 32-bit for abs(inp)*/
/*******************************************/
/*  row 0   */  m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp5);      /*col =0,1,2,3*/
/*  row 1   */  m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);      /*col =0,1,2,3*/
/*  row 2   */  m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp7);      /*col =0,1,2,3*/
/*  row 3   */  m_src_temp8 = _mm_cvtepi16_epi32(m_src_temp8);      /*col =0,1,2,3*/

/******************************/
/* tmp = tmp * (quant_coeff); */
/******************************/

/*row =0*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale);    /*  col =0, 1, 2, 3 */

/*row =1*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);    /*  col =0, 1, 2, 3 */

/*row =2*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale);    /*  col =0, 1, 2, 3 */

/*row =3*/  m_src_temp8 = _mm_mullo_epi32(m_src_temp8, m_scale);    /*  col =0, 1, 2, 3 */

/*********************************************************************/
/* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
/*********************************************************************/

/*row =0*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_temp);           /*  col =0, 1, 2, 3 */

/*row =1*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_temp);           /*  col =0, 1, 2, 3 */

/*row =2*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_temp);           /*  col =0, 1, 2, 3 */

/*row =3*/  m_src_temp8 = _mm_add_epi32(m_src_temp8, m_temp);           /*  col =0, 1, 2, 3 */

/***************************/
/* tmp = tmp >> q_bits;    */
/***************************/

/*row =0*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */

/*row =1*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);          /*  col =0, 1, 2, 3 */

/*row =2*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);      /*  col =0, 1, 2, 3 */

/*row =3*/  m_src_temp8 = _mm_srli_epi32(m_src_temp8, q_bits);      /*  col =0, 1, 2, 3 */


/**********************/
/* tmp = tmp * sign;  */
/**********************/

/*row=0*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign0);     /* col= [0][1][2][3]*/

/*row=1*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign1);         /* col= [0][1][2][3]*/

/*row=2*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign2); /* col= [0][1][2][3]*/

/*row=3*/   m_src_temp8 = _mm_sign_epi32(m_src_temp8, m_sign3);     /* col= [0][1][2][3]*/

/************************************/
/* out = (WORD16) CLIP_S16(tmp);    */
/************************************/

    m_src_temp0 = _mm_packs_epi32(m_src_temp5, m_src_temp6);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
    m_src_temp1 = _mm_packs_epi32(m_src_temp7, m_src_temp8);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

    m_src_temp5 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
    m_src_temp6 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

    m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);   /*  row =1 [0][1][2][3] */
    m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);   /*  row =3 [0][1][2][3] */

    _mm_storel_epi64 ((__m128i *)(pi2_dst+((0)*dst_strd)),m_src_temp0);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((1)*dst_strd)),m_src_temp2);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((2)*dst_strd)),m_src_temp1);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((3)*dst_strd)),m_src_temp3);

    if(!(_mm_test_all_ones (m_src_temp5))||!(_mm_test_all_ones (m_src_temp6)))
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

        *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}




/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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

WORD32 ihevc_quant_4x4_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15, m_src_temp16, m_src_temp17;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign, m_sign1;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 2;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier ;

    m_temp = _mm_cvtsi32_si128(q_add);
    m_temp = _mm_slli_epi64(m_temp, (q_bits - QUANT_ROUND_FACTOR_Q));
    m_temp = _mm_unpacklo_epi64(m_temp, m_temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((0)*trans_size)));    /* quant_coeff */
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/

            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((1)*trans_size)));    /* quant_coeff */
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/

            m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((2)*trans_size)));    /* quant_coeff */
            m_src_temp5 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/

            m_src_temp6 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((3)*trans_size)));    /* quant_coeff */
            m_src_temp7 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row=3 */

            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);              /* pi2_quant_coeff [0][1][2][3] row =0*/
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);              /* pi2_quant_coeff [0][1][2][3] row =1*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);              /* pi2_quant_coeff [0][1][2][3] row =2*/
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);              /* pi2_quant_coeff [0][1][2][3] row =3*/

            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_scale);        /* row =0 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_scale);        /* row =1 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);        /* row =2 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);        /* row =3 quant_coeff for col =0, 1, 2, 3 */

            m_src_temp1= _mm_unpacklo_epi64(m_src_temp1, m_src_temp3); /* row =0, 1*/ /*32-bit resolution*/
            m_src_temp3 = _mm_unpacklo_epi64(m_src_temp5, m_src_temp7); /* row =2, 3*/

            m_src_temp5 = _mm_abs_epi16(m_src_temp1);                   /* 16bit resolution*/
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);

/* row =0, 1, 2, 3*/
            m_sign = _mm_cmpgt_epi16(m_zero, m_src_temp1);
            m_sign1 = _mm_cmpgt_epi16(m_zero, m_src_temp3);

            m_sign = _mm_slli_epi16(m_sign, 1);
            m_sign1 = _mm_slli_epi16(m_sign1, 1);

            m_sign = _mm_add_epi16(m_sign, m_one);
            m_sign1 = _mm_add_epi16(m_sign1, m_one);    /* sign value */

/* converting 16 bit to 64-bit for abs(inp)*/
/* for row =0, 1*/
            m_src_temp8 = _mm_cvtepi16_epi64(m_src_temp5);      /*col =0,1*/
            m_src_temp9 = _mm_srli_si128(m_src_temp5, 4);
            m_src_temp9 = _mm_cvtepi16_epi64(m_src_temp9);      /*col =2,3*/

            m_src_temp10 = _mm_srli_si128(m_src_temp5, 8);
            m_src_temp10 = _mm_cvtepi16_epi64(m_src_temp10);    /*col =0,1*/
            m_src_temp11 = _mm_srli_si128(m_src_temp5, 12);
            m_src_temp11 = _mm_cvtepi16_epi64(m_src_temp11);    /*col =2,3*/

/* for row =2, 3*/
            m_src_temp12 = _mm_cvtepi16_epi64(m_src_temp7);     /*col =0,1*/
            m_src_temp13 = _mm_srli_si128(m_src_temp7, 4);
            m_src_temp13 = _mm_cvtepi16_epi64(m_src_temp13);    /*col =2,3*/

            m_src_temp14 = _mm_srli_si128(m_src_temp7, 8);
            m_src_temp14 = _mm_cvtepi16_epi64(m_src_temp14);    /*col =0,1*/
            m_src_temp15 = _mm_srli_si128(m_src_temp7, 12);
            m_src_temp15 = _mm_cvtepi16_epi64(m_src_temp15);    /*col =2,3*/

/* converting 32 bit to 64-bit for quant_coeff[]*/
/* for row =0, 1*/
            m_src_temp5 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =0,1*/
            m_src_temp0 = _mm_srli_si128(m_src_temp0, 4);
            m_src_temp0 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =2,3*/

            m_src_temp7 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =0,1*/
            m_src_temp2 = _mm_srli_si128(m_src_temp2, 4);
            m_src_temp2 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =2,3*/

/* for row =2, 3*/
            m_src_temp16 = _mm_cvtepi32_epi64(m_src_temp4);     /*col =0,1*/
            m_src_temp4 = _mm_srli_si128(m_src_temp4, 4);
            m_src_temp4 = _mm_cvtepi32_epi64(m_src_temp4);      /*col =2,3*/

            m_src_temp17 = _mm_cvtepi32_epi64(m_src_temp6);     /*col =0,1*/
            m_src_temp6 = _mm_srli_si128(m_src_temp6, 4);
            m_src_temp6 = _mm_cvtepi32_epi64(m_src_temp6);      /*col =2,3*/

/* tmp = tmp * (quant_coeff); */

/*row =0*/  m_src_temp8 = _mm_mul_epi32(m_src_temp8, m_src_temp5);      /*  col =0, 1 */
            m_src_temp9 = _mm_mul_epi32(m_src_temp9, m_src_temp0);      /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_mul_epi32(m_src_temp10, m_src_temp7);    /*  col =0, 1 */
            m_src_temp11 = _mm_mul_epi32(m_src_temp11, m_src_temp2);    /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_mul_epi32(m_src_temp12, m_src_temp16);   /*  col =0, 1 */
            m_src_temp13 = _mm_mul_epi32(m_src_temp13, m_src_temp4);    /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_mul_epi32(m_src_temp14, m_src_temp17);   /*  col =0, 1 */
            m_src_temp15 = _mm_mul_epi32(m_src_temp15, m_src_temp6);    /*  col =2, 3 */

/* tmp = tmp + (((LWORD64)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */

/*row =0*/  m_src_temp8 = _mm_add_epi64(m_src_temp8, m_temp);           /*  col =0, 1 */
            m_src_temp9 = _mm_add_epi64(m_src_temp9, m_temp);           /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_add_epi64(m_src_temp10, m_temp);         /*  col =0, 1 */
            m_src_temp11 = _mm_add_epi64(m_src_temp11, m_temp);         /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_add_epi64(m_src_temp12, m_temp);         /*  col =0, 1 */
            m_src_temp13 = _mm_add_epi64(m_src_temp13, m_temp);         /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_add_epi64(m_src_temp14, m_temp);         /*  col =0, 1 */
            m_src_temp15 = _mm_add_epi64(m_src_temp15, m_temp);         /*  col =2, 3 */

/* tmp = tmp >> q_bits;    */
/*row =0*/  m_src_temp8 = _mm_srli_epi64(m_src_temp8, q_bits);          /*  col =0, 1 */
            m_src_temp9 = _mm_srli_epi64(m_src_temp9, q_bits);          /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_srli_epi64(m_src_temp10, q_bits);        /*  col =0, 1 */
            m_src_temp11 = _mm_srli_epi64(m_src_temp11, q_bits);        /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_srli_epi64(m_src_temp12, q_bits);            /*  col =0, 1 */
            m_src_temp13 = _mm_srli_epi64(m_src_temp13, q_bits);            /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_srli_epi64(m_src_temp14, q_bits);        /*  col =0, 1 */
            m_src_temp15 = _mm_srli_epi64(m_src_temp15, q_bits);        /*  col =2, 3 */

/* unpack */

            m_src_temp8 = _mm_shuffle_epi32(m_src_temp8, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp9 = _mm_shuffle_epi32(m_src_temp9, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp10 = _mm_shuffle_epi32(m_src_temp10, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp11 = _mm_shuffle_epi32(m_src_temp11, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp12 = _mm_shuffle_epi32(m_src_temp12, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp13 = _mm_shuffle_epi32(m_src_temp13, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp14 = _mm_shuffle_epi32(m_src_temp14, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp15 = _mm_shuffle_epi32(m_src_temp15, _MM_SHUFFLE(0, 0, 2, 0) );


/*32-bit result*/
/*row=0*/   m_src_temp0 = _mm_unpacklo_epi64 (m_src_temp8,  m_src_temp9);/*col = 0, 1, 2, 3*/
/*row=1*/   m_src_temp1 = _mm_unpacklo_epi64 (m_src_temp10, m_src_temp11);/*col = 0, 1, 2, 3*/
/*row=2*/   m_src_temp2 = _mm_unpacklo_epi64 (m_src_temp12, m_src_temp13);/*col = 0, 1, 2, 3*/
/*row=3*/   m_src_temp3 = _mm_unpacklo_epi64 (m_src_temp14, m_src_temp15);/*col = 0, 1, 2, 3*/

/* sign value coverting 16-bit to 32-bit*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_sign);  /* sign col= [0][1][2][3]*/
            m_sign = _mm_srli_si128(m_sign, 8);
            m_sign = _mm_cvtepi16_epi32(m_sign);       /* sign col= [0][1][2][3]*/

            m_src_temp5 = _mm_cvtepi16_epi32(m_sign1);  /* sign col= [0][1][2][3]*/
            m_sign1 = _mm_srli_si128(m_sign1, 8);
            m_sign1 = _mm_cvtepi16_epi32(m_sign1);     /* sign col= [0][1][2][3]*/

/* tmp = tmp * sign;  */
/*row=0*/   m_src_temp0 = _mm_sign_epi32(m_src_temp0, m_src_temp4); /* col= [0][1][2][3]*/
/*row=1*/   m_src_temp1 = _mm_sign_epi32(m_src_temp1, m_sign);      /* col= [0][1][2][3]*/
/*row=2*/   m_src_temp2 = _mm_sign_epi32(m_src_temp2, m_src_temp5); /* col= [0][1][2][3]*/
/*row=3*/   m_src_temp3 = _mm_sign_epi32(m_src_temp3, m_sign1);     /* col= [0][1][2][3]*/

/* out = (WORD16) CLIP_S16(tmp);    */
            m_src_temp0 = _mm_packs_epi32(m_src_temp0, m_src_temp1);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp2, m_src_temp3);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp8 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp9 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp8))||!(_mm_test_all_ones (m_src_temp9)))
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
#if ZERO_ROW
                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
#endif
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
#if ZERO_ROW
        block_row += 4;
#endif
        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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


WORD32 ihevc_quant_flat_scale_mat_4x4_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5, m_src_temp6, m_src_temp7, m_src_temp8;
    __m128i m_src_temp9, m_src_temp10, m_src_temp11;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    //__m128i m_sign;
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;
    WORD32 temp;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col, temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 2;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_temp = _mm_set1_epi32(temp);

    m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((0)*src_strd)));     /* inp row =0*/
    m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((1)*src_strd)));     /* inp row =1*/
    m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((2)*src_strd)));     /* inp row =2*/
    m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+((3)*src_strd)));     /* inp row= 3 */


    m_src_temp4 = _mm_abs_epi16(m_src_temp0);   /*  abs(row0)   */
    m_src_temp5 = _mm_abs_epi16(m_src_temp1);   /*  abs(row1)   */
    m_src_temp6 = _mm_abs_epi16(m_src_temp2);   /*  abs(row2)   */
    m_src_temp7 = _mm_abs_epi16(m_src_temp3);   /*  abs(row3)   */

    m_src_temp8  = _mm_cmpgt_epi16(m_zero, m_src_temp0);    /*  sign(row0)  */
    m_src_temp9  = _mm_cmpgt_epi16(m_zero, m_src_temp1);    /*  sign(row1)  */
    m_src_temp10 = _mm_cmpgt_epi16(m_zero, m_src_temp2);    /*  sign(row2)  */
    m_src_temp11 = _mm_cmpgt_epi16(m_zero, m_src_temp3);    /*  sign(row3)  */

    m_src_temp8  = _mm_slli_epi16(m_src_temp8,  1);
    m_src_temp9  = _mm_slli_epi16(m_src_temp9 , 1);
    m_src_temp10 = _mm_slli_epi16(m_src_temp10, 1);
    m_src_temp11 = _mm_slli_epi16(m_src_temp11, 1);

    m_src_temp8  = _mm_add_epi16(m_src_temp8,  m_one);  /*  sign(row0)  */
    m_src_temp9  = _mm_add_epi16(m_src_temp9 , m_one);  /*  sign(row1)  */
    m_src_temp10 = _mm_add_epi16(m_src_temp10, m_one);  /*  sign(row2)  */
    m_src_temp11 = _mm_add_epi16(m_src_temp11, m_one);  /*  sign(row3)  */

    /*  Convert sign from 16 to 32 bits */
    m_sign0 = _mm_cvtepi16_epi32(m_src_temp8 );
    m_sign1 = _mm_cvtepi16_epi32(m_src_temp9 );
    m_sign2 = _mm_cvtepi16_epi32(m_src_temp10);
    m_sign3 = _mm_cvtepi16_epi32(m_src_temp11);

    /*******************************************/
    /* converting 16 bit to 32-bit for abs(inp)*/
    /*******************************************/
    /*  row 0   */  m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);      /*col =0,1,2,3*/
    /*  row 1   */  m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp5);      /*col =0,1,2,3*/
    /*  row 2   */  m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);      /*col =0,1,2,3*/
    /*  row 3   */  m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp7);      /*col =0,1,2,3*/

    /******************************/
    /* tmp = tmp * (quant_coeff); */
    /******************************/

    /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);    /*  col =0, 1, 2, 3 */

    /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale);    /*  col =0, 1, 2, 3 */

    /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);    /*  col =0, 1, 2, 3 */

    /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale);    /*  col =0, 1, 2, 3 */

    /*********************************************************************/
    /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
    /*********************************************************************/

    /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_temp);           /*  col =0, 1, 2, 3 */

    /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_temp);           /*  col =0, 1, 2, 3 */

    /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_temp);           /*  col =0, 1, 2, 3 */

    /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_temp);           /*  col =0, 1, 2, 3 */

    /***************************/
    /* tmp = tmp >> q_bits;    */
    /***************************/

    /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */

    /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */

    /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);      /*  col =0, 1, 2, 3 */

    /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);      /*  col =0, 1, 2, 3 */


    /**********************/
    /* tmp = tmp * sign;  */
    /**********************/

    /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);     /* col= [0][1][2][3]*/

    /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/

    /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2); /* col= [0][1][2][3]*/

    /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);     /* col= [0][1][2][3]*/

    /************************************/
    /* out = (WORD16) CLIP_S16(tmp);    */
    /************************************/

    m_src_temp0 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
    m_src_temp1 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

    *(csbf + block_col) = 0;

    m_src_temp5 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
    m_src_temp6 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

    m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);   /*  row =1 [0][1][2][3] */
    m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);   /*  row =3 [0][1][2][3] */

    _mm_storel_epi64 ((__m128i *)(pi2_dst+((0)*dst_strd)),m_src_temp0);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((1)*dst_strd)),m_src_temp2);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((2)*dst_strd)),m_src_temp1);
    _mm_storel_epi64 ((__m128i *)(pi2_dst+((3)*dst_strd)),m_src_temp3);

    if(!(_mm_test_all_ones (m_src_temp5))||!(_mm_test_all_ones (m_src_temp6)))
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

    *zero_col = ~temp_zero_col; //final zero_col storing
    *zero_row = ~temp_zero_row; //final zero_row storing
    return cbf;
}




/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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


WORD32 ihevc_quant_8x8_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15, m_src_temp16, m_src_temp17;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign, m_sign1;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 3;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier ;

    m_temp = _mm_cvtsi32_si128(q_add);
    m_temp = _mm_slli_epi64(m_temp, (q_bits - QUANT_ROUND_FACTOR_Q));
    m_temp = _mm_unpacklo_epi64(m_temp, m_temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((0)*trans_size)));    /* quant_coeff */
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/

            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((1)*trans_size)));    /* quant_coeff */
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/

            m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((2)*trans_size)));    /* quant_coeff */
            m_src_temp5 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/

            m_src_temp6 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((3)*trans_size)));    /* quant_coeff */
            m_src_temp7 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row=3 */

            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);              /* pi2_quant_coeff [0][1][2][3] row =0*/
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);              /* pi2_quant_coeff [0][1][2][3] row =1*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);              /* pi2_quant_coeff [0][1][2][3] row =2*/
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);              /* pi2_quant_coeff [0][1][2][3] row =3*/

            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_scale);        /* row =0 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_scale);        /* row =1 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);        /* row =2 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);        /* row =3 quant_coeff for col =0, 1, 2, 3 */

            m_src_temp1= _mm_unpacklo_epi64(m_src_temp1, m_src_temp3); /* row =0, 1*/ /*32-bit resolution*/
            m_src_temp3 = _mm_unpacklo_epi64(m_src_temp5, m_src_temp7); /* row =2, 3*/

            m_src_temp5 = _mm_abs_epi16(m_src_temp1);                   /* 16bit resolution*/
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);

/* row =0, 1, 2, 3*/
            m_sign = _mm_cmpgt_epi16(m_zero, m_src_temp1);
            m_sign1 = _mm_cmpgt_epi16(m_zero, m_src_temp3);

            m_sign = _mm_slli_epi16(m_sign, 1);
            m_sign1 = _mm_slli_epi16(m_sign1, 1);

            m_sign = _mm_add_epi16(m_sign, m_one);
            m_sign1 = _mm_add_epi16(m_sign1, m_one);    /* sign value */

/* converting 16 bit to 64-bit for abs(inp)*/
/* for row =0, 1*/
            m_src_temp8 = _mm_cvtepi16_epi64(m_src_temp5);      /*col =0,1*/
            m_src_temp9 = _mm_srli_si128(m_src_temp5, 4);
            m_src_temp9 = _mm_cvtepi16_epi64(m_src_temp9);      /*col =2,3*/

            m_src_temp10 = _mm_srli_si128(m_src_temp5, 8);
            m_src_temp10 = _mm_cvtepi16_epi64(m_src_temp10);    /*col =0,1*/
            m_src_temp11 = _mm_srli_si128(m_src_temp5, 12);
            m_src_temp11 = _mm_cvtepi16_epi64(m_src_temp11);    /*col =2,3*/

/* for row =2, 3*/
            m_src_temp12 = _mm_cvtepi16_epi64(m_src_temp7);     /*col =0,1*/
            m_src_temp13 = _mm_srli_si128(m_src_temp7, 4);
            m_src_temp13 = _mm_cvtepi16_epi64(m_src_temp13);    /*col =2,3*/

            m_src_temp14 = _mm_srli_si128(m_src_temp7, 8);
            m_src_temp14 = _mm_cvtepi16_epi64(m_src_temp14);    /*col =0,1*/
            m_src_temp15 = _mm_srli_si128(m_src_temp7, 12);
            m_src_temp15 = _mm_cvtepi16_epi64(m_src_temp15);    /*col =2,3*/

/* converting 32 bit to 64-bit for quant_coeff[]*/
/* for row =0, 1*/
            m_src_temp5 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =0,1*/
            m_src_temp0 = _mm_srli_si128(m_src_temp0, 4);
            m_src_temp0 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =2,3*/

            m_src_temp7 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =0,1*/
            m_src_temp2 = _mm_srli_si128(m_src_temp2, 4);
            m_src_temp2 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =2,3*/

/* for row =2, 3*/
            m_src_temp16 = _mm_cvtepi32_epi64(m_src_temp4);     /*col =0,1*/
            m_src_temp4 = _mm_srli_si128(m_src_temp4, 4);
            m_src_temp4 = _mm_cvtepi32_epi64(m_src_temp4);      /*col =2,3*/

            m_src_temp17 = _mm_cvtepi32_epi64(m_src_temp6);     /*col =0,1*/
            m_src_temp6 = _mm_srli_si128(m_src_temp6, 4);
            m_src_temp6 = _mm_cvtepi32_epi64(m_src_temp6);      /*col =2,3*/

/* tmp = tmp * (quant_coeff); */

/*row =0*/  m_src_temp8 = _mm_mul_epi32(m_src_temp8, m_src_temp5);      /*  col =0, 1 */
            m_src_temp9 = _mm_mul_epi32(m_src_temp9, m_src_temp0);      /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_mul_epi32(m_src_temp10, m_src_temp7);    /*  col =0, 1 */
            m_src_temp11 = _mm_mul_epi32(m_src_temp11, m_src_temp2);    /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_mul_epi32(m_src_temp12, m_src_temp16);   /*  col =0, 1 */
            m_src_temp13 = _mm_mul_epi32(m_src_temp13, m_src_temp4);    /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_mul_epi32(m_src_temp14, m_src_temp17);   /*  col =0, 1 */
            m_src_temp15 = _mm_mul_epi32(m_src_temp15, m_src_temp6);    /*  col =2, 3 */

/* tmp = tmp + (((LWORD64)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */

/*row =0*/  m_src_temp8 = _mm_add_epi64(m_src_temp8, m_temp);           /*  col =0, 1 */
            m_src_temp9 = _mm_add_epi64(m_src_temp9, m_temp);           /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_add_epi64(m_src_temp10, m_temp);         /*  col =0, 1 */
            m_src_temp11 = _mm_add_epi64(m_src_temp11, m_temp);         /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_add_epi64(m_src_temp12, m_temp);         /*  col =0, 1 */
            m_src_temp13 = _mm_add_epi64(m_src_temp13, m_temp);         /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_add_epi64(m_src_temp14, m_temp);         /*  col =0, 1 */
            m_src_temp15 = _mm_add_epi64(m_src_temp15, m_temp);         /*  col =2, 3 */

/* tmp = tmp >> q_bits;    */
/*row =0*/  m_src_temp8 = _mm_srli_epi64(m_src_temp8, q_bits);          /*  col =0, 1 */
            m_src_temp9 = _mm_srli_epi64(m_src_temp9, q_bits);          /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_srli_epi64(m_src_temp10, q_bits);        /*  col =0, 1 */
            m_src_temp11 = _mm_srli_epi64(m_src_temp11, q_bits);        /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_srli_epi64(m_src_temp12, q_bits);            /*  col =0, 1 */
            m_src_temp13 = _mm_srli_epi64(m_src_temp13, q_bits);            /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_srli_epi64(m_src_temp14, q_bits);        /*  col =0, 1 */
            m_src_temp15 = _mm_srli_epi64(m_src_temp15, q_bits);        /*  col =2, 3 */

/* unpack */

            m_src_temp8 = _mm_shuffle_epi32(m_src_temp8, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp9 = _mm_shuffle_epi32(m_src_temp9, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp10 = _mm_shuffle_epi32(m_src_temp10, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp11 = _mm_shuffle_epi32(m_src_temp11, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp12 = _mm_shuffle_epi32(m_src_temp12, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp13 = _mm_shuffle_epi32(m_src_temp13, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp14 = _mm_shuffle_epi32(m_src_temp14, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp15 = _mm_shuffle_epi32(m_src_temp15, _MM_SHUFFLE(0, 0, 2, 0) );


/*32-bit result*/
/*row=0*/   m_src_temp0 = _mm_unpacklo_epi64 (m_src_temp8,  m_src_temp9);/*col = 0, 1, 2, 3*/
/*row=1*/   m_src_temp1 = _mm_unpacklo_epi64 (m_src_temp10, m_src_temp11);/*col = 0, 1, 2, 3*/
/*row=2*/   m_src_temp2 = _mm_unpacklo_epi64 (m_src_temp12, m_src_temp13);/*col = 0, 1, 2, 3*/
/*row=3*/   m_src_temp3 = _mm_unpacklo_epi64 (m_src_temp14, m_src_temp15);/*col = 0, 1, 2, 3*/

/* sign value coverting 16-bit to 32-bit*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_sign);  /* sign col= [0][1][2][3]*/
            m_sign = _mm_srli_si128(m_sign, 8);
            m_sign = _mm_cvtepi16_epi32(m_sign);       /* sign col= [0][1][2][3]*/

            m_src_temp5 = _mm_cvtepi16_epi32(m_sign1);  /* sign col= [0][1][2][3]*/
            m_sign1 = _mm_srli_si128(m_sign1, 8);
            m_sign1 = _mm_cvtepi16_epi32(m_sign1);     /* sign col= [0][1][2][3]*/

/* tmp = tmp * sign;  */
/*row=0*/   m_src_temp0 = _mm_sign_epi32(m_src_temp0, m_src_temp4); /* col= [0][1][2][3]*/
/*row=1*/   m_src_temp1 = _mm_sign_epi32(m_src_temp1, m_sign);      /* col= [0][1][2][3]*/
/*row=2*/   m_src_temp2 = _mm_sign_epi32(m_src_temp2, m_src_temp5); /* col= [0][1][2][3]*/
/*row=3*/   m_src_temp3 = _mm_sign_epi32(m_src_temp3, m_sign1);     /* col= [0][1][2][3]*/

/* out = (WORD16) CLIP_S16(tmp);    */
            m_src_temp0 = _mm_packs_epi32(m_src_temp0, m_src_temp1);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp2, m_src_temp3);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp8 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp9 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp8))||!(_mm_test_all_ones (m_src_temp9)))
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
#if ZERO_ROW
                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
#endif

            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
#if ZERO_ROW
        block_row += 4;
#endif
        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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



WORD32 ihevc_quant_flat_scale_mat_8x8_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;
    WORD32 temp;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 3;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */; ;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_temp = _mm_set1_epi32(temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */


            m_src_temp4 = _mm_abs_epi16(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi16(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi16(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi16(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi16(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi16(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi16(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi16(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi16(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi16(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi16(m_src_temp11, 1);

            m_src_temp8  = _mm_add_epi16(m_src_temp8,  m_one);  /*  sign(row0)  */
            m_src_temp9  = _mm_add_epi16(m_src_temp9 , m_one);  /*  sign(row1)  */
            m_src_temp10 = _mm_add_epi16(m_src_temp10, m_one);  /*  sign(row2)  */
            m_src_temp11 = _mm_add_epi16(m_src_temp11, m_one);  /*  sign(row3)  */

            /*  Convert sign from 16 to 32 bits */
            m_sign0 = _mm_cvtepi16_epi32(m_src_temp8 );
            m_sign1 = _mm_cvtepi16_epi32(m_src_temp9 );
            m_sign2 = _mm_cvtepi16_epi32(m_src_temp10);
            m_sign3 = _mm_cvtepi16_epi32(m_src_temp11);

            /*******************************************/
            /* converting 16 bit to 32-bit for abs(inp)*/
            /*******************************************/
            /*  row 0   */  m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);      /*col =0,1,2,3*/
            /*  row 1   */  m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp5);      /*col =0,1,2,3*/
            /*  row 2   */  m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);      /*col =0,1,2,3*/
            /*  row 3   */  m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp7);      /*col =0,1,2,3*/

            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/

            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale);    /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/

            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_temp);           /*  col =0, 1, 2, 3 */

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/

            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);      /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);      /*  col =0, 1, 2, 3 */


            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/

            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);     /* col= [0][1][2][3]*/

            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/

            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2); /* col= [0][1][2][3]*/

            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);     /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/

            m_src_temp0 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp5 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp6 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);   /*  row =1 [0][1][2][3] */
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);   /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp5))||!(_mm_test_all_ones (m_src_temp6)))
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

                //break;
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
        block_row += 4;

        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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
WORD32 ihevc_quant_16x16_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15, m_src_temp16, m_src_temp17;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign, m_sign1;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 4;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier ;

    m_temp = _mm_cvtsi32_si128(q_add);
    m_temp = _mm_slli_epi64(m_temp, (q_bits - QUANT_ROUND_FACTOR_Q));
    m_temp = _mm_unpacklo_epi64(m_temp, m_temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((0)*trans_size)));    /* quant_coeff */
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/

            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((1)*trans_size)));    /* quant_coeff */
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/

            m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((2)*trans_size)));    /* quant_coeff */
            m_src_temp5 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/

            m_src_temp6 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((3)*trans_size)));    /* quant_coeff */
            m_src_temp7 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row=3 */

            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);              /* pi2_quant_coeff [0][1][2][3] row =0*/
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);              /* pi2_quant_coeff [0][1][2][3] row =1*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);              /* pi2_quant_coeff [0][1][2][3] row =2*/
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);              /* pi2_quant_coeff [0][1][2][3] row =3*/

            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_scale);        /* row =0 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_scale);        /* row =1 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);        /* row =2 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);        /* row =3 quant_coeff for col =0, 1, 2, 3 */

            m_src_temp1= _mm_unpacklo_epi64(m_src_temp1, m_src_temp3); /* row =0, 1*/ /*32-bit resolution*/
            m_src_temp3 = _mm_unpacklo_epi64(m_src_temp5, m_src_temp7); /* row =2, 3*/

            m_src_temp5 = _mm_abs_epi16(m_src_temp1);                   /* 16bit resolution*/
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);

/* row =0, 1, 2, 3*/
            m_sign = _mm_cmpgt_epi16(m_zero, m_src_temp1);
            m_sign1 = _mm_cmpgt_epi16(m_zero, m_src_temp3);

            m_sign = _mm_slli_epi16(m_sign, 1);
            m_sign1 = _mm_slli_epi16(m_sign1, 1);

            m_sign = _mm_add_epi16(m_sign, m_one);
            m_sign1 = _mm_add_epi16(m_sign1, m_one);    /* sign value */

/* converting 16 bit to 64-bit for abs(inp)*/
/* for row =0, 1*/
            m_src_temp8 = _mm_cvtepi16_epi64(m_src_temp5);      /*col =0,1*/
            m_src_temp9 = _mm_srli_si128(m_src_temp5, 4);
            m_src_temp9 = _mm_cvtepi16_epi64(m_src_temp9);      /*col =2,3*/

            m_src_temp10 = _mm_srli_si128(m_src_temp5, 8);
            m_src_temp10 = _mm_cvtepi16_epi64(m_src_temp10);    /*col =0,1*/
            m_src_temp11 = _mm_srli_si128(m_src_temp5, 12);
            m_src_temp11 = _mm_cvtepi16_epi64(m_src_temp11);    /*col =2,3*/

/* for row =2, 3*/
            m_src_temp12 = _mm_cvtepi16_epi64(m_src_temp7);     /*col =0,1*/
            m_src_temp13 = _mm_srli_si128(m_src_temp7, 4);
            m_src_temp13 = _mm_cvtepi16_epi64(m_src_temp13);    /*col =2,3*/

            m_src_temp14 = _mm_srli_si128(m_src_temp7, 8);
            m_src_temp14 = _mm_cvtepi16_epi64(m_src_temp14);    /*col =0,1*/
            m_src_temp15 = _mm_srli_si128(m_src_temp7, 12);
            m_src_temp15 = _mm_cvtepi16_epi64(m_src_temp15);    /*col =2,3*/

/* converting 32 bit to 64-bit for quant_coeff[]*/
/* for row =0, 1*/
            m_src_temp5 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =0,1*/
            m_src_temp0 = _mm_srli_si128(m_src_temp0, 4);
            m_src_temp0 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =2,3*/

            m_src_temp7 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =0,1*/
            m_src_temp2 = _mm_srli_si128(m_src_temp2, 4);
            m_src_temp2 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =2,3*/

/* for row =2, 3*/
            m_src_temp16 = _mm_cvtepi32_epi64(m_src_temp4);     /*col =0,1*/
            m_src_temp4 = _mm_srli_si128(m_src_temp4, 4);
            m_src_temp4 = _mm_cvtepi32_epi64(m_src_temp4);      /*col =2,3*/

            m_src_temp17 = _mm_cvtepi32_epi64(m_src_temp6);     /*col =0,1*/
            m_src_temp6 = _mm_srli_si128(m_src_temp6, 4);
            m_src_temp6 = _mm_cvtepi32_epi64(m_src_temp6);      /*col =2,3*/

/* tmp = tmp * (quant_coeff); */

/*row =0*/  m_src_temp8 = _mm_mul_epi32(m_src_temp8, m_src_temp5);      /*  col =0, 1 */
            m_src_temp9 = _mm_mul_epi32(m_src_temp9, m_src_temp0);      /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_mul_epi32(m_src_temp10, m_src_temp7);    /*  col =0, 1 */
            m_src_temp11 = _mm_mul_epi32(m_src_temp11, m_src_temp2);    /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_mul_epi32(m_src_temp12, m_src_temp16);   /*  col =0, 1 */
            m_src_temp13 = _mm_mul_epi32(m_src_temp13, m_src_temp4);    /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_mul_epi32(m_src_temp14, m_src_temp17);   /*  col =0, 1 */
            m_src_temp15 = _mm_mul_epi32(m_src_temp15, m_src_temp6);    /*  col =2, 3 */

/* tmp = tmp + (((LWORD64)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */

/*row =0*/  m_src_temp8 = _mm_add_epi64(m_src_temp8, m_temp);           /*  col =0, 1 */
            m_src_temp9 = _mm_add_epi64(m_src_temp9, m_temp);           /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_add_epi64(m_src_temp10, m_temp);         /*  col =0, 1 */
            m_src_temp11 = _mm_add_epi64(m_src_temp11, m_temp);         /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_add_epi64(m_src_temp12, m_temp);         /*  col =0, 1 */
            m_src_temp13 = _mm_add_epi64(m_src_temp13, m_temp);         /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_add_epi64(m_src_temp14, m_temp);         /*  col =0, 1 */
            m_src_temp15 = _mm_add_epi64(m_src_temp15, m_temp);         /*  col =2, 3 */

/* tmp = tmp >> q_bits;    */
/*row =0*/  m_src_temp8 = _mm_srli_epi64(m_src_temp8, q_bits);          /*  col =0, 1 */
            m_src_temp9 = _mm_srli_epi64(m_src_temp9, q_bits);          /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_srli_epi64(m_src_temp10, q_bits);        /*  col =0, 1 */
            m_src_temp11 = _mm_srli_epi64(m_src_temp11, q_bits);        /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_srli_epi64(m_src_temp12, q_bits);            /*  col =0, 1 */
            m_src_temp13 = _mm_srli_epi64(m_src_temp13, q_bits);            /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_srli_epi64(m_src_temp14, q_bits);        /*  col =0, 1 */
            m_src_temp15 = _mm_srli_epi64(m_src_temp15, q_bits);        /*  col =2, 3 */

/* unpack */

            m_src_temp8 = _mm_shuffle_epi32(m_src_temp8, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp9 = _mm_shuffle_epi32(m_src_temp9, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp10 = _mm_shuffle_epi32(m_src_temp10, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp11 = _mm_shuffle_epi32(m_src_temp11, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp12 = _mm_shuffle_epi32(m_src_temp12, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp13 = _mm_shuffle_epi32(m_src_temp13, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp14 = _mm_shuffle_epi32(m_src_temp14, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp15 = _mm_shuffle_epi32(m_src_temp15, _MM_SHUFFLE(0, 0, 2, 0) );


/*32-bit result*/
/*row=0*/   m_src_temp0 = _mm_unpacklo_epi64 (m_src_temp8,  m_src_temp9);/*col = 0, 1, 2, 3*/
/*row=1*/   m_src_temp1 = _mm_unpacklo_epi64 (m_src_temp10, m_src_temp11);/*col = 0, 1, 2, 3*/
/*row=2*/   m_src_temp2 = _mm_unpacklo_epi64 (m_src_temp12, m_src_temp13);/*col = 0, 1, 2, 3*/
/*row=3*/   m_src_temp3 = _mm_unpacklo_epi64 (m_src_temp14, m_src_temp15);/*col = 0, 1, 2, 3*/

/* sign value coverting 16-bit to 32-bit*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_sign);  /* sign col= [0][1][2][3]*/
            m_sign = _mm_srli_si128(m_sign, 8);
            m_sign = _mm_cvtepi16_epi32(m_sign);       /* sign col= [0][1][2][3]*/

            m_src_temp5 = _mm_cvtepi16_epi32(m_sign1);  /* sign col= [0][1][2][3]*/
            m_sign1 = _mm_srli_si128(m_sign1, 8);
            m_sign1 = _mm_cvtepi16_epi32(m_sign1);     /* sign col= [0][1][2][3]*/

/* tmp = tmp * sign;  */
/*row=0*/   m_src_temp0 = _mm_sign_epi32(m_src_temp0, m_src_temp4); /* col= [0][1][2][3]*/
/*row=1*/   m_src_temp1 = _mm_sign_epi32(m_src_temp1, m_sign);      /* col= [0][1][2][3]*/
/*row=2*/   m_src_temp2 = _mm_sign_epi32(m_src_temp2, m_src_temp5); /* col= [0][1][2][3]*/
/*row=3*/   m_src_temp3 = _mm_sign_epi32(m_src_temp3, m_sign1);     /* col= [0][1][2][3]*/

/* out = (WORD16) CLIP_S16(tmp);    */
            m_src_temp0 = _mm_packs_epi32(m_src_temp0, m_src_temp1);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp2, m_src_temp3);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp8 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp9 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp8))||!(_mm_test_all_ones (m_src_temp9)))
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
#if ZERO_ROW
                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
#endif

            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
#if ZERO_ROW
        block_row += 4;
#endif
        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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

WORD32 ihevc_quant_flat_scale_mat_16x16_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;
    WORD32 temp;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 4;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_temp = _mm_set1_epi32(temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */


            m_src_temp4 = _mm_abs_epi16(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi16(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi16(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi16(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi16(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi16(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi16(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi16(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi16(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi16(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi16(m_src_temp11, 1);

            m_src_temp8  = _mm_add_epi16(m_src_temp8,  m_one);  /*  sign(row0)  */
            m_src_temp9  = _mm_add_epi16(m_src_temp9 , m_one);  /*  sign(row1)  */
            m_src_temp10 = _mm_add_epi16(m_src_temp10, m_one);  /*  sign(row2)  */
            m_src_temp11 = _mm_add_epi16(m_src_temp11, m_one);  /*  sign(row3)  */

            /*  Convert sign from 16 to 32 bits */
            m_sign0 = _mm_cvtepi16_epi32(m_src_temp8 );
            m_sign1 = _mm_cvtepi16_epi32(m_src_temp9 );
            m_sign2 = _mm_cvtepi16_epi32(m_src_temp10);
            m_sign3 = _mm_cvtepi16_epi32(m_src_temp11);

            /*******************************************/
            /* converting 16 bit to 32-bit for abs(inp)*/
            /*******************************************/
            /*  row 0   */  m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);      /*col =0,1,2,3*/
            /*  row 1   */  m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp5);      /*col =0,1,2,3*/
            /*  row 2   */  m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);      /*col =0,1,2,3*/
            /*  row 3   */  m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp7);      /*col =0,1,2,3*/

            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/

            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale);    /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/

            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_temp);           /*  col =0, 1, 2, 3 */

            /***************************/
/* tmp = tmp >> q_bits;    */
            /***************************/

            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);      /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);      /*  col =0, 1, 2, 3 */


            /**********************/
/* tmp = tmp * sign;  */
            /**********************/

            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);     /* col= [0][1][2][3]*/

            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/

            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2); /* col= [0][1][2][3]*/

            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);     /* col= [0][1][2][3]*/

            /************************************/
/* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/

            m_src_temp0 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp5 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp6 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);   /*  row =1 [0][1][2][3] */
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);   /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp5))||!(_mm_test_all_ones (m_src_temp6)))
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

                //break;
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update


        block_col++;
        }
        block_col = 0;
        block_row += 4;

        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}



/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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
WORD32 ihevc_quant_32x32_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;
    __m128i m_src_temp12, m_src_temp13, m_src_temp14, m_src_temp15, m_src_temp16, m_src_temp17;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign, m_sign1;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 5;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier ;

    m_temp = _mm_cvtsi32_si128(q_add);
    m_temp = _mm_slli_epi64(m_temp, (q_bits - QUANT_ROUND_FACTOR_Q));
    m_temp = _mm_unpacklo_epi64(m_temp, m_temp);

    /*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((0)*trans_size)));    /* quant_coeff */
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/

            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((1)*trans_size)));    /* quant_coeff */
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/

            m_src_temp4 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((2)*trans_size)));    /* quant_coeff */
            m_src_temp5 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/

            m_src_temp6 = _mm_loadu_si128((__m128i *) (pi2_quant_coeff+j+((3)*trans_size)));    /* quant_coeff */
            m_src_temp7 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row=3 */

            m_src_temp0 = _mm_cvtepi16_epi32(m_src_temp0);              /* pi2_quant_coeff [0][1][2][3] row =0*/
            m_src_temp2 = _mm_cvtepi16_epi32(m_src_temp2);              /* pi2_quant_coeff [0][1][2][3] row =1*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);              /* pi2_quant_coeff [0][1][2][3] row =2*/
            m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);              /* pi2_quant_coeff [0][1][2][3] row =3*/

            m_src_temp0 = _mm_mullo_epi32(m_src_temp0, m_scale);        /* row =0 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp2 = _mm_mullo_epi32(m_src_temp2, m_scale);        /* row =1 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);        /* row =2 quant_coeff for col =0, 1, 2, 3 */
            m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);        /* row =3 quant_coeff for col =0, 1, 2, 3 */

            m_src_temp1= _mm_unpacklo_epi64(m_src_temp1, m_src_temp3); /* row =0, 1*/ /*32-bit resolution*/
            m_src_temp3 = _mm_unpacklo_epi64(m_src_temp5, m_src_temp7); /* row =2, 3*/

            m_src_temp5 = _mm_abs_epi16(m_src_temp1);                   /* 16bit resolution*/
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);

/* row =0, 1, 2, 3*/
            m_sign = _mm_cmpgt_epi16(m_zero, m_src_temp1);
            m_sign1 = _mm_cmpgt_epi16(m_zero, m_src_temp3);

            m_sign = _mm_slli_epi16(m_sign, 1);
            m_sign1 = _mm_slli_epi16(m_sign1, 1);

            m_sign = _mm_add_epi16(m_sign, m_one);
            m_sign1 = _mm_add_epi16(m_sign1, m_one);    /* sign value */

/* converting 16 bit to 64-bit for abs(inp)*/
/* for row =0, 1*/
            m_src_temp8 = _mm_cvtepi16_epi64(m_src_temp5);      /*col =0,1*/
            m_src_temp9 = _mm_srli_si128(m_src_temp5, 4);
            m_src_temp9 = _mm_cvtepi16_epi64(m_src_temp9);      /*col =2,3*/

            m_src_temp10 = _mm_srli_si128(m_src_temp5, 8);
            m_src_temp10 = _mm_cvtepi16_epi64(m_src_temp10);    /*col =0,1*/
            m_src_temp11 = _mm_srli_si128(m_src_temp5, 12);
            m_src_temp11 = _mm_cvtepi16_epi64(m_src_temp11);    /*col =2,3*/

/* for row =2, 3*/
            m_src_temp12 = _mm_cvtepi16_epi64(m_src_temp7);     /*col =0,1*/
            m_src_temp13 = _mm_srli_si128(m_src_temp7, 4);
            m_src_temp13 = _mm_cvtepi16_epi64(m_src_temp13);    /*col =2,3*/

            m_src_temp14 = _mm_srli_si128(m_src_temp7, 8);
            m_src_temp14 = _mm_cvtepi16_epi64(m_src_temp14);    /*col =0,1*/
            m_src_temp15 = _mm_srli_si128(m_src_temp7, 12);
            m_src_temp15 = _mm_cvtepi16_epi64(m_src_temp15);    /*col =2,3*/

/* converting 32 bit to 64-bit for quant_coeff[]*/
/* for row =0, 1*/
            m_src_temp5 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =0,1*/
            m_src_temp0 = _mm_srli_si128(m_src_temp0, 4);
            m_src_temp0 = _mm_cvtepi32_epi64(m_src_temp0);      /*col =2,3*/

            m_src_temp7 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =0,1*/
            m_src_temp2 = _mm_srli_si128(m_src_temp2, 4);
            m_src_temp2 = _mm_cvtepi32_epi64(m_src_temp2);      /*col =2,3*/

/* for row =2, 3*/
            m_src_temp16 = _mm_cvtepi32_epi64(m_src_temp4);     /*col =0,1*/
            m_src_temp4 = _mm_srli_si128(m_src_temp4, 4);
            m_src_temp4 = _mm_cvtepi32_epi64(m_src_temp4);      /*col =2,3*/

            m_src_temp17 = _mm_cvtepi32_epi64(m_src_temp6);     /*col =0,1*/
            m_src_temp6 = _mm_srli_si128(m_src_temp6, 4);
            m_src_temp6 = _mm_cvtepi32_epi64(m_src_temp6);      /*col =2,3*/

/* tmp = tmp * (quant_coeff); */

/*row =0*/  m_src_temp8 = _mm_mul_epi32(m_src_temp8, m_src_temp5);      /*  col =0, 1 */
            m_src_temp9 = _mm_mul_epi32(m_src_temp9, m_src_temp0);      /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_mul_epi32(m_src_temp10, m_src_temp7);    /*  col =0, 1 */
            m_src_temp11 = _mm_mul_epi32(m_src_temp11, m_src_temp2);    /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_mul_epi32(m_src_temp12, m_src_temp16);   /*  col =0, 1 */
            m_src_temp13 = _mm_mul_epi32(m_src_temp13, m_src_temp4);    /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_mul_epi32(m_src_temp14, m_src_temp17);   /*  col =0, 1 */
            m_src_temp15 = _mm_mul_epi32(m_src_temp15, m_src_temp6);    /*  col =2, 3 */

/* tmp = tmp + (((LWORD64)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */

/*row =0*/  m_src_temp8 = _mm_add_epi64(m_src_temp8, m_temp);           /*  col =0, 1 */
            m_src_temp9 = _mm_add_epi64(m_src_temp9, m_temp);           /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_add_epi64(m_src_temp10, m_temp);         /*  col =0, 1 */
            m_src_temp11 = _mm_add_epi64(m_src_temp11, m_temp);         /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_add_epi64(m_src_temp12, m_temp);         /*  col =0, 1 */
            m_src_temp13 = _mm_add_epi64(m_src_temp13, m_temp);         /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_add_epi64(m_src_temp14, m_temp);         /*  col =0, 1 */
            m_src_temp15 = _mm_add_epi64(m_src_temp15, m_temp);         /*  col =2, 3 */

/* tmp = tmp >> q_bits;    */
/*row =0*/  m_src_temp8 = _mm_srli_epi64(m_src_temp8, q_bits);          /*  col =0, 1 */
            m_src_temp9 = _mm_srli_epi64(m_src_temp9, q_bits);          /*  col =2, 3 */
/*row =1*/  m_src_temp10 = _mm_srli_epi64(m_src_temp10, q_bits);        /*  col =0, 1 */
            m_src_temp11 = _mm_srli_epi64(m_src_temp11, q_bits);        /*  col =2, 3 */
/*row =2*/  m_src_temp12 = _mm_srli_epi64(m_src_temp12, q_bits);            /*  col =0, 1 */
            m_src_temp13 = _mm_srli_epi64(m_src_temp13, q_bits);            /*  col =2, 3 */
/*row =3*/  m_src_temp14 = _mm_srli_epi64(m_src_temp14, q_bits);        /*  col =0, 1 */
            m_src_temp15 = _mm_srli_epi64(m_src_temp15, q_bits);        /*  col =2, 3 */

/* unpack */

            m_src_temp8 = _mm_shuffle_epi32(m_src_temp8, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp9 = _mm_shuffle_epi32(m_src_temp9, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp10 = _mm_shuffle_epi32(m_src_temp10, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp11 = _mm_shuffle_epi32(m_src_temp11, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp12 = _mm_shuffle_epi32(m_src_temp12, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp13 = _mm_shuffle_epi32(m_src_temp13, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp14 = _mm_shuffle_epi32(m_src_temp14, _MM_SHUFFLE(0, 0, 2, 0) );
            m_src_temp15 = _mm_shuffle_epi32(m_src_temp15, _MM_SHUFFLE(0, 0, 2, 0) );


/*32-bit result*/
/*row=0*/   m_src_temp0 = _mm_unpacklo_epi64 (m_src_temp8,  m_src_temp9);/*col = 0, 1, 2, 3*/
/*row=1*/   m_src_temp1 = _mm_unpacklo_epi64 (m_src_temp10, m_src_temp11);/*col = 0, 1, 2, 3*/
/*row=2*/   m_src_temp2 = _mm_unpacklo_epi64 (m_src_temp12, m_src_temp13);/*col = 0, 1, 2, 3*/
/*row=3*/   m_src_temp3 = _mm_unpacklo_epi64 (m_src_temp14, m_src_temp15);/*col = 0, 1, 2, 3*/

/* sign value coverting 16-bit to 32-bit*/
            m_src_temp4 = _mm_cvtepi16_epi32(m_sign);  /* sign col= [0][1][2][3]*/
            m_sign = _mm_srli_si128(m_sign, 8);
            m_sign = _mm_cvtepi16_epi32(m_sign);       /* sign col= [0][1][2][3]*/

            m_src_temp5 = _mm_cvtepi16_epi32(m_sign1);  /* sign col= [0][1][2][3]*/
            m_sign1 = _mm_srli_si128(m_sign1, 8);
            m_sign1 = _mm_cvtepi16_epi32(m_sign1);     /* sign col= [0][1][2][3]*/

/* tmp = tmp * sign;  */
/*row=0*/   m_src_temp0 = _mm_sign_epi32(m_src_temp0, m_src_temp4); /* col= [0][1][2][3]*/
/*row=1*/   m_src_temp1 = _mm_sign_epi32(m_src_temp1, m_sign);      /* col= [0][1][2][3]*/
/*row=2*/   m_src_temp2 = _mm_sign_epi32(m_src_temp2, m_src_temp5); /* col= [0][1][2][3]*/
/*row=3*/   m_src_temp3 = _mm_sign_epi32(m_src_temp3, m_sign1);     /* col= [0][1][2][3]*/

/* out = (WORD16) CLIP_S16(tmp);    */
            m_src_temp0 = _mm_packs_epi32(m_src_temp0, m_src_temp1);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp2, m_src_temp3);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp8 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp9 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp8))||!(_mm_test_all_ones (m_src_temp9)))
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
#if ZERO_ROW
                /* zero row update */ /* temp_zero_row = ~zero_row */
                temp_zero_row = (temp_zero_row) | (0xF << block_row);
                // zero row can be optimized further. Now clearing the
                // entire 4 bits corresponding to 4 rows of 4x4 block
                // even if any 4x4 csbf is set
#endif

            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;
#if ZERO_ROW
        block_row += 4;
#endif
        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}


/**
 *******************************************************************************
 *
 * @brief
 *  This function performs quantization
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
WORD32 ihevc_quant_flat_scale_mat_32x32_sse42(WORD16 *pi2_coeffs,
                         WORD16 *pi2_quant_coeff,
                         WORD16 *pi2_dst,
                         WORD32 qp_div,/* qpscaled / 6 */
                         WORD32 qp_rem,/* qpscaled % 6 */
                         WORD32 q_add,
                         WORD32 src_strd,
                         WORD32 dst_strd,
                         UWORD8 *csbf,
                         WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;
    WORD32 cbf = 0;

    __m128i m_src_temp0, m_src_temp1, m_src_temp2, m_src_temp3, m_src_temp4, m_src_temp5;
    __m128i m_src_temp6, m_src_temp7, m_src_temp8, m_src_temp9, m_src_temp10, m_src_temp11;

    __m128i m_scale = _mm_set1_epi32(g_ihevc_quant_scales[qp_rem]);
    __m128i m_temp ;
    __m128i m_one = _mm_set1_epi16(1);
    __m128i m_sign0, m_sign1, m_sign2, m_sign3;
    __m128i m_zero = _mm_set1_epi32(0);

    WORD16 quant_multiplier = 4 ; /* because quant_coeff are multiplied by 16. Instead of multiplying, we can reduce the division factor q_bits by 4 */
    WORD16 bit_depth = 8;
    WORD32  q_bits, transform_shift;
    WORD32 temp;

    WORD32 block_col=0;
    WORD32 block_row=0;
    WORD32 temp_zero_col,temp_zero_row;

    temp_zero_col = 0;
    temp_zero_row = 0;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */
    /* Quant initialization */
    log2_size = 5;

    /* q_bits and q_add calculation*/
    /* To be moved outside in neon. To be computer once per transform call */
    transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size;
    q_bits = QUANT_SHIFT + qp_div + transform_shift + SCALING_Q_SHIFT - quant_multiplier - FLAT_RESCALE_MAT_Q_SHIFT /* 2048 */;
    temp = (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q));

    m_temp = _mm_set1_epi32(temp);

/*  for loop starts from here */

    for(i = 0; i < trans_size; i+=4)
    {
        for(j = 0; j < trans_size; j +=4)
        {
            m_src_temp0 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((0)*src_strd)));       /* inp row =0*/
            m_src_temp1 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((1)*src_strd)));       /* inp row =1*/
            m_src_temp2 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((2)*src_strd)));       /* inp row =2*/
            m_src_temp3 = _mm_loadu_si128((__m128i *) (pi2_coeffs+j+((3)*src_strd)));       /* inp row= 3 */


            m_src_temp4 = _mm_abs_epi16(m_src_temp0);   /*  abs(row0)   */
            m_src_temp5 = _mm_abs_epi16(m_src_temp1);   /*  abs(row1)   */
            m_src_temp6 = _mm_abs_epi16(m_src_temp2);   /*  abs(row2)   */
            m_src_temp7 = _mm_abs_epi16(m_src_temp3);   /*  abs(row3)   */

            m_src_temp8  = _mm_cmpgt_epi16(m_zero, m_src_temp0);    /*  sign(row0)  */
            m_src_temp9  = _mm_cmpgt_epi16(m_zero, m_src_temp1);    /*  sign(row1)  */
            m_src_temp10 = _mm_cmpgt_epi16(m_zero, m_src_temp2);    /*  sign(row2)  */
            m_src_temp11 = _mm_cmpgt_epi16(m_zero, m_src_temp3);    /*  sign(row3)  */

            m_src_temp8  = _mm_slli_epi16(m_src_temp8,  1);
            m_src_temp9  = _mm_slli_epi16(m_src_temp9 , 1);
            m_src_temp10 = _mm_slli_epi16(m_src_temp10, 1);
            m_src_temp11 = _mm_slli_epi16(m_src_temp11, 1);

            m_src_temp8  = _mm_add_epi16(m_src_temp8,  m_one);  /*  sign(row0)  */
            m_src_temp9  = _mm_add_epi16(m_src_temp9 , m_one);  /*  sign(row1)  */
            m_src_temp10 = _mm_add_epi16(m_src_temp10, m_one);  /*  sign(row2)  */
            m_src_temp11 = _mm_add_epi16(m_src_temp11, m_one);  /*  sign(row3)  */

            /*  Convert sign from 16 to 32 bits */
            m_sign0 = _mm_cvtepi16_epi32(m_src_temp8 );
            m_sign1 = _mm_cvtepi16_epi32(m_src_temp9 );
            m_sign2 = _mm_cvtepi16_epi32(m_src_temp10);
            m_sign3 = _mm_cvtepi16_epi32(m_src_temp11);

            /*******************************************/
            /* converting 16 bit to 32-bit for abs(inp)*/
            /*******************************************/
            /*  row 0   */  m_src_temp4 = _mm_cvtepi16_epi32(m_src_temp4);      /*col =0,1,2,3*/
            /*  row 1   */  m_src_temp5 = _mm_cvtepi16_epi32(m_src_temp5);      /*col =0,1,2,3*/
            /*  row 2   */  m_src_temp6 = _mm_cvtepi16_epi32(m_src_temp6);      /*col =0,1,2,3*/
            /*  row 3   */  m_src_temp7 = _mm_cvtepi16_epi32(m_src_temp7);      /*col =0,1,2,3*/

            /******************************/
            /* tmp = tmp * (quant_coeff); */
            /******************************/

            /*row =0*/  m_src_temp4 = _mm_mullo_epi32(m_src_temp4, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_mullo_epi32(m_src_temp5, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_mullo_epi32(m_src_temp6, m_scale);    /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_mullo_epi32(m_src_temp7, m_scale);    /*  col =0, 1, 2, 3 */

            /*********************************************************************/
            /* tmp = tmp + (((WORD32)q_add) << (q_bits - QUANT_ROUND_FACTOR_Q)); */
            /*********************************************************************/

            /*row =0*/  m_src_temp4 = _mm_add_epi32(m_src_temp4, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_add_epi32(m_src_temp5, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_add_epi32(m_src_temp6, m_temp);           /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_add_epi32(m_src_temp7, m_temp);           /*  col =0, 1, 2, 3 */

            /***************************/
            /* tmp = tmp >> q_bits;    */
            /***************************/

            /*row =0*/  m_src_temp4 = _mm_srli_epi32(m_src_temp4, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =1*/  m_src_temp5 = _mm_srli_epi32(m_src_temp5, q_bits);          /*  col =0, 1, 2, 3 */

            /*row =2*/  m_src_temp6 = _mm_srli_epi32(m_src_temp6, q_bits);      /*  col =0, 1, 2, 3 */

            /*row =3*/  m_src_temp7 = _mm_srli_epi32(m_src_temp7, q_bits);      /*  col =0, 1, 2, 3 */


            /**********************/
            /* tmp = tmp * sign;  */
            /**********************/

            /*row=0*/   m_src_temp4 = _mm_sign_epi32(m_src_temp4, m_sign0);     /* col= [0][1][2][3]*/

            /*row=1*/   m_src_temp5 = _mm_sign_epi32(m_src_temp5, m_sign1);         /* col= [0][1][2][3]*/

            /*row=2*/   m_src_temp6 = _mm_sign_epi32(m_src_temp6, m_sign2); /* col= [0][1][2][3]*/

            /*row=3*/   m_src_temp7 = _mm_sign_epi32(m_src_temp7, m_sign3);     /* col= [0][1][2][3]*/

            /************************************/
            /* out = (WORD16) CLIP_S16(tmp);    */
            /************************************/

            m_src_temp0 = _mm_packs_epi32(m_src_temp4, m_src_temp5);    /* row = 0 out[0][1][2][3] row =1 [4][5][6][7] */
            m_src_temp1 = _mm_packs_epi32(m_src_temp6, m_src_temp7);    /* row = 2 out[0][1][2][3] row =3 [4][5][6][7] */

            *(csbf + block_col) = 0;

            m_src_temp5 = _mm_cmpeq_epi32 (m_zero, m_src_temp0);
            m_src_temp6 = _mm_cmpeq_epi32 (m_zero, m_src_temp1);

            m_src_temp2 = _mm_srli_si128(m_src_temp0, 8);   /*  row =1 [0][1][2][3] */
            m_src_temp3 = _mm_srli_si128(m_src_temp1, 8);   /*  row =3 [0][1][2][3] */

            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((0)*dst_strd)),m_src_temp0);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((1)*dst_strd)),m_src_temp2);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((2)*dst_strd)),m_src_temp1);
            _mm_storel_epi64 ((__m128i *)(pi2_dst+j+((3)*dst_strd)),m_src_temp3);

            if(!(_mm_test_all_ones (m_src_temp5))||!(_mm_test_all_ones (m_src_temp6)))
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

                //break;
            }

            cbf = cbf || (*(csbf + block_col)); // cbf update

        block_col++;
        }
        block_col = 0;

        block_row += 4;

        csbf += csbf_strd;
        pi2_coeffs += (4*src_strd);
        pi2_dst += (4*dst_strd);
        pi2_quant_coeff += (4*trans_size);

        }
        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
        return cbf;
}

