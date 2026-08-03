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


WORD32 ihevc_quant_4x4_ttype1(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */ /* temp_zero_row = ~zero_row */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_4x4(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now setting the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_8x8(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 3;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }

    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_16x16(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 4;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_32x32(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 5;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_4x4_ttype1(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */ /* temp_zero_row = ~zero_row */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_4x4(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now setting the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_8x8(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 3;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }

    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_16x16(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 4;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_32x32(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 5;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);    */
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_4x4_ttype1_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT(pi2_dst[j], i4_temp,
                      pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */ /* temp_zero_row = ~zero_row */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_4x4_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT(pi2_dst[j], i4_temp,
                      pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now setting the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_8x8_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 3;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT(pi2_dst[j], i4_temp,
                      pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }

    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_16x16_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 4;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT(pi2_dst[j], i4_temp,
                      pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_32x32_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 5;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT(pi2_dst[j], i4_temp,
                      pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_4x4_ttype1_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT_NO_WEIGHTMAT(pi2_dst[j], i4_temp,
                      g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */ /* temp_zero_row = ~zero_row */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_4x4_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT_NO_WEIGHTMAT(pi2_dst[j], i4_temp,
                      g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
                    }
                    if(*(csbf + block_col) == 1)
                    {
                        /* zero_col update *//* temp_zero_col = ~zero_col */
                        temp_zero_col = (temp_zero_col) | (0xF << block_col * 4);
                        // zero col can be optimized further. Now clearing the
                        // entire 4 bits corresponding to 4 colums of 4x4 block
                        // even if any 4x4 csbf is set

                        /* zero row update */
                        temp_zero_row = (temp_zero_row) | (0xF << block_row);
                        // zero row can be optimized further. Now setting the
                        // entire 4 bits corresponding to 4 rows of 4x4 block
                        // even if any 4x4 csbf is set

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_8x8_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 3;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT_NO_WEIGHTMAT(pi2_dst[j], i4_temp,
                      g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }

    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_16x16_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 4;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);*/
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT_NO_WEIGHTMAT(pi2_dst[j], i4_temp,
                      g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

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


WORD32 ihevc_quant_flat_scale_mat_32x32_rdoq(WORD16 *pi2_coeffs,
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

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_32;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 5;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 i4_temp = pi2_coeffs[j];
            /*QUANT(pi2_dst[j], pi2_coeffs[j],
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);    */
            /* modified by 1028 */
            QUANT_NO_WEIGHTMAT(pi2_dst[j], pi2_coeffs[j],
                  g_ihevc_quant_scales[qp_rem], qp_div,
                  log2_size, q_add);
            if(abs(pi2_dst[j]) > 1)
            {
                QUANT_NO_WEIGHTMAT(pi2_dst[j], i4_temp,
                      g_ihevc_quant_scales[qp_rem], qp_div,
                      log2_size, ((1 << QUANT_ROUND_FACTOR_Q)/2));
            }
        }

        pi2_coeffs += src_strd;
        pi2_dst += dst_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 temp_zero_col = 0;
        WORD32 temp_zero_row = 0;

        pi2_dst = pi2_dst_orig;

        for(block_row = 0; block_row < trans_size; block_row += 4)
        {
            //block_col is incrementing by 1 for easy update of csbf pointer
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * dst_strd + block_col * 4;
                *(csbf + block_col) = 0;

                for(row = 0; row < 4; row++)
                {
                    for(col = 0; col < 4; col++)
                    {
                        if(pi2_block[row * dst_strd + col] != 0)
                        {
                            *(csbf + block_col) = 1;
                            break;
                        }
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

                        break;
                    }
                }

                cbf = cbf || (*(csbf + block_col)); // cbf update
            }
            csbf += csbf_strd;
        }

        *zero_col = ~temp_zero_col; //final zero_col storing
        *zero_row = ~temp_zero_row; //final zero_row storing
    }

    return cbf;
}
