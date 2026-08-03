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
 *  ihevc_chroma_resi_trans_quant.c
 *
 * @brief
 *  Contains function definitions for residual, forward transform and
 * quantization of chroma interleaved data.
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *   - ihevc_chroma_resi_trans_quant_4x4()
 *   - ihevc_chroma_resi_trans_quant_8x8()
 *   - ihevc_chroma_resi_trans_quant_16x16()
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
#include "ihevc_chroma_resi_trans_quant.h"
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

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform on
 * input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction
 * followed by forward transform and  quantization
 *
 * @param[in] pu1_src
 *  Input 4x4 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing forward transform
 *  1st stage output
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
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
 * @param[in] pred_strd
 *  Prediction Stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_chroma_resi_trans_quant_4x4(UWORD8 *pu1_src,
                                       UWORD8 *pu1_pred,
                                       WORD16 *pi2_tmp,
                                       WORD16 *pi2_quant_coeff,
                                       WORD16 *pi2_dst,
                                       WORD32 qp_div,/* qpscaled / 6 */
                                       WORD32 qp_rem,/* qpscaled % 6 */
                                       WORD32 q_add,
                                       WORD32 src_strd,
                                       WORD32 pred_strd,
                                       WORD32 dst_strd,
                                       WORD32 *csbf)
{
    WORD32 i;
    WORD32 e[2], o[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD16 *pi2_tmp_orig;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    pi2_tmp_orig = pi2_tmp;
    trans_size = TRANS_SIZE_4;

    /* Residue + Forward Transform 1st stage */
    shift = 1; // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;
        /* e and o */
        resi_tmp_1 = pu1_src[0 * 2] - pu1_pred[0 * 2];
        resi_tmp_2 = pu1_src[3 * 2] - pu1_pred[3 * 2];
        e[0] = resi_tmp_1 + resi_tmp_2;
        o[0] = resi_tmp_1 - resi_tmp_2;

        resi_tmp_1 = pu1_src[1 * 2] - pu1_pred[1 * 2];
        resi_tmp_2 = pu1_src[2 * 2] - pu1_pred[2 * 2];
        e[1] = resi_tmp_1 + resi_tmp_2;
        o[1] = resi_tmp_1 - resi_tmp_2;

        pi2_tmp[0] = (g_ai2_ihevc_trans_4[0][0] * e[0]
                        + g_ai2_ihevc_trans_4[0][1] * e[1] + add) >> shift;
        pi2_tmp[2 * trans_size] = (g_ai2_ihevc_trans_4[2][0] * e[0]
                        + g_ai2_ihevc_trans_4[2][1] * e[1] + add) >> shift;
        pi2_tmp[trans_size] = (g_ai2_ihevc_trans_4[1][0] * o[0]
                        + g_ai2_ihevc_trans_4[1][1] * o[1] + add) >> shift;
        pi2_tmp[3 * trans_size] = (g_ai2_ihevc_trans_4[3][0] * o[0]
                        + g_ai2_ihevc_trans_4[3][1] * o[1] + add) >> shift;

        pu1_src += src_strd;
        pu1_pred += pred_strd;
        pi2_tmp++;
    }

    pi2_tmp = pi2_tmp_orig;
    /* Forward transform 2nd stage and Quantization */
    {
        WORD32 log2_trans_size;

        /* Trans initializations */
        shift = 8; // log2(iHeight) + 6
        add = 1 << (shift - 1);

        /* Quant initialization */
        log2_trans_size = 0 + 2;

        for(i = 0; i < trans_size; i++)
        {
            WORD32 trans_out;
            /* e and o */
            e[0] = pi2_tmp[0] + pi2_tmp[3];
            o[0] = pi2_tmp[0] - pi2_tmp[3];
            e[1] = pi2_tmp[1] + pi2_tmp[2];
            o[1] = pi2_tmp[1] - pi2_tmp[2];

            trans_out = (g_ai2_ihevc_trans_4[0][0] * e[0]
                            + g_ai2_ihevc_trans_4[0][1] * e[1] + add) >> shift;
            QUANT(pi2_dst[0*dst_strd], trans_out,
                  pi2_quant_coeff[0*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_4[2][0] * e[0]
                            + g_ai2_ihevc_trans_4[2][1] * e[1] + add) >> shift;
            QUANT(pi2_dst[2*dst_strd], trans_out,
                  pi2_quant_coeff[2*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_4[1][0] * o[0]
                            + g_ai2_ihevc_trans_4[1][1] * o[1] + add) >> shift;
            QUANT(pi2_dst[1*dst_strd], trans_out,
                  pi2_quant_coeff[1*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_4[3][0] * o[0]
                            + g_ai2_ihevc_trans_4[3][1] * o[1] + add) >> shift;
            QUANT(pi2_dst[3*dst_strd], trans_out,
                  pi2_quant_coeff[3*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            pi2_tmp += trans_size;
            pi2_quant_coeff++;
            pi2_dst++;
        }
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * dst_strd + col];
                val = (val == 0);
                *csbf = *csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    csbf++;
                    *csbf = 0;
                    shift = 31;
                }
            }

        }
    }
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
 * followed by forward transform and  quantization
 *
 * @param[in] pu1_src
 *  Input 8x8 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary 8x8 buffer for storing forward transform
 *  1st stage output
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
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
 *  Prediction Stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_chroma_resi_trans_quant_8x8(UWORD8 *pu1_src,
                                       UWORD8 *pu1_pred,
                                       WORD16 *pi2_tmp,
                                       WORD16 *pi2_quant_coeff,
                                       WORD16 *pi2_dst,
                                       WORD32 qp_div,/* qpscaled / 6 */
                                       WORD32 qp_rem,/* qpscaled % 6 */
                                       WORD32 q_add,
                                       WORD32 src_strd,
                                       WORD32 pred_strd,
                                       WORD32 dst_strd,
                                       WORD32 *csbf)
{
    WORD32 i, k;
    WORD32 e[4], o[4];
    WORD32 ee[2], eo[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD16 *pi2_tmp_orig;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    pi2_tmp_orig = pi2_tmp;
    trans_size = TRANS_SIZE_8;
    /* Residue + Forward Transform 1st stage */
    shift = 2; // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;
        /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
        /* e and o*/
        for(k = 0; k < 4; k++)
        {
            resi_tmp_1 = pu1_src[k * 2] - pu1_pred[k * 2];
            resi_tmp_2 = pu1_src[(7 - k) * 2] - pu1_pred[(7 - k) * 2];
            e[k] = resi_tmp_1 + resi_tmp_2;
            o[k] = resi_tmp_1 - resi_tmp_2;
        }
        /* ee and eo */
        ee[0] = e[0] + e[3];
        eo[0] = e[0] - e[3];
        ee[1] = e[1] + e[2];
        eo[1] = e[1] - e[2];

        pi2_tmp[0] = (g_ai2_ihevc_trans_8[0][0] * ee[0]
                        + g_ai2_ihevc_trans_8[0][1] * ee[1] + add) >> shift;
        pi2_tmp[4 * trans_size] = (g_ai2_ihevc_trans_8[4][0] * ee[0]
                        + g_ai2_ihevc_trans_8[4][1] * ee[1] + add) >> shift;
        pi2_tmp[2 * trans_size] = (g_ai2_ihevc_trans_8[2][0] * eo[0]
                        + g_ai2_ihevc_trans_8[2][1] * eo[1] + add) >> shift;
        pi2_tmp[6 * trans_size] = (g_ai2_ihevc_trans_8[6][0] * eo[0]
                        + g_ai2_ihevc_trans_8[6][1] * eo[1] + add) >> shift;

        pi2_tmp[trans_size] = (g_ai2_ihevc_trans_8[1][0] * o[0]
                        + g_ai2_ihevc_trans_8[1][1] * o[1]
                        + g_ai2_ihevc_trans_8[1][2] * o[2]
                        + g_ai2_ihevc_trans_8[1][3] * o[3] + add) >> shift;
        pi2_tmp[3 * trans_size] = (g_ai2_ihevc_trans_8[3][0] * o[0]
                        + g_ai2_ihevc_trans_8[3][1] * o[1]
                        + g_ai2_ihevc_trans_8[3][2] * o[2]
                        + g_ai2_ihevc_trans_8[3][3] * o[3] + add) >> shift;
        pi2_tmp[5 * trans_size] = (g_ai2_ihevc_trans_8[5][0] * o[0]
                        + g_ai2_ihevc_trans_8[5][1] * o[1]
                        + g_ai2_ihevc_trans_8[5][2] * o[2]
                        + g_ai2_ihevc_trans_8[5][3] * o[3] + add) >> shift;
        pi2_tmp[7 * trans_size] = (g_ai2_ihevc_trans_8[7][0] * o[0]
                        + g_ai2_ihevc_trans_8[7][1] * o[1]
                        + g_ai2_ihevc_trans_8[7][2] * o[2]
                        + g_ai2_ihevc_trans_8[7][3] * o[3] + add) >> shift;

        pu1_src += src_strd;
        pu1_pred += pred_strd;
        pi2_tmp++;
    }

    pi2_tmp = pi2_tmp_orig;
    /* Forward transform 2nd stage and Quantization */
    {
        WORD32 log2_trans_size;

        /* Trans initializations */
        shift = 9; // log2(iHeight) + 6
        add = 1 << (shift - 1);
        /* Quant initialization */
        log2_trans_size = 3;

        for(i = 0; i < trans_size; i++)
        {
            WORD32 trans_out;
            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            /* e and o*/
            for(k = 0; k < 4; k++)
            {
                e[k] = pi2_tmp[k] + pi2_tmp[7 - k];
                o[k] = pi2_tmp[k] - pi2_tmp[7 - k];
            }
            /* ee and eo */
            ee[0] = e[0] + e[3];
            eo[0] = e[0] - e[3];
            ee[1] = e[1] + e[2];
            eo[1] = e[1] - e[2];

            trans_out = (g_ai2_ihevc_trans_8[0][0] * ee[0]
                            + g_ai2_ihevc_trans_8[0][1] * ee[1] + add) >> shift;
            QUANT(pi2_dst[0*dst_strd], trans_out,
                  pi2_quant_coeff[0*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[4][0] * ee[0]
                            + g_ai2_ihevc_trans_8[4][1] * ee[1] + add) >> shift;

            QUANT(pi2_dst[4*dst_strd], trans_out,
                  pi2_quant_coeff[4*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[2][0] * eo[0]
                            + g_ai2_ihevc_trans_8[2][1] * eo[1] + add) >> shift;

            QUANT(pi2_dst[2*dst_strd], trans_out,
                  pi2_quant_coeff[2*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[6][0] * eo[0]
                            + g_ai2_ihevc_trans_8[6][1] * eo[1] + add) >> shift;
            QUANT(pi2_dst[6*dst_strd], trans_out,
                  pi2_quant_coeff[6*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[1][0] * o[0]
                            + g_ai2_ihevc_trans_8[1][1] * o[1]
                            + g_ai2_ihevc_trans_8[1][2] * o[2]
                            + g_ai2_ihevc_trans_8[1][3] * o[3] + add) >> shift;
            QUANT(pi2_dst[1*dst_strd], trans_out,
                  pi2_quant_coeff[1*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[3][0] * o[0]
                            + g_ai2_ihevc_trans_8[3][1] * o[1]
                            + g_ai2_ihevc_trans_8[3][2] * o[2]
                            + g_ai2_ihevc_trans_8[3][3] * o[3] + add) >> shift;
            QUANT(pi2_dst[3*dst_strd], trans_out,
                  pi2_quant_coeff[3*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[5][0] * o[0]
                            + g_ai2_ihevc_trans_8[5][1] * o[1]
                            + g_ai2_ihevc_trans_8[5][2] * o[2]
                            + g_ai2_ihevc_trans_8[5][3] * o[3] + add) >> shift;
            QUANT(pi2_dst[5*dst_strd], trans_out,
                  pi2_quant_coeff[5*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_8[7][0] * o[0]
                            + g_ai2_ihevc_trans_8[7][1] * o[1]
                            + g_ai2_ihevc_trans_8[7][2] * o[2]
                            + g_ai2_ihevc_trans_8[7][3] * o[3] + add) >> shift;
            QUANT(pi2_dst[7*dst_strd], trans_out,
                  pi2_quant_coeff[7*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);
            pi2_tmp += trans_size;
            pi2_quant_coeff++;
            pi2_dst++;
        }
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * dst_strd + col];
                val = (val == 0);
                *csbf = *csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    csbf++;
                    *csbf = 0;
                    shift = 31;
                }
            }

        }
    }
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
 * followed by forward transform and  quantization
 *
 * @param[in] pu1_src
 *  Input 16x16 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary 16x16 buffer for storing forward transform
 *  1st stage output
 *
 * @param[in] pi2_dequant_coeff
 *  Dequant Coeffs
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
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
 *  Prediction Stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_chroma_resi_trans_quant_16x16(UWORD8 *pu1_src,
                                         UWORD8 *pu1_pred,
                                         WORD16 *pi2_tmp,
                                         WORD16 *pi2_quant_coeff,
                                         WORD16 *pi2_dst,
                                         WORD32 qp_div,/* qpscaled / 6 */
                                         WORD32 qp_rem,/* qpscaled % 6 */
                                         WORD32 q_add,
                                         WORD32 src_strd,
                                         WORD32 pred_strd,
                                         WORD32 dst_strd,
                                         WORD32 *csbf)
{
    WORD32 i, k;
    WORD32 e[8], o[8];
    WORD32 ee[4], eo[4];
    WORD32 eee[2], eeo[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD16 *pi2_tmp_orig;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    pi2_tmp_orig = pi2_tmp;
    trans_size = TRANS_SIZE_16;
    /* Residue + Forward Transform 1st stage */
    shift = 3; // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;
        /* e and o*/
        for(k = 0; k < 8; k++)
        {
            resi_tmp_1 = pu1_src[k * 2] - pu1_pred[k * 2];
            resi_tmp_2 = pu1_src[(15 - k) * 2] - pu1_pred[(15 - k) * 2];
            e[k] = resi_tmp_1 + resi_tmp_2;
            o[k] = resi_tmp_1 - resi_tmp_2;
        }
        /* ee and eo */
        for(k = 0; k < 4; k++)
        {
            ee[k] = e[k] + e[7 - k];
            eo[k] = e[k] - e[7 - k];
        }
        /* eee and eeo */
        eee[0] = ee[0] + ee[3];
        eeo[0] = ee[0] - ee[3];
        eee[1] = ee[1] + ee[2];
        eeo[1] = ee[1] - ee[2];

        pi2_tmp[0] = (g_ai2_ihevc_trans_16[0][0] * eee[0]
                        + g_ai2_ihevc_trans_16[0][1] * eee[1] + add) >> shift;
        pi2_tmp[8 * trans_size] = (g_ai2_ihevc_trans_16[8][0] * eee[0]
                        + g_ai2_ihevc_trans_16[8][1] * eee[1] + add) >> shift;
        pi2_tmp[4 * trans_size] = (g_ai2_ihevc_trans_16[4][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[4][1] * eeo[1] + add) >> shift;
        pi2_tmp[12 * trans_size] = (g_ai2_ihevc_trans_16[12][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[12][1] * eeo[1] + add) >> shift;

        for(k = 2; k < 16; k += 4)
        {
            pi2_tmp[k * trans_size] = (g_ai2_ihevc_trans_16[k][0] * eo[0]
                            + g_ai2_ihevc_trans_16[k][1] * eo[1]
                            + g_ai2_ihevc_trans_16[k][2] * eo[2]
                            + g_ai2_ihevc_trans_16[k][3] * eo[3] + add)
                            >> shift;
        }

        for(k = 1; k < 16; k += 2)
        {
            pi2_tmp[k * trans_size] = (g_ai2_ihevc_trans_16[k][0] * o[0]
                            + g_ai2_ihevc_trans_16[k][1] * o[1]
                            + g_ai2_ihevc_trans_16[k][2] * o[2]
                            + g_ai2_ihevc_trans_16[k][3] * o[3]
                            + g_ai2_ihevc_trans_16[k][4] * o[4]
                            + g_ai2_ihevc_trans_16[k][5] * o[5]
                            + g_ai2_ihevc_trans_16[k][6] * o[6]
                            + g_ai2_ihevc_trans_16[k][7] * o[7] + add) >> shift;
        }
        pu1_src += src_strd;
        pu1_pred += pred_strd;
        pi2_tmp++;
    }

    pi2_tmp = pi2_tmp_orig;
    /* Forward transform 2nd stage and Quantization */
    {
        WORD32 log2_trans_size;

        /* Trans initializations */
        shift = 10; // log2(iHeight) + 6
        add = 1 << (shift - 1);

        /* Quant initialization */
        log2_trans_size = 4;

        for(i = 0; i < TRANS_SIZE_16; i++)
        {
            WORD32 trans_out;
            /* e and o*/
            for(k = 0; k < 8; k++)
            {
                e[k] = pi2_tmp[k] + pi2_tmp[15 - k];
                o[k] = pi2_tmp[k] - pi2_tmp[15 - k];
            }
            /* ee and eo */
            for(k = 0; k < 4; k++)
            {
                ee[k] = e[k] + e[7 - k];
                eo[k] = e[k] - e[7 - k];
            }
            /* eee and eeo */
            eee[0] = ee[0] + ee[3];
            eeo[0] = ee[0] - ee[3];
            eee[1] = ee[1] + ee[2];
            eeo[1] = ee[1] - ee[2];

            trans_out = (g_ai2_ihevc_trans_16[0][0] * eee[0]
                            + g_ai2_ihevc_trans_16[0][1] * eee[1] + add)
                            >> shift;
            QUANT(pi2_dst[0*dst_strd], trans_out,
                  pi2_quant_coeff[0*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_16[8][0] * eee[0]
                            + g_ai2_ihevc_trans_16[8][1] * eee[1] + add)
                            >> shift;
            QUANT(pi2_dst[8*dst_strd], trans_out,
                  pi2_quant_coeff[8*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_16[4][0] * eeo[0]
                            + g_ai2_ihevc_trans_16[4][1] * eeo[1] + add)
                            >> shift;
            QUANT(pi2_dst[4*dst_strd], trans_out,
                  pi2_quant_coeff[4*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            trans_out = (g_ai2_ihevc_trans_16[12][0] * eeo[0]
                            + g_ai2_ihevc_trans_16[12][1] * eeo[1] + add)
                            >> shift;
            QUANT(pi2_dst[12*dst_strd], trans_out,
                  pi2_quant_coeff[12*trans_size]*g_ihevc_quant_scales[qp_rem],
                  qp_div, log2_trans_size, q_add);

            for(k = 2; k < 16; k += 4)
            {
                trans_out = (g_ai2_ihevc_trans_16[k][0] * eo[0]
                                + g_ai2_ihevc_trans_16[k][1] * eo[1]
                                + g_ai2_ihevc_trans_16[k][2] * eo[2]
                                + g_ai2_ihevc_trans_16[k][3] * eo[3] + add)
                                >> shift;
                QUANT(pi2_dst[k*dst_strd],
                      trans_out,
                      pi2_quant_coeff[k*trans_size]*g_ihevc_quant_scales[qp_rem],
                      qp_div, log2_trans_size, q_add);

            }

            for(k = 1; k < 16; k += 2)
            {
                trans_out = (g_ai2_ihevc_trans_16[k][0] * o[0]
                                + g_ai2_ihevc_trans_16[k][1] * o[1]
                                + g_ai2_ihevc_trans_16[k][2] * o[2]
                                + g_ai2_ihevc_trans_16[k][3] * o[3]
                                + g_ai2_ihevc_trans_16[k][4] * o[4]
                                + g_ai2_ihevc_trans_16[k][5] * o[5]
                                + g_ai2_ihevc_trans_16[k][6] * o[6]
                                + g_ai2_ihevc_trans_16[k][7] * o[7] + add)
                                >> shift;
                QUANT(pi2_dst[k*dst_strd],
                      trans_out,
                      pi2_quant_coeff[k*trans_size]*g_ihevc_quant_scales[qp_rem],
                      qp_div, log2_trans_size, q_add);
            }
            pi2_tmp += trans_size;
            pi2_quant_coeff++;
            pi2_dst++;
        }
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * dst_strd + col];
                val = (val == 0);
                *csbf = *csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    csbf++;
                    *csbf = 0;
                    shift = 31;
                }
            }

        }
    }
}

