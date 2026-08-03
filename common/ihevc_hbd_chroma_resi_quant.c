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
 *  ihevc_chroma_resi_quant.c
 *
 * @brief
 *  Contains function definitions for residual  and quantization  of chroma
 * interleaved data.
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *   - ihevc_chroma_resi_quant_4x4()
 *   - ihevc_chroma_resi_quant_8x8()
 *   - ihevc_chroma_resi_quant_16x16()
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
#include "ihevc_chroma_resi_quant.h"
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
 *  This function performs residue and quantization  on input pixels
 *
 * @par Description:
 *  Calculates the residue by subtracting the prediction  from input pixels
 * and does quantization on output
 *
 * @param[in] pu2_src
 *  Input 4x4 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_quant_coeff
 *  Scaling matrix
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] pi4_csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_resi_quant_4x4(UWORD16 *pu2_src,
                                 UWORD16 *pu2_pred,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 i4_qp_div,/* qpscaled / 6 */
                                 WORD32 i4_qp_rem,/* qpscaled % 6 */
                                 WORD32 i4_q_add,
                                 WORD32 i4_src_strd,
                                 WORD32 i4_pred_strd,
                                 WORD32 i4_dst_strd,
                                 WORD32 *pi4_csbf,
                                 UWORD8 u1_bit_depth)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_4;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 2;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 residue;

            residue = (pu2_src[j * 2] - pu2_pred[j * 2]);
            QUANT_HBD(pi2_dst[j], residue,
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[i4_qp_rem], i4_qp_div,
                  log2_size, i4_q_add, u1_bit_depth);
        }

        pu2_src += i4_src_strd;
        pi2_dst += i4_dst_strd;
        pu2_pred += i4_pred_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *pi4_csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * i4_dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * i4_dst_strd + col];
                val = (val == 0);
                *pi4_csbf = *pi4_csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    pi4_csbf++;
                    *pi4_csbf = 0;
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
 *  This function performs residue and quantization  on input pixels
 *
 * @par Description:
 *  Calculates the residue by subtracting the prediction  from input pixels
 * and does quantization on output
 *
 * @param[in] pu2_src
 *  Input 8x8 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_quant_coeff
 *  Scaling matrix
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] pi4_csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_resi_quant_8x8(UWORD16 *pu2_src,
                                 UWORD16 *pu2_pred,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 i4_qp_div,/* qpscaled / 6 */
                                 WORD32 i4_qp_rem,/* qpscaled % 6 */
                                 WORD32 i4_q_add,
                                 WORD32 i4_src_strd,
                                 WORD32 i4_pred_strd,
                                 WORD32 i4_dst_strd,
                                 WORD32 *pi4_csbf,
                                 UWORD8 u1_bit_depth)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_8;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 3;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 residue;

            residue = (pu2_src[j * 2] - pu2_pred[j * 2]);
            QUANT_HBD(pi2_dst[j], residue,
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[i4_qp_rem], i4_qp_div,
                  log2_size, i4_q_add, u1_bit_depth);
        }

        pu2_src += i4_src_strd;
        pi2_dst += i4_dst_strd;
        pu2_pred += i4_pred_strd;
        pi2_quant_coeff += trans_size;
    }

    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *pi4_csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * i4_dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * i4_dst_strd + col];
                val = (val == 0);
                *pi4_csbf = *pi4_csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    pi4_csbf++;
                    *pi4_csbf = 0;
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
 *  This function performs residue and quantization  on input pixels
 *
 * @par Description:
 *  Calculates the residue by subtracting the prediction  from input pixels
 * and does quantization on output
 *
 * @param[in] pu2_src
 *  Input 16x16 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_quant_coeff
 *  Scaling matrix
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
 *
 * @param[in] i4_qp_div
 *  Quantization parameter / 6
 *
 * @param[in] i4_qp_rem
 *  Quantization parameter % 6
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] pi4_csbf
 *  coded sub block flag
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */

void ihevc_hbd_chroma_resi_quant_16x16(UWORD16 *pu2_src,
                                   UWORD16 *pu2_pred,
                                   WORD16 *pi2_quant_coeff,
                                   WORD16 *pi2_dst,
                                   WORD32 i4_qp_div,/* qpscaled / 6 */
                                   WORD32 i4_qp_rem,/* qpscaled % 6 */
                                   WORD32 i4_q_add,
                                   WORD32 i4_src_strd,
                                   WORD32 i4_pred_strd,
                                   WORD32 i4_dst_strd,
                                   WORD32 *pi4_csbf,
                                   UWORD8 u1_bit_depth)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD32 log2_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;

    trans_size = TRANS_SIZE_16;

    /* Residue and Quantization */

    /* Quant initialization */
    log2_size = 4;

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            WORD32 residue;

            residue = (pu2_src[j * 2] - pu2_pred[j * 2]);
            QUANT_HBD(pi2_dst[j], residue,
                  pi2_quant_coeff[j] * g_ihevc_quant_scales[i4_qp_rem], i4_qp_div,
                  log2_size, i4_q_add, u1_bit_depth);
        }

        pu2_src += i4_src_strd;
        pi2_dst += i4_dst_strd;
        pu2_pred += i4_pred_strd;
        pi2_quant_coeff += trans_size;
    }
    /* CSBF update */
    {
        WORD32 block_row, block_col, val;
        WORD32 row, col;
        WORD16 *pi2_block;
        WORD32 shift;

        pi2_dst = pi2_dst_orig;

        *pi4_csbf = 0;
        shift = 31;
        for(block_row = 0; block_row < trans_size / 4; block_row++)
        {
            for(block_col = 0; block_col < trans_size / 4; block_col++)
            {
                pi2_block = pi2_dst + block_row * 4 * i4_dst_strd + block_col * 4;

                val = 0;
                for(row = 0; row < 4; row++)
                    for(col = 0; col < 4; col++)
                        val = val | pi2_block[row * i4_dst_strd + col];
                val = (val == 0);
                *pi4_csbf = *pi4_csbf | (val << shift);
                shift--;
                if(shift < 0)
                {
                    pi4_csbf++;
                    *pi4_csbf = 0;
                    shift = 31;
                }
            }

        }
    }

}
