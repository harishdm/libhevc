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
 *  ihevc_chroma_resi.c
 *
 * @brief
 *  Contains function definitions for residual  of chroma interleaved data.
 *
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_chroma_resi_4x4()
 *  - ihevc_chroma_resi_8x8()
 *  - ihevc_chroma_resi_16x16()
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
#include "ihevc_chroma_resi.h"
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
 *  This function performs residue on input pixels
 *
 * @par Description:
 *  Gets residue by subtracting prediction from source
 *
 * @param[in] pu1_src
 *  Input 4x4 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
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


void ihevc_chroma_resi_4x4(UWORD8 *pu1_src,
                           UWORD8 *pu1_pred,
                           WORD16 *pi2_dst,
                           WORD32 src_strd,
                           WORD32 pred_strd,
                           WORD32 dst_strd,
                           WORD32 *csbf)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_4;

    /* Residue */

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            pi2_dst[j] = (pu1_src[j * 2] - pu1_pred[j * 2]);
        }

        pu1_src += src_strd;
        pi2_dst += dst_strd;
        pu1_pred += pred_strd;
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
 *  This function performs residue on input pixels
 *
 * @par Description:
 *  Gets residue by subtracting prediction from source
 *
 * @param[in] pu1_src
 *  Input 8x8 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
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


void ihevc_chroma_resi_8x8(UWORD8 *pu1_src,
                           UWORD8 *pu1_pred,
                           WORD16 *pi2_dst,
                           WORD32 src_strd,
                           WORD32 pred_strd,
                           WORD32 dst_strd,
                           WORD32 *csbf)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_8;

    /* Residue */

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            pi2_dst[j] = (pu1_src[j * 2] - pu1_pred[j * 2]);
        }

        pu1_src += src_strd;
        pi2_dst += dst_strd;
        pu1_pred += pred_strd;
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
 *  This function performs residue on input pixels
 *
 * @par Description:
 *  Gets residue by subtracting prediction from source
 *
 * @param[in] pu1_src
 *  Input 16x16 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
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


void ihevc_chroma_resi_16x16(UWORD8 *pu1_src,
                             UWORD8 *pu1_pred,
                             WORD16 *pi2_dst,
                             WORD32 src_strd,
                             WORD32 pred_strd,
                             WORD32 dst_strd,
                             WORD32 *csbf)
{
    WORD32 i, j;
    WORD32 trans_size;
    WORD16 *pi2_dst_orig;

    pi2_dst_orig = pi2_dst;
    trans_size = TRANS_SIZE_16;

    /* Residue */

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            pi2_dst[j] = (pu1_src[j * 2] - pu1_pred[j * 2]);
        }

        pu1_src += src_strd;
        pi2_dst += dst_strd;
        pu1_pred += pred_strd;
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

