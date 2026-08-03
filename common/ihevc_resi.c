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
 *  ihevc_resi.c
 *
 * @brief
 *  Contains function definitions for residual
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_resi_4x4_ttype1()
 *  - ihevc_resi_4x4()
 *  - ihevc_resi_8x8()
 *  - ihevc_resi_16x16()
 *  - ihevc_resi_32x32()
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
#include "ihevc_resi.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"

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

void ihevc_resi_4x4_ttype1(UWORD8 *pu1_src,
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
            pi2_dst[j] = (pu1_src[j] - pu1_pred[j]);
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

void ihevc_resi_4x4(UWORD8 *pu1_src,
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
            pi2_dst[j] = (pu1_src[j] - pu1_pred[j]);
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
 * @param[] pi2_dst
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

void ihevc_resi_8x8(UWORD8 *pu1_src,
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
            pi2_dst[j] = (pu1_src[j] - pu1_pred[j]);
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

void ihevc_resi_16x16(UWORD8 *pu1_src,
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
            pi2_dst[j] = (pu1_src[j] - pu1_pred[j]);
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
 *  Input 32x32 pixels
 *
 * @param[in] pu1_pred
 *  Prediction data
 *
 * @param[out] pi2_dst
 *  Output 32x32 coefficients
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

void ihevc_resi_32x32(UWORD8 *pu1_src,
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

    trans_size = TRANS_SIZE_32;

    /* Residue */

    for(i = 0; i < trans_size; i++)
    {
        for(j = 0; j < trans_size; j++)
        {
            pi2_dst[j] = (pu1_src[j] - pu1_pred[j]);
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

