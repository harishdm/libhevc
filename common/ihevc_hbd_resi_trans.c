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
 *  ihevc_resi_trans.c
 *
 * @brief
 *  Contains function definitions for residual and  forward transform
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_resi_trans_4x4_ttype1()
 *  - ihevc_resi_trans_4x4()
 *  - ihevc_resi_trans_8x8()
 *  - ihevc_resi_trans_16x16()
 *  - ihevc_resi_trans_32x32()
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
#include "ihevc_resi_trans.h"
#include "ihevc_func_selector.h"
#include "ihevc_trans_macros.h"

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs residue calculation and forward  transform type 1
 * on input pixels
 *
 * @par Description:
 *  Performs residue calculation by subtracting source and  prediction and
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 4x4 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_4x4_ttype1(UWORD16 *pu2_src,
                                        UWORD16 *pu2_pred,
                                        WORD32 *pi4_temp,
                                        WORD16 *pi2_dst,
                                        WORD32 i4_src_strd,
                                        WORD32 i4_pred_strd,
                                        WORD32 i4_dst_strd_chr_flag,
                                        UWORD8 u1_bit_depth)
{
    WORD32 i, c[4];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32  u4_blk_sad = 0;
 //   WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

 //   chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_4;
    log2_trans_size = 2;

    /* Residue + Forward Transform 1st stage */
    shift = log2_trans_size - 1 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2, resi_tmp_3;

        // Intermediate Variables
        resi_tmp_1 = pu2_src[0] - pu2_pred[0];
        resi_tmp_2 = pu2_src[3] - pu2_pred[3];
        c[0] = resi_tmp_1 + resi_tmp_2;
        u4_blk_sad += abs(resi_tmp_1) + abs(resi_tmp_2);

        resi_tmp_1 = pu2_src[1] - pu2_pred[1];
        resi_tmp_2 = pu2_src[3] - pu2_pred[3];
        c[1] = resi_tmp_1 + resi_tmp_2;
        u4_blk_sad += abs(resi_tmp_1);

        resi_tmp_1 = pu2_src[0] - pu2_pred[0];
        resi_tmp_2 = pu2_src[1] - pu2_pred[1];
        c[2] = resi_tmp_1 - resi_tmp_2;

        resi_tmp_1 = pu2_src[2] - pu2_pred[2];
        c[3] = 74 * resi_tmp_1;
        u4_blk_sad += abs(resi_tmp_1);

        pi4_temp[0] = (29 * c[0] + 55 * c[1] + c[3] + add) >> shift;

        resi_tmp_1 = pu2_src[0] - pu2_pred[0];
        resi_tmp_2 = pu2_src[1] - pu2_pred[1];
        resi_tmp_3 = pu2_src[3] - pu2_pred[3];
        pi4_temp[trans_size] =
                        (74 * (resi_tmp_1 + resi_tmp_2 - resi_tmp_3) + add)
                                        >> shift;
        pi4_temp[2 * trans_size] = (29 * c[2] + 55 * c[0] - c[3] + add) >> shift;
        pi4_temp[3 * trans_size] = (55 * c[2] - 29 * c[1] + c[3] + add) >> shift;

        pu2_src += i4_src_strd;
        pu2_pred += i4_pred_strd;
        pi4_temp++;
    }

    pi4_temp = pi4_tmp_orig;

    /* Forward transform 2nd stage */
    shift = log2_trans_size + 6; // log2(iHeight) + 6
    add = 1 << (shift - 1);

    for(i = 0; i < TRANS_SIZE_4; i++)
    {
        // Intermediate Variables
        c[0] = pi4_temp[0] + pi4_temp[3];
        c[1] = pi4_temp[1] + pi4_temp[3];
        c[2] = pi4_temp[0] - pi4_temp[1];
        c[3] = 74 * pi4_temp[2];

        pi2_dst[0] = (29 * c[0] + 55 * c[1] + c[3] + add) >> shift;
        pi2_dst[i4_dst_strd] = (74 * (pi4_temp[0] + pi4_temp[1] - pi4_temp[3]) + add)
                        >> shift;
        pi2_dst[2 * i4_dst_strd] = (29 * c[2] + 55 * c[0] - c[3] + add) >> shift;
        pi2_dst[3 * i4_dst_strd] = (55 * c[2] - 29 * c[1] + c[3] + add) >> shift;

        pi4_temp += trans_size;
        pi2_dst++;
    }

    return u4_blk_sad;
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
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 4x4 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 4x4
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_4x4(UWORD16 *pu2_src,
                          UWORD16 *pu2_pred,
                          WORD32 *pi4_temp,
                          WORD16 *pi2_dst,
                          WORD32 i4_src_strd,
                          WORD32 i4_pred_strd,
                          WORD32 i4_dst_strd_chr_flag,
                          UWORD8 u1_bit_depth)
{
    WORD32 i;
    WORD32 e[2], o[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad=0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_4;
    log2_trans_size = 2;

    /* Residue + Forward Transform 1st stage */
    //shift = log2_trans_size - 1 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    //add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;

        /* e and o */
        resi_tmp_1 = pu2_src[0 + 0*chroma_flag] - pu2_pred[0 + 0*chroma_flag];
        resi_tmp_2 = pu2_src[3 + 3*chroma_flag] - pu2_pred[3 + 3*chroma_flag];
        e[0] = resi_tmp_1 + resi_tmp_2;
        o[0] = resi_tmp_1 - resi_tmp_2;
        u4_blk_sad += abs(resi_tmp_1);
        u4_blk_sad += abs(resi_tmp_2);

        resi_tmp_1 = pu2_src[1 + 1*chroma_flag] - pu2_pred[1 + 1*chroma_flag];
        resi_tmp_2 = pu2_src[2 + 2*chroma_flag] - pu2_pred[2 + 2*chroma_flag];
        e[1] = resi_tmp_1 + resi_tmp_2;
        o[1] = resi_tmp_1 - resi_tmp_2;
        u4_blk_sad += abs(resi_tmp_1);
        u4_blk_sad += abs(resi_tmp_2);

        pi4_temp[0] = (g_ai2_ihevc_trans_4[0][0] * e[0]
                        + g_ai2_ihevc_trans_4[0][1] * e[1]);// + add) >> shift;
        pi4_temp[2 * trans_size] = (g_ai2_ihevc_trans_4[2][0] * e[0]
                        + g_ai2_ihevc_trans_4[2][1] * e[1]);// + add) >> shift;
        pi4_temp[trans_size] = (g_ai2_ihevc_trans_4[1][0] * o[0]
                        + g_ai2_ihevc_trans_4[1][1] * o[1]);// + add) >> shift;
        pi4_temp[3 * trans_size] = (g_ai2_ihevc_trans_4[3][0] * o[0]
                        + g_ai2_ihevc_trans_4[3][1] * o[1]);// + add) >> shift;

        pu2_src += i4_src_strd;
        pu2_pred += i4_pred_strd;
        pi4_temp++;
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = 9 + (u1_bit_depth - 8); // log2(iHeight) + 6
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {

        /* e and o */
        e[0] = pi4_temp[0] + pi4_temp[3];
        o[0] = pi4_temp[0] - pi4_temp[3];
        e[1] = pi4_temp[1] + pi4_temp[2];
        o[1] = pi4_temp[1] - pi4_temp[2];

        pi2_dst[0] = (g_ai2_ihevc_trans_4[0][0] * e[0]
                        + g_ai2_ihevc_trans_4[0][1] * e[1] + add) >> shift;
        pi2_dst[2 * i4_dst_strd] = (g_ai2_ihevc_trans_4[2][0] * e[0]
                        + g_ai2_ihevc_trans_4[2][1] * e[1] + add) >> shift;
        pi2_dst[i4_dst_strd] = (g_ai2_ihevc_trans_4[1][0] * o[0]
                        + g_ai2_ihevc_trans_4[1][1] * o[1] + add) >> shift;
        pi2_dst[3 * i4_dst_strd] = (g_ai2_ihevc_trans_4[3][0] * o[0]
                        + g_ai2_ihevc_trans_4[3][1] * o[1] + add) >> shift;

        pi4_temp += trans_size;
        pi2_dst++;
    }

    return u4_blk_sad;
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
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 8x8 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 8x8
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_8x8(UWORD16 *pu2_src,
                          UWORD16 *pu2_pred,
                          WORD32 *pi4_temp,
                          WORD16 *pi2_dst,
                          WORD32 i4_src_strd,
                          WORD32 i4_pred_strd,
                          WORD32 i4_dst_strd_chr_flag,
                          UWORD8 u1_bit_depth)
{
    WORD32 i, k;
    WORD32 e[4], o[4];
    WORD32 ee[2], eo[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
//    WORD16 *pi2_tmp;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad=0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_8;
    log2_trans_size = 3;
    /* Residue + Forward Transform 1st stage */
    //shift = log2_trans_size - 1 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    //add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;

        /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
        /* e and o*/
        for(k = 0; k < 4; k++)
        {
            resi_tmp_1 = pu2_src[k*(1+chroma_flag)] - pu2_pred[k*(1+chroma_flag)];
            resi_tmp_2 = pu2_src[(7-k)*(1+chroma_flag)] - pu2_pred[(7-k)*(1+chroma_flag)];
            e[k] = resi_tmp_1 + resi_tmp_2;
            o[k] = resi_tmp_1 - resi_tmp_2;
            u4_blk_sad += abs(resi_tmp_1) + abs(resi_tmp_2);
        }
        /* ee and eo */
        ee[0] = e[0] + e[3];
        eo[0] = e[0] - e[3];
        ee[1] = e[1] + e[2];
        eo[1] = e[1] - e[2];

        pi4_temp[0] = (g_ai2_ihevc_trans_8[0][0] * ee[0]
                        + g_ai2_ihevc_trans_8[0][1] * ee[1]);   // + add) >> shift;
        pi4_temp[4 * trans_size] = (g_ai2_ihevc_trans_8[4][0] * ee[0]
                        + g_ai2_ihevc_trans_8[4][1] * ee[1]);   // + add) >> shift;
        pi4_temp[2 * trans_size] = (g_ai2_ihevc_trans_8[2][0] * eo[0]
                        + g_ai2_ihevc_trans_8[2][1] * eo[1]);   // + add) >> shift;
        pi4_temp[6 * trans_size] = (g_ai2_ihevc_trans_8[6][0] * eo[0]
                        + g_ai2_ihevc_trans_8[6][1] * eo[1]);   // + add) >> shift;

        pi4_temp[trans_size] = (g_ai2_ihevc_trans_8[1][0] * o[0]
                        + g_ai2_ihevc_trans_8[1][1] * o[1]
                        + g_ai2_ihevc_trans_8[1][2] * o[2]
                        + g_ai2_ihevc_trans_8[1][3] * o[3]);    // + add) >> shift;
        pi4_temp[3 * trans_size] = (g_ai2_ihevc_trans_8[3][0] * o[0]
                        + g_ai2_ihevc_trans_8[3][1] * o[1]
                        + g_ai2_ihevc_trans_8[3][2] * o[2]
                        + g_ai2_ihevc_trans_8[3][3] * o[3]);    // + add) >> shift;
        pi4_temp[5 * trans_size] = (g_ai2_ihevc_trans_8[5][0] * o[0]
                        + g_ai2_ihevc_trans_8[5][1] * o[1]
                        + g_ai2_ihevc_trans_8[5][2] * o[2]
                        + g_ai2_ihevc_trans_8[5][3] * o[3]);    // + add) >> shift;
        pi4_temp[7 * trans_size] = (g_ai2_ihevc_trans_8[7][0] * o[0]
                        + g_ai2_ihevc_trans_8[7][1] * o[1]
                        + g_ai2_ihevc_trans_8[7][2] * o[2]
                        + g_ai2_ihevc_trans_8[7][3] * o[3]);    // + add) >> shift;

        pu2_src += i4_src_strd;
        pu2_pred += i4_pred_strd;
        pi4_temp++;
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = (u1_bit_depth - 8) + 11; // log2(iHeight) + 6
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
        /* e and o*/
        for(k = 0; k < 4; k++)
        {
            e[k] = pi4_temp[k] + pi4_temp[7 - k];
            o[k] = pi4_temp[k] - pi4_temp[7 - k];
        }
        /* ee and eo */
        ee[0] = e[0] + e[3];
        eo[0] = e[0] - e[3];
        ee[1] = e[1] + e[2];
        eo[1] = e[1] - e[2];

        pi2_dst[0] = (g_ai2_ihevc_trans_8[0][0] * ee[0]
                        + g_ai2_ihevc_trans_8[0][1] * ee[1] + add) >> shift;
        pi2_dst[4 * i4_dst_strd] = (g_ai2_ihevc_trans_8[4][0] * ee[0]
                        + g_ai2_ihevc_trans_8[4][1] * ee[1] + add) >> shift;
        pi2_dst[2 * i4_dst_strd] = (g_ai2_ihevc_trans_8[2][0] * eo[0]
                        + g_ai2_ihevc_trans_8[2][1] * eo[1] + add) >> shift;
        pi2_dst[6 * i4_dst_strd] = (g_ai2_ihevc_trans_8[6][0] * eo[0]
                        + g_ai2_ihevc_trans_8[6][1] * eo[1] + add) >> shift;

        pi2_dst[i4_dst_strd] = (g_ai2_ihevc_trans_8[1][0] * o[0]
                        + g_ai2_ihevc_trans_8[1][1] * o[1]
                        + g_ai2_ihevc_trans_8[1][2] * o[2]
                        + g_ai2_ihevc_trans_8[1][3] * o[3] + add) >> shift;
        pi2_dst[3 * i4_dst_strd] = (g_ai2_ihevc_trans_8[3][0] * o[0]
                        + g_ai2_ihevc_trans_8[3][1] * o[1]
                        + g_ai2_ihevc_trans_8[3][2] * o[2]
                        + g_ai2_ihevc_trans_8[3][3] * o[3] + add) >> shift;
        pi2_dst[5 * i4_dst_strd] = (g_ai2_ihevc_trans_8[5][0] * o[0]
                        + g_ai2_ihevc_trans_8[5][1] * o[1]
                        + g_ai2_ihevc_trans_8[5][2] * o[2]
                        + g_ai2_ihevc_trans_8[5][3] * o[3] + add) >> shift;
        pi2_dst[7 * i4_dst_strd] = (g_ai2_ihevc_trans_8[7][0] * o[0]
                        + g_ai2_ihevc_trans_8[7][1] * o[1]
                        + g_ai2_ihevc_trans_8[7][2] * o[2]
                        + g_ai2_ihevc_trans_8[7][3] * o[3] + add) >> shift;

        pi4_temp += trans_size;
        pi2_dst++;
    }

    return u4_blk_sad;
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
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 16x16 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 16x16
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_16x16(UWORD16 *pu2_src,
                            UWORD16 *pu2_pred,
                            WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 i4_src_strd,
                            WORD32 i4_pred_strd,
                            WORD32 i4_dst_strd_chr_flag,
                            UWORD8 u1_bit_depth)
{
    WORD32 i, k;
    WORD32 e[8], o[8];
    WORD32 ee[4], eo[4];
    WORD32 eee[2], eeo[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad = 0;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_16;
    log2_trans_size = 4;
    /* Residue + Forward Transform 1st stage */
    shift = (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;
        /* e and o*/
        for(k = 0; k < 8; k++)
        {
            resi_tmp_1 = pu2_src[k*(1+chroma_flag)] - pu2_pred[k*(1+chroma_flag)];
            resi_tmp_2 = pu2_src[(15-k)*(1+chroma_flag)] - pu2_pred[(15-k)*(1+chroma_flag)];
            e[k] = resi_tmp_1 + resi_tmp_2;
            o[k] = resi_tmp_1 - resi_tmp_2;
            u4_blk_sad += abs(resi_tmp_1) + abs(resi_tmp_2);
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

        pi4_temp[0] = (g_ai2_ihevc_trans_16[0][0] * eee[0]
                        + g_ai2_ihevc_trans_16[0][1] * eee[1] + add) >> shift;
        pi4_temp[8 * trans_size] = (g_ai2_ihevc_trans_16[8][0] * eee[0]
                        + g_ai2_ihevc_trans_16[8][1] * eee[1] + add) >> shift;
        pi4_temp[4 * trans_size] = (g_ai2_ihevc_trans_16[4][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[4][1] * eeo[1] + add) >> shift;
        pi4_temp[12 * trans_size] = (g_ai2_ihevc_trans_16[12][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[12][1] * eeo[1] + add) >> shift;

        for(k = 2; k < 16; k += 4)
        {
            pi4_temp[k * trans_size] = (g_ai2_ihevc_trans_16[k][0] * eo[0]
                            + g_ai2_ihevc_trans_16[k][1] * eo[1]
                            + g_ai2_ihevc_trans_16[k][2] * eo[2]
                            + g_ai2_ihevc_trans_16[k][3] * eo[3] + add)>> shift;

        }

        for(k = 1; k < 16; k += 2)
        {
            pi4_temp[k * trans_size] = (g_ai2_ihevc_trans_16[k][0] * o[0]
                            + g_ai2_ihevc_trans_16[k][1] * o[1]
                            + g_ai2_ihevc_trans_16[k][2] * o[2]
                            + g_ai2_ihevc_trans_16[k][3] * o[3]
                            + g_ai2_ihevc_trans_16[k][4] * o[4]
                            + g_ai2_ihevc_trans_16[k][5] * o[5]
                            + g_ai2_ihevc_trans_16[k][6] * o[6]
                            + g_ai2_ihevc_trans_16[k][7] * o[7] + add) >> shift;
        }
        pu2_src += i4_src_strd;
        pu2_pred += i4_pred_strd;
        pi4_temp++;
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = 13; //log2_trans_size - 1 + log2_trans_size + 6;
    add = 1 << (shift - 1);

    for(i = 0; i < TRANS_SIZE_16; i++)
    {
        /* e and o*/
        for(k = 0; k < 8; k++)
        {
            e[k] = pi4_temp[k] + pi4_temp[15 - k];
            o[k] = pi4_temp[k] - pi4_temp[15 - k];
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

        pi2_dst[0] = (g_ai2_ihevc_trans_16[0][0] * eee[0]
                        + g_ai2_ihevc_trans_16[0][1] * eee[1] + add) >> shift;
        pi2_dst[8 * i4_dst_strd] = (g_ai2_ihevc_trans_16[8][0] * eee[0]
                        + g_ai2_ihevc_trans_16[8][1] * eee[1] + add) >> shift;
        pi2_dst[4 * i4_dst_strd] = (g_ai2_ihevc_trans_16[4][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[4][1] * eeo[1] + add) >> shift;
        pi2_dst[12 * i4_dst_strd] = (g_ai2_ihevc_trans_16[12][0] * eeo[0]
                        + g_ai2_ihevc_trans_16[12][1] * eeo[1] + add) >> shift;

        for(k = 2; k < 16; k += 4)
        {
            pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_16[k][0] * eo[0]
                            + g_ai2_ihevc_trans_16[k][1] * eo[1]
                            + g_ai2_ihevc_trans_16[k][2] * eo[2]
                            + g_ai2_ihevc_trans_16[k][3] * eo[3] + add)
                            >> shift;
        }

        for(k = 1; k < 16; k += 2)
        {
            pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_16[k][0] * o[0]
                            + g_ai2_ihevc_trans_16[k][1] * o[1]
                            + g_ai2_ihevc_trans_16[k][2] * o[2]
                            + g_ai2_ihevc_trans_16[k][3] * o[3]
                            + g_ai2_ihevc_trans_16[k][4] * o[4]
                            + g_ai2_ihevc_trans_16[k][5] * o[5]
                            + g_ai2_ihevc_trans_16[k][6] * o[6]
                            + g_ai2_ihevc_trans_16[k][7] * o[7] + add) >> shift;
        }

        pi4_temp += trans_size;
        pi2_dst++;
    }

    return u4_blk_sad;
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
 * followed by forward transform
 *
 * @param[in] pu2_src
 *  Input 32x32 pixels
 *
 * @param[in] pu2_pred
 *  Prediction data
 *
 * @param[in] pi2_tmp
 *  Temporary buffer of size 32x32
 *
 * @param[out] pi2_dst
 *  Output 32x32 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_pred_strd
 *  Prediction Stride
 *
 * @param[in] i4_dst_strd_chr_flag
 *  Output Stride and Chroma Flag packed in the MS and LS 16-bit
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
UWORD32 ihevc_hbd_resi_trans_32x32(UWORD16 *pu2_src,
                            UWORD16 *pu2_pred,
                            WORD32 *pi4_temp,
                            WORD16 *pi2_dst,
                            WORD32 i4_src_strd,
                            WORD32 i4_pred_strd,
                            WORD32 i4_dst_strd_chr_flag,
                            UWORD8 u1_bit_depth)
{
    WORD32 i, k;
    WORD32 e[16], o[16];
    WORD32 ee[8], eo[8];
    WORD32 eee[4], eeo[4];
    WORD32 eeee[2], eeeo[2];
    WORD32 add, shift;
    WORD32 trans_size;
    WORD32 *pi4_tmp_orig;
    WORD16 *pi2_dst_orig;
    UWORD32 u4_blk_sad = 0 ;
    WORD32 chroma_flag;
    WORD32 i4_dst_strd;
    WORD32 log2_trans_size;

    chroma_flag = i4_dst_strd_chr_flag & 1;
    i4_dst_strd = i4_dst_strd_chr_flag >> 16;

    pi2_dst_orig = pi2_dst;
    pi4_tmp_orig = pi4_temp;
    trans_size = TRANS_SIZE_32;
    log2_trans_size = 5;
    /* Residue + Forward Transform 1st stage */
    /* Made to zero to match with intrinsics */
    shift = 2 + (u1_bit_depth - 8); // log2(iWidth) - 1 + g_uiBitIncrement
    add = 1 << (shift - 1);

    for(i = 0; i < trans_size; i++)
    {
        WORD32 resi_tmp_1, resi_tmp_2;
        /* e and o*/
        for(k = 0; k < 16; k++)
        {
            resi_tmp_1 = pu2_src[k] - pu2_pred[k];
            resi_tmp_2 = pu2_src[31 - k] - pu2_pred[31 - k];
            e[k] = resi_tmp_1 + resi_tmp_2;
            o[k] = resi_tmp_1 - resi_tmp_2;
            u4_blk_sad += abs(resi_tmp_1) + abs(resi_tmp_2);
        }
        /* ee and eo */
        for(k = 0; k < 8; k++)
        {
            ee[k] = e[k] + e[15 - k];
            eo[k] = e[k] - e[15 - k];
        }
        /* eee and eeo */
        for(k = 0; k < 4; k++)
        {
            eee[k] = ee[k] + ee[7 - k];
            eeo[k] = ee[k] - ee[7 - k];
        }
        /* eeee and eeeo */
        eeee[0] = eee[0] + eee[3];
        eeeo[0] = eee[0] - eee[3];
        eeee[1] = eee[1] + eee[2];
        eeeo[1] = eee[1] - eee[2];

        pi4_temp[0] = (g_ai2_ihevc_trans_32[0][0] * eeee[0]
                        + g_ai2_ihevc_trans_32[0][1] * eeee[1] + add) >> shift;
        pi4_temp[16 * trans_size] = (g_ai2_ihevc_trans_32[16][0] * eeee[0]
                        + g_ai2_ihevc_trans_32[16][1] * eeee[1] + add) >> shift;
        pi4_temp[8 * trans_size] = (g_ai2_ihevc_trans_32[8][0] * eeeo[0]
                        + g_ai2_ihevc_trans_32[8][1] * eeeo[1] + add) >> shift;
        pi4_temp[24 * trans_size] = (g_ai2_ihevc_trans_32[24][0] * eeeo[0]
                        + g_ai2_ihevc_trans_32[24][1] * eeeo[1] + add) >> shift;
        for(k = 4; k < 32; k += 8)
        {
            pi4_temp[k * trans_size] = (g_ai2_ihevc_trans_32[k][0] * eeo[0]
                            + g_ai2_ihevc_trans_32[k][1] * eeo[1]
                            + g_ai2_ihevc_trans_32[k][2] * eeo[2]
                            + g_ai2_ihevc_trans_32[k][3] * eeo[3] + add)>> shift;
        }
        for(k = 2; k < 32; k += 4)
        {
            pi4_temp[k * trans_size] = (g_ai2_ihevc_trans_32[k][0] * eo[0]
                            + g_ai2_ihevc_trans_32[k][1] * eo[1]
                            + g_ai2_ihevc_trans_32[k][2] * eo[2]
                            + g_ai2_ihevc_trans_32[k][3] * eo[3]
                            + g_ai2_ihevc_trans_32[k][4] * eo[4]
                            + g_ai2_ihevc_trans_32[k][5] * eo[5]
                            + g_ai2_ihevc_trans_32[k][6] * eo[6]
                            + g_ai2_ihevc_trans_32[k][7] * eo[7] + add)>> shift;
        }
        for(k = 1; k < 32; k += 2)
        {
            pi4_temp[k * trans_size] = (g_ai2_ihevc_trans_32[k][0] * o[0]
                            + g_ai2_ihevc_trans_32[k][1] * o[1]
                            + g_ai2_ihevc_trans_32[k][2] * o[2]
                            + g_ai2_ihevc_trans_32[k][3] * o[3]
                            + g_ai2_ihevc_trans_32[k][4] * o[4]
                            + g_ai2_ihevc_trans_32[k][5] * o[5]
                            + g_ai2_ihevc_trans_32[k][6] * o[6]
                            + g_ai2_ihevc_trans_32[k][7] * o[7]
                            + g_ai2_ihevc_trans_32[k][8] * o[8]
                            + g_ai2_ihevc_trans_32[k][9] * o[9]
                            + g_ai2_ihevc_trans_32[k][10] * o[10]
                            + g_ai2_ihevc_trans_32[k][11] * o[11]
                            + g_ai2_ihevc_trans_32[k][12] * o[12]
                            + g_ai2_ihevc_trans_32[k][13] * o[13]
                            + g_ai2_ihevc_trans_32[k][14] * o[14]
                            + g_ai2_ihevc_trans_32[k][15] * o[15] + add) >> shift;
        }
        pu2_src += i4_src_strd;
        pu2_pred += i4_pred_strd;
        pi4_temp++;
    }

    pi4_temp = pi4_tmp_orig;
    /* Forward Transform 2nd stage */
    shift = log2_trans_size + 8; // log2(iHeight) + 6
    add = 1 << (shift - 1);

    for(i = 0; i < TRANS_SIZE_32; i++)
    {
        /* e and o*/
        for(k = 0; k < 16; k++)
        {
            e[k] = pi4_temp[k] + pi4_temp[31 - k];
            o[k] = pi4_temp[k] - pi4_temp[31 - k];
        }
        /* ee and eo */
        for(k = 0; k < 8; k++)
        {
            ee[k] = e[k] + e[15 - k];
            eo[k] = e[k] - e[15 - k];
        }
        /* eee and eeo */
        for(k = 0; k < 4; k++)
        {
            eee[k] = ee[k] + ee[7 - k];
            eeo[k] = ee[k] - ee[7 - k];
        }
        /* eeee and eeeo */
        eeee[0] = eee[0] + eee[3];
        eeeo[0] = eee[0] - eee[3];
        eeee[1] = eee[1] + eee[2];
        eeeo[1] = eee[1] - eee[2];

        pi2_dst[0] = (g_ai2_ihevc_trans_32[0][0] * eeee[0]
                        + g_ai2_ihevc_trans_32[0][1] * eeee[1] + add) >> shift;
        pi2_dst[16 * i4_dst_strd] = (g_ai2_ihevc_trans_32[16][0] * eeee[0]
                        + g_ai2_ihevc_trans_32[16][1] * eeee[1] + add) >> shift;
        pi2_dst[8 * i4_dst_strd] = (g_ai2_ihevc_trans_32[8][0] * eeeo[0]
                        + g_ai2_ihevc_trans_32[8][1] * eeeo[1] + add) >> shift;
        pi2_dst[24 * i4_dst_strd] = (g_ai2_ihevc_trans_32[24][0] * eeeo[0]
                        + g_ai2_ihevc_trans_32[24][1] * eeeo[1] + add) >> shift;
        for(k = 4; k < 32; k += 8)
        {
            pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_32[k][0] * eeo[0]
                            + g_ai2_ihevc_trans_32[k][1] * eeo[1]
                            + g_ai2_ihevc_trans_32[k][2] * eeo[2]
                            + g_ai2_ihevc_trans_32[k][3] * eeo[3] + add)
                            >> shift;
        }
        for(k = 2; k < 32; k += 4)
        {
            pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_32[k][0] * eo[0]
                            + g_ai2_ihevc_trans_32[k][1] * eo[1]
                            + g_ai2_ihevc_trans_32[k][2] * eo[2]
                            + g_ai2_ihevc_trans_32[k][3] * eo[3]
                            + g_ai2_ihevc_trans_32[k][4] * eo[4]
                            + g_ai2_ihevc_trans_32[k][5] * eo[5]
                            + g_ai2_ihevc_trans_32[k][6] * eo[6]
                            + g_ai2_ihevc_trans_32[k][7] * eo[7] + add)
                            >> shift;
        }
        for(k = 1; k < 32; k += 2)
        {
            pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_32[k][0] * o[0]
                            + g_ai2_ihevc_trans_32[k][1] * o[1]
                            + g_ai2_ihevc_trans_32[k][2] * o[2]
                            + g_ai2_ihevc_trans_32[k][3] * o[3]
                            + g_ai2_ihevc_trans_32[k][4] * o[4]
                            + g_ai2_ihevc_trans_32[k][5] * o[5]
                            + g_ai2_ihevc_trans_32[k][6] * o[6]
                            + g_ai2_ihevc_trans_32[k][7] * o[7]
                            + g_ai2_ihevc_trans_32[k][8] * o[8]
                            + g_ai2_ihevc_trans_32[k][9] * o[9]
                            + g_ai2_ihevc_trans_32[k][10] * o[10]
                            + g_ai2_ihevc_trans_32[k][11] * o[11]
                            + g_ai2_ihevc_trans_32[k][12] * o[12]
                            + g_ai2_ihevc_trans_32[k][13] * o[13]
                            + g_ai2_ihevc_trans_32[k][14] * o[14]
                            + g_ai2_ihevc_trans_32[k][15] * o[15] + add)
                            >> shift;
        }

        pi4_temp += trans_size;
        pi2_dst++;
    }

    return u4_blk_sad;
}
