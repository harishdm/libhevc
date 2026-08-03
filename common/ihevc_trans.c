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
 *  ihevc_trans.c
 *
 * @brief
 *  Contains function definitions single stage  forward transform
 *
 * @author
 *  100470
 *
 * @par List of Functions:
 *  - ihevc_trans_4x4_ttype1()
 *  - ihevc_trans_4x4()
 *  - ihevc_trans_8x8()
 *  - ihevc_trans_16x16()
 *  - ihevc_trans_32x32()
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

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  forward transform type 1 (DST) for
 * 4x4 input block
 *
 * @par Description:
 *  Performs single stage 4x4 forward transform type 1  by utilizing the
 * symmetry of transformation matrix  and reducing number of multiplications
 * wherever  possible but keeping the number of operations
 * (addition,multiplication and shift)same
 *
 * @param[in] pi2_src
 *  Input 4x4 pixels
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
 *
 * @param[in] i4_zero_rows
 *  Zero rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_trans_4x4_ttype1(WORD16 *pi2_src,
                            WORD16 *pi2_dst,
                            WORD32 i4_src_strd,
                            WORD32 i4_dst_strd,
                            WORD32 i4_shift,
                            WORD32 i4_zero_rows)
{
    WORD32 i, j, c[4];
    WORD32 add;

    add = 1 << (i4_shift - 1);

    for(i = 0; i < TRANS_SIZE_4; i++)
    {
        /* Checking for Zero Rows */
        if((i4_zero_rows & 1) == 1)
        {
            pi2_src += i4_src_strd;
            for(j = 0; j < TRANS_SIZE_4; j++)
                pi2_dst[j * i4_dst_strd] = 0;
            pi2_dst++;
        }
        else
        {
            // Intermediate Variables
            c[0] = pi2_src[0] + pi2_src[3];
            c[1] = pi2_src[1] + pi2_src[3];
            c[2] = pi2_src[0] - pi2_src[1];
            c[3] = 74 * pi2_src[2];

            pi2_dst[0] = (29 * c[0] + 55 * c[1] + c[3] + add) >> i4_shift;
            pi2_dst[i4_dst_strd] = (74 * (pi2_src[0] + pi2_src[1] - pi2_src[3])
                            + add) >> i4_shift;
            pi2_dst[2 * i4_dst_strd] = (29 * c[2] + 55 * c[0] - c[3] + add)
                            >> i4_shift;
            pi2_dst[3 * i4_dst_strd] = (55 * c[2] - 29 * c[1] + c[3] + add)
                            >> i4_shift;

            pi2_src += i4_src_strd;
            pi2_dst++;
        }
        i4_zero_rows = i4_zero_rows >> 1;
    }
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  forward transform for 4x4 input
 * block
 *
 * @par Description:
 *  Performs single stage 4x4 forward transform by utilizing  the symmetry of
 * transformation matrix and reducing number  of multiplications wherever
 * possible but keeping the  number of operations(addition,multiplication and
 * shift)  same
 *
 * @param[in] pi2_src
 *  Input 4x4 pixels
 *
 * @param[out] pi2_dst
 *  Output 4x4 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
 *
 * @param[in] i4_zero_rows
 *  Zero rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_trans_4x4(WORD16 *pi2_src,
                     WORD16 *pi2_dst,
                     WORD32 i4_src_strd,
                     WORD32 i4_dst_strd,
                     WORD32 i4_shift,
                     WORD32 i4_zero_rows)
{
    WORD32 i, j;
    WORD32 e[2], o[2];
    WORD32 add;

    add = 1 << (i4_shift - 1);

    for(i = 0; i < TRANS_SIZE_4; i++)
    {
        /* Checking for Zero Rows */
        if((i4_zero_rows & 1) == 1)
        {
            pi2_src += i4_src_strd;
            for(j = 0; j < TRANS_SIZE_4; j++)
                pi2_dst[j * i4_dst_strd] = 0;
            pi2_dst++;
        }
        else
        {
            /* e and o */
            e[0] = pi2_src[0] + pi2_src[3];
            o[0] = pi2_src[0] - pi2_src[3];
            e[1] = pi2_src[1] + pi2_src[2];
            o[1] = pi2_src[1] - pi2_src[2];

            pi2_dst[0] = (g_ai2_ihevc_trans_4[0][0] * e[0]
                            + g_ai2_ihevc_trans_4[0][1] * e[1] + add)
                            >> i4_shift;
            pi2_dst[2 * i4_dst_strd] = (g_ai2_ihevc_trans_4[2][0] * e[0]
                            + g_ai2_ihevc_trans_4[2][1] * e[1] + add)
                            >> i4_shift;
            pi2_dst[i4_dst_strd] = (g_ai2_ihevc_trans_4[1][0] * o[0]
                            + g_ai2_ihevc_trans_4[1][1] * o[1] + add)
                            >> i4_shift;
            pi2_dst[3 * i4_dst_strd] = (g_ai2_ihevc_trans_4[3][0] * o[0]
                            + g_ai2_ihevc_trans_4[3][1] * o[1] + add)
                            >> i4_shift;

            pi2_src += i4_src_strd;
            pi2_dst++;
        }
        i4_zero_rows = i4_zero_rows >> 1;
    }
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  forward transform for 8x8 input
 * block
 *
 * @par Description:
 *  Performs single stage 8x8 forward transform by utilizing  the symmetry of
 * transformation matrix and reducing number  of multiplications wherever
 * possible but keeping the  number of operations(addition,multiplication and
 * shift)  same
 *
 * @param[in] pi2_src
 *  Input 8x8 pixels
 *
 * @param[out] pi2_dst
 *  Output 8x8 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
 *
 * @param[in] i4_zero_rows
 *  Zero rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_trans_8x8(WORD16 *pi2_src,
                     WORD16 *pi2_dst,
                     WORD32 i4_src_strd,
                     WORD32 i4_dst_strd,
                     WORD32 i4_shift,
                     WORD32 i4_zero_rows)
{
    WORD32 i, j, k;
    WORD32 e[4], o[4];
    WORD32 ee[2], eo[2];
    WORD32 add;

    add = 1 << (i4_shift - 1);

    for(i = 0; i < TRANS_SIZE_8; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_rows & 1) == 1)
        {
            pi2_src += i4_src_strd;
            for(j = 0; j < TRANS_SIZE_8; j++)
                pi2_dst[j * i4_dst_strd] = 0;
            pi2_dst++;
        }
        else
        {
            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            /* e and o*/
            for(k = 0; k < 4; k++)
            {
                e[k] = pi2_src[k] + pi2_src[7 - k];
                o[k] = pi2_src[k] - pi2_src[7 - k];
            }
            /* ee and eo */
            ee[0] = e[0] + e[3];
            eo[0] = e[0] - e[3];
            ee[1] = e[1] + e[2];
            eo[1] = e[1] - e[2];

            pi2_dst[0] = (g_ai2_ihevc_trans_8[0][0] * ee[0]
                            + g_ai2_ihevc_trans_8[0][1] * ee[1] + add)
                            >> i4_shift;
            pi2_dst[4 * i4_dst_strd] = (g_ai2_ihevc_trans_8[4][0] * ee[0]
                            + g_ai2_ihevc_trans_8[4][1] * ee[1] + add)
                            >> i4_shift;
            pi2_dst[2 * i4_dst_strd] = (g_ai2_ihevc_trans_8[2][0] * eo[0]
                            + g_ai2_ihevc_trans_8[2][1] * eo[1] + add)
                            >> i4_shift;
            pi2_dst[6 * i4_dst_strd] = (g_ai2_ihevc_trans_8[6][0] * eo[0]
                            + g_ai2_ihevc_trans_8[6][1] * eo[1] + add)
                            >> i4_shift;

            pi2_dst[i4_dst_strd] = (g_ai2_ihevc_trans_8[1][0] * o[0]
                            + g_ai2_ihevc_trans_8[1][1] * o[1]
                            + g_ai2_ihevc_trans_8[1][2] * o[2]
                            + g_ai2_ihevc_trans_8[1][3] * o[3] + add)
                            >> i4_shift;
            pi2_dst[3 * i4_dst_strd] = (g_ai2_ihevc_trans_8[3][0] * o[0]
                            + g_ai2_ihevc_trans_8[3][1] * o[1]
                            + g_ai2_ihevc_trans_8[3][2] * o[2]
                            + g_ai2_ihevc_trans_8[3][3] * o[3] + add)
                            >> i4_shift;
            pi2_dst[5 * i4_dst_strd] = (g_ai2_ihevc_trans_8[5][0] * o[0]
                            + g_ai2_ihevc_trans_8[5][1] * o[1]
                            + g_ai2_ihevc_trans_8[5][2] * o[2]
                            + g_ai2_ihevc_trans_8[5][3] * o[3] + add)
                            >> i4_shift;
            pi2_dst[7 * i4_dst_strd] = (g_ai2_ihevc_trans_8[7][0] * o[0]
                            + g_ai2_ihevc_trans_8[7][1] * o[1]
                            + g_ai2_ihevc_trans_8[7][2] * o[2]
                            + g_ai2_ihevc_trans_8[7][3] * o[3] + add)
                            >> i4_shift;

            pi2_src += i4_src_strd;
            pi2_dst++;
        }
        i4_zero_rows = i4_zero_rows >> 1;
    }
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  forward transform for 16x16 input
 * block
 *
 * @par Description:
 *  Performs single stage 16x16 forward transform by  utilizing the symmetry
 * of transformation matrix  and reducing number of multiplications wherever
 * possible  but keeping the number of operations  (addition,multiplication
 * and shift) same
 *
 * @param[in] pi2_src
 *  Input 16x16 pixels
 *
 * @param[out] pi2_dst
 *  Output 16x16 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
 *
 * @param[in] i4_zero_rows
 *  Zero rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_trans_16x16(WORD16 *pi2_src,
                       WORD16 *pi2_dst,
                       WORD32 i4_src_strd,
                       WORD32 i4_dst_strd,
                       WORD32 i4_shift,
                       WORD32 i4_zero_rows)
{
    WORD32 i, j, k;
    WORD32 e[8], o[8];
    WORD32 ee[4], eo[4];
    WORD32 eee[2], eeo[2];
    WORD32 add;

    add = 1 << (i4_shift - 1);

    for(i = 0; i < TRANS_SIZE_16; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_rows & 1) == 1)
        {
            pi2_src += i4_src_strd;
            for(j = 0; j < TRANS_SIZE_16; j++)
                pi2_dst[j * i4_dst_strd] = 0;
            pi2_dst++;
        }
        else
        {
            /* e and o*/
            for(k = 0; k < 8; k++)
            {
                e[k] = pi2_src[k] + pi2_src[15 - k];
                o[k] = pi2_src[k] - pi2_src[15 - k];
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
                            + g_ai2_ihevc_trans_16[0][1] * eee[1] + add)
                            >> i4_shift;
            pi2_dst[8 * i4_dst_strd] = (g_ai2_ihevc_trans_16[8][0] * eee[0]
                            + g_ai2_ihevc_trans_16[8][1] * eee[1] + add)
                            >> i4_shift;
            pi2_dst[4 * i4_dst_strd] = (g_ai2_ihevc_trans_16[4][0] * eeo[0]
                            + g_ai2_ihevc_trans_16[4][1] * eeo[1] + add)
                            >> i4_shift;
            pi2_dst[12 * i4_dst_strd] = (g_ai2_ihevc_trans_16[12][0] * eeo[0]
                            + g_ai2_ihevc_trans_16[12][1] * eeo[1] + add)
                            >> i4_shift;

            for(k = 2; k < 16; k += 4)
            {
                pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_16[k][0] * eo[0]
                                + g_ai2_ihevc_trans_16[k][1] * eo[1]
                                + g_ai2_ihevc_trans_16[k][2] * eo[2]
                                + g_ai2_ihevc_trans_16[k][3] * eo[3] + add)
                                >> i4_shift;
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
                                + g_ai2_ihevc_trans_16[k][7] * o[7] + add)
                                >> i4_shift;
            }
            pi2_src += i4_src_strd;
            pi2_dst++;
        }
        i4_zero_rows = i4_zero_rows >> 1;
    }
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs Single stage  forward transform for 32x32 input
 * block
 *
 * @par Description:
 *  Performs single stage 32x32 forward transform by  utilizing the symmetry
 * of transformation matrix and  reducing number of multiplications wherever
 * possible  but keeping the number of operations  (addition,multiplication
 * and shift) same
 *
 * @param[in] pi2_src
 *  Input 32x32 pixels
 *
 * @param[out] pi2_dst
 *  Output 32x32 coefficients
 *
 * @param[in] i4_src_strd
 *  Input stride
 *
 * @param[in] i4_dst_strd
 *  Output Stride
 *
 * @param[in] i4_shift
 *  Output shift
 *
 * @param[in] i4_zero_rows
 *  Zero rows in pi2_src
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */


void ihevc_trans_32x32(WORD16 *pi2_src,
                       WORD16 *pi2_dst,
                       WORD32 i4_src_strd,
                       WORD32 i4_dst_strd,
                       WORD32 i4_shift,
                       WORD32 i4_zero_rows)
{
    WORD32 i, j, k;
    WORD32 e[16], o[16];
    WORD32 ee[8], eo[8];
    WORD32 eee[4], eeo[4];
    WORD32 eeee[2], eeeo[2];
    WORD32 add;

    add = 1 << (i4_shift - 1);

    for(i = 0; i < TRANS_SIZE_32; i++)
    {
        /* Checking for Zero Cols */
        if((i4_zero_rows & 1) == 1)
        {
            pi2_src += i4_src_strd;
            for(j = 0; j < TRANS_SIZE_32; j++)
                pi2_dst[j * i4_dst_strd] = 0;
            pi2_dst++;
        }
        else
        {
            /* e and o*/
            for(k = 0; k < 16; k++)
            {
                e[k] = pi2_src[k] + pi2_src[31 - k];
                o[k] = pi2_src[k] - pi2_src[31 - k];
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
                            + g_ai2_ihevc_trans_32[0][1] * eeee[1] + add)
                            >> i4_shift;
            pi2_dst[16 * i4_dst_strd] = (g_ai2_ihevc_trans_32[16][0] * eeee[0]
                            + g_ai2_ihevc_trans_32[16][1] * eeee[1] + add)
                            >> i4_shift;
            pi2_dst[8 * i4_dst_strd] = (g_ai2_ihevc_trans_32[8][0] * eeeo[0]
                            + g_ai2_ihevc_trans_32[8][1] * eeeo[1] + add)
                            >> i4_shift;
            pi2_dst[24 * i4_dst_strd] = (g_ai2_ihevc_trans_32[24][0] * eeeo[0]
                            + g_ai2_ihevc_trans_32[24][1] * eeeo[1] + add)
                            >> i4_shift;
            for(k = 4; k < 32; k += 8)
            {
                pi2_dst[k * i4_dst_strd] = (g_ai2_ihevc_trans_32[k][0] * eeo[0]
                                + g_ai2_ihevc_trans_32[k][1] * eeo[1]
                                + g_ai2_ihevc_trans_32[k][2] * eeo[2]
                                + g_ai2_ihevc_trans_32[k][3] * eeo[3] + add)
                                >> i4_shift;
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
                                >> i4_shift;
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
                                >> i4_shift;
            }

            pi2_src += i4_src_strd;
            pi2_dst++;
        }
        i4_zero_rows = i4_zero_rows >> 1;
    }
}

