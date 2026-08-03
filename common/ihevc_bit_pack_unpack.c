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
*  ihevc_bit_pack_unpack.c
*
* @brief
*  Contains Bit Pack Unpack functions
*
* @author
*  100377
*
* @remarks
*  None
*
*******************************************************************************
*/

#include "ihevc_typedefs.h"
#include "ihevc_platform_macros.h"
#include "ihevc_func_selector.h"
#include "ihevc_defs.h"

#if 1 //old implementation
/*!
******************************************************************************
* \if Function name : ihevc_pack_10bit \endif
*
* \brief
*   Packs 10 bit data into a linear packed buffer without any holes
*  destination is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_pack_10bit(UWORD16 *pu2_src,
    WORD32  src_strd,
    UWORD8  *pu1_dst,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i,i4_sw;

    for(i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 4)
        {
            /* 4 pixels (40 bits) are packed into 5 bytes */
            *pu1_dst++ = (pu2_src[i4_sw    ] >> 2);
            *pu1_dst++ = (pu2_src[i4_sw + 1] >> 4) | (pu2_src[i4_sw    ] << 6);
            *pu1_dst++ = (pu2_src[i4_sw + 1] << 4) | (pu2_src[i4_sw + 2] >> 6);
            *pu1_dst++ = (pu2_src[i4_sw + 2] << 2) | (pu2_src[i4_sw + 3] >> 8);
            *pu1_dst++ = (pu2_src[i4_sw + 3]     );
        }
        pu2_src += src_strd;
    }
    return;
}

/*!
******************************************************************************
* \if Function name : ihevc_pack_12bit \endif
*
* \brief
*   Packs 12 bit data into a linear packed buffer without any holes
*  destination is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_pack_12bit(UWORD16 *pu2_src,
    WORD32  src_strd,
    UWORD8  *pu1_dst,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i,i4_sw;

    for(i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 2)
        {
            /* 2 pixels (24 bits) are packed into 3 bytes */
            *pu1_dst++ = (pu2_src[i4_sw    ] >> 4);
            *pu1_dst++ = (pu2_src[i4_sw + 1] >> 8) | (pu2_src[i4_sw] << 4);
            *pu1_dst++ = (pu2_src[i4_sw + 1]     );
        }
        pu2_src += src_strd;
    }
    return;
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_10bit \endif
*
* \brief
*  UnPacks linear packed buffer without any holes to 10 bit (2 byte) buffer
*  source is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_unpack_10bit(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i,i4_sw;

    for(i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 4)
        {
            UWORD16 u2_val;

            /* 5 bytes are unpacked into 4 pixels (40 bits) */
            u2_val             = (pu1_src[0] << 2);
            u2_val             = u2_val | (pu1_src[1] >> 6);
            pu2_dst[i4_sw + 0] = u2_val & 0x03FF;

            u2_val             = (pu1_src[1] << 4);
            u2_val             = u2_val | (pu1_src[2] >> 4);
            pu2_dst[i4_sw + 1] = u2_val & 0x03FF;

            u2_val             = (pu1_src[2] << 6);
            u2_val             = u2_val | (pu1_src[3] >> 2);
            pu2_dst[i4_sw + 2] = u2_val & 0x03FF;

            u2_val             = (pu1_src[3] << 8);
            u2_val             = u2_val | (pu1_src[4]);
            pu2_dst[i4_sw + 3] = u2_val & 0x03FF;

            pu1_src += 5;
        }
        pu2_dst += dst_strd;
    }
    return;
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_12bit \endif
*
* \brief
*  UnPacks linear packed buffer without any holes to 12 bit (2 byte) buffer
*  source is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_unpack_12bit(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i,i4_sw;

    for(i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 2)
        {
            UWORD16 u2_val;

            /* 3 bytes are unpacked into 2 pixels (24 bits) */
            u2_val             = (pu1_src[0] << 4);
            u2_val             = u2_val | (pu1_src[1] >> 4);
            pu2_dst[i4_sw + 0] = u2_val & 0x0FFF;

            u2_val             = (pu1_src[1] << 8) ;
            u2_val             = u2_val | (pu1_src[2]);
            pu2_dst[i4_sw + 1] = u2_val & 0x0FFF;

            pu1_src += 3;
        }
        pu2_dst += dst_strd;
    }
    return;
}

#else
/*!
******************************************************************************
* \if Function name : ihevc_pack_10bit \endif
*
* \brief
*   Packs 10 bit data into a linear packed buffer without any holes
*  destination is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_pack_10bit(UWORD16 *src,
    WORD32 src_strd,
    UWORD8 *dst,
    WORD32 wd,
    WORD32 ht)
{
    WORD32 i, i4_dw, i4_sw;
    WORD32 dst_strd;

    dst_strd = (wd * 5)/4;

    for (i = 0; i < ht; i++)
    {
        i4_dw = 0;
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 8)
        {
            dst[i4_dw] = src[i4_sw];
            dst[i4_dw + 1] = src[i4_sw + 1];
            dst[i4_dw + 2] = src[i4_sw + 2];
            dst[i4_dw + 3] = src[i4_sw + 3];
            dst[i4_dw + 4] = src[i4_sw + 4];
            dst[i4_dw + 5] = src[i4_sw + 5];
            dst[i4_dw + 6] = src[i4_sw + 6];
            dst[i4_dw + 7] = src[i4_sw + 7];
            dst[i4_dw + 8] = ((src[i4_sw] >> 8) & 0x03) | ((src[i4_sw + 1] >> 6) & 0x0C) | ((src[i4_sw + 2] >> 4) & 0x30) | ((src[i4_sw + 3] >> 2) & 0xC0);
            dst[i4_dw + 9] = ((src[i4_sw + 4] >> 8) & 0x03) | ((src[i4_sw + 5] >> 6) & 0x0C) | ((src[i4_sw + 6] >> 4) & 0x30) | ((src[i4_sw + 7] >> 2) & 0xC0);

            i4_dw = i4_dw + 10;
        }
        dst += dst_strd;
        src += src_strd;
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_10bit \endif
*
* \brief
*  UnPacks linear packed buffer without any holes to 10 bit (2 byte) buffer
*  source is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_unpack_10bit(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i, i4_sw;

    for (i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 8)
        {
            /* 10 bytes are unpacked into 8 pixels (80 bits) */
            pu2_dst[i4_sw + 0] = pu1_src[0] | ((pu1_src[8] << 8) & 0x0300);
            pu2_dst[i4_sw + 1] = pu1_src[1] | ((pu1_src[8] << 6) & 0x0300);
            pu2_dst[i4_sw + 2] = pu1_src[2] | ((pu1_src[8] << 4) & 0x0300);
            pu2_dst[i4_sw + 3] = pu1_src[3] | ((pu1_src[8] << 2) & 0x0300);
            pu2_dst[i4_sw + 4] = pu1_src[4] | ((pu1_src[9] << 8) & 0x0300);
            pu2_dst[i4_sw + 5] = pu1_src[5] | ((pu1_src[9] << 6) & 0x0300);
            pu2_dst[i4_sw + 6] = pu1_src[6] | ((pu1_src[9] << 4) & 0x0300);
            pu2_dst[i4_sw + 7] = pu1_src[7] | ((pu1_src[9] << 2) & 0x0300);

            pu1_src += 10;
        }
        pu2_dst += dst_strd;
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_pack_12bit \endif
*
* \brief
*   Packs 12 bit data into a linear packed buffer without any holes
*  destination is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_pack_12bit(UWORD16 *src,
    WORD32 src_strd,
    UWORD8 *dst,
    WORD32 wd,
    WORD32 ht)
{
    WORD32 i, i4_dw, i4_sw;
    WORD32 dst_strd;

    dst_strd = (wd * 3) / 2;

    for (i = 0; i < ht; i++)
    {
        i4_dw = 0;
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 8)
        {
            dst[i4_dw] = src[i4_sw];
            dst[i4_dw + 1] = src[i4_sw + 1];
            dst[i4_dw + 2] = src[i4_sw + 2];
            dst[i4_dw + 3] = src[i4_sw + 3];
            dst[i4_dw + 4] = src[i4_sw + 4];
            dst[i4_dw + 5] = src[i4_sw + 5];
            dst[i4_dw + 6] = src[i4_sw + 6];
            dst[i4_dw + 7] = src[i4_sw + 7];
            dst[i4_dw + 8] = ((src[i4_sw] >> 8) & 0x0F) | ((src[i4_sw + 1] >> 4) & 0xF0);
            dst[i4_dw + 9] = ((src[i4_sw + 2] >> 8) & 0x0F) | ((src[i4_sw + 3] >> 4) & 0xF0);
            dst[i4_dw + 10] = ((src[i4_sw + 4] >> 8) & 0x0F) | ((src[i4_sw + 5] >> 4) & 0xF0);
            dst[i4_dw + 11] = ((src[i4_sw + 6] >> 8) & 0x0F) | ((src[i4_sw + 7] >> 4) & 0xF0);

            i4_dw = i4_dw + 12;
        }
        dst += dst_strd;
        src += src_strd;
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_12bit \endif
*
* \brief
*  UnPacks linear packed buffer without any holes to 12 bit (2 byte) buffer
*  source is assumed to be linear buffer without any strides
*
* \return
*    None
*
* \author
*  Ittiam
*
*****************************************************************************
*/
void ihevc_unpack_12bit(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 i4_i, i4_sw;

    for (i4_i = 0; i4_i < ht; i4_i++)
    {
        for (i4_sw = 0; i4_sw < wd; i4_sw = i4_sw + 8)
        {
            /* 12 bytes are unpacked into 8 pixels (80 bits) */
            pu2_dst[i4_sw + 0] = pu1_src[0] | ((pu1_src[8] << 8) & 0x0F00);
            pu2_dst[i4_sw + 1] = pu1_src[1] | ((pu1_src[8] << 4) & 0x0F00);
            pu2_dst[i4_sw + 2] = pu1_src[2] | ((pu1_src[9] << 8) & 0x0F00);
            pu2_dst[i4_sw + 3] = pu1_src[3] | ((pu1_src[9] << 4) & 0x0F00);
            pu2_dst[i4_sw + 4] = pu1_src[4] | ((pu1_src[10] << 8) & 0x0F00);
            pu2_dst[i4_sw + 5] = pu1_src[5] | ((pu1_src[10] << 4) & 0x0F00);
            pu2_dst[i4_sw + 6] = pu1_src[6] | ((pu1_src[11] << 8) & 0x0F00);
            pu2_dst[i4_sw + 7] = pu1_src[7] | ((pu1_src[11] << 4) & 0x0F00);

            pu1_src += 12;
        }
        pu2_dst += dst_strd;
    }
}
#endif

