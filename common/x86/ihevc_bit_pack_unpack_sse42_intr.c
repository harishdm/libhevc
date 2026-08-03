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
*  ihevc_bit_pack_unpack_x86_intr.c
*
* @brief
*  Contains Bit Pack Unpack intrinsics
*
* @author
*  Ittiam
*
* @par List of Functions:
*  ihevc_intra_pred_chroma_planar_sse42()
*
*
*
* @remarks
*  None
*
*******************************************************************************
*/

/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/

#include <assert.h>

#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_debug.h"
#include "ihevc_func_selector.h"
#include "ihevc_platform_macros.h"

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

/*!
******************************************************************************
* \if Function name : ihevc_pack_10bit_sse42 \endif
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
void ihevc_pack_10bit_sse42(UWORD16 *pu2_src,
    WORD32 src_strd,
    UWORD8 *pu1_dst,
    WORD32 wd,
    WORD32 ht)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    UWORD8 *pu1_dst_copy;
    WORD32 dst_strd;

    __m128i mask1_8x16b, mask2_8x16b;
    __m128i src1_8x16b, dst1_8x16b;
    __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
    __m128i temp5_8x16b, temp6_8x16b;

    /*Assumption*/
    ASSERT(0 == wd % 8);

    /*Initialisation*/
    pu2_src_copy = pu2_src;
    pu1_dst_copy = pu1_dst;

    dst_strd = (wd * 5)/4;

    PREFETCH((WORD8 const *)pu2_src, _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + 2 * src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + 3 * src_strd), _MM_HINT_T0)

    mask1_8x16b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0E0C0A08, 0x06040200);
    mask2_8x16b = _mm_set_epi32(0xFFFFFFFF, 0xFFFF0A02, 0xFFFFFFFF, 0xFFFFFFFF);

    for (row = 0; row < ht; row++)
    {
        pu2_src = pu2_src_copy + row * src_strd;    /*pointer update*/
        pu1_dst = pu1_dst_copy + row * dst_strd;    /*pointer update*/

        PREFETCH((WORD8 const *)(pu2_src + (row + 4)*src_strd), _MM_HINT_T0)

        for (col = 0; col < wd; col += 8)
        {
            PREFETCH((WORD8 const *)(pu2_src + 32), _MM_HINT_T0)

            src1_8x16b = _mm_loadu_si128((__m128i const*)(pu2_src));

            temp1_8x16b = _mm_shuffle_epi8(src1_8x16b, mask1_8x16b);
            temp2_8x16b = _mm_srli_epi16(src1_8x16b, 8);

            temp3_8x16b = _mm_slli_epi64(temp2_8x16b, 16);
            temp4_8x16b = _mm_slli_epi64(temp2_8x16b, 2);
            temp5_8x16b = _mm_srli_epi64(temp2_8x16b, 12);
            temp6_8x16b = _mm_srli_epi64(temp2_8x16b, 26);

            temp2_8x16b = _mm_or_si128(temp3_8x16b, temp4_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp5_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp6_8x16b);

            temp2_8x16b = _mm_shuffle_epi8(temp2_8x16b, mask2_8x16b);

            dst1_8x16b = _mm_or_si128(temp2_8x16b, temp1_8x16b);

            _mm_storeu_si128((__m128i *)(pu1_dst), dst1_8x16b);

            pu2_src += 8;       /*pointer update*/
            pu1_dst += 10;      /*pointer update*/
        }
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_10bit_sse42 \endif
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
void ihevc_unpack_10bit_sse42(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 row, col, src_strd;
    UWORD8 *pu1_src_copy;
    UWORD16 *pu2_dst_copy;

    __m128i mask1_8x16b, mask2_8x16b, mask3_8x16b;
    __m128i src1_8x16b, dst1_8x16b;
    __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
    __m128i temp5_8x16b;

    /*Assumption*/
    ASSERT(0 == wd % 8);

    /*Initialisation*/
    pu1_src_copy = pu1_src;
    pu2_dst_copy = pu2_dst;

    src_strd = (wd * 5) / 4;

    PREFETCH((WORD8 const *)pu1_src, _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + 2 * src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + 3 * src_strd), _MM_HINT_T0)

    mask1_8x16b = _mm_set_epi32(0xFF07FF06, 0xFF05FF04, 0xFF03FF02, 0xFF01FF00);
    mask2_8x16b = _mm_set_epi32(0x09FFFFFF, 0xFFFFFFFF, 0x08FFFFFF, 0xFFFFFFFF);
    mask3_8x16b = _mm_set_epi32(0x0FFF0DFF, 0x0BFF09FF, 0x07FF05FF, 0x03FF01FF);

    for (row = 0; row < ht; row++)
    {
        pu1_src = pu1_src_copy + row * src_strd;    /*pointer update*/
        pu2_dst = pu2_dst_copy + row * dst_strd;    /*pointer update*/

        PREFETCH((WORD8 const *)(pu1_src + (row + 4)*src_strd), _MM_HINT_T0)

        for (col = 0; col < wd; col += 8)
        {
            src1_8x16b = _mm_loadu_si128((__m128i const*)(pu1_src));

            temp1_8x16b = _mm_shuffle_epi8(src1_8x16b, mask1_8x16b);
            temp2_8x16b = _mm_shuffle_epi8(src1_8x16b, mask2_8x16b);

            temp3_8x16b = _mm_srli_epi64(temp2_8x16b, 42);
            temp4_8x16b = _mm_srli_epi64(temp2_8x16b, 28);
            temp5_8x16b = _mm_srli_epi64(temp2_8x16b, 14);

            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp3_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp4_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp5_8x16b);

            temp3_8x16b = _mm_srli_epi16(temp2_8x16b, 6);
            temp3_8x16b = _mm_shuffle_epi8(temp3_8x16b, mask3_8x16b);

            dst1_8x16b = _mm_or_si128(temp1_8x16b, temp3_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_dst), dst1_8x16b);

            pu1_src += 10;      /*pointer update*/
            pu2_dst += 8;       /*pointer update*/
        }
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_pack_12bit_sse42 \endif
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
void ihevc_pack_12bit_sse42(UWORD16 *pu2_src,
    WORD32 src_strd,
    UWORD8 *pu1_dst,
    WORD32 wd,
    WORD32 ht)
{
    WORD32 row, col;
    UWORD16 *pu2_src_copy;
    UWORD8 *pu1_dst_copy;
    WORD32 dst_strd;

    __m128i mask1_8x16b, mask2_8x16b;
    __m128i src1_8x16b, dst1_8x16b;
    __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
    __m128i temp5_8x16b, temp6_8x16b;

    /*Assumption*/
    ASSERT(0 == wd % 8);

    /*Initialisation*/
    pu2_src_copy = pu2_src;
    pu1_dst_copy = pu1_dst;

    dst_strd = (wd * 3) / 2;

    PREFETCH((WORD8 const *)pu2_src, _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + 2 * src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu2_src + 3 * src_strd), _MM_HINT_T0)

    mask1_8x16b = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0E0C0A08, 0x06040200);
    mask2_8x16b = _mm_set_epi32(0xFFFFFFFF, 0x0B0A0302, 0xFFFFFFFF, 0xFFFFFFFF);

    for (row = 0; row < ht; row++)
    {
        pu2_src = pu2_src_copy + row * src_strd;    /*pointer update*/
        pu1_dst = pu1_dst_copy + row * dst_strd;    /*pointer update*/

        PREFETCH((WORD8 const *)(pu2_src + (row + 4)*src_strd), _MM_HINT_T0)

        for (col = 0; col < wd; col += 8)
        {
            PREFETCH((WORD8 const *)(pu2_src + 32), _MM_HINT_T0)

                src1_8x16b = _mm_loadu_si128((__m128i const*)(pu2_src));

            temp1_8x16b = _mm_shuffle_epi8(src1_8x16b, mask1_8x16b);
            temp2_8x16b = _mm_srli_epi16(src1_8x16b, 8);

            temp3_8x16b = _mm_slli_epi64(temp2_8x16b, 16);
            temp4_8x16b = _mm_slli_epi64(temp2_8x16b, 4);
            temp5_8x16b = _mm_srli_epi64(temp2_8x16b, 8);
            temp6_8x16b = _mm_srli_epi64(temp2_8x16b, 20);

            temp2_8x16b = _mm_or_si128(temp3_8x16b, temp4_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp5_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp6_8x16b);

            temp2_8x16b = _mm_shuffle_epi8(temp2_8x16b, mask2_8x16b);

            dst1_8x16b = _mm_or_si128(temp2_8x16b, temp1_8x16b);

            _mm_storeu_si128((__m128i *)(pu1_dst), dst1_8x16b);

            pu2_src += 8;       /*pointer update*/
            pu1_dst += 12;      /*pointer update*/

        }
    }
}

/*!
******************************************************************************
* \if Function name : ihevc_unpack_12bit_sse42 \endif
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
void ihevc_unpack_12bit_sse42(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht)
{
    WORD32 row, col, src_strd;
    UWORD8 *pu1_src_copy;
    UWORD16 *pu2_dst_copy;

    __m128i mask1_8x16b, mask2_8x16b, mask3_8x16b;
    __m128i src1_8x16b, dst1_8x16b;
    __m128i temp1_8x16b, temp2_8x16b, temp3_8x16b, temp4_8x16b;
    __m128i temp5_8x16b;

    /*Assumption*/
    ASSERT(0 == wd % 8);

    /*Initialisation*/
    pu1_src_copy = pu1_src;
    pu2_dst_copy = pu2_dst;

    src_strd = (wd * 3) / 2;

    PREFETCH((WORD8 const *)pu1_src, _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + 2 * src_strd), _MM_HINT_T0)
    PREFETCH((WORD8 const *)(pu1_src + 3 * src_strd), _MM_HINT_T0)

    mask1_8x16b = _mm_set_epi32(0xFF07FF06, 0xFF05FF04, 0xFF03FF02, 0xFF01FF00);
    mask2_8x16b = _mm_set_epi32(0x0B0AFFFF, 0xFFFFFFFF, 0x0908FFFF, 0xFFFFFFFF);
    mask3_8x16b = _mm_set_epi32(0x0FFF0DFF, 0x0BFF09FF, 0x07FF05FF, 0x03FF01FF);

    for (row = 0; row < ht; row++)
    {
        pu1_src = pu1_src_copy + row * src_strd;    /*pointer update*/
        pu2_dst = pu2_dst_copy + row * dst_strd;    /*pointer update*/

        PREFETCH((WORD8 const *)(pu1_src + (row + 4)*src_strd), _MM_HINT_T0)

        for (col = 0; col < wd; col += 8)
        {
            src1_8x16b = _mm_loadu_si128((__m128i const*)(pu1_src));

            temp1_8x16b = _mm_shuffle_epi8(src1_8x16b, mask1_8x16b);
            temp2_8x16b = _mm_shuffle_epi8(src1_8x16b, mask2_8x16b);

            temp3_8x16b = _mm_srli_epi64(temp2_8x16b, 36);
            temp4_8x16b = _mm_srli_epi64(temp2_8x16b, 24);
            temp5_8x16b = _mm_srli_epi64(temp2_8x16b, 12);

            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp3_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp4_8x16b);
            temp2_8x16b = _mm_or_si128(temp2_8x16b, temp5_8x16b);

            temp3_8x16b = _mm_srli_epi16(temp2_8x16b, 4);
            temp3_8x16b = _mm_shuffle_epi8(temp3_8x16b, mask3_8x16b);

            dst1_8x16b = _mm_or_si128(temp1_8x16b, temp3_8x16b);

            _mm_storeu_si128((__m128i *)(pu2_dst), dst1_8x16b);

            pu1_src += 12;      /*pointer update*/
            pu2_dst += 8;       /*pointer update*/
        }
    }
}
