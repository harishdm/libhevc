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
*  ihevc_bit_pack_unpack.h
*
* @brief
*  Bit pack and unpack function definations
*
* @author
*  100377
*
* @remarks
*  None
*
*******************************************************************************
*/
#ifndef _IHEVC_BIT_PACK_UNPACK_H_
#define _IHEVC_BIT_PACK_UNPACK_H_

typedef void ihevc_pack_10bit_ft(UWORD16 *src,
    WORD32 src_strd,
    UWORD8 *dst,
    WORD32 wd,
    WORD32 ht);

typedef void ihevc_unpack_10bit_ft(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht);

typedef void ihevc_pack_12bit_ft(UWORD16 *src,
    WORD32 src_strd,
    UWORD8 *dst,
    WORD32 wd,
    WORD32 ht);

typedef void ihevc_unpack_12bit_ft(UWORD16 *pu2_dst,
    WORD32  dst_strd,
    UWORD8  *pu1_src,
    WORD32   wd,
    WORD32   ht);

/* C functions */
ihevc_pack_10bit_ft   ihevc_pack_10bit;
ihevc_pack_12bit_ft   ihevc_pack_12bit;
ihevc_unpack_10bit_ft ihevc_unpack_10bit;
ihevc_unpack_12bit_ft ihevc_unpack_12bit;


/* SSE4.2 Intrinsics functions */
ihevc_pack_10bit_ft   ihevc_pack_10bit_sse42;
ihevc_pack_12bit_ft   ihevc_pack_12bit_sse42;
ihevc_unpack_10bit_ft ihevc_unpack_10bit_sse42;
ihevc_unpack_12bit_ft ihevc_unpack_12bit_sse42;

/* AVX Intrinsics functions */
ihevc_pack_10bit_ft   ihevc_pack_10bit_avx;
ihevc_pack_12bit_ft   ihevc_pack_12bit_avx;
ihevc_unpack_10bit_ft ihevc_unpack_10bit_avx;
ihevc_unpack_12bit_ft ihevc_unpack_12bit_avx;

/* AVX2 Intrinsics functions */
ihevc_pack_10bit_ft   ihevc_pack_10bit_avx2;
ihevc_pack_12bit_ft   ihevc_pack_12bit_avx2;
ihevc_unpack_10bit_ft ihevc_unpack_10bit_avx2;
ihevc_unpack_12bit_ft ihevc_unpack_12bit_avx2;

#endif /*_IHEVC_BIT_PACK_UNPACK_H_*/
