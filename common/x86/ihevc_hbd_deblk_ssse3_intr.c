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
*  ihevc_deblck_atom_intr.c
*
* @brief
*  Contains function definitions for deblocking filters
*
* @author
*  Ittiam
*
* @par List of Functions:
*   - ihevc_deblk_422chroma_horz_ssse3()
*   - ihevc_deblk_422chroma_vert_ssse3()
*
* @remarks
*  None
*
*******************************************************************************
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "ihevc_typedefs.h"
#include "ihevc_platform_macros.h"
#include "ihevc_macros.h"
#include "ihevc_deblk.h"
#include "ihevc_deblk_tables.h"
#include "ihevc_debug.h"

#include "ihevc_tables_x86_intr.h"

#include <immintrin.h>

void ihevc_deblk_422chroma_vert_ssse3(UWORD8 *pu1_src,
                                      WORD32 src_strd,
                                      WORD32 quant_param_p,
                                      WORD32 quant_param_q,
                                      WORD32 qp_offset_u,
                                      WORD32 qp_offset_v,
                                      WORD32 tc_offset_div2,
                                      WORD32 filter_flag_p,
                                      WORD32 filter_flag_q)
{
    WORD32 qp_indx_u, qp_chroma_u;
    WORD32 qp_indx_v, qp_chroma_v;
    WORD32 tc_indx_u, tc_u;
    WORD32 tc_indx_v, tc_v;

    __m128i src_row_0_16x8b, tmp_pxl_0_16x8b, src_row_2_16x8b, tmp_pxl_1_16x8b;
    ASSERT(filter_flag_p || filter_flag_q);

    /* chroma processing is done only if BS is 2             */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_chroma_u = MIN(qp_indx_u, 51);

    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_chroma_v = MIN(qp_indx_v, 51);

    tc_indx_u = CLIP3(qp_chroma_u + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_u = gai4_ihevc_tc_table[tc_indx_u];

    tc_indx_v = CLIP3(qp_chroma_v + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_v = gai4_ihevc_tc_table[tc_indx_v];

    if(0 == tc_u && 0 == tc_v)
    {
        return;
    }
    src_row_0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src - 4));
    tmp_pxl_0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src + src_strd - 4));
    src_row_2_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src + 2 * src_strd - 4));
    tmp_pxl_1_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src + 3 * src_strd - 4));

    {
        LWORD64 mask_tc, mask_flag, mask;
        __m128i delta_vu0_16x8b, delta_vu1_16x8b;
        __m128i mask_tc_16x8, mask_16x8b, mask_flag_p_16x8b, mask_flag_q_16x8b;
        __m128i min_0_16x8b;
        __m128i const_16x8b;
        mask_flag = (((LWORD64)filter_flag_p) << 31) | (((LWORD64)filter_flag_q) << 63);
        mask_tc = (((LWORD64)tc_v) << 16) | ((LWORD64)tc_u);
        mask = 0xffff00000000ffffLL;

        src_row_0_16x8b = _mm_unpacklo_epi64(src_row_0_16x8b, tmp_pxl_0_16x8b);
        src_row_2_16x8b = _mm_unpacklo_epi64(src_row_2_16x8b, tmp_pxl_1_16x8b);

        mask_16x8b = _mm_load_si128((__m128i *)(shuffle_uv));
        // qv11 qu11 qv10 qu10 qv01 qu01 qv00 qu00 pv10 pu10 pv11 pu11 pv00 pu00 pv01 pu01
        // qv31 qu31 qv30 qu30 qv21 qu21 qv20 qu20 pv30 pu30 pv31 pu31 pv20 pu20 pv21 pu21
        delta_vu0_16x8b = _mm_shuffle_epi8(src_row_0_16x8b, mask_16x8b);
        delta_vu1_16x8b = _mm_shuffle_epi8(src_row_2_16x8b, mask_16x8b);

        tmp_pxl_0_16x8b = _mm_unpacklo_epi64(delta_vu0_16x8b, delta_vu1_16x8b);
        tmp_pxl_1_16x8b = _mm_unpackhi_epi64(delta_vu0_16x8b, delta_vu1_16x8b);
        // pv30 pv31 pu30 pu31 pv20 pv21 pu20 pu21 pv10 pv11 pu10 pu11 pv00 pv01 pu00 pu01
        // qv31 qv30 qu31 qu30 qv21 qv20 qu21 qu20 qv11 qv10 qu11 qu10 qv01 qv00 qu01 qu00
        delta_vu0_16x8b = _mm_load_si128((__m128i *)delta0);
        delta_vu1_16x8b = _mm_load_si128((__m128i *)delta1);

        delta_vu0_16x8b = _mm_maddubs_epi16(tmp_pxl_0_16x8b, delta_vu0_16x8b);
        delta_vu1_16x8b = _mm_maddubs_epi16(tmp_pxl_1_16x8b, delta_vu1_16x8b);

        //generating offset 4
        const_16x8b = _mm_cmpeq_epi16(tmp_pxl_0_16x8b, tmp_pxl_0_16x8b);
        // filter flag mask and tc mask
        mask_tc_16x8 = _mm_loadl_epi64((__m128i *)(&mask_tc));
        mask_flag_q_16x8b = _mm_loadl_epi64((__m128i *)(&mask_flag));

        mask_tc_16x8 = _mm_shuffle_epi32(mask_tc_16x8, 0x00);
        mask_flag_q_16x8b = _mm_srai_epi32(mask_flag_q_16x8b, 31);
        //-tc
        min_0_16x8b = _mm_sign_epi16(mask_tc_16x8, const_16x8b);
        //converting const 1
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        //filterp and filterq flag
        mask_flag_p_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x00);
        mask_flag_q_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x55);

        //modified delta with a filter (1 -4 4 -1) available in 16 bit
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, delta_vu1_16x8b);
        //converting const 4
        const_16x8b = _mm_slli_epi16(const_16x8b, 2);

        mask_16x8b = _mm_loadl_epi64((__m128i *)(&mask));
        //offset addition
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, const_16x8b);
        //eliminating q1
        tmp_pxl_1_16x8b = _mm_slli_epi16(tmp_pxl_1_16x8b, 8);

        const_16x8b = _mm_setzero_si128();
        //filter after normalisation
        delta_vu0_16x8b = _mm_srai_epi16(delta_vu0_16x8b, 3);
        mask_16x8b = _mm_shuffle_epi32(mask_16x8b, 0x44);

        //clipping MAX
        delta_vu0_16x8b = _mm_min_epi16(delta_vu0_16x8b, mask_tc_16x8);
        //getting p0 and eliminating p1
        tmp_pxl_0_16x8b = _mm_srli_epi16(tmp_pxl_0_16x8b, 8);
        //clipping MIN
        delta_vu0_16x8b = _mm_max_epi16(delta_vu0_16x8b, min_0_16x8b);
        //getting q0
        tmp_pxl_1_16x8b = _mm_srli_epi16(tmp_pxl_1_16x8b, 8);
        //masking filter flag
        delta_vu1_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_q_16x8b);
        delta_vu0_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_p_16x8b);

        // q-delta ,p+delta
        tmp_pxl_1_16x8b = _mm_sub_epi16(tmp_pxl_1_16x8b, delta_vu1_16x8b);
        tmp_pxl_0_16x8b = _mm_add_epi16(tmp_pxl_0_16x8b, delta_vu0_16x8b);
        //merging q0 and p0 of respective rows
        delta_vu1_16x8b = _mm_unpackhi_epi32(tmp_pxl_0_16x8b, tmp_pxl_1_16x8b);
        delta_vu0_16x8b = _mm_unpacklo_epi32(tmp_pxl_0_16x8b, tmp_pxl_1_16x8b);
        // row 0 and row 1 packed , row2 and row3 packed
        delta_vu0_16x8b = _mm_packus_epi16(delta_vu0_16x8b, const_16x8b);
        delta_vu1_16x8b = _mm_packus_epi16(delta_vu1_16x8b, const_16x8b);
        //removing older pixel values
        src_row_0_16x8b = _mm_and_si128(src_row_0_16x8b, mask_16x8b);
        src_row_2_16x8b = _mm_and_si128(src_row_2_16x8b, mask_16x8b);
        //arranging modified pixels
        delta_vu0_16x8b = _mm_shuffle_epi32(delta_vu0_16x8b, 0xd8);
        delta_vu1_16x8b = _mm_shuffle_epi32(delta_vu1_16x8b, 0xd8);
        delta_vu0_16x8b = _mm_slli_epi64(delta_vu0_16x8b, 16);
        delta_vu1_16x8b = _mm_slli_epi64(delta_vu1_16x8b, 16);
        //plugging the modified values
        src_row_0_16x8b = _mm_or_si128(src_row_0_16x8b, delta_vu0_16x8b);
        src_row_2_16x8b = _mm_or_si128(src_row_2_16x8b, delta_vu1_16x8b);

        //geting values for row1 and row 3
        tmp_pxl_0_16x8b = _mm_srli_si128(src_row_0_16x8b, 8);
        tmp_pxl_1_16x8b = _mm_srli_si128(src_row_2_16x8b, 8);

        _mm_storel_epi64((__m128i *)(pu1_src - 4), src_row_0_16x8b);
        _mm_storel_epi64((__m128i *)((pu1_src - 4) + src_strd), tmp_pxl_0_16x8b);
        _mm_storel_epi64((__m128i *)((pu1_src - 4) + 2 * src_strd), src_row_2_16x8b);
        _mm_storel_epi64((__m128i *)((pu1_src - 4) + 3 * src_strd), tmp_pxl_1_16x8b);
    }
}


void ihevc_deblk_422chroma_horz_ssse3(UWORD8 *pu1_src,
                                      WORD32 src_strd,
                                      WORD32 quant_param_p,
                                      WORD32 quant_param_q,
                                      WORD32 qp_offset_u,
                                      WORD32 qp_offset_v,
                                      WORD32 tc_offset_div2,
                                      WORD32 filter_flag_p,
                                      WORD32 filter_flag_q)
{
    WORD32 qp_indx_u, qp_chroma_u;
    WORD32 qp_indx_v, qp_chroma_v;
    WORD32 tc_indx_u, tc_u;
    WORD32 tc_indx_v, tc_v;

    __m128i tmp_p0_16x8b, src_p0_16x8b, src_q0_16x8b, tmp_q0_16x8b;

    ASSERT(filter_flag_p || filter_flag_q);

    /* chroma processing is done only if BS is 2             */
    /* this function is assumed to be called only if BS is 2 */
    qp_indx_u = qp_offset_u + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_chroma_u = MIN(qp_indx_u, 51);

    qp_indx_v = qp_offset_v + ((quant_param_p + quant_param_q + 1) >> 1);
    qp_chroma_v = MIN(qp_indx_v, 51);

    tc_indx_u = CLIP3(qp_chroma_u + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_u = gai4_ihevc_tc_table[tc_indx_u];

    tc_indx_v = CLIP3(qp_chroma_v + 2 + (tc_offset_div2 << 1), 0, 53);
    tc_v = gai4_ihevc_tc_table[tc_indx_v];

    if(0 == tc_u && 0 == tc_v)
    {
        return;
    }
    tmp_p0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src - 2 * src_strd));
    src_p0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src - src_strd));
    src_q0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src));
    tmp_q0_16x8b = _mm_loadl_epi64((__m128i *)(pu1_src + src_strd));

    {
        LWORD64 mask_tc, mask_flag;
        __m128i delta_vu0_16x8b, delta_vu1_16x8b;
        __m128i mask_tc_16x8, mask_16x8b, mask_flag_p_16x8b, mask_flag_q_16x8b;
        __m128i min_0_16x8b;
        __m128i const_16x8b;
        mask_flag = (((LWORD64)filter_flag_p) << 31) | (((LWORD64)filter_flag_q) << 63);
        mask_tc = (((LWORD64)tc_v) << 16) | ((LWORD64)tc_u);

        tmp_p0_16x8b = _mm_unpacklo_epi8(tmp_p0_16x8b, src_p0_16x8b);
        tmp_q0_16x8b = _mm_unpacklo_epi8(src_q0_16x8b, tmp_q0_16x8b);

        // pv30 pv31 pu30 pu31 pv20 pv21 pu20 pu21 pv10 pv11 pu10 pu11 pv00 pv01 pu00 pu01
        // qv31 qv30 qu31 qu30 qv21 qv20 qu21 qu20 qv11 qv10 qu11 qu10 qv01 qv00 qu01 qu00
        delta_vu0_16x8b = _mm_load_si128((__m128i *)delta0);
        delta_vu1_16x8b = _mm_load_si128((__m128i *)delta1);

        delta_vu0_16x8b = _mm_maddubs_epi16(tmp_p0_16x8b, delta_vu0_16x8b);
        delta_vu1_16x8b = _mm_maddubs_epi16(tmp_q0_16x8b, delta_vu1_16x8b);

        // filter flag mask and tc mask
        mask_tc_16x8 = _mm_loadl_epi64((__m128i *)(&mask_tc));
        mask_flag_q_16x8b = _mm_loadl_epi64((__m128i *)(&mask_flag));

        //generating offset 4
        const_16x8b = _mm_cmpeq_epi16(tmp_p0_16x8b, tmp_p0_16x8b);
        // filter flag mask and tc mask
        mask_tc_16x8 = _mm_shuffle_epi32(mask_tc_16x8, 0x00);
        mask_flag_q_16x8b = _mm_srai_epi32(mask_flag_q_16x8b, 31);
        //-tc
        min_0_16x8b = _mm_sign_epi16(mask_tc_16x8, const_16x8b);
        //converting const 1
        const_16x8b = _mm_srli_epi16(const_16x8b, 15);

        //filterp
        mask_flag_p_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x00);

        //converting const 4
        const_16x8b = _mm_slli_epi16(const_16x8b, 2);
        //modified delta with a filter (1 -4 4 -1) available in 16 bit
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, delta_vu1_16x8b);

        //filterq flag
        mask_flag_q_16x8b = _mm_shuffle_epi32(mask_flag_q_16x8b, 0x55);
        //offset addition
        delta_vu0_16x8b = _mm_add_epi16(delta_vu0_16x8b, const_16x8b);
        mask_16x8b = _mm_setzero_si128();
        //filter after normalisation
        delta_vu0_16x8b = _mm_srai_epi16(delta_vu0_16x8b, 3);

        //converting p0 to 16bit
        src_p0_16x8b = _mm_unpacklo_epi8(src_p0_16x8b, mask_16x8b);
        //clipping MAX
        delta_vu0_16x8b = _mm_min_epi16(delta_vu0_16x8b, mask_tc_16x8);
        //converting q0 to 16bit
        src_q0_16x8b = _mm_unpacklo_epi8(src_q0_16x8b, mask_16x8b);
        //clipping MIN
        delta_vu0_16x8b = _mm_max_epi16(delta_vu0_16x8b, min_0_16x8b);

        //masking filter flag
        delta_vu1_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_q_16x8b);
        delta_vu0_16x8b = _mm_and_si128(delta_vu0_16x8b, mask_flag_p_16x8b);

        // q-delta ,p+delta
        src_q0_16x8b = _mm_sub_epi16(src_q0_16x8b, delta_vu1_16x8b);
        src_p0_16x8b = _mm_add_epi16(src_p0_16x8b, delta_vu0_16x8b);

        // p0 and q0 packed
        src_q0_16x8b = _mm_packus_epi16(src_q0_16x8b, mask_16x8b);
        src_p0_16x8b = _mm_packus_epi16(src_p0_16x8b, mask_16x8b);

        _mm_storel_epi64((__m128i *)(pu1_src - src_strd), src_p0_16x8b);
        _mm_storel_epi64((__m128i *)(pu1_src), src_q0_16x8b);
    }
}
