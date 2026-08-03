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
*  ihevc_quant.h
*
* @brief
*  Functions declarations for quantization
*
* @author
*  Ittiam
*
* @remarks
*  None
*
*******************************************************************************
*/


#ifndef _IHEVC_QUANT_H_
#define _IHEVC_QUANT_H_

typedef WORD32 ihevc_quant_4x4_ttype1_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row);

typedef WORD32 ihevc_quant_4x4_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_8x8_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_16x16_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_32x32_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_4x4_ttype1_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_4x4_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_8x8_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_16x16_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_32x32_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_4x4_ttype1_rdoq_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row);

typedef WORD32 ihevc_quant_4x4_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_8x8_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_16x16_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_32x32_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_4x4_ttype1_rdoq_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_4x4_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_8x8_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_16x16_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);

typedef WORD32 ihevc_quant_flat_scale_mat_32x32_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row);
typedef WORD32 ihevc_hbd_quant_4x4_ttype1_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row,
                              UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_4x4_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row,
                       UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_8x8_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row,
                       UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_16x16_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row,
                         UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_32x32_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row,
                         UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth
    );

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_4x4_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth
    );

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_8x8_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth
    );

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_16x16_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth
    );

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_32x32_ft(WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_4x4_ttype1_rdoq_ft(WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_4x4_rdoq_ft(WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_8x8_rdoq_ft(WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_16x16_rdoq_ft(WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_32x32_rdoq_ft(WORD16 *pi2_coeffs,
                            WORD16 *pi2_quant_coeff,
                            WORD16 *pi2_dst,
                            WORD32 qp_div,/* qpscaled / 6 */
                            WORD32 qp_rem,/* qpscaled % 6 */
                            WORD32 q_add,
                            WORD32 src_strd,
                            WORD32 dst_strd,
                            UWORD8 *csbf,
                            WORD32 csbf_strd,
                              WORD32 *zero_col,
                              WORD32 *zero_row,
                              UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_rdoq_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_dst,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    UWORD8 bit_depth
    );

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_4x4_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row,
                       UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_8x8_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                       WORD32 *zero_col,
                       WORD32 *zero_row,
                       UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_16x16_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row,
                         UWORD8 bit_depth);

typedef WORD32 ihevc_hbd_quant_flat_scale_mat_32x32_rdoq_ft(WORD16 *pi2_coeffs,
                     WORD16 *pi2_quant_coeff,
                     WORD16 *pi2_dst,
                     WORD32 qp_div,/* qpscaled / 6 */
                     WORD32 qp_rem,/* qpscaled % 6 */
                     WORD32 q_add,
                     WORD32 src_strd,
                     WORD32 dst_strd,
                     UWORD8 *csbf,
                     WORD32 csbf_strd,
                         WORD32 *zero_col,
                         WORD32 *zero_row,
                         UWORD8 bit_depth);

ihevc_quant_4x4_ttype1_ft ihevc_quant_4x4_ttype1;
ihevc_quant_4x4_ft ihevc_quant_4x4;
ihevc_quant_8x8_ft ihevc_quant_8x8;
ihevc_quant_16x16_ft ihevc_quant_16x16;
ihevc_quant_32x32_ft ihevc_quant_32x32;
ihevc_quant_flat_scale_mat_4x4_ttype1_ft ihevc_quant_flat_scale_mat_4x4_ttype1;
ihevc_quant_flat_scale_mat_4x4_ft ihevc_quant_flat_scale_mat_4x4;
ihevc_quant_flat_scale_mat_8x8_ft ihevc_quant_flat_scale_mat_8x8;
ihevc_quant_flat_scale_mat_16x16_ft ihevc_quant_flat_scale_mat_16x16;
ihevc_quant_flat_scale_mat_32x32_ft ihevc_quant_flat_scale_mat_32x32;
ihevc_quant_4x4_ttype1_rdoq_ft ihevc_quant_4x4_ttype1_rdoq;
ihevc_quant_4x4_rdoq_ft ihevc_quant_4x4_rdoq;
ihevc_quant_8x8_rdoq_ft ihevc_quant_8x8_rdoq;
ihevc_quant_16x16_rdoq_ft ihevc_quant_16x16_rdoq;
ihevc_quant_32x32_rdoq_ft ihevc_quant_32x32_rdoq;
ihevc_quant_flat_scale_mat_4x4_ttype1_rdoq_ft ihevc_quant_flat_scale_mat_4x4_ttype1_rdoq;
ihevc_quant_flat_scale_mat_4x4_rdoq_ft ihevc_quant_flat_scale_mat_4x4_rdoq;
ihevc_quant_flat_scale_mat_8x8_rdoq_ft ihevc_quant_flat_scale_mat_8x8_rdoq;
ihevc_quant_flat_scale_mat_16x16_rdoq_ft ihevc_quant_flat_scale_mat_16x16_rdoq;
ihevc_quant_flat_scale_mat_32x32_rdoq_ft ihevc_quant_flat_scale_mat_32x32_rdoq;

ihevc_quant_4x4_ttype1_ft ihevc_quant_4x4_ttype1_sse42;
ihevc_quant_4x4_ft ihevc_quant_4x4_sse42;
ihevc_quant_8x8_ft ihevc_quant_8x8_sse42;
ihevc_quant_16x16_ft ihevc_quant_16x16_sse42;
ihevc_quant_32x32_ft ihevc_quant_32x32_sse42;
ihevc_quant_flat_scale_mat_4x4_ttype1_ft ihevc_quant_flat_scale_mat_4x4_ttype1_sse42;
ihevc_quant_flat_scale_mat_4x4_ft ihevc_quant_flat_scale_mat_4x4_sse42;
ihevc_quant_flat_scale_mat_8x8_ft ihevc_quant_flat_scale_mat_8x8_sse42;
ihevc_quant_flat_scale_mat_16x16_ft ihevc_quant_flat_scale_mat_16x16_sse42;
ihevc_quant_flat_scale_mat_32x32_ft ihevc_quant_flat_scale_mat_32x32_sse42;

ihevc_hbd_quant_4x4_ttype1_ft ihevc_hbd_quant_4x4_ttype1;
ihevc_hbd_quant_4x4_ft ihevc_hbd_quant_4x4;
ihevc_hbd_quant_8x8_ft ihevc_hbd_quant_8x8;
ihevc_hbd_quant_16x16_ft ihevc_hbd_quant_16x16;
ihevc_hbd_quant_32x32_ft ihevc_hbd_quant_32x32;
ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_ft ihevc_hbd_quant_flat_scale_mat_4x4_ttype1;
ihevc_hbd_quant_flat_scale_mat_4x4_ft ihevc_hbd_quant_flat_scale_mat_4x4;
ihevc_hbd_quant_flat_scale_mat_8x8_ft ihevc_hbd_quant_flat_scale_mat_8x8;
ihevc_hbd_quant_flat_scale_mat_16x16_ft ihevc_hbd_quant_flat_scale_mat_16x16;
ihevc_hbd_quant_flat_scale_mat_32x32_ft ihevc_hbd_quant_flat_scale_mat_32x32;

ihevc_hbd_quant_4x4_ttype1_rdoq_ft ihevc_hbd_quant_4x4_ttype1_rdoq;
ihevc_hbd_quant_4x4_rdoq_ft ihevc_hbd_quant_4x4_rdoq;
ihevc_hbd_quant_8x8_rdoq_ft ihevc_hbd_quant_8x8_rdoq;
ihevc_hbd_quant_16x16_rdoq_ft ihevc_hbd_quant_16x16_rdoq;
ihevc_hbd_quant_32x32_rdoq_ft ihevc_hbd_quant_32x32_rdoq;
ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_rdoq_ft ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_rdoq;
ihevc_hbd_quant_flat_scale_mat_4x4_rdoq_ft ihevc_hbd_quant_flat_scale_mat_4x4_rdoq;
ihevc_hbd_quant_flat_scale_mat_8x8_rdoq_ft ihevc_hbd_quant_flat_scale_mat_8x8_rdoq;
ihevc_hbd_quant_flat_scale_mat_16x16_rdoq_ft ihevc_hbd_quant_flat_scale_mat_16x16_rdoq;
ihevc_hbd_quant_flat_scale_mat_32x32_rdoq_ft ihevc_hbd_quant_flat_scale_mat_32x32_rdoq;


ihevc_hbd_quant_4x4_ttype1_ft ihevc_hbd_quant_4x4_ttype1_sse42;
ihevc_hbd_quant_4x4_ft ihevc_hbd_quant_4x4_sse42;
ihevc_hbd_quant_8x8_ft ihevc_hbd_quant_8x8_sse42;
ihevc_hbd_quant_16x16_ft ihevc_hbd_quant_16x16_sse42;
ihevc_hbd_quant_32x32_ft ihevc_hbd_quant_32x32_sse42;
ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_ft ihevc_hbd_quant_flat_scale_mat_4x4_ttype1_sse42;
ihevc_hbd_quant_flat_scale_mat_4x4_ft ihevc_hbd_quant_flat_scale_mat_4x4_sse42;
ihevc_hbd_quant_flat_scale_mat_8x8_ft ihevc_hbd_quant_flat_scale_mat_8x8_sse42;
ihevc_hbd_quant_flat_scale_mat_16x16_ft ihevc_hbd_quant_flat_scale_mat_16x16_sse42;
ihevc_hbd_quant_flat_scale_mat_32x32_ft ihevc_hbd_quant_flat_scale_mat_32x32_sse42;

#endif /*_IHEVC_QUANT_H_*/
