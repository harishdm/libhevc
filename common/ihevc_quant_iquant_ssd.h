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
*  ihevc_quant_iquant_ssd.h
*
* @brief
*  Functions declarations for quantization, followed by Inverse
*  quantization to find transform domain SSD
*
* @author
*  Ittiam
*
* @remarks
*  None
*
*******************************************************************************
*/


#ifndef _IHEVC_QUANT_IQUANT_SSD_H_
#define _IHEVC_QUANT_IQUANT_SSD_H_

/*venkat:*/
#define SMOOTHENING_Q_IQ_32x32  0
#define SMOOTHENING_Q_IQ_16x16  0
#define SMOOTHENING_Q_IQ_8x8    0

typedef WORD32 ihevc_quant_iquant_ssd_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
#if 0//QP_MORE_THAN_51
    WORD32 unclipped_qp_div,/* qpscaled / 6 */
    WORD32 unclipped_qp_rem,/* qpscaled % 6 */
#endif
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost
    );

typedef ihevc_quant_iquant_ssd_ft ihevc_quant_iquant_ssd_rdoq_ft;

typedef ihevc_quant_iquant_ssd_ft ihevc_quant_iquant_ssd_flat_scale_mat_ft;

typedef ihevc_quant_iquant_ssd_ft ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft;

typedef ihevc_quant_iquant_ssd_ft ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft;

typedef ihevc_quant_iquant_ssd_ft ihevc_q_iq_ssd_var_rnd_fact_ft;

typedef WORD32 ihevc_hbd_quant_iquant_ssd_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
#if 0//QP_MORE_THAN_51
    WORD32 unclipped_qp_div,/* qpscaled / 6 */
    WORD32 unclipped_qp_rem,/* qpscaled % 6 */
#endif
    WORD32 q_add,
    WORD32 *pi4_quant_round_factor_0_1,
    WORD32 *pi4_quant_round_factor_1_2,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    );

typedef WORD32 ihevc_hbd_quant_iquant_ssd_rdoq_ft
    (
    WORD16 *pi2_coeffs,
    WORD16 *pi2_quant_coeff,
    WORD16 *pi2_q_dst,
    WORD16 *pi2_iq_dst,
    WORD32  trans_size,
    WORD32 qp_div,/* qpscaled / 6 */
    WORD32 qp_rem,/* qpscaled % 6 */
    WORD32 q_add,
    WORD32 src_strd,
    WORD32 dst_q_strd,
    WORD32 dst_iq_strd,
    UWORD8 *csbf,
    WORD32 csbf_strd,
    WORD32 *zero_col,
    WORD32 *zero_row,
    WORD16 *pi2_dequant_coeff,
    LWORD64 *pi8_cost,
    WORD32 i4_bit_depth
    );

typedef ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_quant_iquant_ssd_rdoq_ft;

typedef ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft;

typedef ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft;

typedef ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft;

typedef ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_q_iq_ssd_var_rnd_fact_ft;

/* C function declarations */
ihevc_quant_iquant_ssd_ft ihevc_quant_iquant_ssd;
ihevc_quant_iquant_ssd_rdoq_ft ihevc_quant_iquant_ssd_rdoq;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_smoothing;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_ssd_flat_scale_mat_rdoq;
ihevc_q_iq_ssd_var_rnd_fact_ft ihevc_q_iq_ssd_var_rnd_fact;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact;

ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_quant_iquant_ssd;
ihevc_hbd_quant_iquant_ssd_rdoq_ft ihevc_hbd_quant_iquant_ssd_rdoq;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq;
ihevc_hbd_q_iq_ssd_var_rnd_fact_ft ihevc_hbd_q_iq_ssd_var_rnd_fact;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact;

ihevc_quant_iquant_ssd_ft ihevc_quant_iquant;
ihevc_quant_iquant_ssd_rdoq_ft ihevc_quant_iquant_rdoq;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_flat_scale_mat;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_flat_scale_mat_rdoq;
ihevc_q_iq_ssd_var_rnd_fact_ft ihevc_q_iq_var_rnd_fact;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_flat_scale_mat_var_rnd_fact;

ihevc_hbd_quant_iquant_ssd_ft ihevc_hbd_quant_iquant;
ihevc_hbd_quant_iquant_ssd_rdoq_ft ihevc_hbd_quant_iquant_rdoq;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_flat_scale_mat;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_flat_scale_mat_rdoq;
ihevc_hbd_q_iq_ssd_var_rnd_fact_ft ihevc_hbd_q_iq_var_rnd_fact;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_flat_scale_mat_var_rnd_fact;

/* SSE42 function declarations */
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_sse42;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_smoothing_sse42;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_sse42;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_sse42;

ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_flat_scale_mat_sse42;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_flat_scale_mat_rdoq_sse42;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_flat_scale_mat_var_rnd_fact_sse42;

ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_sse42;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_sse42;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_sse42;

ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_flat_scale_mat_sse42;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_flat_scale_mat_rdoq_sse42;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_flat_scale_mat_var_rnd_fact_sse42;

/* AVX function declarations */
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_avx;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_smoothing_avx;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_avx;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_avx;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_flat_scale_mat_avx;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_flat_scale_mat_rdoq_avx;
ihevc_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_q_iq_flat_scale_mat_var_rnd_fact_avx;

ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_avx;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_avx;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_avx;

ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_flat_scale_mat_avx;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_hbd_quant_iquant_flat_scale_mat_rdoq_avx;
ihevc_hbd_q_iq_ssd_flat_scale_mat_var_rnd_fact_ft ihevc_hbd_q_iq_flat_scale_mat_var_rnd_fact_avx;

#ifndef DISABLE_AVX2
/* AVX2 function declarations */
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_avx2;
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_smoothing_avx2;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_avx2;

ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_flat_scale_mat_avx2;
ihevc_quant_iquant_ssd_flat_scale_mat_rdoq_ft ihevc_quant_iquant_flat_scale_mat_rdoq_avx2;

ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_avx2;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_ssd_flat_scale_mat_rdoq_avx2;
ihevc_hbd_quant_iquant_ssd_flat_scale_mat_ft ihevc_hbd_quant_iquant_flat_scale_mat_rdoq_avx2;
#endif

#ifdef ARM_ASM_FUNC
ihevc_quant_iquant_ssd_flat_scale_mat_ft ihevc_quant_iquant_ssd_flat_scale_mat_a9q;
#endif

#endif /*_IHEVC_QUANT_IQUANT_SSD_H_*/
