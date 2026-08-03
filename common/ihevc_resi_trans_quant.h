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
*  ihevc_resi_trans_quant.h
*
* @brief
*  Functions declarations for residual,forward  transform and quantization
*
*
* @author
*  Ittiam
*
* @remarks
*  None
*
*******************************************************************************
*/
#ifndef _IHEVC_RESI_TRANS_QUANT_H_
#define _IHEVC_RESI_TRANS_QUANT_H_

typedef UWORD32 ihevc_resi_trans_quant_4x4_ttype1_ft(UWORD8 *pu1_src,
                                       UWORD8 *pu1_pred,
                                    WORD32 *pi4_tmp,
                                       WORD16 *pi2_quant_coeff,
                                       WORD16 *pi2_dst,
                                       WORD32 qp_div,/* qpscaled / 6 */
                                       WORD32 qp_rem,/* qpscaled % 6 */
                                       WORD32 q_add,
                                       WORD32 src_strd,
                                       WORD32 pred_strd,
                                       WORD32 dst_strd,
                                    UWORD8 *csbf,
                                    WORD32 csbf_strd,
                                    WORD32 *zero_col,
                                    WORD32 *zero_row,
                                    WORD32 *cbf);
typedef void ihevc_hbd_resi_trans_quant_4x4_ttype1_ft(UWORD16 *pu2_src,
                                       UWORD16 *pu2_pred,
                                       WORD16 *pi2_tmp,
                                       WORD16 *pi2_quant_coeff,
                                       WORD16 *pi2_dst,
                                       WORD32 qp_div,/* qpscaled / 6 */
                                       WORD32 qp_rem,/* qpscaled % 6 */
                                       WORD32 q_add,
                                       WORD32 src_strd,
                                       WORD32 pred_strd,
                                       WORD32 dst_strd,
                                       WORD32 *csbf,
                                       UWORD8 bit_depth);
typedef UWORD32 ihevc_resi_trans_quant_4x4_ft(UWORD8 *pu1_src,
                               UWORD8 *pu1_pred,
                                    WORD32 *pi4_tmp,
                               WORD16 *pi2_quant_coeff,
                               WORD16 *pi2_dst,
                               WORD32 qp_div,/* qpscaled / 6 */
                               WORD32 qp_rem,/* qpscaled % 6 */
                               WORD32 q_add,
                               WORD32 src_strd,
                               WORD32 pred_strd,
                               WORD32 dst_strd,
                                    UWORD8 *csbf,
                                    WORD32 csbf_strd,
                                    WORD32 *zero_col,
                                    WORD32 *zero_row,
                                    WORD32 *cbf);
typedef void ihevc_hbd_resi_trans_quant_4x4_ft(UWORD16 *pu2_src,
                               UWORD16 *pu2_pred,
                               WORD16 *pi2_tmp,
                               WORD16 *pi2_quant_coeff,
                               WORD16 *pi2_dst,
                               WORD32 qp_div,/* qpscaled / 6 */
                               WORD32 qp_rem,/* qpscaled % 6 */
                               WORD32 q_add,
                               WORD32 src_strd,
                               WORD32 pred_strd,
                               WORD32 dst_strd,
                               WORD32 *csbf,
                               UWORD8 bit_depth);
typedef UWORD32 ihevc_resi_trans_quant_8x8_ft(UWORD8 *pu1_src,
                               UWORD8 *pu1_pred,
                                    WORD32 *pi4_tmp,
                               WORD16 *pi2_quant_coeff,
                               WORD16 *pi2_dst,
                               WORD32 qp_div,/* qpscaled / 6 */
                               WORD32 qp_rem,/* qpscaled % 6 */
                               WORD32 q_add,
                               WORD32 src_strd,
                               WORD32 pred_strd,
                               WORD32 dst_strd,
                                    UWORD8 *csbf,
                                    WORD32 csbf_strd,
                                    WORD32 *zero_col,
                                    WORD32 *zero_row,
                                    WORD32 *cbf);
typedef void ihevc_hbd_resi_trans_quant_8x8_ft(UWORD16 *pu2_src,
                               UWORD16 *pu2_pred,
                               WORD16 *pi2_tmp,
                               WORD16 *pi2_quant_coeff,
                               WORD16 *pi2_dst,
                               WORD32 qp_div,/* qpscaled / 6 */
                               WORD32 qp_rem,/* qpscaled % 6 */
                               WORD32 q_add,
                               WORD32 src_strd,
                               WORD32 pred_strd,
                               WORD32 dst_strd,
                               WORD32 *csbf,
                               UWORD8 bit_depth);
typedef UWORD32 ihevc_resi_trans_quant_16x16_ft(UWORD8 *pu1_src,
                                 UWORD8 *pu1_pred,
                                    WORD32 *pi4_tmp,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 qp_div,/* qpscaled / 6 */
                                 WORD32 qp_rem,/* qpscaled % 6 */
                                 WORD32 q_add,
                                 WORD32 src_strd,
                                 WORD32 pred_strd,
                                 WORD32 dst_strd,
                                    UWORD8 *csbf,
                                    WORD32 csbf_strd,
                                    WORD32 *zero_col,
                                    WORD32 *zero_row,
                                    WORD32 *cbf);
typedef void ihevc_hbd_resi_trans_quant_16x16_ft(UWORD16 *pu2_src,
                                 UWORD16 *pu2_pred,
                                 WORD16 *pi2_tmp,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 qp_div,/* qpscaled / 6 */
                                 WORD32 qp_rem,/* qpscaled % 6 */
                                 WORD32 q_add,
                                 WORD32 src_strd,
                                 WORD32 pred_strd,
                                 WORD32 dst_strd,
                                 WORD32 *csbf,
                                 UWORD8 bit_depth);
typedef UWORD32 ihevc_resi_trans_quant_32x32_ft(UWORD8 *pu1_src,
                                 UWORD8 *pu1_pred,
                                    WORD32 *pi4_tmp,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 qp_div,/* qpscaled / 6 */
                                 WORD32 qp_rem,/* qpscaled % 6 */
                                 WORD32 q_add,
                                 WORD32 src_strd,
                                 WORD32 pred_strd,
                                 WORD32 dst_strd,
                                    UWORD8 *csbf,
                                    WORD32 csbf_strd,
                                    WORD32 *zero_col,
                                    WORD32 *zero_row,
                                    WORD32 *cbf);
typedef void ihevc_hbd_resi_trans_quant_32x32_ft(UWORD16 *pu2_src,
                                 UWORD16 *pu2_pred,
                                 WORD16 *pi2_tmp,
                                 WORD16 *pi2_quant_coeff,
                                 WORD16 *pi2_dst,
                                 WORD32 qp_div,/* qpscaled / 6 */
                                 WORD32 qp_rem,/* qpscaled % 6 */
                                 WORD32 q_add,
                                 WORD32 src_strd,
                                 WORD32 pred_strd,
                                 WORD32 dst_strd,
                                 WORD32 *csbf,
                                 UWORD8 bit_depth);

ihevc_resi_trans_quant_4x4_ttype1_ft ihevc_resi_trans_quant_4x4_ttype1;
ihevc_hbd_resi_trans_quant_4x4_ttype1_ft ihevc_hbd_resi_trans_quant_4x4_ttype1;
ihevc_resi_trans_quant_4x4_ft ihevc_resi_trans_quant_4x4;
ihevc_hbd_resi_trans_quant_4x4_ft ihevc_hbd_resi_trans_quant_4x4;
ihevc_resi_trans_quant_8x8_ft ihevc_resi_trans_quant_8x8;
ihevc_resi_trans_quant_8x8_ft ihevc_resi_trans_quant_8x8_avx2;
ihevc_hbd_resi_trans_quant_8x8_ft ihevc_hbd_resi_trans_quant_8x8;
ihevc_resi_trans_quant_16x16_ft ihevc_resi_trans_quant_16x16;
ihevc_resi_trans_quant_16x16_ft ihevc_resi_trans_quant_16x16_avx2;
ihevc_hbd_resi_trans_quant_16x16_ft ihevc_hbd_resi_trans_quant_16x16;
ihevc_resi_trans_quant_32x32_ft ihevc_resi_trans_quant_32x32;
ihevc_resi_trans_quant_32x32_ft ihevc_resi_trans_quant_32x32_avx2;
ihevc_hbd_resi_trans_quant_32x32_ft ihevc_hbd_resi_trans_quant_32x32;


#endif /*_IHEVC_RESI_TRANS_QUANT_H_*/
