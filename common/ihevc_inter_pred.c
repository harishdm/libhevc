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
 *  ihevc_inter_pred.c
 *
 * @brief
 *  Calculates the prediction samples for a given cbt
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *   - ihevc_inter_pred()
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
#include <stdio.h>
#include <stdlib.h>
#include "ihevc_typedefs.h"
#include "ihevc_inter_pred.h"
#include "ihevc_defs.h"
#include "ihevc_structs.h"
#include "ihevc_weighted_pred.h"
WORD8 luma_filter[4][NTAPS_LUMA] =
    {
        { 0, 0, 0, 64, 0, 0, 0, 0 },
          { -1, 4, -10, 58, 17, -5, 1, 0 },
          { -1, 4, -11, 40, 40, -11, 4, -1 },
          { 0, 1, -5, 17, 58, -10, 4, -1 } };

/* The filter uses only the first four elements in each array */
WORD8 chroma_filter[8][NTAPS_LUMA] =
    {
        { 0, 64, 0, 0, 0, 0, 0, 0 },
          { -2, 58, 10, -2, 0, 0, 0, 0 },
          { -4, 54, 16, -2, 0, 0, 0, 0 },
          { -6, 46, 28, -4, 0, 0, 0, 0 },
          { -4, 36, 36, -4, 0, 0, 0, 0 },
          { -4, 28, 46, -6, 0, 0, 0, 0 },
          { -2, 16, 54, -4, 0, 0, 0, 0 },
          { -2, 10, 58, -2, 0, 0, 0, 0 } };

/*Chroma motion vector*/
typedef struct
{
    /* Horizontal Motion Vector */
    WORD16 i2_mvcx;

    /* Vertical Motion Vector */
    WORD16 i2_mvcy;
} mvc_t;

typedef void (*pf_inter_pred)(void *,
                              void *,
                              WORD32,
                              WORD32,
                              WORD8 *,
                              WORD32,
                              WORD32);

const pf_inter_pred func_array[22] =
    { NULL, (pf_inter_pred)&ihevc_inter_pred_luma_copy,
      (pf_inter_pred)&ihevc_inter_pred_luma_vert,
      (pf_inter_pred)&ihevc_inter_pred_luma_horz,
      (pf_inter_pred)&ihevc_inter_pred_luma_horz_w16out,

      (pf_inter_pred)&ihevc_inter_pred_luma_copy_w16out,
      (pf_inter_pred)&ihevc_inter_pred_luma_vert_w16out,
      (pf_inter_pred)&ihevc_inter_pred_luma_horz_w16out,
      (pf_inter_pred)&ihevc_inter_pred_luma_horz_w16out,

      (pf_inter_pred)&ihevc_inter_pred_luma_vert_w16inp,
      (pf_inter_pred)&ihevc_inter_pred_luma_vert_w16inp_w16out,

      NULL,
      (pf_inter_pred)&ihevc_inter_pred_chroma_copy,
      (pf_inter_pred)&ihevc_inter_pred_chroma_vert,
      (pf_inter_pred)&ihevc_inter_pred_chroma_horz,
      (pf_inter_pred)&ihevc_inter_pred_chroma_horz_w16out,

      (pf_inter_pred)&ihevc_inter_pred_chroma_copy_w16out,
      (pf_inter_pred)&ihevc_inter_pred_chroma_vert_w16out,
      (pf_inter_pred)&ihevc_inter_pred_chroma_horz_w16out,
      (pf_inter_pred)&ihevc_inter_pred_chroma_horz_w16out,

      (pf_inter_pred)&ihevc_inter_pred_chroma_vert_w16inp,
      (pf_inter_pred)&ihevc_inter_pred_chroma_vert_w16inp_w16out };


/* Not in the original code */
WORD32 ieee_rand(WORD32 L, WORD32 H);
/* end - Not in the original code */


/**
*******************************************************************************
*
* @brief
*  Inter prediction CTB level function
*
* @par Description:
*  For a given CTB, Inter prediction followed by weighted  prediction is
* done for all the PUs present in the CTB
*
* @param[in] ps_ctb
*  Pointer to the CTB context
*
* @returns
*
* @remarks
*
*
*******************************************************************************
*/

void ihevc_inter_pred_ctb(ctb_t *ps_ctb)
{

    UWORD8 *ref_pic_luma_l0, *ref_pic_chroma_l0;
    UWORD8 *ref_pic_luma_l1, *ref_pic_chroma_l1;
    UWORD8 *ref_pic_l0, *ref_pic_l1;

    WORD32 pu_indx;
    WORD32 pu_x, pu_y;
    WORD32 pu_wd, pu_ht;

    WORD32 clr_indx;
    WORD32 ntaps;

    WORD32 ai2_xint[2], ai2_yint[2];
    WORD32 ai2_xfrac[2], ai2_yfrac[2];

    WORD32 weighted_pred, bi_pred; /*Calculated from the reference indices*/

    WORD32 ref_strd; /*given in the codec context*/
    UWORD8 *pu1_dst_luma, *pu1_dst_chroma; /*given in the codec context*/
    /* pu1_dst_luma and pu1_dst_chroma are assumed to be the addresses of the
     corresponding top samples */
    UWORD8 *pu1_dst;

    WORD16 *pi2_tmp1, *pi2_tmp2, *pi2_tmp3; /*given in the codec context*/
    WORD32 tmp_strd;

    WORD32 wgt0, wgt1, off0, off1, shift, lvl_shift1, lvl_shift2;

    pf_inter_pred func_ptr1, func_ptr2, func_ptr3, func_ptr4;
    WORD32 func_indx1, func_indx2, func_indx3, func_indx4;
    void *func_src;
    void *func_dst;
    WORD32 func_src_strd;
    WORD32 func_dst_strd;
    WORD8 *func_coeff;
    WORD32 func_wd;
    WORD32 func_ht;

    mvc_t s_l0_mvc, s_l1_mvc;

    WORD8 (*coeff)[8];

    /* Not in the original code */
    WORD32 row, col;

    s_l0_mvc.i2_mvcx = ieee_rand(0, REF_WIDTH);
    s_l0_mvc.i2_mvcy = ieee_rand(0, REF_HEIGHT / 2);
    s_l1_mvc.i2_mvcx = ieee_rand(0, REF_WIDTH);
    s_l1_mvc.i2_mvcy = ieee_rand(0, REF_HEIGHT / 2);

    ref_pic_luma_l0 = (UWORD8*)malloc(sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT);
    ref_pic_luma_l1 = (UWORD8*)malloc(sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT);
    pu1_dst_luma = (UWORD8*)malloc(sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT);

    ref_pic_chroma_l0 = (UWORD8*)malloc(
                    sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT / 2);
    ref_pic_chroma_l1 = (UWORD8*)malloc(
                    sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT / 2);
    pu1_dst_chroma = (UWORD8*)malloc(
                    sizeof(UWORD8) * REF_WIDTH * REF_HEIGHT / 2);

    for(row = 0; row < REF_HEIGHT; row++)
    {
        for(col = 0; col < REF_WIDTH; col++)
        {
            ref_pic_luma_l0[row * REF_WIDTH + col] = ieee_rand(0, 255);
            ref_pic_luma_l1[row * REF_WIDTH + col] = ieee_rand(0, 255);
        }
    }

    for(row = 0; row < REF_HEIGHT / 2; row++)
    {
        for(col = 0; col < REF_WIDTH; col++)
        {
            ref_pic_chroma_l0[row * REF_WIDTH + col] = ieee_rand(0, 255);
            ref_pic_chroma_l1[row * REF_WIDTH + col] = ieee_rand(0, 255);
        }
    }

    pi2_tmp1 = (WORD16*)malloc(sizeof(WORD16) * REF_WIDTH * REF_HEIGHT);
    pi2_tmp2 = (WORD16*)malloc(sizeof(WORD16) * REF_WIDTH * REF_HEIGHT);
    pi2_tmp3 = (WORD16*)malloc(sizeof(WORD16) * REF_WIDTH * REF_HEIGHT);

    ref_strd = REF_WIDTH;
    tmp_strd = REF_WIDTH;

    /* end - Not in the original code */

    for(pu_indx = 0; pu_indx < ps_ctb->i4_pu_cnt; pu_indx++, ps_ctb->ps_pu++)
    {
        pu_x = ps_ctb->i4_pos_x + (ps_ctb->ps_pu->b4_pos_x << 2);
        pu_y = ps_ctb->i4_pos_y + (ps_ctb->ps_pu->b4_pos_y << 2);

        pu_wd = (ps_ctb->ps_pu->b4_wd + 1) << 2;
        pu_ht = (ps_ctb->ps_pu->b4_ht + 1) << 2;

        /* Not in the original code */
        weighted_pred = 1;
        bi_pred = 1;

        wgt0 = ieee_rand(255, 255);
        wgt1 = ieee_rand(255, 255);
        off0 = ieee_rand(128, 127);
        off1 = ieee_rand(128, 127);
        shift = ieee_rand(0, 7);
        shift += 6;
        /* end - Not in the original code */

        /*luma and chroma components*/
        for(clr_indx = 0; clr_indx < 2; clr_indx++)
        {
            /*reference pic list needed for ref_pic_luma, ref_pic_chroma*/
            /*weighted_pred and bi_pred are to be calculated*/
            /*motion vectors for chroma are to be calculated*/

            if(clr_indx == 0)
            {
                ai2_xint[0] = pu_x + (ps_ctb->ps_pu->mv.s_l0_mv.i2_mvx >> 2);
                ai2_yint[0] = pu_y + (ps_ctb->ps_pu->mv.s_l0_mv.i2_mvy >> 2);

                ai2_xint[1] = pu_x + (ps_ctb->ps_pu->mv.s_l1_mv.i2_mvx >> 2);
                ai2_yint[1] = pu_y + (ps_ctb->ps_pu->mv.s_l1_mv.i2_mvy >> 2);

                ai2_xfrac[0] = ps_ctb->ps_pu->mv.s_l0_mv.i2_mvx & 3;
                ai2_yfrac[0] = ps_ctb->ps_pu->mv.s_l0_mv.i2_mvy & 3;

                ai2_xfrac[1] = ps_ctb->ps_pu->mv.s_l1_mv.i2_mvx & 3;
                ai2_yfrac[1] = ps_ctb->ps_pu->mv.s_l1_mv.i2_mvy & 3;

                ref_pic_l0 = ref_pic_luma_l0 + ai2_xint[0] * ref_strd
                                + ai2_yint[0];
                ref_pic_l1 = ref_pic_luma_l1 + ai2_xint[1] * ref_strd
                                + ai2_yint[1];
                pu1_dst = pu1_dst_luma + pu_x * ref_strd + pu_y;

                ntaps = NTAPS_LUMA;
                coeff = luma_filter;
            }

            else
            {
                /* xint is upshifted by 1 because the chroma components are  */
                /* interleaved which is not the assumption made by standard  */
                ai2_xint[0] = (pu_x / 2 + (s_l0_mvc.i2_mvcx >> 3)) << 1;
                ai2_yint[0] = pu_y / 2 + (s_l0_mvc.i2_mvcy >> 3);

                ai2_xint[1] = (pu_x / 2 + (s_l1_mvc.i2_mvcx >> 3)) << 1;
                ai2_yint[1] = pu_y / 2 + (s_l1_mvc.i2_mvcy >> 3);

                ai2_xfrac[0] = s_l0_mvc.i2_mvcx & 7;
                ai2_yfrac[0] = s_l0_mvc.i2_mvcy & 7;

                ai2_xfrac[1] = s_l1_mvc.i2_mvcx & 7;
                ai2_yfrac[1] = s_l1_mvc.i2_mvcy & 7;

                ref_pic_l0 = ref_pic_chroma_l0 + ai2_xint[0] * ref_strd
                                + ai2_yint[0];
                ref_pic_l1 = ref_pic_chroma_l1 + ai2_xint[1] * ref_strd
                                + ai2_yint[1];
                pu1_dst = pu1_dst_chroma + pu_x * ref_strd + pu_y / 2;

                ntaps = NTAPS_CHROMA;
                coeff = chroma_filter;
            }

            func_indx1 = 4 * (weighted_pred || bi_pred) + 1 + 11 * clr_indx;
            func_indx1 += ai2_xfrac[0] ? 2 : 0;
            func_indx1 += ai2_yfrac[0] ? 1 : 0;

            func_indx2 = (ai2_xfrac[0] && ai2_yfrac[0])
                            * (9 + (weighted_pred || bi_pred)) + 11 * clr_indx;

            func_indx3 = 4 * (weighted_pred || bi_pred) + 1 + 11 * clr_indx;
            func_indx3 += ai2_xfrac[1] ? 2 : 0;
            func_indx3 += ai2_yfrac[1] ? 1 : 0;

            func_indx4 = (ai2_xfrac[1] && ai2_yfrac[1])
                            * (9 + (weighted_pred || bi_pred)) + 11 * clr_indx;

            func_ptr1 = func_array[func_indx1];
            func_ptr2 = func_array[func_indx2];
            func_ptr3 = func_array[bi_pred * func_indx3];
            func_ptr4 = func_array[bi_pred * func_indx4];

            /*Function 1*/
            func_src = (ai2_xfrac[0] && ai2_yfrac[0]) ?
                            ref_pic_l0 - (ntaps / 2 - 1) * ref_strd :
                            ref_pic_l0;
            func_dst = (weighted_pred || bi_pred) ?
                            (void *)pi2_tmp1 : (void *)pu1_dst;
            if(ai2_xfrac[0] && ai2_yfrac[0])
            {
                func_dst = pi2_tmp3;
            }
            func_src_strd = ref_strd;
            func_dst_strd = (weighted_pred || bi_pred
                            || (ai2_xfrac[0] && ai2_yfrac[0])) ?
                            tmp_strd : ref_strd;
            func_coeff = ai2_xfrac[0] ?
                            coeff[ai2_xfrac[0]] : coeff[ai2_yfrac[0]];
            func_wd = pu_wd >> clr_indx;
            func_ht = pu_ht >> clr_indx;
            func_ht += (ai2_xfrac[0] && ai2_yfrac[0]) ? ntaps - 1 : 0;

            func_ptr1(func_src, func_dst, func_src_strd, func_dst_strd,
                      func_coeff, func_ht, func_wd);

            /*Function 2*/
            if(func_ptr2 != NULL)
            {
                func_src = pi2_tmp3 + (ntaps / 2 - 1) * ref_strd;
                func_dst = (weighted_pred || bi_pred) ?
                                (void *)pi2_tmp1 : (void *)pu1_dst;
                func_src_strd = tmp_strd;
                func_dst_strd = (weighted_pred || bi_pred) ?
                                tmp_strd : ref_strd;
                func_coeff = coeff[ai2_yfrac[0]];
                func_wd = pu_wd >> clr_indx;
                func_ht = pu_ht >> clr_indx;

                func_ptr2(func_src, func_dst, func_src_strd, func_dst_strd,
                          func_coeff, func_ht, func_wd);
            }

            if(func_ptr3 != NULL)
            {
                func_src = (ai2_xfrac[1] && ai2_yfrac[1]) ?
                                ref_pic_l1 - (ntaps / 2 - 1) * ref_strd :
                                ref_pic_l1;
                func_dst = (ai2_xfrac[1] && ai2_yfrac[1]) ? pi2_tmp3 : pi2_tmp2;
                func_src_strd = ref_strd;
                func_dst_strd = tmp_strd;
                func_coeff = ai2_xfrac[1] ?
                                coeff[ai2_xfrac[1]] : coeff[ai2_yfrac[1]];
                func_wd = pu_wd >> clr_indx;
                func_ht = pu_ht >> clr_indx;
                func_ht += (ai2_xfrac[1] && ai2_yfrac[1]) ? ntaps - 1 : 0;

                func_ptr3(func_src, func_dst, func_src_strd, func_dst_strd,
                          func_coeff, func_ht, func_wd);
            }

            if(func_ptr4 != NULL)
            {
                func_src = pi2_tmp3 + (ntaps / 2 - 1) * ref_strd;
                func_dst = pi2_tmp2;
                func_src_strd = tmp_strd;
                func_dst_strd = tmp_strd;
                func_coeff = coeff[ai2_yfrac[1]];
                func_wd = pu_wd >> clr_indx;
                func_ht = pu_ht >> clr_indx;

                func_ptr4(func_src, func_dst, func_src_strd, func_dst_strd,
                          func_coeff, func_ht, func_wd);
            }

            if((weighted_pred != 0) && (bi_pred != 0))
            {
                /*weights, offsets and shift are to be calculated*/
                lvl_shift1 = clr_indx ? 0 : (1 << 13);
                lvl_shift2 = clr_indx ? 0 : (1 << 13);

                ihevc_weighted_pred_bi(pi2_tmp1, pi2_tmp2, pu1_dst, tmp_strd,
                                       tmp_strd, ref_strd, wgt0, off0, wgt1,
                                       off1, shift, lvl_shift1, lvl_shift2,
                                       pu_ht >> clr_indx, pu_wd >> clr_indx);
            }

            if((weighted_pred != 0) && (bi_pred == 0))
            {
                /*weight, offset and shift are to be calculated*/
                lvl_shift1 = clr_indx ? 0 : (1 << 13);

                ihevc_weighted_pred_uni(pi2_tmp1, pu1_dst, tmp_strd, ref_strd,
                                        wgt0, off0, shift, lvl_shift1,
                                        pu_ht >> clr_indx, pu_wd >> clr_indx);
            }

            if((weighted_pred == 0) && (bi_pred != 0))
            {
                lvl_shift1 = clr_indx ? 0 : (1 << 13);
                lvl_shift2 = clr_indx ? 0 : (1 << 13);

                ihevc_weighted_pred_bi_default(pi2_tmp1, pi2_tmp2, pu1_dst,
                                               tmp_strd, tmp_strd, ref_strd,
                                               lvl_shift1, lvl_shift2,
                                               pu_ht >> clr_indx,
                                               pu_wd >> clr_indx);
            }
        }
    }
}
