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
*  ihevc_ilf_padding_frame.c
*
* @brief
*  Does frame level loop filtering (deblocking and SAO) and padding
*
* @author
*  Srinivas T
*
* @par List of Functions:
*   - ihevc_ilf_pad_frame()
*
* @remarks
*  None
*
*******************************************************************************
*/


#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
#include "ihevcd_cxa.h"
#include "ithread.h"

#include "ihevc_defs.h"
#include "ihevc_debug.h"
#include "ihevc_defs.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_cabac_tables.h"

#include "ihevc_error.h"
#include "ihevc_common_tables.h"

#include "ihevcd_trace.h"
#include "ihevcd_defs.h"
#include "ihevcd_function_selector.h"
#include "ihevcd_structs.h"
#include "ihevcd_error.h"
#include "ihevcd_nal.h"
#include "ihevcd_bitstream.h"
#include "ihevcd_job_queue.h"
#include "ihevcd_utils.h"

#include "ihevc_deblk.h"
#include "ihevc_deblk_tables.h"
#include "ihevcd_profile.h"
#include "ihevcd_deblk.h"
#include "ihevcd_sao.h"
#include "ihevc_padding.h"

void ihevcd_ilf_pad_frame(deblk_ctxt_t *ps_deblk_ctxt, sao_ctxt_t *ps_sao_ctxt)
{
    sps_t *ps_sps;
    slice_header_t *ps_slice_hdr;
    codec_t *ps_codec;
    WORD32 i4_ctb_x, i4_ctb_y;
    WORD32 ctb_size;
    WORD32 i4_pixel_size_y;
    WORD32 i4_pixel_size_uv;

    ps_sps = ps_deblk_ctxt->ps_sps;
    ps_slice_hdr = ps_deblk_ctxt->ps_slice_hdr;
    ps_codec = ps_deblk_ctxt->ps_codec;
    ctb_size = (1 << ps_sps->i1_log2_ctb_size);
    i4_pixel_size_y     = ps_codec->i4_pixel_size_y;
    i4_pixel_size_uv    = ps_codec->i4_pixel_size_uv;

    for(i4_ctb_y = 0; i4_ctb_y < ps_sps->i2_pic_ht_in_ctb; i4_ctb_y++)
    {
        for(i4_ctb_x = 0; i4_ctb_x < ps_sps->i2_pic_wd_in_ctb; i4_ctb_x++)
        {
            WORD32 i4_is_last_ctb_x = 0;
            WORD32 i4_is_last_ctb_y = 0;

            /*TODO:
             *  Slice header also has to be updated
             *  */
            ps_deblk_ctxt->i4_ctb_x = i4_ctb_x;
            ps_deblk_ctxt->i4_ctb_y = i4_ctb_y;

            ps_sao_ctxt->i4_ctb_x = i4_ctb_x;
            ps_sao_ctxt->i4_ctb_y = i4_ctb_y;

            if((0 == ps_slice_hdr->i1_slice_disable_deblocking_filter_flag) &&
               (0 == ps_codec->i4_disable_deblk_pic))
            {
                ihevcd_deblk_ctb(ps_deblk_ctxt, i4_is_last_ctb_x, i4_is_last_ctb_y);

                /* If the last CTB in the row was a complete CTB then deblocking has to be called from remaining pixels, since deblocking
                 * is applied on a shifted CTB structure
                 */
                if(i4_ctb_x == ps_sps->i2_pic_wd_in_ctb - 1)
                {
                    WORD32 last_x_pos;
                    i4_is_last_ctb_x = 1;
                    i4_is_last_ctb_y = 0;


                    last_x_pos = (ps_sps->i2_pic_wd_in_ctb << ps_sps->i1_log2_ctb_size);
                    if(last_x_pos  ==  ps_sps->i2_pic_width_in_luma_samples)
                    {
                        ihevcd_deblk_ctb(ps_deblk_ctxt, i4_is_last_ctb_x, i4_is_last_ctb_y);
                    }
                }


                /* If the last CTB in the column was a complete CTB then deblocking has to be called from remaining pixels, since deblocking
                 * is applied on a shifted CTB structure
                 */
                if(i4_ctb_y == ps_sps->i2_pic_ht_in_ctb - 1)
                {
                    WORD32 last_y_pos;
                    i4_is_last_ctb_y = 1;
                    i4_is_last_ctb_x = 0;

                    last_y_pos = (ps_sps->i2_pic_ht_in_ctb << ps_sps->i1_log2_ctb_size);
                    if(last_y_pos == ps_sps->i2_pic_height_in_luma_samples)
                    {
                        ihevcd_deblk_ctb(ps_deblk_ctxt, i4_is_last_ctb_x, i4_is_last_ctb_y);
                    }
                }

            } /* End of deblock */

            if(ps_slice_hdr->i1_slice_sao_luma_flag || ps_slice_hdr->i1_slice_sao_chroma_flag)
            {
                ftype_sao_ctb   *pf_sao_ctb = (ftype_sao_ctb *)(ps_sao_ctxt->pf_sao_ctb);
                pf_sao_ctb(ps_sao_ctxt);
            }

            /* Call padding if required */
            {
                UWORD8 *pu1_cur_ctb_luma = ps_deblk_ctxt->pu1_cur_pic_luma +
                        (i4_ctb_x * ctb_size + i4_ctb_y * ctb_size * ps_codec->i4_strd) * i4_pixel_size_y;
                WORD32  i4_sub_ht_c = 2;
                UWORD8 *pu1_cur_ctb_chroma = ps_deblk_ctxt->pu1_cur_pic_chroma +
                        (i4_ctb_x * ctb_size + (i4_ctb_y * ctb_size * ps_codec->i4_strd / i4_sub_ht_c)) * i4_pixel_size_uv;

                if(0 == i4_ctb_x)
                {
                    WORD32 pad_ht_luma;
                    WORD32 pad_ht_chroma;

                    pad_ht_luma = ctb_size;
                    pad_ht_luma += (ps_sps->i2_pic_ht_in_ctb - 1) == i4_ctb_y ? 8 : 0;
                    pad_ht_chroma = ctb_size / i4_sub_ht_c;
                    pad_ht_chroma += (ps_sps->i2_pic_ht_in_ctb - 1) == i4_ctb_y ? 8 : 0;
                    /* Pad left after 1st CTB is processed */
                    if (1 == i4_pixel_size_y)
                    {
                        ps_codec->s_func_selector.ihevc_pad_left_luma_fptr(pu1_cur_ctb_luma - 8 * ps_codec->i4_strd,
                            ps_codec->i4_strd, pad_ht_luma, PAD_LEFT);
                    } else {
                        ps_codec->s_func_selector.ihevc_hbd_pad_left_luma_fptr((UWORD16 *)pu1_cur_ctb_luma - 8 * ps_codec->i4_strd,
                                                ps_codec->i4_strd, pad_ht_luma, PAD_LEFT);
                    }
                    if (1 == i4_pixel_size_uv)
                    {
                        ps_codec->s_func_selector.ihevc_pad_left_chroma_fptr(pu1_cur_ctb_chroma - 8 * ps_codec->i4_strd,
                            ps_codec->i4_strd, pad_ht_chroma, PAD_LEFT);
                    } else {
                        ps_codec->s_func_selector.ihevc_hbd_pad_left_chroma_fptr((UWORD16 *)pu1_cur_ctb_chroma - 8 * ps_codec->i4_strd, ps_codec->i4_strd,
                                                pad_ht_chroma, PAD_LEFT);
                    }
                }
                else if((ps_sps->i2_pic_wd_in_ctb - 1) == i4_ctb_x)
                {
                    WORD32 pad_ht_luma;
                    WORD32 pad_ht_chroma;
                    WORD32 cols_remaining = ps_sps->i2_pic_width_in_luma_samples - (i4_ctb_x << ps_sps->i1_log2_ctb_size);

                    pad_ht_luma = ctb_size;
                    pad_ht_luma += (ps_sps->i2_pic_ht_in_ctb - 1) == i4_ctb_y ? 8 : 0;
                    pad_ht_chroma = ctb_size / i4_sub_ht_c;
                    pad_ht_chroma += (ps_sps->i2_pic_ht_in_ctb - 1) == i4_ctb_y ? 8 : 0;
                    /* Pad right after last CTB in the current row is processed */
                    if (1 == i4_pixel_size_y) {
                        ps_codec->s_func_selector.ihevc_pad_right_luma_fptr(pu1_cur_ctb_luma + cols_remaining - 8 * ps_codec->i4_strd,
                            ps_codec->i4_strd, pad_ht_luma, PAD_RIGHT);
                    } else {
                        ps_codec->s_func_selector.ihevc_hbd_pad_right_luma_fptr((UWORD16 *)pu1_cur_ctb_luma + cols_remaining - 8 * ps_codec->i4_strd,
                                                    ps_codec->i4_strd, pad_ht_luma, PAD_RIGHT);
                    }
                    if (1 == i4_pixel_size_uv) {
                        ps_codec->s_func_selector.ihevc_pad_right_chroma_fptr(pu1_cur_ctb_chroma + cols_remaining - 8 * ps_codec->i4_strd,
                            ps_codec->i4_strd, pad_ht_chroma, PAD_RIGHT);
                    } else {
                        ihevc_hbd_pad_right_chroma((UWORD16 *)pu1_cur_ctb_chroma + cols_remaining - 8 * ps_codec->i4_strd,
                                                    ps_codec->i4_strd, pad_ht_chroma, PAD_RIGHT);
                    }


                    if((ps_sps->i2_pic_ht_in_ctb - 1) == i4_ctb_y)
                    {
                        UWORD8 *pu1_buf;
                        /* Since SAO is shifted by 8x8, chroma padding can not be done till second row is processed */
                        /* Hence moving top padding to to end of frame, Moving it to second row also results in problems when there is only one row */
                        /* Pad top after padding left and right for current rows after processing 1st CTB row */
                        if (1 == i4_pixel_size_y) {
                            ihevc_pad_top(ps_deblk_ctxt->pu1_cur_pic_luma - PAD_LEFT, ps_codec->i4_strd,
                                            ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_TOP);
                        } else {
                            ihevc_hbd_pad_top((UWORD16 *)ps_deblk_ctxt->pu1_cur_pic_luma - PAD_LEFT, ps_codec->i4_strd,
                                            ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_TOP);
                        }
                        if (1 == i4_pixel_size_uv) {
                            ihevc_pad_top(ps_deblk_ctxt->pu1_cur_pic_chroma - PAD_LEFT, ps_codec->i4_strd,
                                            ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_TOP / i4_sub_ht_c);
                        }
                        else {
                            ihevc_hbd_pad_top((UWORD16 *)ps_deblk_ctxt->pu1_cur_pic_chroma - PAD_LEFT, ps_codec->i4_strd,
                                            ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_TOP / i4_sub_ht_c);
                        }

                        pu1_buf = ps_deblk_ctxt->pu1_cur_pic_luma +
                            (ps_codec->i4_strd * ps_sps->i2_pic_height_in_luma_samples - PAD_LEFT) * i4_pixel_size_y;
                        /* Pad top after padding left and right for current rows after processing 1st CTB row */
                        if (1 == i4_pixel_size_y) {
                            ihevc_pad_bottom(pu1_buf, ps_codec->i4_strd, ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_BOT);
                        } else {
                            ihevc_hbd_pad_bottom((UWORD16 *)pu1_buf, ps_codec->i4_strd,
                                                    ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_BOT);
                        }

                        pu1_buf = ps_deblk_ctxt->pu1_cur_pic_chroma +
                            (ps_codec->i4_strd * (ps_sps->i2_pic_height_in_luma_samples / i4_sub_ht_c) - PAD_LEFT) * i4_pixel_size_uv;
                        if (1 == i4_pixel_size_uv) {
                            ihevc_pad_bottom(pu1_buf, ps_codec->i4_strd,
                                                ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_BOT / i4_sub_ht_c);
                        } else {
                            ihevc_hbd_pad_bottom((UWORD16 *)pu1_buf, ps_codec->i4_strd,
                                                    ps_sps->i2_pic_width_in_luma_samples + PAD_WD, PAD_BOT / i4_sub_ht_c);
                        }

                    } /* End of if ctb in last row */

                } /* End of if last ctb in a row */

            } /* End of padding */

        } /* End of loop over ctb_x */

    } /* End of loop over ctb_y */

} /* End of ihevcd_ilf_pad_frame */
