///*****************************************************************************
//*
//* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
//*
//* Licensed under the Apache License, Version 2.0 (the "License");
//* you may not use this file except in compliance with the License.
//* You may obtain a copy of the License at:
//*
//* http://www.apache.org/licenses/LICENSE-2.0
//*
//* Unless required by applicable law or agreed to in writing, software
//* distributed under the License is distributed on an "AS IS" BASIS,
//* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//* See the License for the specific language governing permissions and
//* limitations under the License.
//*
//*****************************************************************************/
///**
//******************************************************************************
//* //file
//*  ihevc_inter_pred_filters_luma_vert.s
//*
//* //brief
//*  contains function definitions for inter prediction  interpolation.
//* functions are coded using neon  intrinsics and can be compiled using

//* rvct
//*
//* //author
//*  parthiban v
//*
//* //par list of functions:
//*
//*  - ihevc_inter_pred_luma_vert()
//*
//* //remarks
//*  none
//*
//*******************************************************************************
//*/

///* all the functions here are replicated from ihevc_inter_pred_filters.c and modified to */
///* include reconstruction */



///**
//*******************************************************************************
//*
//* //brief
//*     interprediction luma filter for vertical input
//*
//* //par description:
//*    applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
//*    the elements pointed by 'pu1_src' and  writes to the location pointed by
//*    'pu1_dst'  the output is downshifted by 6 and clipped to 8 bits
//*    assumptions : the function is optimized considering the fact width is
//*    multiple of 4 or 8. and height as multiple of 2.
//*
//* //param[in] pu1_src
//*  uword8 pointer to the source
//*
//* //param[out] pu1_dst
//*  uword8 pointer to the destination
//*
//* //param[in] src_strd
//*  integer source stride
//*
//* //param[in] dst_strd
//*  integer destination stride
//*
//* //param[in] pi1_coeff
//*  word8 pointer to the filter coefficients
//*
//* //param[in] ht
//*  integer height of the array
//*
//* //param[in] wd
//*  integer width of the array
//*
//* //returns
//*
//* //remarks
//*  none
//*
//*******************************************************************************
//*/

//void ihevc_inter_pred_luma_vert (
//                            uword8 *pu1_src,
//                            uword8 *pu1_dst,
//                            word32 src_strd,
//                            word32 dst_strd,
//                            word8 *pi1_coeff,
//                            word32 ht,
//                            word32 wd   )

//**************variables vs registers*****************************************
//    x0 => *pu1_src
//    x1 => *pu1_dst
//    x2 =>  src_strd
//    x6 =>  dst_strd
//    x12 => *pi1_coeff
//    x5 =>  ht
//    x3 =>  wd
.text
.align 4

.include "ihevc_neon_macros.s"

.globl ihevc_inter_pred_luma_vert_av8

.type ihevc_inter_pred_luma_vert_av8, %function

ihevc_inter_pred_luma_vert_av8:

// stmfd sp!, {x4-x12, x14}                             //stack stores the values of the arguments

    stp         x19, x20,[sp,#-16]!

    mov         x15,x4                                  // pi1_coeff
    mov         x16,x5                                  // ht
    mov         x17,x6                                  // wd

    mov         x12,x15                                 //load pi1_coeff
    mov         x6,x3
    mov         x5,x17                                  //load wd
    ld1         {v0.8b},[x12]                           //coeff = vld1_s8(pi1_coeff)
    sub         x12,x2,x2,lsl #2                        //src_ctrd & pi1_coeff
    abs         v0.8b, v0.8b                            //vabs_s8(coeff)
    add         x0,x0,x12                               //x0->pu1_src    x12->pi1_coeff
    mov         x3,x16                                  //load ht
    subs        x7,x3,#0                                //x3->ht
    dup         v22.16b, v0.b[0]                        //coeffabs_0 = vdup_lane_u8(coeffabs, 0)//
    cmp         x5,#8
    dup         v23.16b, v0.b[1]                        //coeffabs_1 = vdup_lane_u8(coeffabs, 1)//
    dup         v24.16b, v0.b[2]                        //coeffabs_2 = vdup_lane_u8(coeffabs, 2)//
    dup         v25.16b, v0.b[3]                        //coeffabs_3 = vdup_lane_u8(coeffabs, 3)//
    dup         v26.16b, v0.b[4]                        //coeffabs_4 = vdup_lane_u8(coeffabs, 4)//
    dup         v27.16b, v0.b[5]                        //coeffabs_5 = vdup_lane_u8(coeffabs, 5)//
    dup         v28.16b, v0.b[6]                        //coeffabs_6 = vdup_lane_u8(coeffabs, 6)//
    dup         v29.16b, v0.b[7]                        //coeffabs_7 = vdup_lane_u8(coeffabs, 7)//
    blt         core_loop_wd_4                          //core loop wd 4 jump
    stp         x0,x1, [sp, #-16]!

    bic         x4,x5,#7                                //x5 ->wd
    sub         x20,x4,x6,lsl #2                        //x6->dst_strd    x5    ->wd
    neg         x9, x20
    sub         x20,x4,x2,lsl #2                        //x2->src_strd
    neg         x8, x20
    lsr         x3, x5, #3                              //divide by 8
    mul         x7, x7, x3                              //multiply height by width
    sub         x7, x7,#4                               //subtract by one for epilog

    lsl         x11,x2,#1
    add         x13,x2,x2,lsl #1

prolog:

    add         x3,x0,x2                                //pu1_src_tmp += src_strd//
    ld1         {v1.8b},[x3],x2                         //src_tmp2 = vld1_u8(pu1_src_tmp)//
    ld1         {v0.8b},[x0],#8                         //src_tmp1 = vld1_u8(pu1_src_tmp)//
    subs        x4,x4,#8
    ld1         {v2.8b},[x3],x2                         //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umull       v19.8h, v1.8b, v23.8b                   //mul_res1 = vmull_u8(src_tmp2, coeffabs_1)//
    ld1         {v3.8b},[x3],x2                         //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v0.8b, v22.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp1, coeffabs_0)//
    ld1         {v4.8b},[x3],x2                         //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v2.8b, v24.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp3, coeffabs_2)//
    ld1         {v5.8b},[x3],x2                         //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlal       v19.8h, v3.8b, v25.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp4, coeffabs_3)//
    ld1         {v6.8b},[x3],x2                         //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal       v19.8h, v4.8b, v26.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp1, coeffabs_4)//
    ld1         {v7.8b},[x3],x2                         //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v5.8b, v27.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp2, coeffabs_5)//
    ld1         {v16.8b},[x3],x2                        //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlal       v19.8h, v6.8b, v28.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp3, coeffabs_6)//
    ld1         {v17.8b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v7.8b, v29.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp4, coeffabs_7)//

    umull       v20.8h, v2.8b, v23.8b                   //mul_res2 = vmull_u8(src_tmp3, coeffabs_1)//
    ld1         {v18.8b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlsl       v20.8h, v1.8b, v22.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp2, coeffabs_0)//
    add         x20,x0,x8
    umlsl       v20.8h, v3.8b, v24.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp4, coeffabs_2)//
    csel        x0, x20, x0,le
    umlal       v20.8h, v4.8b, v25.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp1, coeffabs_3)//
    bic         x20,x5,#7                               //x5 ->wd
    umlal       v20.8h, v5.8b, v26.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp2, coeffabs_4)//
    csel        x4, x20, x4,le
    umlsl       v20.8h, v6.8b, v27.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp3, coeffabs_5)//
    prfm        PLDL1KEEP,[x3,x20]
    umlal       v20.8h, v7.8b, v28.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp4, coeffabs_6)//
    prfm        PLDL1KEEP,[x3,x2]
    umlsl       v20.8h, v16.8b, v29.8b                  //mul_res2 = vmlsl_u8(mul_res2, src_tmp1, coeffabs_7)//

    prfm        PLDL1KEEP,[x3,x11]
    umull       v21.8h, v3.8b, v23.8b
    prfm        PLDL1KEEP,[x3,x13]
    umlsl       v21.8h, v2.8b, v22.8b
    add         x3,x0,x2                                //pu1_src_tmp += src_strd//
    umlsl       v21.8h, v4.8b, v24.8b
    sqrshrun    v19.8b, v19.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//
    umlal       v21.8h, v5.8b, v25.8b
    ld1         {v1.8b},[x3],x2                         //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal       v21.8h, v6.8b, v26.8b
    ld1         {v0.8b},[x0],#8                         //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlsl       v21.8h, v7.8b, v27.8b
    add         x14,x1,x6
    umlal       v21.8h, v16.8b, v28.8b
    sqrshrun    v20.8b, v20.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//
    umlsl       v21.8h, v17.8b, v29.8b
    ld1         {v2.8b},[x3],x2                         //src_tmp3 = vld1_u8(pu1_src_tmp)//

    st1         {v19.8b},[x1],#8                        //vst1_u8(pu1_dst,sto_res)//
    add         x20,x1,x9
    umull       v30.8h, v4.8b, v23.8b
    csel        x1, x20, x1,le
    umlsl       v30.8h, v3.8b, v22.8b
    umlsl       v30.8h, v5.8b, v24.8b
    ld1         {v3.8b},[x3],x2                         //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v6.8b, v25.8b
    ld1         {v4.8b},[x3],x2                         //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v7.8b, v26.8b
    st1         {v20.8b},[x14],x6                       //vst1_u8(pu1_dst_tmp,sto_res)//
    ld1         {v5.8b},[x3],x2                         //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlsl       v30.8h, v16.8b, v27.8b
    subs        x7,x7,#4
    ld1         {v6.8b},[x3],x2                         //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v17.8b, v28.8b
    sqrshrun    v21.8b, v21.8h,#6
    ld1         {v7.8b},[x3],x2                         //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlsl       v30.8h, v18.8b, v29.8b

    blt         epilog_end                              //jumps to epilog_end
    beq         epilog                                  //jumps to epilog

kernel_8:

    ld1         {v16.8b},[x3],x2                        //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umull       v19.8h, v1.8b, v23.8b                   //mul_res1 = vmull_u8(src_tmp2, coeffabs_1)//
    subs        x4,x4,#8
    umlsl       v19.8h, v0.8b, v22.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp1, coeffabs_0)//
    add         x20,x0,x8
    umlsl       v19.8h, v2.8b, v24.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp3, coeffabs_2)//
    ld1         {v17.8b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlal       v19.8h, v3.8b, v25.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp4, coeffabs_3)//
    csel        x0, x20, x0,le
    umlal       v19.8h, v4.8b, v26.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp1, coeffabs_4)//
    st1         {v21.8b},[x14],x6
    bic         x20,x5,#7                               //x5 ->wd
    umlsl       v19.8h, v5.8b, v27.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp2, coeffabs_5)//
    csel        x4, x20, x4,le
    umlal       v19.8h, v6.8b, v28.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp3, coeffabs_6)//
    ld1         {v18.8b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v7.8b, v29.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp4, coeffabs_7)//
    add         x3,x0,x2                                //pu1_src_tmp += src_strd//
    sqrshrun    v30.8b, v30.8h,#6

    umull       v20.8h, v2.8b, v23.8b                   //mul_res2 = vmull_u8(src_tmp3, coeffabs_1)//
    ld1         {v0.16b},[x0]                           //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlsl       v20.8h, v1.8b, v22.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp2, coeffabs_0)//
    add         x0,x0,#8
    umlsl       v20.8h, v3.8b, v24.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp4, coeffabs_2)//
    umlal       v20.8h, v4.8b, v25.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp1, coeffabs_3)//
    st1         {v30.8b},[x14],x6
    umlal       v20.8h, v5.8b, v26.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp2, coeffabs_4)//
    ld1         {v1.16b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlsl       v20.8h, v6.8b, v27.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp3, coeffabs_5)//
    add         x14,x1,#0
    umlal       v20.8h, v7.8b, v28.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp4, coeffabs_6)//
    add         x1, x1, #8
    umlsl       v20.8h, v16.8b, v29.8b                  //mul_res2 = vmlsl_u8(mul_res2, src_tmp1, coeffabs_7)//
    sqrshrun    v19.8b, v19.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//

    add         x20,x1,x9
    umull       v21.8h, v3.8b, v23.8b
    csel        x1, x20, x1,le
    umlsl       v21.8h, v2.8b, v22.8b
    ld1         {v2.16b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlsl       v21.8h, v4.8b, v24.8b
    add         x10, x3, x2, lsl #3                     // 11*strd
    umlal       v21.8h, v5.8b, v25.8b
    prfm        PLDL1KEEP,[x10,x20]                     //11+ 0
    umlal       v21.8h, v6.8b, v26.8b
    st1         {v19.8b},[x14],x6                       //vst1_u8(pu1_dst,sto_res)//
    umlsl       v21.8h, v7.8b, v27.8b
    prfm        PLDL1KEEP,[x10,x2]                      //11+ 1*strd
    umlal       v21.8h, v16.8b, v28.8b
    prfm        PLDL1KEEP,[x10,x11]                     //11+ 2*strd
    umlsl       v21.8h, v17.8b, v29.8b
    prfm        PLDL1KEEP,[x10,x13]                     //11+ 3*strd
    sqrshrun    v20.8b, v20.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//

    umull       v30.8h, v4.8b, v23.8b
    lsr         x19,x5,#4
    umlsl       v30.8h, v3.8b, v22.8b
    cmp         x19,#2
    ld1         {v3.16b},[x3],x2                        //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlsl       v30.8h, v5.8b, v24.8b
    ld1         {v4.16b},[x3],x2                        //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v6.8b, v25.8b
    ld1         {v5.16b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v7.8b, v26.8b
    st1         {v20.8b},[x14],x6                       //vst1_u8(pu1_dst_tmp,sto_res)//
    umlsl       v30.8h, v16.8b, v27.8b
    ld1         {v6.16b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal       v30.8h, v17.8b, v28.8b
    ld1         {v7.16b},[x3],x2                        //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlsl       v30.8h, v18.8b, v29.8b
    sqrshrun    v21.8b, v21.8h,#6

    bge         kernel_start

    subs        x7,x7,#4
    bgt         kernel_8                                //jumps to kernel_8
    ble         epilog


kernel_start:
    sub        x7,x7,#4
    add         x0,x0,#8

kernel_16:

    ld1         {v16.16b},[x3],x2                       //src_tmp1 = vld1_u8(pu1_src_tmp)//
    subs        x4,x4,#16
    umull       v19.8h, v1.8b, v23.8b                   //mul_res1 = vmull_u8(src_tmp2, coeffabs_1)//
    add         x20,x0,x8
    umlsl       v19.8h, v0.8b, v22.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp1, coeffabs_0)//
    csel        x0, x20, x0,le
    umlsl       v19.8h, v2.8b, v24.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp3, coeffabs_2)//
    ld1         {v17.16b},[x3],x2                       //src_tmp2 = vld1_u8(pu1_src_tmp)//
    bic         x20,x5,#7                               //x5 ->wd
    umlal       v19.8h, v3.8b, v25.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp4, coeffabs_3)//
    csel        x4, x20, x4,le
    umlal       v19.8h, v4.8b, v26.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp1, coeffabs_4)//
    st1         {v21.8b},[x14],x6
    umlsl       v19.8h, v5.8b, v27.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp2, coeffabs_5)//
    ld1         {v18.16b},[x3],x2                       //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal       v19.8h, v6.8b, v28.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp3, coeffabs_6)//
    add         x3,x0,x2                                //pu1_src_tmp += src_strd//
    umlsl       v19.8h, v7.8b, v29.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp4, coeffabs_7)//
    sqrshrun    v30.8b, v30.8h,#6


    umull       v20.8h, v2.8b, v23.8b                   //mul_res2 = vmull_u8(src_tmp3, coeffabs_1)//
    umlsl       v20.8h, v1.8b, v22.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp2, coeffabs_0)//
    umlsl       v20.8h, v3.8b, v24.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp4, coeffabs_2)//
    umlal       v20.8h, v4.8b, v25.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp1, coeffabs_3)//
    umlal       v20.8h, v5.8b, v26.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp2, coeffabs_4)//
    st1         {v30.8b},[x14],x6
    add         x14,x1,#0
    umlsl       v20.8h, v6.8b, v27.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp3, coeffabs_5)//
    add         x1, x1, #16                             //+16
    umlal       v20.8h, v7.8b, v28.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp4, coeffabs_6)//
    add         x20,x1,x9
    umlsl       v20.8h, v16.8b, v29.8b                  //mul_res2 = vmlsl_u8(mul_res2, src_tmp1, coeffabs_7)//
    sqrshrun    v19.8b, v19.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//


    umull       v21.8h, v3.8b, v23.8b
    csel        x1, x20, x1,le
    umlsl       v21.8h, v2.8b, v22.8b
    add         x19,x14,#8
    umlsl       v21.8h, v4.8b, v24.8b
    add         x10, x3, x2, lsl #3                     // 9*strd(1+8)
    umlal       v21.8h, v5.8b, v25.8b
    st1         {v19.8b},[x14],x6                       //vst1_u8(pu1_dst,sto_res)//
    add         x10,x10,x2,lsl #1                       //11*stride (9+2)
    umlal       v21.8h, v6.8b, v26.8b
    prfm        PLDL1KEEP,[x10,x20]                     //11+ 0
    umlsl       v21.8h, v7.8b, v27.8b
    prfm        PLDL1KEEP,[x10,x2]                      //11+ 1*strd
    umlal       v21.8h, v16.8b, v28.8b
    prfm        PLDL1KEEP,[x10,x11]                     //11+ 2*strd
    umlsl       v21.8h, v17.8b, v29.8b
    prfm        PLDL1KEEP,[x10,x13]                     //11+ 3*strd
    sqrshrun    v20.8b, v20.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//

    umull       v30.8h, v4.8b, v23.8b
    umlsl       v30.8h, v3.8b, v22.8b
    umlsl       v30.8h, v5.8b, v24.8b
    umlal       v30.8h, v6.8b, v25.8b
    umlal       v30.8h, v7.8b, v26.8b
    umlsl       v30.8h, v16.8b, v27.8b
    st1         {v20.8b},[x14],x6                       //vst1_u8(pu1_dst_tmp,sto_res)//
    umlal       v30.8h, v17.8b, v28.8b
    umlsl       v30.8h, v18.8b, v29.8b
    sqrshrun    v21.8b, v21.8h,#6


//******** 16th elements processing start

    umull2       v19.8h, v1.16b, v23.16b                //mul_res1 = vmull_u8(src_tmp2, coeffabs_1)//
    umlsl2       v19.8h, v0.16b, v22.16b                //mul_res1 = vmlsl_u8(mul_res1, src_tmp1, coeffabs_0)//
    umlsl2       v19.8h, v2.16b, v24.16b                //mul_res1 = vmlsl_u8(mul_res1, src_tmp3, coeffabs_2)//
    ld1         {v0.16b},[x0],#16                       //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlal2       v19.8h, v3.16b, v25.16b                //mul_res1 = vmlal_u8(mul_res1, src_tmp4, coeffabs_3)//
    umlal2       v19.8h, v4.16b, v26.16b                //mul_res1 = vmlal_u8(mul_res1, src_tmp1, coeffabs_4)//
    umlsl2       v19.8h, v5.16b, v27.16b                //mul_res1 = vmlsl_u8(mul_res1, src_tmp2, coeffabs_5)//
    st1         {v21.8b},[x14],x6
    umlal2       v19.8h, v6.16b, v28.16b                //mul_res1 = vmlal_u8(mul_res1, src_tmp3, coeffabs_6)//
    umlsl2       v19.8h, v7.16b, v29.16b                //mul_res1 = vmlsl_u8(mul_res1, src_tmp4, coeffabs_7)//
    sqrshrun    v30.8b, v30.8h,#6

    umull2       v20.8h, v2.16b, v23.16b                //mul_res2 = vmull_u8(src_tmp3, coeffabs_1)//
    umlsl2       v20.8h, v1.16b, v22.16b                //mul_res2 = vmlsl_u8(mul_res2, src_tmp2, coeffabs_0)//
    umlsl2       v20.8h, v3.16b, v24.16b                //mul_res2 = vmlsl_u8(mul_res2, src_tmp4, coeffabs_2)//
    ld1         {v1.16b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlal2       v20.8h, v4.16b, v25.16b                //mul_res2 = vmlal_u8(mul_res2, src_tmp1, coeffabs_3)//
    umlal2       v20.8h, v5.16b, v26.16b                //mul_res2 = vmlal_u8(mul_res2, src_tmp2, coeffabs_4)//
    umlsl2       v20.8h, v6.16b, v27.16b                //mul_res2 = vmlsl_u8(mul_res2, src_tmp3, coeffabs_5)//
    st1         {v30.8b},[x14],x6
    umlal2       v20.8h, v7.16b, v28.16b                //mul_res2 = vmlal_u8(mul_res2, src_tmp4, coeffabs_6)//
    mov         x14,x19
    umlsl2       v20.8h,v16.16b, v29.16b                //mul_res2 = vmlsl_u8(mul_res2, src_tmp1, coeffabs_7)//
    sqrshrun    v19.8b, v19.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//

    umull2       v21.8h, v3.16b, v23.16b
    umlsl2       v21.8h, v2.16b, v22.16b
    umlsl2       v21.8h, v4.16b, v24.16b
    ld1         {v2.16b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal2       v21.8h, v5.16b, v25.16b
    umlal2       v21.8h, v6.16b, v26.16b
    umlsl2       v21.8h, v7.16b, v27.16b
    st1         {v19.8b},[x14],x6                       //vst1_u8(pu1_dst,sto_res)//
    umlal2       v21.8h,v16.16b, v28.16b
    umlsl2       v21.8h,v17.16b, v29.16b
    sqrshrun    v20.8b, v20.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//

    umull2       v30.8h, v4.16b, v23.16b
    umlsl2       v30.8h, v3.16b, v22.16b
    umlsl2       v30.8h, v5.16b, v24.16b
    ld1         {v3.16b},[x3],x2                        //src_tmp4 = vld1_u8(pu1_src_tmp)//
    umlal2       v30.8h, v6.16b, v25.16b
    st1         {v20.8b},[x14],x6                       //vst1_u8(pu1_dst_tmp,sto_res)//
    ld1         {v4.16b},[x3],x2                        //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlal2       v30.8h, v7.16b, v26.16b
    ld1         {v5.16b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlsl2       v30.8h,v16.16b, v27.16b
    ld1         {v6.16b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//
    umlal2       v30.8h,v17.16b, v28.16b
    ld1         {v7.16b},[x3],x2                        //src_tmp4 = vld1_u8(pu1_src_tmp)//
    subs        x7,x7,#8
    umlsl2       v30.8h,v18.16b, v29.16b
    sqrshrun    v21.8b, v21.8h,#6

    bgt         kernel_16                               //jumps to kernel_8
    ble         epilog_end



epilog:

    st1         {v21.8b},[x14],x6
    umull       v19.8h, v1.8b, v23.8b                   //mul_res1 = vmull_u8(src_tmp2, coeffabs_1)//
    ld1         {v16.8b},[x3],x2                        //src_tmp1 = vld1_u8(pu1_src_tmp)//
    umlsl       v19.8h, v0.8b, v22.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp1, coeffabs_0)//
    umlsl       v19.8h, v2.8b, v24.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp3, coeffabs_2)//
    umlal       v19.8h, v3.8b, v25.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp4, coeffabs_3)//
    sqrshrun    v30.8b, v30.8h,#6
    umlal       v19.8h, v4.8b, v26.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp1, coeffabs_4)//
    umlsl       v19.8h, v5.8b, v27.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp2, coeffabs_5)//
    umlal       v19.8h, v6.8b, v28.8b                   //mul_res1 = vmlal_u8(mul_res1, src_tmp3, coeffabs_6)//
    umlsl       v19.8h, v7.8b, v29.8b                   //mul_res1 = vmlsl_u8(mul_res1, src_tmp4, coeffabs_7)//

    umull       v20.8h, v2.8b, v23.8b                   //mul_res2 = vmull_u8(src_tmp3, coeffabs_1)//
    st1         {v30.8b},[x14],x6
    umlsl       v20.8h, v1.8b, v22.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp2, coeffabs_0)//
    sqrshrun    v19.8b, v19.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//
    umlsl       v20.8h, v3.8b, v24.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp4, coeffabs_2)//
    ld1         {v17.8b},[x3],x2                        //src_tmp2 = vld1_u8(pu1_src_tmp)//
    umlal       v20.8h, v4.8b, v25.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp1, coeffabs_3)//
    umlal       v20.8h, v5.8b, v26.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp2, coeffabs_4)//
    umlsl       v20.8h, v6.8b, v27.8b                   //mul_res2 = vmlsl_u8(mul_res2, src_tmp3, coeffabs_5)//
    umlal       v20.8h, v7.8b, v28.8b                   //mul_res2 = vmlal_u8(mul_res2, src_tmp4, coeffabs_6)//
    umlsl       v20.8h, v16.8b, v29.8b                  //mul_res2 = vmlsl_u8(mul_res2, src_tmp1, coeffabs_7)//

    umull       v21.8h, v3.8b, v23.8b
    add         x14,x1,x6
    umlsl       v21.8h, v2.8b, v22.8b
    st1         {v19.8b},[x1],#8                        //vst1_u8(pu1_dst,sto_res)//
    umlsl       v21.8h, v4.8b, v24.8b
    sqrshrun    v20.8b, v20.8h,#6                       //sto_res = vqmovun_s16(sto_res_tmp)//
    umlal       v21.8h, v5.8b, v25.8b
    umlal       v21.8h, v6.8b, v26.8b
    umlsl       v21.8h, v7.8b, v27.8b
    umlal       v21.8h, v16.8b, v28.8b
    umlsl       v21.8h, v17.8b, v29.8b
    ld1         {v18.8b},[x3],x2                        //src_tmp3 = vld1_u8(pu1_src_tmp)//

    umull       v30.8h, v4.8b, v23.8b
    st1         {v20.8b},[x14],x6                       //vst1_u8(pu1_dst_tmp,sto_res)//
    umlsl       v30.8h, v3.8b, v22.8b
    umlsl       v30.8h, v5.8b, v24.8b
    sqrshrun    v21.8b, v21.8h,#6
    umlal       v30.8h, v6.8b, v25.8b
    umlal       v30.8h, v7.8b, v26.8b
    umlsl       v30.8h, v16.8b, v27.8b
    umlal       v30.8h, v17.8b, v28.8b
    umlsl       v30.8h, v18.8b, v29.8b

epilog_end:
    st1         {v21.8b},[x14],x6
    sqrshrun    v30.8b, v30.8h,#6

    st1         {v30.8b},[x14],x6


end_loops:
    tst         x5,#7
    ldp         x0,x1, [sp],#16

//  ldmeqfd sp!,{x4-x12,x15}                            //reload the registers from sp
    bne         lbl409
    ldp         x19, x20,[sp], #16

    ret
lbl409:
    mov         x5, #4
    add         x0, x0, #8
    add         x1, x1, #8
    mov         x7, #16

core_loop_wd_4:
    sub         x20,x5,x6,lsl #2                        //x6->dst_strd    x5    ->wd
    neg         x9, x20
    sub         x20,x5,x2,lsl #2                        //x2->src_strd
    neg         x8, x20
    movi        v4.8b, #0

outer_loop_wd_4:
    subs        x12,x5,#0
    ble         end_inner_loop_wd_4                     //outer loop jump

inner_loop_wd_4:
    add         x3,x0,x2
    ld1         {v4.s}[1],[x3],x2                       //src_tmp1 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp1, 1)//
    subs        x12,x12,#4
    dup         v5.2s, v4.s[1]                          //src_tmp2 = vdup_lane_u32(src_tmp1, 1)//
    ld1         {v5.s}[1],[x3],x2                       //src_tmp2 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp2, 1)//
    ld1         {v4.s}[0],[x0]                          //src_tmp1 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp1, 0)//
    umull       v0.8h, v5.8b, v23.8b                    //mul_res1 = vmull_u8(vreinterpret_u8_u32(src_tmp2), coeffabs_1)//

    dup         v6.2s, v5.s[1]                          //src_tmp3 = vdup_lane_u32(src_tmp2, 1)//
    add         x0,x0,#4
    ld1         {v6.s}[1],[x3],x2                       //src_tmp3 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp3, 1)//
    umlsl       v0.8h, v4.8b, v22.8b                    //mul_res1 = vmlsl_u8(mul_res1, vreinterpret_u8_u32(src_tmp1), coeffabs_0)//
    dup         v7.2s, v6.s[1]                          //src_tmp4 = vdup_lane_u32(src_tmp3, 1)//
    ld1         {v7.s}[1],[x3],x2                       //src_tmp4 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp4, 1)//
    umlsl       v0.8h, v6.8b, v24.8b                    //mul_res1 = vmlsl_u8(mul_res1, vreinterpret_u8_u32(src_tmp3), coeffabs_2)//
    umull       v19.8h, v7.8b, v23.8b
    dup         v4.2s, v7.s[1]                          //src_tmp1 = vdup_lane_u32(src_tmp4, 1)//
    umull       v2.8h, v7.8b, v25.8b                    //mul_res2 = vmull_u8(vreinterpret_u8_u32(src_tmp4), coeffabs_3)//
    ld1         {v4.s}[1],[x3],x2                       //src_tmp1 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp1, 1)//

    umlsl       v19.8h, v6.8b, v22.8b
    umlal       v0.8h, v4.8b, v26.8b                    //mul_res1 = vmlal_u8(mul_res1, vreinterpret_u8_u32(src_tmp1), coeffabs_4)//
    dup         v5.2s, v4.s[1]                          //src_tmp2 = vdup_lane_u32(src_tmp1, 1)//
    umlsl       v19.8h, v4.8b, v24.8b
    ld1         {v5.s}[1],[x3],x2                       //src_tmp2 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp2, 1)//
    umlsl       v2.8h, v5.8b, v27.8b                    //mul_res2 = vmlsl_u8(mul_res2, vreinterpret_u8_u32(src_tmp2), coeffabs_5)//
    dup         v6.2s, v5.s[1]                          //src_tmp3 = vdup_lane_u32(src_tmp2, 1)//
    umlal       v19.8h, v5.8b, v25.8b
    ld1         {v6.s}[1],[x3],x2                       //src_tmp3 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp3, 1)//
    umlal       v0.8h, v6.8b, v28.8b                    //mul_res1 = vmlal_u8(mul_res1, vreinterpret_u8_u32(src_tmp3), coeffabs_6)//

    dup         v7.2s, v6.s[1]                          //src_tmp4 = vdup_lane_u32(src_tmp3, 1)//
    umlal       v19.8h, v6.8b, v26.8b
    ld1         {v7.s}[1],[x3],x2                       //src_tmp4 = vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp4, 1)//
    umlsl       v2.8h, v7.8b, v29.8b                    //mul_res2 = vmlsl_u8(mul_res2, vreinterpret_u8_u32(src_tmp4), coeffabs_7)//
    dup         v4.2s, v7.s[1]
    add         v0.8h,  v0.8h ,  v2.8h                  //mul_res1 = vaddq_u16(mul_res1, mul_res2)//

    umlsl       v19.8h, v7.8b, v27.8b
    ld1         {v4.s}[1],[x3],x2
    umlal       v19.8h, v4.8b, v28.8b
    dup         v5.2s, v4.s[1]
    sqrshrun    v0.8b, v0.8h,#6                         //sto_res = vqmovun_s16(sto_res_tmp)//
    ld1         {v5.s}[1],[x3]
    add         x3,x1,x6
    st1         {v0.s}[0],[x1]                          //vst1_lane_u32((uint32_t *)pu1_dst, vreinterpret_u32_u8(sto_res), 0)//

    umlsl       v19.8h, v5.8b, v29.8b
    st1         {v0.s}[1],[x3],x6                       //vst1_lane_u32((uint32_t *)pu1_dst_tmp, vreinterpret_u32_u8(sto_res), 1)//
    sqrshrun    v19.8b, v19.8h,#6
    st1         {v19.s}[0],[x3],x6
    add         x1,x1,#4
    st1         {v19.s}[1],[x3]
    bgt         inner_loop_wd_4

end_inner_loop_wd_4:
    subs        x7,x7,#4
    add         x1,x1,x9
    add         x0,x0,x8
    bgt         outer_loop_wd_4

//  ldmfd sp!, {x4-x12, x15}                            //reload the registers from sp
    ldp         x19, x20,[sp], #16

    ret

