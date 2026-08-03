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
//*******************************************************************************
//* //file
//*  ihevc_inter_pred_chroma_vert_neon_w16inp_neon.s
//*
//* //brief
//*  contains function definitions for inter prediction  interpolation.
//* functions are coded using neon  intrinsics and can be compiled using

//* rvct
//*
//* //author
//*  yogeswaran rs / parthiban
//*
//* //par list of functions:
//*
//*
//* //remarks
//*  none
//*
//*******************************************************************************
//*/
///**
///**
//*******************************************************************************
//*
//* //brief
//*       chroma interprediction filter for 16bit vertical input.
//*
//* //par description:
//*    applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
//*    the elements pointed by 'pu1_src' and  writes to the location pointed by
//*    'pu1_dst'  input is 16 bits  the filter output is downshifted by 12 and
//*    clipped to lie  between 0 and 255   assumptions : the function is
//*    optimized considering the fact width and  height are multiple of 2.
//*
//* //param[in] pi2_src
//*  word16 pointer to the source
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
//void ihevc_inter_pred_chroma_vert_w16inp(word16 *pi2_src,
//                                          uword8 *pu1_dst,
//                                          word32 src_strd,
//                                          word32 dst_strd,
//                                          word8 *pi1_coeff,
//                                          word32 ht,
//                                          word32 wd)
//**************variables vs registers*****************************************
//x0 => *pu1_src
//x1 => *pi2_dst
//x2 =>  src_strd
//x3 =>  dst_strd

.text
.align 4

.include "ihevc_neon_macros.s"

.globl ihevc_inter_pred_chroma_vert_w16inp_av8

.type ihevc_inter_pred_chroma_vert_w16inp_av8, %function

ihevc_inter_pred_chroma_vert_w16inp_av8:

    // stmfd sp!, {x4-x12, x14}                    //stack stores the values of the arguments

    lsl         x2,x2,#1                    //src_strd = 2* src_strd
    ld1         {v0.8b},[x4]                //loads pi1_coeff
    sub         x4,x0,x2                    //pu1_src - src_strd
    sxtl        v0.8h, v0.8b                //long the value

    tst         x6,#3                       //checks wd  == 2
    dup         v16.8h, v0.h[0]             //coeff_0
    dup         v17.8h, v0.h[1]             //coeff_1
    dup         v18.8h, v0.h[2]             //coeff_2
    dup         v19.8h, v0.h[3]             //coeff_3

    bgt         core_loop_ht_2              //jumps to loop handles wd 2

    tst         x5,#3                       //checks ht == mul of 4
    beq         core_loop_ht_4              //jumps to loop handles ht mul of 4

core_loop_ht_2:
    lsl         x7,x2,#1                    //2*src_strd
    lsl         x12,x3,#1                   //2*dst_strd
    lsl         x9,x6,#2                    //4*wd
    sub         x6,x12,x6,lsl #1            //2*dst_strd - 2*wd
    sub         x8,x7,x9                    //2*src_strd - 4*wd
    mov         x12,x9                      //4wd

inner_loop_ht_2:
    add         x0,x4,x2                    //increments pi2_src
    ld1         {v0.4h},[x4],#8             //loads pu1_src
    smull       v0.4s, v0.4h, v16.4h        //vmull_s16(src_tmp1, coeff_0)
    subs        x12,x12,#8                  //2wd + 8
    ld1         {v2.4h},[x0],x2             //loads pi2_src
    smull       v7.4s, v2.4h, v16.4h        //vmull_s16(src_tmp2, coeff_0)
    ld1         {v3.4h},[x0],x2             //loads pi2_src
    smlal       v0.4s, v2.4h, v17.4h
    ld1         {v6.4h},[x0],x2
    smlal       v7.4s, v3.4h, v17.4h
    ld1         {v2.4h},[x0]
    add         x7,x1,x3                    //pu1_dst + dst_strd
    smlal       v0.4s, v3.4h, v18.4h
    smlal       v7.4s, v6.4h, v18.4h
    smlal       v0.4s, v6.4h, v19.4h
    smlal       v7.4s, v2.4h, v19.4h
    sqshrn      v0.4h, v0.4s,#6             //right shift
    sqshrn      v30.4h, v7.4s,#6            //right shift
    sqrshrun    v0.8b, v0.8h,#6             //rounding shift
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift
    st1         {v0.s}[0],[x1],#4           //stores the loaded value
    st1         {v30.s}[0],[x7]             //stores the loaded value
    bgt         inner_loop_ht_2             //inner loop -again

    //inner loop ends
    subs        x5,x5,#2                    //increments ht
    add         x1,x1,x6                    //pu1_dst += 2*dst_strd - 2*wd
    mov         x12,x9                      //4wd
    add         x4,x4,x8                    //pi1_src_tmp1 += 2*src_strd - 4*wd
    bgt         inner_loop_ht_2             //loop again

    b           end_loops                   //jumps to end

core_loop_ht_4:
    lsl         x7,x2,#2                    //4*src_strd
    lsl         x12,x3,#2                   //4*dst_strd
    lsr         x11, x6, #1                 //divide by 2
    sub         x14,x12,x6,lsl #1           //4*dst_strd - 2*wd
    sub         x8,x7,x6,lsl #2             //4*src_strd - 4*wd

    mul         x12, x5 , x11               //multiply height by width
    sub         x12, x12,#4                 //subtract by one for epilog
    lsl         x11, x6, #1                 //2*wd

prolog:
    add         x0,x4,x2                    //increments pi2_src
    ld1         {v0.4h},[x4],#8             //loads pu1_src
    ld1         {v1.4h},[x0],x2             //loads pi2_src
    ld1         {v2.4h},[x0],x2             //loads pi2_src
    ld1         {v3.4h},[x0],x2

    smull       v30.4s, v0.4h, v16.4h       //vmull_s16(src_tmp1, coeff_0)
    subs        x11,x11,#4
    smlal       v30.4s, v1.4h, v17.4h
    add         x9,x1,x3                    //pu1_dst + dst_strd
    smlal       v30.4s, v2.4h, v18.4h
    smlal       v30.4s, v3.4h, v19.4h
    ld1         {v4.4h},[x0],x2

    smull       v28.4s, v1.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x13,x4,x8
    smlal       v28.4s, v2.4h, v17.4h
    csel        x4, x13, x4,le
    ld1         {v5.4h},[x0],x2
    smlal       v28.4s, v3.4h, v18.4h
    lsl         x13,x6,#1
    smlal       v28.4s, v4.4h, v19.4h
    csel        x11, x13, x11,le
    sqshrn      v30.4h, v30.4s,#6           //right shift
    ld1         {v6.4h},[x0],x2

    smull       v26.4s, v2.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x0,x4,x2
    smlal       v26.4s, v3.4h, v17.4h
    ld1         {v0.4h},[x4],#8             //loads pu1_src
    smlal       v26.4s, v4.4h, v18.4h
    smlal       v26.4s, v5.4h, v19.4h
    ld1         {v1.4h},[x0],x2             //loads pi2_src
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift
    sqshrn      v28.4h, v28.4s,#6           //right shift

    smull       v24.4s, v3.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    ld1         {v2.4h},[x0],x2             //loads pi2_src
    smlal       v24.4s, v4.4h, v17.4h
    st1         {v30.s}[0],[x1],#4          //stores the loaded value
    smlal       v24.4s, v5.4h, v18.4h
    add         x13,x1,x14
    smlal       v24.4s, v6.4h, v19.4h
    csel        x1, x13, x1,le
    ld1         {v3.4h},[x0],x2

    sqshrn      v26.4h, v26.4s,#6           //right shift
    subs        x12,x12,#4
    sqrshrun    v28.8b, v28.8h,#6           //rounding shift

    beq         epilog                      //jumps to epilog

kernel_4:
    smull       v30.4s, v0.4h, v16.4h       //vmull_s16(src_tmp1, coeff_0)
    subs        x11,x11,#4
    smlal       v30.4s, v1.4h, v17.4h
    smlal       v30.4s, v2.4h, v18.4h
    ld1         {v4.4h},[x0],x2
    smlal       v30.4s, v3.4h, v19.4h
    st1         {v28.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v24.4h, v24.4s,#6           //right shift
    sqrshrun    v26.8b, v26.8h,#6           //rounding shift

    smull       v28.4s, v1.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x13,x4,x8
    smlal       v28.4s, v2.4h, v17.4h
    csel        x4, x13, x4,le
    smlal       v28.4s, v3.4h, v18.4h
    ld1         {v5.4h},[x0],x2
    lsl         x13,x6,#1
    smlal       v28.4s, v4.4h, v19.4h
    csel        x11, x13, x11,le
    st1         {v26.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v30.4h, v30.4s,#6           //right shift
    sqrshrun    v24.8b, v24.8h,#6           //rounding shift
    ld1         {v6.4h},[x0],x2

    smull       v26.4s, v2.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x0,x4,x2
    smlal       v26.4s, v3.4h, v17.4h
    ld1         {v0.8h},[x4]             //loads pu1_src
    smlal       v26.4s, v4.4h, v18.4h
    add         x4,x4,#8
    smlal       v26.4s, v5.4h, v19.4h
    st1         {v24.s}[0],[x9]             //stores the loaded value
    sqshrn      v28.4h, v28.4s,#6           //right shift
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift

    ld1         {v1.8h},[x0],x2             //loads pi2_src
    smull       v24.4s, v3.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x9,x1,x3                    //pu1_dst + dst_strd
    smlal       v24.4s, v4.4h, v17.4h
    ld1         {v2.8h},[x0],x2             //loads pi2_src
    lsr         x15,x6,#2
    smlal       v24.4s, v5.4h, v18.4h
    smlal       v24.4s, v6.4h, v19.4h
    st1         {v30.s}[0],[x1],#4          //stores the loaded value
    sqshrn      v26.4h, v26.4s,#6           //right shift
    ld1         {v3.8h},[x0],x2
    add         x13,x1,x14
    sqrshrun    v28.8b, v28.8h,#6           //rounding shift
    csel        x1, x13, x1,le

    cmp         x15,#2
    bge         kernel_8_start

    subs        x12,x12,#4
    bgt         kernel_4                    //jumps to kernel_4
    ble         epilog

kernel_8_start:
    add         x4,x4,#8
    subs        x12,x12,#4

kernel_8:

    smull       v30.4s, v0.4h, v16.4h       //vmull_s16(src_tmp1, coeff_0)
    subs        x11,x11,#8
    smlal       v30.4s, v1.4h, v17.4h
    smlal       v30.4s, v2.4h, v18.4h
    ld1         {v4.8h},[x0],x2
    smlal       v30.4s, v3.4h, v19.4h
    st1         {v28.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v24.4h, v24.4s,#6           //right shift
    sqrshrun    v26.8b, v26.8h,#6           //rounding shift

    smull       v28.4s, v1.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x13,x4,x8
    smlal       v28.4s, v2.4h, v17.4h
    csel        x4, x13, x4,le
    smlal       v28.4s, v3.4h, v18.4h
    lsl         x13,x6,#1
    ld1         {v5.8h},[x0],x2
    smlal       v28.4s, v4.4h, v19.4h
    csel        x11, x13, x11,le
    st1         {v26.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v30.4h, v30.4s,#6           //right shift
    sqrshrun    v24.8b, v24.8h,#6           //rounding shift

    smull       v26.4s, v2.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    smlal       v26.4s, v3.4h, v17.4h
    smlal       v26.4s, v4.4h, v18.4h
    ld1         {v6.8h},[x0],x2
    smlal       v26.4s, v5.4h, v19.4h
    st1         {v24.s}[0],[x9]             //stores the loaded value
    sqshrn      v28.4h, v28.4s,#6           //right shift
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift

    smull       v24.4s, v3.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    add         x0,x4,x2
    smlal       v24.4s, v4.4h, v17.4h
    add         x9,x1,x3                    //pu1_dst + dst_strd
    smlal       v24.4s, v5.4h, v18.4h
    add         x15,x1,#4
    smlal       v24.4s, v6.4h, v19.4h
    st1         {v30.s}[0],[x1]          //stores the loaded value
    sqshrn      v26.4h, v26.4s,#6           //right shift
    add         x1,x1,#8
    sqrshrun    v28.8b, v28.8h,#6           //rounding shift
    add         x13,x1,x14

//******** 8th elements processing start

    smull2       v30.4s, v0.8h, v16.8h       //vmull_s16(src_tmp1, coeff_0)
    csel        x1, x13, x1,le
    smlal2       v30.4s, v1.8h, v17.8h
    smlal2       v30.4s, v2.8h, v18.8h
    smlal2       v30.4s, v3.8h, v19.8h
    st1         {v28.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v24.4h, v24.4s,#6           //right shift
    ld1         {v0.8h},[x4],#16             //loads pu1_src
    sqrshrun    v26.8b, v26.8h,#6           //rounding shift

    smull2       v28.4s, v1.8h, v16.8h       //vmull_s16(src_tmp2, coeff_0)
    smlal2       v28.4s, v2.8h, v17.8h
    smlal2       v28.4s, v3.8h, v18.8h
    smlal2       v28.4s, v4.8h, v19.8h
    st1         {v26.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v30.4h, v30.4s,#6           //right shift
    ld1         {v1.8h},[x0],x2             //loads pi2_src
    sqrshrun    v24.8b, v24.8h,#6           //rounding shift

    smull2       v26.4s, v2.8h, v16.8h       //vmull_s16(src_tmp2, coeff_0)
    smlal2       v26.4s, v3.8h, v17.8h
    ld1         {v2.8h},[x0],x2             //loads pi2_src
    smlal2       v26.4s, v4.8h, v18.8h
    smlal2       v26.4s, v5.8h, v19.8h
    st1         {v24.s}[0],[x9]             //stores the loaded value
    sqshrn      v28.4h, v28.4s,#6           //right shift
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift

    smull2       v24.4s, v3.8h, v16.8h       //vmull_s16(src_tmp2, coeff_0)
    smlal2       v24.4s, v4.8h, v17.8h
    smlal2       v24.4s, v5.8h, v18.8h
    mov         x9,x15
    smlal2       v24.4s, v6.8h, v19.8h
    st1         {v30.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v26.4h, v26.4s,#6           //right shift
    ld1         {v3.8h},[x0],x2
    subs        x12,x12,#8
    sqrshrun    v28.8b, v28.8h,#6           //rounding shift

    bgt         kernel_8                    //jumps to kernel_4
    ble         epilog_end

epilog:
    smull       v30.4s, v0.4h, v16.4h       //vmull_s16(src_tmp1, coeff_0)
    smlal       v30.4s, v1.4h, v17.4h
    ld1         {v4.4h},[x0],x2
    smlal       v30.4s, v2.4h, v18.4h
    st1         {v28.s}[0],[x9],x3          //stores the loaded value
    smlal       v30.4s, v3.4h, v19.4h
    sqshrn      v24.4h, v24.4s,#6           //right shift
    sqrshrun    v26.8b, v26.8h,#6           //rounding shift

    smull       v28.4s, v1.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    smlal       v28.4s, v2.4h, v17.4h
    smlal       v28.4s, v3.4h, v18.4h
    ld1         {v5.4h},[x0],x2
    smlal       v28.4s, v4.4h, v19.4h
    st1         {v26.s}[0],[x9],x3          //stores the loaded value
    sqshrn      v30.4h, v30.4s,#6           //right shift
    sqrshrun    v24.8b, v24.8h,#6           //rounding shift

    smull       v26.4s, v2.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    smlal       v26.4s, v3.4h, v17.4h
    ld1         {v6.4h},[x0],x2
    smlal       v26.4s, v4.4h, v18.4h
    smlal       v26.4s, v5.4h, v19.4h
    st1         {v24.s}[0],[x9]             //stores the loaded value
    sqshrn      v28.4h, v28.4s,#6           //right shift
    add         x9,x1,x3                    //pu1_dst + dst_strd
    sqrshrun    v30.8b, v30.8h,#6           //rounding shift

    smull       v24.4s, v3.4h, v16.4h       //vmull_s16(src_tmp2, coeff_0)
    smlal       v24.4s, v4.4h, v17.4h
    smlal       v24.4s, v5.4h, v18.4h
    smlal       v24.4s, v6.4h, v19.4h
    st1         {v30.s}[0],[x1],#4          //stores the loaded value

    sqrshrun    v28.8b, v28.8h,#6           //rounding shift
    sqshrn      v26.4h, v26.4s,#6           //right shift

epilog_end:

    st1         {v28.s}[0],[x9],x3          //stores the loaded value
    sqrshrun    v26.8b, v26.8h,#6           //rounding shift
    sqshrn      v24.4h, v24.4s,#6           //right shift
    st1         {v26.s}[0],[x9],x3          //stores the loaded value
    sqrshrun    v24.8b, v24.8h,#6           //rounding shift

    st1         {v24.s}[0],[x9]             //stores the loaded value

end_loops:
    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
//    ldp         x19, x20,[sp],#16

    ret




