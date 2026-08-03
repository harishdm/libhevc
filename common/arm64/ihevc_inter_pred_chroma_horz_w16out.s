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
//*  ihevc_inter_pred_chroma_horz_neon.s
//*
//* //brief
//*  contains function definitions for inter prediction  interpolation.
//* functions are coded using neon  intrinsics and can be compiled using

//* rvct
//*
//* //author
//*  yogeswaran rs / akshaya mukund
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
//*******************************************************************************
//*
//* //brief
//*       chroma interprediction filter to store horizontal 16bit ouput
//*
//* //par description:
//*    applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
//*    to the elements pointed by 'pu1_src' and  writes to the location pointed
//*    by 'pu1_dst'  no downshifting or clipping is done and the output is  used
//*    as an input for vertical filtering or weighted  prediction
//*
//* //param[in] pu1_src
//*  uword8 pointer to the source
//*
//* //param[out] pi2_dst
//*  word16 pointer to the destination
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
//void ihevc_inter_pred_chroma_horz_w16out(uword8 *pu1_src,
//                                          word16 *pi2_dst,
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

.globl ihevc_inter_pred_chroma_horz_w16out_av8


.type ihevc_inter_pred_chroma_horz_w16out_av8, %function

ihevc_inter_pred_chroma_horz_w16out_av8:

    // stmfd sp!, {x4-x12, x14}                    //stack stores the values of the arguments

    stp         d10,d11,[sp,#-16]!
    stp         d12,d13,[sp,#-16]!
    stp         d14,d15,[sp,#-16]!
    stp         x19, x20,[sp,#-16]!
    stp         x21, x22,[sp,#-16]!

    mov         x15,x4 // pi1_coeff
    mov         x16,x5 // ht
    mov         x17,x6 // wd

    mov         x4,x15                      //loads pi1_coeff
    mov         x6,x16                      //loads ht
    mov         x10,x17                     //loads wd

    lsl         x21,x2,#1
    lsl         x22,x2,#2
    ld1         {v0.8b},[x4]                //coeff = vld1_s8(pi1_coeff)
    subs        x14,x6,#0                   //checks for ht == 0
    abs         v2.8b, v0.8b                //vabs_s8(coeff)
    mov         x11, #2
    ble         end_loops

    dup         v24.16b, v2.b[0]             //coeffabs_0 = vdup_lane_u8(coeffabs, 0)
    sub         x12,x0,#2                   //pu1_src - 2
    dup         v25.16b, v2.b[1]             //coeffabs_1 = vdup_lane_u8(coeffabs, 1)
    add         x4,x12,x2                   //pu1_src_tmp2_8 = pu1_src + src_strd
    dup         v26.16b, v2.b[2]             //coeffabs_2 = vdup_lane_u8(coeffabs, 2)
    tst         x10,#3                      //checks wd for multiples of 4
    lsl         x5, x10, #1                 //2wd
    dup         v27.16b, v2.b[3]             //coeffabs_3 = vdup_lane_u8(coeffabs, 3)
    and         x7,x14,#1                   //added                //calculating ht_residue ht_residue = (ht & 1)
    sub         x14,x14,x7                  //added                //decrement height by ht_residue(residue value is calculated outside)
    bne         outer_loop_4                // this branching happens when the width is 2 or 6

    cmp         x10,#12
    beq         skip_16

    cmp         x10,#8
    bge         outer_loop_16

skip_16:
    tst         x6,#3
    sub         x9,x0,#2
    beq         outer_loop_ht_4             //this branching happens when the height is a a multiple of 4

    b           outer_loop_8

outer_loop_16:
    add         x4,x12,x2
    and         x0, x12, #31
    prfm        PLDL1KEEP,[x12,x21]
    add         x19,x12,#8
    ld1         { v0.4s},[x12],x11          //vector load pu1_src

    mov         x10,x5                      //2wd
    mul         x14, x14 , x10
    prfm        PLDL1KEEP,[x4,x21]
    mov         x9,#10
    sub         x20,x3,#8
    ld1         { v2.4s},[x12],x11          //vector load pu1_src
    neg         x6, x20

    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    sub         x8,x3,#8
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    lsl         x6,x6,#1
    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    sub         x20,x5,x3,lsl #1
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    neg         x3, x20

    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    lsl         x8,x8,#1
    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    sub         x20,x5,x2,lsl #1
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    neg         x7, x20
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    cmp         x14,#32

    beq         epilog_end
    sub         x14, x14,#64

inner_loop_16:

    st1         { v30.8h}, [x1],#16
    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    prfm        PLDL1KEEP,[x12,x22]
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    prfm        PLDL1KEEP,[x4,x22]
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    subs        x10,x10,#16
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    add         x20,x12,x7

    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    csel        x12, x20, x12,eq
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    st1         { v28.8h}, [x1],x8
    add         x20,x12,x2
    ld1         { v0.4s},[x12],x11          //vector load pu1_src
    ld1         { v2.4s},[x12],x11          //vector load pu1_src
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    csel        x4, x20, x4,eq
    st1         { v22.8h},[x1],#16          //store the result pu1_dst
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    csel        x10, x5, x10,eq             //2wd

    st1         { v20.8h},[x1],x6           //store the result pu1_dst
    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x1,x3,lsl #1
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    csel        x1, x20, x1,eq
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    subs        x14,x14,#32                 //decrement the ht loop
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    bgt         inner_loop_16

    add         x14,x14,#64
    cmp         x14,#32
    beq         epilog_end

epilog:
    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    st1         { v30.8h}, [x1],#16
    subs        x10,x10,#16                 //decrement the wd loop
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    add         x20,x12,x7
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    csel        x12, x20, x12,eq
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    csel        x10, x5, x10,eq             //2wd

    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x12,x2
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v0.4s},[x12],x11          //vector load pu1_src
    csel        x4, x20, x4,eq
    st1         { v28.8h}, [x1],x8
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v2.4s},[x12],x11          //vector load pu1_src
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    st1         { v22.8h},[x1],#16          //store the result pu1_dst
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         { v20.8h},[x1],x6           //store the result pu1_dst
    add         x20,x1,x3,lsl #1
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    csel        x1, x20, x1,eq

epilog_end:

    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    mov         x21,x16                      //loads ht
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    and         x7,x21,#1
    st1         { v30.8h}, [x1],#16
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    cmp         x7,#0
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    mov         x10,x5
    st1         { v28.8h}, [x1],x8
    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x12,x2,lsl #1
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    csel        x12, x20, x12,ne
    st1         { v22.8h},[x1],#16          //store the result pu1_dst
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    sub         x20,x12,x5
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    csel        x12, x20, x12,ne
    st1         { v20.8h},[x1],x6           //store the result pu1_dst
    add         x20,x1,x3,lsl #1
    csel        x1, x20, x1,ne

    bgt         loop_residue_4

    b           end_loops

outer_loop_8:
    mov         x13,#8
    add         x4,x12,x2                   //pu1_src + src_strd

    add         x6,x1,x3,lsl #1             //pu1_dst + dst_strd
    mov         x10,x5                      //2wd

inner_loop_8:

    ld1         {v0.16b},[x12],x13           //vector load pu1_src
    ld1         {v4.16b},[x4],x13            //vector load pu1_src
    EXT         v1.16b,v0.16b,v0.16b,#2
    EXT         v2.16b,v0.16b,v0.16b,#4
    EXT         v3.16b,v0.16b,v0.16b,#6

    umull       v29.8h, v1.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v29.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    EXT         v5.16b,v4.16b,v4.16b,#2
    EXT         v6.16b,v4.16b,v4.16b,#4
    EXT         v7.16b,v4.16b,v4.16b,#6
    st1         {v29.8h}, [x1],#16

    umull       v10.8h, v5.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v10.8h, v6.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v10.8h, v7.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    st1         {v10.8h},[x6],#16           //store the result pu1_dst
    subs        x10,x10,#8                  //decrement the wd loop

    bgt         inner_loop_8

    sub         x12,x12,x5
    subs        x14,x14,#2                  //decrement the ht loop
    sub         x1,x1,x5,lsl #1
    add         x12,x12,x2,lsl #1
    add         x1,x1,x3,lsl #2
    bgt         outer_loop_8

    cmp         x7,#0
    mov         x10,x5
    bgt         loop_residue_4

    b           end_loops

outer_loop_ht_4:

    mov         x10,x5

prologue_ht_4:
    lsl         x8, x3, #1

inner_loop_ht_4:

    mov         x12,x9
    ld1         {v0.16b},[x12],x2           //(1)vector load pu1_src
    ld1         {v4.16b},[x12],x2           //(2)vector load pu1_src

    EXT         v1.16b,v0.16b,v0.16b,#2
    mov         x4,x1
    EXT         v2.16b,v0.16b,v0.16b,#4
    sub         x0, x2, #6                  // not sure if x0 needs to be preserved
    EXT         v3.16b,v0.16b,v0.16b,#6

    umull       v29.8h, v1.8b, v25.8b       //(1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v29.8h, v0.8b, v24.8b       //(1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //(1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //(1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         {v14.16b},[x12],x2           //(2)vector load pu1_src
    EXT         v5.16b,v4.16b,v4.16b,#2
    EXT         v6.16b,v4.16b,v4.16b,#4
    EXT         v7.16b,v4.16b,v4.16b,#6

    umull       v10.8h, v5.8b, v25.8b       //(2)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //(2)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v10.8h, v6.8b, v26.8b       //(2)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v10.8h, v7.8b, v27.8b       //(2)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    ld1         {v18.16b},[x12],x2           //(2)vector load pu1_src

    EXT         v15.16b,v14.16b,v14.16b,#2
    EXT         v16.16b,v14.16b,v14.16b,#4
    EXT         v17.16b,v14.16b,v14.16b,#6
    EXT         v19.16b,v18.16b,v18.16b,#2
    add         x9,x9,#8                    //(core loop)
    EXT         v20.16b,v18.16b,v18.16b,#4
    subs        x10,x10,#8                  //(prologue)decrement the wd loop
    EXT         v21.16b,v18.16b,v18.16b,#6

    beq         epilogue

core_loop:
    st1         {v29.8h},[x4],x8            //(1)store the result pu1_dst
    mov         x12,x9
    ld1         {v0.16b},[x12],x2           //(1)vector load pu1_src

    umull       v12.8h, v15.8b, v25.8b      //(3)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v12.8h, v14.8b, v24.8b      //(3)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v12.8h, v16.8b, v26.8b      //(3)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v12.8h, v17.8b, v27.8b      //(3)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    st1         {v10.8h},[x4],x8            //(2)store the result pu1_dst
    umull       v22.8h, v19.8b, v25.8b      //(4)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v22.8h, v18.8b, v24.8b      //(4)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    add         x1,x1,#16                   //(core loop)
    umlal       v22.8h, v20.8b, v26.8b      //(4)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    add         x9,x9,#8                    //(core loop)
    umlsl       v22.8h, v21.8b, v27.8b      //(4)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    ld1         {v4.16b},[x12],x2           //(2)vector load pu1_src

    st1         {v12.8h},[x4],x8            //(3)store the result pu1_dst
    EXT         v1.16b,v0.16b,v0.16b,#2
    EXT         v2.16b,v0.16b,v0.16b,#4
    EXT         v3.16b,v0.16b,v0.16b,#6

    umull       v29.8h, v1.8b, v25.8b       //(1_1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    subs        x10,x10,#8                  //(core loop)
    umlsl       v29.8h, v0.8b, v24.8b       //(1_1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //(1_1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //(1_1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    ld1         {v14.16b},[x12],x2           //(2)vector load pu1_src

    EXT         v5.16b,v4.16b,v4.16b,#2
    EXT         v6.16b,v4.16b,v4.16b,#4
    EXT         v7.16b,v4.16b,v4.16b,#6

    umull       v10.8h, v5.8b, v25.8b       //(2_1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //(2_1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         {v18.16b},[x12],x2           //(2)vector load pu1_src
    umlal       v10.8h, v6.8b, v26.8b       //(2_1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v10.8h, v7.8b, v27.8b       //(2_1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    EXT         v15.16b,v14.16b,v14.16b,#2
    EXT         v16.16b,v14.16b,v14.16b,#4
    EXT         v17.16b,v14.16b,v14.16b,#6
    st1         {v22.8h}, [x4], x8          //(4)store the result pu1_dst
    EXT         v19.16b,v18.16b,v18.16b,#2
    EXT         v20.16b,v18.16b,v18.16b,#4
    mov         x4, x1                      //(core loop)
    EXT         v21.16b,v18.16b,v18.16b,#6

    bgt         core_loop                   //loopback

epilogue:
    st1         {v29.8h},[x4], x8           //(1)store the result pu1_dst
    umull       v12.8h, v15.8b, v25.8b      //(3)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x1,x1,#16                   //(core loop)
    umlsl       v12.8h, v14.8b, v24.8b      //(3)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    sub         x9,x9,x5
    umlal       v12.8h, v16.8b, v26.8b      //(3)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         {v10.8h},[x4], x8           //(2)store the result pu1_dst
    subs        x14,x14,#4                  //decrement the ht loop
    umlsl       v12.8h, v17.8b, v27.8b      //(3)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    umull       v22.8h, v19.8b, v25.8b      //(4)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    sub         x1,x1,x5,lsl #1
    umlsl       v22.8h, v18.8b, v24.8b      //(4)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    st1         {v12.8h},[x4], x8           //(3)store the result pu1_dst
    add         x9,x9,x2,lsl #2
    umlal       v22.8h, v20.8b, v26.8b      //(4)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v22.8h, v21.8b, v27.8b      //(4)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    add         x1,x1,x3,lsl #3

    st1         {v22.8h},[x4], x8           //(4)store the result pu1_dst
    bgt         outer_loop_ht_4

    cmp         x7,#0
    mov         x10,x5
    csel        x12, x9, x12,gt
    csel        x4, x1, x4,gt
    bgt         loop_residue_4

    b           end_loops

outer_loop_4:
    add         x6,x1,x3,lsl #1             //pu1_dst + dst_strd
    mov         x10,x5
    add         x4,x12,x2                   //pu1_src + src_strd
    mov         x13,#4

inner_loop_4:

    ld1         {v20.16b},[x12],x13          //vector load pu1_src
    ld1         {v16.16b},[x4],x13           //vector load pu1_src

    EXT         v21.16b,v20.16b,v20.16b,#2
    EXT         v22.16b,v20.16b,v20.16b,#4
    EXT         v23.16b,v20.16b,v20.16b,#6
    EXT         v17.16b,v16.16b,v16.16b,#2
    EXT         v18.16b,v16.16b,v16.16b,#4
    EXT         v19.16b,v16.16b,v16.16b,#6

    zip1        v0.2s, v20.2s, v16.2s
    zip1        v1.2s, v21.2s, v17.2s
    zip1        v2.2s, v22.2s, v18.2s
    zip1        v3.2s, v23.2s, v19.2s

    umull       v29.8h, v1.8b, v25.8b       //arithmetic operations for ii iteration in the same time
    umlsl       v29.8h, v0.8b, v24.8b
    umlal       v29.8h, v2.8b, v26.8b
    umlsl       v29.8h, v3.8b, v27.8b

    st1         {v29.d}[0],[x1],#8          //store the i iteration result which is in upper part of the register
    subs        x10,x10,#4                  //decrement the wd by 4

    st1         {v29.d}[1],[x6],#8          //store the ii iteration result which is in lower part of the register

    bgt         inner_loop_4

    sub         x12,x12,x5
    subs        x14,x14,#2                  //decrement the ht by 2
    sub         x1,x1,x5,lsl #1
    add         x12,x12,x2,lsl #1
    add         x1,x1,x3,lsl #2
    bgt         outer_loop_4

    cmp         x7,#0
    mov         x10,x5
    beq         end_loops

loop_residue_4:

    mov         x10,x5                      //2wd

loop_residue:
    ld1         {v20.16b},[x12]          //vector load pu1_src
    EXT         v21.16b,v20.16b,v20.16b,#2
    EXT         v22.16b,v20.16b,v20.16b,#4
    EXT         v23.16b,v20.16b,v20.16b,#6

    add         x12,x12,#4

    umull       v29.8h, v21.8b, v25.8b
    umlsl       v29.8h, v20.8b, v24.8b
    umlal       v29.8h, v22.8b, v26.8b
    umlsl       v29.8h, v23.8b, v27.8b

    st1         {v29.1d},[x1]               //store the result pu1_dst
    subs        x10,x10,#4                  //decrement the wd loop
    add         x1,x1,#8                    //pi2_dst + 8

    bgt         loop_residue                //loop again

end_loops:
    ldp         x21, x22,[sp],#16
    ldp         x19, x20,[sp],#16
    ldp         d14,d15,[sp],#16
    ldp         d12,d13,[sp],#16
    ldp         d10,d11,[sp],#16
    ret



