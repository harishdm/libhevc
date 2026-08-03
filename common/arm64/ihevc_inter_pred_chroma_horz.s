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
//*    chroma interprediction filter for horizontal input
//*
//* //par description:
//*    applies a horizontal filter with coefficients pointed to  by 'pi1_coeff'
//*    to the elements pointed by 'pu1_src' and  writes to the location pointed
//*    by 'pu1_dst'  the output is downshifted by 6 and clipped to 8 bits
//*    assumptions : the function is optimized considering the fact width is
//*    multiple of 2,4 or 8. if width is 2, then height  should be multiple of 2.
//*    width 4,8 is optimized further
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

//void ihevc_inter_pred_chroma_horz(uword8 *pu1_src,
//                                   uword8 *pu1_dst,
//                                   word32 src_strd,
//                                   word32 dst_strd,
//                                   word8 *pi1_coeff,
//                                   word32 ht,
//                                   word32 wd)
//**************variables vs registers*****************************************
//x0 => *pu1_src
//x1 => *pi2_dst
//x2 =>  src_strd
//x3 =>  dst_strd

.text
.align 4

.include "ihevc_neon_macros.s"

.globl ihevc_inter_pred_chroma_horz_av8

.type ihevc_inter_pred_chroma_horz_av8, %function

ihevc_inter_pred_chroma_horz_av8:

    // stmfd sp!, {x4-x12, x14}                    //stack stores the values of the arguments

    stp         d9,d10,[sp,#-16]!
    stp         d11,d12,[sp,#-16]!
    stp         d13,d14,[sp,#-16]!
    stp         d8,d15,[sp,#-16]!           // Storing d15 using { sub sp,sp,#8; str d15,[sp] } is giving bus error.
                                            // d8 is used as dummy register and stored along with d15 using stp. d8 is not used in the function.
    stp         x19, x20,[sp,#-16]!

//    mov         x15,x4 // pi1_coeff
//    mov         x16,x5 // ht
//    mov         x17,x6 // wd

//    mov         x4,x15                      //loads pi1_coeff
//    mov         x7,x16                      //loads ht
//    mov         x10,x17                     //loads wd

    mov         x7,x5
    mov         x10,x6

    lsl         x15,x2,#1
    lsl         x16,x2,#2

    ld1         {v0.8b},[x4]                //coeff = vld1_s8(pi1_coeff)
    subs        x14,x7,#0                   //checks for ht == 0
    abs         v2.8b, v0.8b                //vabs_s8(coeff)
    mov         x11,#2
    ble         end_loops

    dup         v24.16b, v2.b[0]             //coeffabs_0 = vdup_lane_u8(coeffabs, 0)
    sub         x12,x0,#2                   //pu1_src - 2
    dup         v25.16b, v2.b[1]             //coeffabs_1 = vdup_lane_u8(coeffabs, 1)
    add         x4,x12,x2                   //pu1_src_tmp2_8 = pu1_src + src_strd
    dup         v26.16b, v2.b[2]             //coeffabs_2 = vdup_lane_u8(coeffabs, 2)

    tst         x10,#3                      //checks wd for multiples
    lsl         x5, x10, #1

    dup         v27.16b, v2.b[3]             //coeffabs_3 = vdup_lane_u8(coeffabs, 3)

    bne         outer_loop_4
    cmp         x10,#12
    beq         skip_16

    cmp         x10,#8
    bge         outer_loop_16
skip_16:
    tst         x7,#3

    sub         x9,x0,#2
    beq         outer_loop_ht_4             //jumps to else condition

    b           outer_loop_8

outer_loop_16:

    add         x4,x12,x2
    and         x0, x12, #31
    prfm        PLDL1KEEP,[x12,x15]
    ld1         { v0.4s},[x12],x11          //vector load pu1_src
    mov         x9,#10
    ld1         { v2.4s},[x12],x11          //vector load pu1_src

    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    mov         x10,x5                      //2wd
    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    mul         x14, x14 , x10
    prfm        PLDL1KEEP,[x4,x15]
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    sub         x20,x3,#16
    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    neg         x6, x20

    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    sub         x20,x5,x3,lsl #1
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    neg         x8, x20
    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    cmp         x14,#32
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    beq         epilog_end
    sub         x14, x14,#64

inner_loop_16:

    sqrshrun    v30.8b, v30.8h,#6
    subs        x10,x10,#16
    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x12,x8
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    csel        x12, x20, x12,eq
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    add         x13,x1,#8
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v30.4h}, [x1],x3
    sqrshrun    v31.8b, v28.8h,#6
    add         x20,x12,x2

    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    csel        x4, x20, x4,eq
    prfm        PLDL1KEEP,[x12,x16]
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v0.4s},[x12],x11          //vector load pu1_src
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v2.4s},[x12],x11          //vector load pu1_src
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v31.4h}, [x13],x3
    sqrshrun    v22.8b, v22.8h,#6
    cmp         x10,#0

    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    prfm        PLDL1KEEP,[x4,x16]
    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    csel        x10, x5, x10,eq             //2wd
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v22.4h},[x1],x6           //store the result pu1_dst
    sqrshrun    v23.8b, v20.8h,#6

    add         x20,x1,x8
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    csel        x1, x20, x1,eq
    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    subs        x14,x14,#32                 //decrement the ht loop
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         { v23.4h},[x13],x6          //store the result pu1_dst
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    bgt         inner_loop_16

    add         x14,x14,#64
    cmp         x14,#32
    beq         epilog_end

epilog:

    sqrshrun    v30.8b, v30.8h,#6
    add         x13,x1,#8
    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    subs        x10,x10,#16                 //decrement the wd loop
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v30.4h}, [x1],x3

    sqrshrun    v31.8b, v28.8h,#6
    add         x20,x12,x8
    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    csel        x12, x20, x12,eq
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    csel        x12, x20, x12,eq
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    csel        x10, x5, x10,eq             //2wd
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    add         x20,x12,x2
    st1         { v31.4h}, [x13],x3

    ld1         { v0.4s},[x12],x11          //vector load pu1_src
    sqrshrun    v22.8b, v22.8h,#6
    ld1         { v2.4s},[x12],x11          //vector load pu1_src
    umull       v30.8h, v2.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    csel        x4, x20, x4,eq
    ld1         { v4.4s},[x12],x11          //vector load pu1_src
    umlsl       v30.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v6.4s},[x12],x9           //vector load pu1_src
    umlal       v30.8h, v4.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v29.4s},[x4],x11          //vector load pu1_src
    umlsl       v30.8h, v6.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v22.4h},[x1],x6           //store the result pu1_dst

    sqrshrun    v23.8b, v20.8h,#6
    ld1         { v10.4s},[x4],x11          //vector load pu1_src
    umull2       v28.8h, v2.16b, v25.16b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    ld1         { v12.4s},[x4],x11          //vector load pu1_src
    umlsl2       v28.8h, v0.16b, v24.16b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v14.4s},[x4],x9           //vector load pu1_src
    umlal2       v28.8h, v4.16b, v26.16b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    add         x20,x1,x8
    umlsl2       v28.8h, v6.16b, v27.16b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v23.4h},[x13],x6          //store the result pu1_dst
    csel        x1, x20, x1,eq

epilog_end:

    sqrshrun    v30.8b, v30.8h,#6
    umull       v22.8h, v10.8b, v25.8b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v22.8h, v29.8b, v24.8b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v22.8h, v12.8b, v26.8b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    add         x13,x1,#8
    umlsl       v22.8h, v14.8b, v27.8b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    st1         { v30.4h}, [x1],x3
    sqrshrun    v31.8b, v28.8h,#6

    sqrshrun    v22.8b, v22.8h,#6
    umull2       v20.8h, v10.16b, v25.16b      //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    st1         { v31.4h}, [x13],x3
    umlsl2       v20.8h, v29.16b, v24.16b      //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal2       v20.8h, v12.16b, v26.16b      //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         { v22.4h},[x1]              //store the result pu1_dst
    umlsl2       v20.8h, v14.16b, v27.16b      //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    sqrshrun    v23.8b, v20.8h,#6
    st1         { v23.4h},[x13]             //store the result pu1_dst

    b           end_loops

outer_loop_8:

    add         x4,x12,x2                   //pu1_src + src_strd
    mov         x20,#8
    ld1         {v0.16b},[x12],x20
    add         x6,x1,x3                    //pu1_dst + dst_strd
    mov         x7,x5

inner_loop_8:

    EXT         v1.16b ,  v0.16b ,  v0.16b,#2
    ld1         {v4.16b},[x4],x20           //vector load pu1_src

    EXT         v2.16b ,  v0.16b ,  v0.16b,#4
    EXT         v3.16b ,  v0.16b ,  v0.16b,#6

    umull       v29.8h, v1.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v29.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    EXT         v5.16b ,  v4.16b ,  v4.16b,#2
    ld1         {v0.16b},[x12],x20
    EXT         v6.16b ,  v4.16b ,  v4.16b,#4
    EXT         v7.16b ,  v4.16b ,  v4.16b,#6
    sqrshrun    v29.8b, v29.8h,#6           //right shift and saturating narrow result 1

    umull       v10.8h, v5.8b, v25.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v10.8h, v6.8b, v26.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         {v29.8b},[x1],#8            //store the result pu1_dst
    umlsl       v10.8h, v7.8b, v27.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    sqrshrun    v10.8b, v10.8h,#6           //right shift and saturating narrow result 2
    subs        x7,x7,#8                    //decrement the wd loop
    st1         {v10.8b},[x6],#8            //store the result pu1_dst
    bgt         inner_loop_8

    sub         x12,x12,x12
    sub         x12,x12,x5
    subs        x14,x14,#2                  //decrement the ht loop
    sub         x1,x1,x5
    add         x12,x12,x2,lsl #1
    add         x1,x1,x3,lsl #1
    bgt         outer_loop_8
    b           end_loops



//height if 4 comes
outer_loop_ht_4:

    mov         x7,x5

prologue_ht_4:

inner_loop_ht_4:

    mov         x12,x9
    mov         x4,x1

    ld1         {v0.16b},[x12],x2           //(1)vector load pu1_src
    EXT         V1.16b,v0.16b,v0.16b,#2
    EXT         V2.16b,v0.16b,v0.16b,#4
    EXT         V3.16b,v0.16b,v0.16b,#6

    ld1         {v4.16b},[x12],x2           //(2)vector load pu1_src
    EXT         V5.16b,v4.16b,v4.16b,#2
    EXT         V6.16b,v4.16b,v4.16b,#4
    EXT         V7.16b,v4.16b,v4.16b,#6

    ld1         {v14.16b},[x12],x2          //(3)vector load pu1_src
    umull       v29.8h, v1.8b, v25.8b       //(1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v29.8h, v0.8b, v24.8b       //(1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //(1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //(1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    EXT         V15.16b,v14.16b,v14.16b,#2
    EXT         V16.16b,v14.16b,v14.16b,#4
    EXT         V17.16b,v14.16b,v14.16b,#6

    ld1         {v18.16b},[x12]          //(4)vector load pu1_src
    umull       v10.8h, v5.8b, v25.8b       //(2)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //(2)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v10.8h, v6.8b, v26.8b       //(2)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v10.8h, v7.8b, v27.8b       //(2)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    EXT         V19.16b,v18.16b,v18.16b,#2
    EXT         V20.16b,v18.16b,v18.16b,#4
    EXT         V21.16b,v18.16b,v18.16b,#6
    sqrshrun    v29.8b, v29.8h,#6           //(1)right shift and saturating narrow result 1

    add         x9,x9,#8                    //(core loop)

    subs        x7,x7,#8                    //(prologue)decrement the wd loop
    beq         epilogue


core_loop:
    mov         x12,x9

    ld1         {v0.16b},[x12],x2           //(1_1)vector load pu1_src

    umull       v12.8h, v15.8b, v25.8b      //(3)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v12.8h, v14.8b, v24.8b      //(3)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v12.8h, v16.8b, v26.8b      //(3)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v12.8h, v17.8b, v27.8b      //(3)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         {v4.16b},[x12],x2           //(2)vector load pu1_src

    EXT         V1.16b,v0.16b,v0.16b,#2
    EXT         V2.16b,v0.16b,v0.16b,#4
    EXT         V3.16b,v0.16b,v0.16b,#6

    st1         {v29.8b},[x4],x3            //(1)store the result pu1_dst
    sqrshrun    v10.8b, v10.8h,#6           //(2)right shift and saturating narrow result 2

    umull       v22.8h, v19.8b, v25.8b      //(4)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v22.8h, v18.8b, v24.8b      //(4)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v22.8h, v20.8b, v26.8b      //(4)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v22.8h, v21.8b, v27.8b      //(4)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         {v14.16b},[x12],x2          //(3)vector load pu1_src

    EXT         V5.16b,v4.16b,v4.16b,#2
    EXT         V6.16b,v4.16b,v4.16b,#4
    EXT         V7.16b,v4.16b,v4.16b,#6

    st1         {v10.8b},[x4],x3            //(2)store the result pu1_dst
    sqrshrun    v12.8b, v12.8h,#6           //(3)right shift and saturating narrow result 1

    umull       v29.8h, v1.8b, v25.8b       //(1_1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v29.8h, v0.8b, v24.8b       //(1_1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v29.8h, v2.8b, v26.8b       //(1_1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v29.8h, v3.8b, v27.8b       //(1_1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    ld1         {v18.16b},[x12]          //(4)vector load pu1_src

    EXT         V15.16b,v14.16b,v14.16b,#2
    EXT         V16.16b,v14.16b,v14.16b,#4
    EXT         V17.16b,v14.16b,v14.16b,#6

    sqrshrun    v22.8b, v22.8h,#6           //(4)right shift and saturating narrow result 2

    add         x9,x9,#8                    //(core loop)

    umull       v10.8h, v5.8b, v25.8b       //(2_1)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v10.8h, v4.8b, v24.8b       //(2_1)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v10.8h, v6.8b, v26.8b       //(2_1)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    umlsl       v10.8h, v7.8b, v27.8b       //(2_1)mul_res = vmlal_u8(src[0_1], coeffabs_1)//

    st1         {v12.8b},[x4],x3            //(3)store the result pu1_dst

    EXT         V19.16b,v18.16b,v18.16b,#2
    EXT         V20.16b,v18.16b,v18.16b,#4
    EXT         V21.16b,v18.16b,v18.16b,#6

    add         x1,x1,#8                    //(core loop)

    subs        x7,x7,#8                    //(core loop)

    st1         {v22.8b},[x4], x3           //(4)store the result pu1_dst
    sqrshrun    v29.8b, v29.8h,#6           //(1_1)right shift and saturating narrow result 1

    mov         x4, x1                      //(core loop)

    bgt         core_loop                   //loopback


epilogue:
    umull       v12.8h, v15.8b, v25.8b      //(3)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v12.8h, v14.8b, v24.8b      //(3)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v12.8h, v16.8b, v26.8b      //(3)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         {v29.8b},[x4],x3            //(1)store the result pu1_dst
    umlsl       v12.8h, v17.8b, v27.8b      //(3)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    sqrshrun    v10.8b, v10.8h,#6           //(2)right shift and saturating narrow result 2

    umull       v22.8h, v19.8b, v25.8b      //(4)mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v22.8h, v18.8b, v24.8b      //(4)mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    umlal       v22.8h, v20.8b, v26.8b      //(4)mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    st1         {v10.8b},[x4],x3            //(2)store the result pu1_dst
    umlsl       v22.8h, v21.8b, v27.8b      //(4)mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    sqrshrun    v12.8b, v12.8h,#6           //(3)right shift and saturating narrow result 1


    add         x1,x1,#8                    //(core loop)

    sqrshrun    v22.8b, v22.8h,#6           //(4)right shift and saturating narrow result 2


    sub         x9,x9,x5
    st1         {v12.8b},[x4],x3            //(3)store the result pu1_dst
    subs        x14,x14,#4                  //decrement the ht loop
    sub         x1,x1,x5
    st1         {v22.8b},[x4], x3           //(4)store the result pu1_dst
    add         x9,x9,x2,lsl #2
    add         x1,x1,x3,lsl #2
    bgt         outer_loop_ht_4
    b           end_loops

outer_loop_4:
    add         x6,x1,x3                    //pu1_dst + dst_strd
    mov         x7,x5
    add         x4,x12,x2                   //pu1_src + src_strd
    mov         x20,#4
inner_loop_4:

    ld1         {v20.16b},[x12],x20          //vector load pu1_src

    EXT         v21.16b,v20.16b,v20.16b,#2

    ld1         {v16.16b},[x4],x20          //vector load pu1_src

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

    sqrshrun    v29.8b, v29.8h,#6           //narrow right shift and saturating the result
    st1         {v29.s}[0],[x1],#4          //store the i iteration result which is in upper part of the register
    subs        x7,x7,#4                    //decrement the wd by 4

    st1         {v29.s}[1],[x6],#4          //store the ii iteration result which is in lower part of the register

    bgt         inner_loop_4

    sub         x12,x12,x5
    subs        x14,x14,#2                  //decrement the ht by 2
    sub         x1,x1,x5
    add         x12,x12,x2,lsl #1
    add         x1,x1,x3,lsl #1
    bgt         outer_loop_4

end_loops:

    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp

    ldp         x19, x20,[sp],#16
    ldp         d8,d15,[sp],#16             // Loading d15 using { ldr d15,[sp]; add sp,sp,#8 } is giving bus error.
                                            // d8 is used as dummy register and loaded along with d15 using ldp. d8 is not used in the function.
    ldp         d13,d14,[sp],#16
    ldp         d11,d12,[sp],#16
    ldp         d9,d10,[sp],#16
    ret








