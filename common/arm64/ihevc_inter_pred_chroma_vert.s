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
//*  ihevc_inter_pred_chroma_vert_neon.s
//*
//* //brief
//*  contains function definitions for inter prediction  interpolation.
//* functions are coded using neon  intrinsics and can be compiled using

//* rvct
//*
//* //author
//*  yogeswaran rs
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
//*   chroma interprediction filter for vertical input
//*
//* //par description:
//*    applies a vertical filter with coefficients pointed to  by 'pi1_coeff' to
//*    the elements pointed by 'pu1_src' and  writes to the location pointed by
//*    'pu1_dst'  the output is down shifted by 6 and clipped to 8 bits
//*    assumptions : the function is optimized considering the fact width is
//*    multiple of 2,4 or 8. and also considering height  should be multiple of 2
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
//void ihevc_inter_pred_chroma_vert(uword8 *pu1_src,
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

.globl ihevc_inter_pred_chroma_vert_av8

.type ihevc_inter_pred_chroma_vert_av8, %function

ihevc_inter_pred_chroma_vert_av8:

    // stmfd sp!,{x4-x12,x14}        //stack stores the values of the arguments

    mov         x12,x4                      //loads pi1_coeff
    mov         x4,x5                       //loads ht

    cmp         x4,#0                       //checks ht == 0
    sub         x0,x0,x2                    //pu1_src - src_strd
    ld1         {v0.8b},[x12]               //loads pi1_coeff

    tst         x6,#3                       //checks (wd & 3)
    abs         v3.8b, v0.8b                //vabs_s8(coeff)
    lsl         x10,x6,#1                   //2*wd
    dup         v0.16b, v3.b[0]              //coeffabs_0
    dup         v1.16b, v3.b[1]              //coeffabs_1
    dup         v2.16b, v3.b[2]              //coeffabs_2
    dup         v3.16b, v3.b[3]              //coeffabs_3

    bgt         outer_loop_wd_2             //jumps to loop handling wd ==2

    tst         x4,#7                       //checks ht for mul of 8
    beq         core_loop_ht_8              //when height is multiple of 8

    lsl         x7,x3,#1                    //2*dst_strd
    sub         x9,x7,x10                   //2*dst_strd - 2wd
    lsl         x12,x2,#1                   //2*src_strd
    sub         x8,x12,x10                  //2*src_strd - 2wd
    mov         x5,x10                      //2wd

inner_loop_ht_2:                            //called when wd is multiple of 4 and ht is 4,2

    add         x6,x0,x2                    //pu1_src +src_strd
    ld1         {v17.8b},[x6],x2            //loads pu1_src
    ld1         {v5.8b},[x0],#8             //loads src
    ld1         {v4.8b},[x6],x2             //loads incremented src
    ld1         {v16.8b},[x6],x2            //loads incremented src
    ld1         {v18.8b},[x6]               //loads the incremented src

    umull       v6.8h, v17.8b, v1.8b        //vmull_u8(vreinterpret_u8_u32(src_tmp2), coeffabs_1)
    subs        x5,x5,#8                    //2wd - 8
    umlsl       v6.8h, v5.8b, v0.8b         //vmlsl_u8(mul_res1, vreinterpret_u8_u32(src_tmp1), coeffabs_0)
    umlal       v6.8h, v4.8b, v2.8b         //vmlal_u8(mul_res1, vreinterpret_u8_u32(src_tmp3), coeffabs_2)
    umlsl       v6.8h, v16.8b, v3.8b

    umull       v4.8h, v4.8b, v1.8b
    umlsl       v4.8h, v17.8b, v0.8b
    umlal       v4.8h, v16.8b, v2.8b
    add         x6,x1,x3                    //pu1_dst + dst_strd
    umlsl       v4.8h, v18.8b, v3.8b

    sqrshrun    v6.8b, v6.8h,#6             //shifts right
    sqrshrun    v4.8b, v4.8h,#6             //shifts right
    st1         {v6.8b},[x1],#8             //stores the loaded value
    st1         {v4.8b},[x6]                //stores the loaded value
    bgt         inner_loop_ht_2             //inner loop again

    subs        x4,x4,#2                    //ht - 2
    add         x1,x1,x9                    //pu1_dst += (2*dst_strd - 2wd)
    mov         x5,x10                      //2wd
    add         x0,x0,x8                    //pu1_src += (2*src_strd - 2wd)
    bgt         inner_loop_ht_2             //loop again

    b           end_loops                   //jumps to end

outer_loop_wd_2:                            //called when width is multiple of 2
    lsl         x5,x3,#1                    //2*dst_strd
    mov         x12,x10                     //2wd
    sub         x9,x5,x10                   //2*dst_strd - 2wd
    lsl         x7,x2,#1                    //2*src_strd
    sub         x8,x7,x10                   //2*src_strd - 2wd

inner_loop_wd_2:

    add         x6,x0,x2                    //pu1_src + src_strd
    ld1         {v6.s}[0],[x0]              //vld1_lane_u32((uint32_t *)pu1_src_tmp, src_tmp1, 0
    subs        x12,x12,#4                  //2wd - 4
    add         x0,x0,#4                    //pu1_src + 4
    ld1         {v6.s}[1],[x6],x2           //loads pu1_src_tmp
    dup         v7.2s, v6.s[1]
    ld1         {v7.s}[1],[x6],x2           //loads pu1_src_tmp
    umull       v4.8h, v7.8b, v1.8b         //vmull_u8(vreinterpret_u8_u32(src_tmp2), coeffabs_1)
    dup         v7.2s, v7.s[1]
    ld1         {v7.s}[1],[x6],x2
    umlsl       v4.8h, v6.8b, v0.8b
    umlal       v4.8h, v7.8b, v2.8b
    dup         v7.2s, v7.s[1]
    ld1         {v7.s}[1],[x6]
    add         x6,x1,x3                    //pu1_dst + dst_strd
    umlsl       v4.8h, v7.8b, v3.8b
    sqrshrun    v4.8b, v4.8h,#6             //vrshrq_n_s16(vreinterpretq_s16_u16(mul_res1),6)
    st1         {v4.s}[0],[x1]              //stores the loaded value
    add         x1,x1,#4                    //pu1_dst += 4
    st1         {v4.s}[1],[x6]              //stores the loaded value

    bgt         inner_loop_wd_2             //inner loop again

    //inner loop ends
    subs        x4,x4,#2                    //ht - 2
    add         x1,x1,x9                    //pu1_dst += 2*dst_strd - 2*wd
    mov         x12,x10                     //2wd
    add         x0,x0,x8                    //pu1_src += 2*src_strd - 2*wd

    bgt         inner_loop_wd_2             //loop again

    b           end_loops                   //jumps to end

core_loop_ht_8:                             //when wd & ht is multiple of 8

    lsl         x12,x3,#2                   //4*dst_strd
    sub         x8,x12,x10                  //4*dst_strd - 2wd
    lsl         x12,x2,#2                   //4*src_strd
    sub         x9,x12,x10                  //4*src_strd - 2wd

    bic         x5,x10,#7                   //x5 ->wd
    lsr         x14, x10, #3                //divide by 8
    mul         x12, x4 , x14               //multiply height by width
    sub         x12, x12,#4                 //subtract by one for epilog

prolog:
    add         x6,x0,x2                    //pu1_src + src_strd
    ld1         {v5.8b},[x6],x2             //loads pu1_src
    ld1         {v4.8b},[x0],#8             //loads the source
    ld1         {v6.8b},[x6],x2             //load and increment
    ld1         {v7.8b},[x6],x2             //load and increment

    umull       v30.8h, v5.8b, v1.8b        //mul with coeff 1
    subs        x5,x5,#8                    //2wd - 8
    umlsl       v30.8h, v4.8b, v0.8b
    add         x7,x1,x3                    //pu1_dst
    umlal       v30.8h, v6.8b, v2.8b
    ld1         {v16.8b},[x6],x2            //load and increment
    add         x13,x0,x9                   //pu1_dst += 4*dst_strd - 2*wd
    umlsl       v30.8h, v7.8b, v3.8b
    csel        x0, x13, x0,le

    umull       v28.8h, v6.8b, v1.8b        //mul_res 2
    bic         x13,x10,#7                  //x5 ->wd
    umlsl       v28.8h, v5.8b, v0.8b
    ld1         {v17.8b},[x6],x2
    csel        x5, x13, x5,le
    umlal       v28.8h, v7.8b, v2.8b
    lsl         x16,x2,#1
    umlsl       v28.8h, v16.8b, v3.8b
    add         x17,x2,x16
    sqrshrun    v30.8b, v30.8h,#6

    ld1         {v18.8b},[x6],x2
    umull       v26.8h, v7.8b, v1.8b
    add         x6,x0,x2                    //pu1_src + src_strd
    umlsl       v26.8h, v6.8b, v0.8b
    ld1         {v4.8b},[x0],#8             //loads the source
    umlal       v26.8h, v16.8b, v2.8b
    umlsl       v26.8h, v17.8b, v3.8b
    st1         {v30.8b},[x1],#8            //stores the loaded value
    sqrshrun    v28.8b, v28.8h,#6
    add         x13,x1,x8                   //pu1_src += 4*src_strd - 2*wd

    ld1         {v5.8b},[x6],x2             //loads pu1_src
    umull       v24.8h, v16.8b, v1.8b
    csel        x1, x13, x1,le
    umlsl       v24.8h, v7.8b, v0.8b
    ld1         {v6.8b},[x6],x2             //load and increment
    subs        x12,x12,#4
    umlal       v24.8h, v17.8b, v2.8b
    lsl         x11,x2,#2
    umlsl       v24.8h, v18.8b, v3.8b
    st1         {v28.8b},[x7],x3            //stores the loaded value
    sub         x13,x2,x2,lsl #3
    sqrshrun    v26.8b, v26.8h,#6
    neg         x11, x13

    ld1         {v7.8b},[x6],x2             //load and increment
    add         x14,x2,x2,lsl #1
    add         x14,x14,x11
    ble         epilog                      //jumps to epilog

kernel_8:

    ld1         {v16.8b},[x6],x2            //load and increment
    umull       v30.8h, v5.8b, v1.8b        //mul with coeff 1
    subs        x5,x5,#8                    //2wd - 8
    umlsl       v30.8h, v4.8b, v0.8b
    add         x13,x0,x9                   //pu1_dst += 4*dst_strd - 2*wd
    st1         {v26.8b},[x7],x3            //stores the loaded value
    umlal       v30.8h, v6.8b, v2.8b
    csel        x0, x13, x0,le
    ld1         {v17.8b},[x6],x2
    umlsl       v30.8h, v7.8b, v3.8b
    bic         x13,x10,#7                  //x10 ->wd
    sqrshrun    v24.8b, v24.8h,#6


    umull       v28.8h, v6.8b, v1.8b        //mul_res 2
    csel        x5, x13, x5,le
    umlsl       v28.8h, v5.8b, v0.8b
    ld1         {v18.8b},[x6],x2
    umlal       v28.8h, v7.8b, v2.8b
    st1         {v24.8b},[x7],x3            //stores the loaded value
    umlsl       v28.8h, v16.8b, v3.8b
    sqrshrun    v30.8b, v30.8h,#6
    add         x7,x1,x3                    //pu1_dst

    umull       v26.8h, v7.8b, v1.8b
    add         x6,x0,x2                    //pu1_src + src_strd
    umlsl       v26.8h, v6.8b, v0.8b
    ld1         {v4.16b},[x0]             //loads the source
    add         x11,x6,x17,lsl #1           //x11 = src + 7 * src_stride x6 = src + stride x17 = 3* src_stride
    umlal       v26.8h, v16.8b, v2.8b
    prfm        PLDL1KEEP,[x11]             //x11 = src + 7 * src_stride
    umlsl       v26.8h, v17.8b, v3.8b
    st1         {v30.8b},[x1],#8            //stores the loaded value
    prfm        PLDL1KEEP,[x11,x2]          //x11 = src + 8 * src_stride
    sqrshrun    v28.8b, v28.8h,#6
    add         x0,x0,#8

    ld1         {v5.16b},[x6],x2             //loads pu1_src
    umull       v24.8h, v16.8b, v1.8b
    prfm        PLDL1KEEP,[x11,x16]         //x11 = src + 9 * src_stride
    umlsl       v24.8h, v7.8b, v0.8b
    prfm        PLDL1KEEP,[x11,x17]         //x11 = src + 10 * src_stride
    ld1         {v6.16b},[x6],x2             //load and increment
    umlal       v24.8h, v17.8b, v2.8b
    add         x13,x1,x8                   //pu1_src += 4*src_strd - 2*wd
    st1         {v28.8b},[x7],x3            //stores the loaded value
    umlsl       v24.8h, v18.8b, v3.8b
    csel        x1, x13, x1,le
    ld1         {v7.16b},[x6],x2             //load and increment
    sqrshrun    v26.8b, v26.8h,#6

    lsr         x15,x10,#4
    cmp         x15,#2
    bge         kernel_16_start

    subs        x12,x12,#4
    bgt         kernel_8                    //jumps to kernel_8
    ble         epilog                    //jumps to kernel_8

kernel_16_start:
    subs        x12,x12,#4
    add         x0,x0,#8
kernel_16:

    umull       v30.8h, v5.8b, v1.8b        //mul with coeff 1
    subs        x5,x5,#16                    //2wd - 8
    umlsl       v30.8h, v4.8b, v0.8b
    add         x13,x0,x9                   //pu1_dst += 4*dst_strd - 2*wd
    umlal       v30.8h, v6.8b, v2.8b
    ld1         {v16.16b},[x6],x2            //load and increment
    csel        x0, x13, x0,le
    umlsl       v30.8h, v7.8b, v3.8b
    st1         {v26.8b},[x7],x3            //stores the loaded value
    sqrshrun    v24.8b, v24.8h,#6
    bic         x13,x10,#7                  //x10 ->wd

    umull       v28.8h, v6.8b, v1.8b        //mul_res 2
    csel        x5, x13, x5,le
    umlsl       v28.8h, v5.8b, v0.8b
    ld1         {v17.16b},[x6],x2
    umlal       v28.8h, v7.8b, v2.8b
    umlsl       v28.8h, v16.8b, v3.8b
    st1         {v24.8b},[x7],x3            //stores the loaded value
    sqrshrun    v30.8b, v30.8h,#6

    umull       v26.8h, v7.8b, v1.8b
    ld1         {v18.16b},[x6],x2
    umlsl       v26.8h, v6.8b, v0.8b
    add         x7,x1,x3                    //pu1_dst
    umlal       v26.8h, v16.8b, v2.8b
    add         x15,x1,#8
    umlsl       v26.8h, v17.8b, v3.8b
    add         x6,x0,x2                    //pu1_src + src_strd
    st1         {v30.8b},[x1]            //stores the loaded value
    sqrshrun    v28.8b, v28.8h,#6

    umull       v24.8h, v16.8b, v1.8b
    add         x11,x6,x17,lsl #1           //x11 = src + 7 * src_stride x6 = src + stride x17 = 3* src_stride
    umlsl       v24.8h, v7.8b, v0.8b
    prfm        PLDL1KEEP,[x11]             //x11 = src + 7 * src_stride
    umlal       v24.8h, v17.8b, v2.8b
    prfm        PLDL1KEEP,[x11,x2]          //x11 = src + 8 * src_stride
    umlsl       v24.8h, v18.8b, v3.8b
    st1         {v28.8b},[x7],x3            //stores the loaded value
    prfm        PLDL1KEEP,[x11,x16]         //x11 = src + 9 * src_stride
    sqrshrun    v26.8b, v26.8h,#6
    prfm        PLDL1KEEP,[x11,x17]         //x11 = src + 10 * src_stride

//******** 16th elements processing start

    umull2       v30.8h, v5.16b, v1.16b        //mul with coeff 1
    add         x1,x1,#16
    umlsl2       v30.8h, v4.16b, v0.16b
    add         x13,x1,x8                   //pu1_src += 4*src_strd - 2*wd
    umlal2       v30.8h, v6.16b, v2.16b
    csel        x1, x13, x1,le
    ld1         {v4.16b},[x0]             //loads the source
    umlsl2       v30.8h, v7.16b, v3.16b
    st1         {v26.8b},[x7],x3            //stores the loaded value
    sqrshrun    v24.8b, v24.8h,#6
    add         x0,x0,#16

    umull2       v28.8h,  v6.16b, v1.16b        //mul_res 2
    umlsl2       v28.8h,  v5.16b, v0.16b
    umlal2       v28.8h,  v7.16b, v2.16b
    ld1         {v5.16b},[x6],x2             //loads pu1_src
    umlsl2       v28.8h, v16.16b, v3.16b
    st1         {v24.8b},[x7],x3            //stores the loaded value
    sqrshrun    v30.8b, v30.8h,#6
    mov         x7,x15

    umull2       v26.8h,  v7.16b, v1.16b
    umlsl2       v26.8h,  v6.16b, v0.16b
    umlal2       v26.8h, v16.16b, v2.16b
    ld1         {v6.16b},[x6],x2             //load and increment
    umlsl2       v26.8h, v17.16b, v3.16b
    st1         {v30.8b},[x7],x2            //stores the loaded value
    sqrshrun    v28.8b, v28.8h,#6

    umull2       v24.8h, v16.16b, v1.16b
    umlsl2       v24.8h,  v7.16b, v0.16b
    subs        x12,x12,#8
    umlal2       v24.8h, v17.16b, v2.16b
    ld1         {v7.16b},[x6],x2             //load and increment
    umlsl2       v24.8h, v18.16b, v3.16b
    st1         {v28.8b},[x7],x3            //stores the loaded value
    sqrshrun    v26.8b, v26.8h,#6

    bgt         kernel_16                    //jumps to kernel_8
    ble         epilog_end                    //jumps to kernel_8

epilog:

    umull       v30.8h, v5.8b, v1.8b        //mul with coeff 1
    umlsl       v30.8h, v4.8b, v0.8b
    umlal       v30.8h, v6.8b, v2.8b
    umlsl       v30.8h, v7.8b, v3.8b
    st1         {v26.8b},[x7],x3            //stores the loaded value
    sqrshrun    v24.8b, v24.8h,#6

    ld1         {v16.8b},[x6],x2            //load and increment
    umull       v28.8h, v6.8b, v1.8b        //mul_res 2
    umlsl       v28.8h, v5.8b, v0.8b
    umlal       v28.8h, v7.8b, v2.8b
    umlsl       v28.8h, v16.8b, v3.8b
    st1         {v24.8b},[x7],x3            //stores the loaded value
    sqrshrun    v30.8b, v30.8h,#6

    ld1         {v17.8b},[x6],x2
    umull       v26.8h, v7.8b, v1.8b
    add         x7,x1,x3                    //pu1_dst
    umlsl       v26.8h, v6.8b, v0.8b
    umlal       v26.8h, v16.8b, v2.8b
    ld1         {v18.8b},[x6],x2
    umlsl       v26.8h, v17.8b, v3.8b
    st1         {v30.8b},[x1],#8            //stores the loaded value
    sqrshrun    v28.8b, v28.8h,#6

    umull       v24.8h, v16.8b, v1.8b
    umlsl       v24.8h, v7.8b, v0.8b
    umlal       v24.8h, v17.8b, v2.8b
    st1         {v28.8b},[x7],x3            //stores the loaded value
    umlsl       v24.8h, v18.8b, v3.8b
    sqrshrun    v26.8b, v26.8h,#6

epilog_end:
    st1         {v26.8b},[x7],x3            //stores the loaded value
    sqrshrun    v24.8b, v24.8h,#6
    st1         {v24.8b},[x7],x3            //stores the loaded value

end_loops:
    // ldmfd sp!,{x4-x12,x15}        //reload the registers from sp

    ret



