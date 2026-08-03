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
//*  ihevc_inter_pred_luma_horz.s
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
//*  - ihevc_inter_pred_luma_horz()
//*
//* //remarks
//*  none
//*
//*******************************************************************************
//*/

///* all the functions here are replicated from ihevc_inter_pred_filters.c and modified to */
///* include reconstruction */
//

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

//void ihevc_inter_pred_luma_horz (
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
//    x3 =>  dst_strd
//    x4 => *pi1_coeff
//    x5 =>  ht
//    x6 =>  wd

.text
.align 4

.include "ihevc_neon_macros.s"

.globl ihevc_inter_pred_luma_horz_av8

.type ihevc_inter_pred_luma_horz_av8, %function

ihevc_inter_pred_luma_horz_av8:

    // stmfd sp!, {x4-x12, x14}                //stack stores the values of the arguments
    push_v_regs
    stp         x19, x20,[sp,#-16]!
    stp         x21, x22,[sp,#-16]!
    //str        x1,[sp,#-4]
    // mov        x7,#8192

    mov         x15,x4 // pi1_coeff
    mov         x16,x5 // ht
    mov         x17,x6 // wd
    lsl         x19,x2,#2

start_loop_count:
    // ldr         x1,[sp,#-4]

    mov         x4,x15                      //loads pi1_coeff
    mov         x8,x16                      //loads ht
    mov         x10,x17                     //loads wd

    ld1         {v0.8b},[x4]                //coeff = vld1_s8(pi1_coeff)
    mov         x11,#1
    subs        x14,x8,#0                   //checks for ht == 0

    abs         v2.8b, v0.8b                //vabs_s8(coeff)

    //ble          end_loops

    dup         v24.16b, v2.b[0]             //coeffabs_0 = vdup_lane_u8(coeffabs, 0)
    sub         x12,x0,#3                   //pu1_src - 3
    dup         v25.16b, v2.b[1]             //coeffabs_1 = vdup_lane_u8(coeffabs, 1)
    add         x4,x12,x2                   //pu1_src_tmp2_8 = pu1_src + src_strd
    dup         v26.16b, v2.b[2]             //coeffabs_2 = vdup_lane_u8(coeffabs, 2)
    sub         x20,x10,x2,lsl #1           //2*src_strd - wd
    neg         x9, x20
    dup         v27.16b, v2.b[3]             //coeffabs_3 = vdup_lane_u8(coeffabs, 3)
    sub         x20,x10,x3,lsl #1           //2*dst_strd - wd
    neg         x8, x20
    dup         v28.16b, v2.b[4]             //coeffabs_4 = vdup_lane_u8(coeffabs, 4)
    dup         v29.16b, v2.b[5]             //coeffabs_5 = vdup_lane_u8(coeffabs, 5)
    // tst          x10,#7                            //checks wd for multiples
    dup         v30.16b, v2.b[6]             //coeffabs_6 = vdup_lane_u8(coeffabs, 6)
    dup         v31.16b, v2.b[7]             //coeffabs_7 = vdup_lane_u8(coeffabs, 7)

    mov         x7,x1
    cmp         x10,#4
    ble         outer_loop_4

    cmp         x10,#24
    mov         x20,#16
    csel        x10, x20, x10,eq
    add         x20, x8,#8
    csel        x8, x20, x8,eq
    add         x20, x9,#8
    csel        x9, x20, x9,eq
    cmp         x10,#16
    bge         outer_loop_16

    cmp         x10,#12
    add         x20, x8,#4
    csel        x8, x20, x8,eq
    add         x20, x9,#4
    csel        x9, x20, x9,eq
    b           outer_loop_8


outer_loop8_residual:
    sub         x12,x0,#3                   //pu1_src - 3
    mov         x1,x7
    mov         x14,#32
    add         x1, x1,#16
    add         x12, x12,#16
    mov         x10,#8
    add         x8, x8,#8
    add         x9, x9,#8

outer_loop_8:
    add         x6,x1,x3                    //pu1_dst + dst_strd
    add         x4,x12,x2                   //pu1_src + src_strd
    subs        x5,x10,#0                   //checks wd
    ble         end_inner_loop_8

    mov         x13,#8

inner_loop_8:
    ld1         {v1.16b},[x12],x13           //vector load pu1_src

    ld1         {v18.16b},[x4],x13


    EXT         v2.16b ,  v1.16b ,  v1.16b,#1
    EXT         v3.16b ,  v1.16b ,  v1.16b,#2
    EXT         v4.16b ,  v1.16b ,  v1.16b,#3
    EXT         v5.16b ,  v1.16b ,  v1.16b,#4
    EXT         v6.16b ,  v1.16b ,  v1.16b,#5
    EXT         v7.16b ,  v1.16b ,  v1.16b,#6
    EXT         v16.16b , v1.16b ,  v1.16b,#7


    umull       v17.8h, v2.8b,v25.8b
    umlal       v17.8h, v4.8b, v27.8b
    umlsl       v17.8h, v1.8b, v24.8b
    umlsl       v17.8h, v3.8b, v26.8b
    umlal       v17.8h, v5.8b, v28.8b
    umlal       v17.8h, v7.8b, v30.8b
    umlsl       v17.8h, v6.8b, v29.8b
    umlsl       v17.8h, v16.8b,v31.8b

    sqrshrun    v17.8b, v17.8h,#6


    EXT         v2.16b ,  v18.16b ,  v18.16b,#1
    EXT         v3.16b ,  v18.16b ,  v18.16b,#2
    EXT         v4.16b ,  v18.16b ,  v18.16b,#3
    EXT         v5.16b ,  v18.16b ,  v18.16b,#4
    st1         {v17.8b},[x1],#8
    EXT         v6.16b ,  v18.16b ,  v18.16b,#5
    EXT         v7.16b ,  v18.16b ,  v18.16b,#6
    EXT         v16.16b , v18.16b ,  v18.16b,#7


    umull       v17.8h, v2.8b,v25.8b
    umlal       v17.8h, v4.8b,v27.8b
    umlsl       v17.8h, v18.8b,v24.8b
    umlsl       v17.8h, v3.8b,v26.8b
    umlal       v17.8h, v5.8b,v28.8b
    umlal       v17.8h, v7.8b,v30.8b
    umlsl       v17.8h, v6.8b,v29.8b
    umlsl       v17.8h, v16.8b,v31.8b
    subs        x5,x5,#8                    //decrement the wd loop
    sqrshrun    v17.8b, v17.8h,#6
    cmp         x5,#4
    st1         {v17.8b},[x6],#8             //store the result pu1_dst

    bgt         inner_loop_8



end_inner_loop_8:
    subs        x14,x14,#2                  //decrement the ht loop
    add         x12,x12,x9                  //increment the src pointer by 2*src_strd-wd
    add         x1,x1,x8                    //increment the dst pointer by 2*dst_strd-wd
    bgt         outer_loop_8


    mov         x10,x17                     //loads wd
    cmp         x10,#12
    beq         outer_loop4_residual

end_loops:

    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
    ldp         x21, x22,[sp], #16
    ldp         x19, x20,[sp], #16
    pop_v_regs
    ret


outer_loop_16:
    mov         x15, #-7
    mov         x13,#1
    stp         x0,x7, [sp, #-16]!

    add         x4,x12,x2                   //pu1_src + src_strd
    and         x0, x12, #31
    sub         x5,x10,#0                   //checks wd
    add         x20,x12, x2, lsl #1
    prfm        PLDL1KEEP,[x20]

    ld1         { v0.4s},[x12],x13           //vector load pu1_src
    ld1         { v1.4s},[x12],x13          //vector load pu1_src
    ld1         { v2.4s},[x12],x13
    umull       v8.8h, v1.8b, v25.8b        //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    add         x6,x1,x3                    //pu1_dst + dst_strd
    ld1         { v3.4s},[x12],x13
    umlal       v8.8h, v3.8b, v27.8b        //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x4, x2, lsl #1
    umlsl       v8.8h, v0.8b, v24.8b        //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    prfm        PLDL1KEEP,[x20]
    ld1         { v4.4s},[x12],x13
    umlsl       v8.8h, v2.8b, v26.8b        //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v5.4s},[x12],x13
    umlal       v8.8h, v4.8b, v28.8b       //mul_res = vmlal_u8(src[0_4], coeffabs_4)//
    ld1         { v6.4s},[x12],x13
    umlsl       v8.8h, v5.8b, v29.8b       //mul_res = vmlsl_u8(src[0_5], coeffabs_5)//
    ld1         { v7.4s},[x12],x13
    umlal       v8.8h, v6.8b, v30.8b       //mul_res = vmlal_u8(src[0_6], coeffabs_6)//
    umlsl       v8.8h, v7.8b, v31.8b       //mul_res = vmlsl_u8(src[0_7], coeffabs_7)//


inner_loop_16:

    umull2       v20.8h, v1.16b, v25.16b
    subs        x5,x5,#16
    umlsl2       v20.8h, v0.16b, v24.16b
    add         x12, x12,#8
    ld1         { v0.4s},[x4],x13            //vector load pu1_src
    umlal2       v20.8h, v3.16b, v27.16b
    sub         x20,x14,#2
    ld1         { v1.4s},[x4],x13           //vector load pu1_src
    umlsl2       v20.8h, v2.16b, v26.16b
    csel        x14, x20, x14,eq
    ld1         { v2.4s},[x4],x13
    umlal2       v20.8h, v4.16b, v28.16b
    ld1         { v3.4s},[x4],x13
    umlsl2       v20.8h, v5.16b, v29.16b
    ld1         { v4.4s},[x4],x13
    umlal2       v20.8h, v6.16b, v30.16b
    ld1         { v5.4s},[x4],x13
    umlsl2       v20.8h, v7.16b, v31.16b
    ld1         { v6.4s},[x4],x13
    sqrshrun    v8.8b, v8.8h,#6             //right shift and saturating narrow result 1



    ld1         { v7.4s},[x4],x13
    umull       v10.8h, v1.8b, v25.8b       //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    add         x4, x4,#8
    umlal       v10.8h, v3.8b, v27.8b       //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    add         x20,x12,x9                  //increment the src pointer by 2*src_strd-wd
    umlsl       v10.8h, v0.8b, v24.8b       //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    csel        x12, x20, x12,eq
    umlsl       v10.8h, v2.8b, v26.8b       //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    add         x20,x12,x2                  //pu1_src + src_strd
    umlal       v10.8h, v4.8b, v28.8b      //mul_res = vmlal_u8(src[0_4], coeffabs_4)//
    csel        x4, x20, x4,eq
    umlsl       v10.8h, v5.8b, v29.8b       //mul_res = vmlsl_u8(src[0_5], coeffabs_5)//
    umlal       v10.8h, v6.8b, v30.8b       //mul_res = vmlal_u8(src[0_6], coeffabs_6)//
    st1         { v8.8b},[x1],#8            //store the result pu1_dst
    umlsl       v10.8h, v7.8b, v31.8b       //mul_res = vmlsl_u8(src[0_7], coeffabs_7)//
    sqrshrun    v9.8b, v20.8h,#6



    umull2       v22.8h, v1.16b, v25.16b
    umlsl2       v22.8h, v0.16b, v24.16b
    umlal2       v22.8h, v3.16b, v27.16b
    prfm        PLDL1KEEP,[x12,x19]
    umlsl2       v22.8h, v2.16b, v26.16b
    st1         { v9.8b},[x1],#8            //store the result pu1_dst
    umlal2       v22.8h, v4.16b, v28.16b
    add         x20,x1,x8
    umlal2       v22.8h, v6.16b, v30.16b
    csel        x1, x20, x1,eq
    umlsl2       v22.8h, v5.16b, v29.16b
    prfm        PLDL1KEEP,[x4,x19]
    umlsl2       v22.8h, v7.16b, v31.16b
    cmp         x14,#0
    sqrshrun    v10.8b, v10.8h,#6           //right shift and saturating narrow result 2



    beq         epilog_16

    ld1         { v0.4s},[x12],x13           //vector load pu1_src
    ld1         { v1.4s},[x12],x13          //vector load pu1_src
    umull       v8.8h, v1.8b, v25.8b        //mul_res = vmlal_u8(src[0_1], coeffabs_1)//
    ld1         { v2.4s},[x12],x13
    ld1         { v3.4s},[x12],x13
    umlal       v8.8h, v3.8b, v27.8b        //mul_res = vmull_u8(src[0_3], coeffabs_3)//
    umlsl       v8.8h, v0.8b, v24.8b        //mul_res = vmlsl_u8(src[0_0], coeffabs_0)//
    ld1         { v4.4s},[x12],x13
    umlsl       v8.8h, v2.8b, v26.8b        //mul_res = vmlsl_u8(src[0_2], coeffabs_2)//
    ld1         { v5.4s},[x12],x13
    umlal       v8.8h, v4.8b, v28.8b       //mul_res = vmlal_u8(src[0_4], coeffabs_4)//
    ld1         { v6.4s},[x12],x13
    umlsl       v8.8h, v5.8b, v29.8b       //mul_res = vmlsl_u8(src[0_5], coeffabs_5)//
    ld1         { v7.4s},[x12],x13
    cmp         x5,#0
    st1         { v10.8b},[x6],#8           //store the result pu1_dst
    umlal       v8.8h, v6.8b, v30.8b       //mul_res = vmlal_u8(src[0_6], coeffabs_6)//
    csel        x5, x10, x5,eq
    umlsl       v8.8h, v7.8b, v31.8b       //mul_res = vmlsl_u8(src[0_7], coeffabs_7)//

    sqrshrun    v11.8b, v22.8h,#6
    add         x20,x1,x3                   //pu1_dst + dst_strd
    st1         { v11.8b},[x6],#8           //store the result pu1_dst
    csel        x6, x20, x6,eq


    b           inner_loop_16

epilog_16:
    sqrshrun    v11.8b, v22.8h,#6
    st1         { v10.8b},[x6],#8           //store the result pu1_dst
    st1         { v11.8b},[x6],#8           //store the result pu1_dst

    ldp         x0,x7, [sp], #16
    mov         x10,x17
    cmp         x10,#24

    beq         outer_loop8_residual

end_loops1:

    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
    ldp         x21, x22,[sp], #16
   ldp         x19, x20,[sp], #16
    pop_v_regs
    ret

outer_loop4_residual:
    sub         x12,x0,#3                   //pu1_src - 3
    mov         x1,x7
    add         x1, x1,#8
    mov         x10,#4
    add         x12, x12,#8
    mov         x14,#16
    add         x8, x8,#4
    add         x9, x9,#4

outer_loop_4:
    add         x6,x1,x3                    //pu1_dst + dst_strd
    add         x4,x12,x2                   //pu1_src + src_strd

    subs        x5,x10,#0                   //checks wd
    ble         end_inner_loop_4

inner_loop_4:
    mov         x13,#8

    ld1         {v20.16b},[x12],x13          //vector load pu1_src
    EXT         v21.16b ,  v20.16b ,  v20.16b,#1
    ld1         {v22.16b},[x4],x13          //vector load pu1_src
    EXT         v23.16b ,  v22.16b ,  v22.16b,#1

    zip1        v0.2s, v20.2s, v22.2s
    zip1        v1.2s, v21.2s, v23.2s

    EXT         v8.16b ,  v20.16b ,  v20.16b,#2
    EXT         v9.16b ,  v20.16b ,  v20.16b,#3
    EXT         v10.16b ,  v22.16b ,  v22.16b,#2
    EXT         v11.16b ,  v22.16b ,  v22.16b,#3

    zip1        v2.2s, v8.2s, v10.2s
    zip1        v3.2s, v9.2s, v11.2s

    EXT         v12.16b ,  v20.16b ,  v20.16b,#4
    EXT         v13.16b ,  v20.16b ,  v20.16b,#5
    EXT         v14.16b ,  v22.16b ,  v22.16b,#4
    EXT         v15.16b ,  v22.16b ,  v22.16b,#5

    zip1        v4.2s, v12.2s, v14.2s
    zip1        v5.2s, v13.2s, v15.2s

    EXT         v16.16b ,  v20.16b ,  v20.16b,#6
    EXT         v17.16b ,  v20.16b ,  v20.16b,#7
    EXT         v18.16b ,  v22.16b ,  v22.16b,#6
    EXT         v19.16b ,  v22.16b ,  v22.16b,#7

    zip1        v6.2s, v16.2s, v18.2s
    zip1        v7.2s, v17.2s, v19.2s


    umull       v8.8h, v1.8b, v25.8b        //arithmetic operations for ii iteration in the same time
    sub         x12,x12,#4
    umlsl       v8.8h, v0.8b, v24.8b
    sub         x4,x4,#4
    umlsl       v8.8h, v2.8b, v26.8b
    umlal       v8.8h, v3.8b, v27.8b
    umlal       v8.8h, v4.8b, v28.8b
    umlsl       v8.8h, v5.8b, v29.8b
    umlal       v8.8h, v6.8b, v30.8b
    subs        x5,x5,#4                    //decrement the wd by 4
    umlsl       v8.8h, v7.8b, v31.8b

    sqrshrun    v8.8b, v8.8h,#6             //narrow right shift and saturating the result
    st1         {v8.s}[0],[x1],#4           //store the i iteration result which is in upper part of the register
    st1         {v8.s}[1],[x6],#4           //store the ii iteration result which is in lower part of the register
    bgt         inner_loop_4

end_inner_loop_4:
    subs        x14,x14,#2                  //decrement the ht by 4
    add         x12,x12,x9                  //increment the input pointer 2*src_strd-wd
    add         x1,x1,x8                    //increment the output pointer 2*dst_strd-wd
    bgt         outer_loop_4
    //subs     x7,x7,#1
    // bgt     start_loop_count

    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
    ldp         x21, x22,[sp], #16
    ldp         x19, x20,[sp], #16
    pop_v_regs
    ret
