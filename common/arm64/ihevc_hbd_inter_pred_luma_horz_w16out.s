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
#//void ihevc_hbd_inter_pred_luma_horz(UWORD16 *pu2_src,
#//                                UWORD16 *pu2_dst,
#//                                WORD32 src_strd,
#//                                WORD32 dst_strd,
#//                                WORD8 *pi1_coeff,
#//                                WORD32 ht,
#//                                WORD32 wd,UWORD8 bit_depth)
#//
#//*************variables vs registers*****************************************
#//    x0 => *pu1_src
#//    x1 => *pu1_dst
#//    x2 =>  src_strd
#//    x3 =>  dst_strd
#//    x4 => *pi1_coeff
#//    x5 =>  ht
#//    x6 =>  wd


.text
.align 4

.include "ihevc_neon_macros.s"

.globl ihevc_hbd_inter_pred_luma_horz_w16out_av8

.type ihevc_hbd_inter_pred_luma_horz_w16out_av8, %function

ihevc_hbd_inter_pred_luma_horz_w16out_av8:

    // stmfd sp!, {x4-x12, x14}                //stack stores the values of the arguments
    //push_v_regs
    stp         x19, x20,[sp,#-16]!
    //str        x1,[sp,#-4]
    // mov        x7,#8192

    mov         x16,x5 // ht
    mov         x17,x6 // wd

start_loop:
    // ldr         x1,[sp,#-4]

    lsl         x2,x2, #1                    //src_strd << 1
    lsl         x3,x3, #1                    //dst_strd << 1
    mov         x8,x16                      //loads ht
    mov         x10,x17                     //loads wd

    ld1         {v0.8b},[x4]                //coeff = vld1_s8(pi1_coeff)
    mov         x11,#2                        // similar to 1 byte in lbd
    subs        x14,x8,#0                   //checks for ht == 0

cmp x10, #8
bge cond_8x

cond_4x:
    sshll       v2.8h, v0.8b, #0            //vmovl_s8(vabs_s8((pi1_coeff)));
    dup         v24.4h, v2.h[0]            //coeffabs_0 = vdup_laneq_s16(coeffabs, 0)
    sub         x12,x0,#6                   //pu2_src - 3
    dup         v25.4h, v2.h[1]            //coeffabs_1 = vdup_laneq_s16(coeffabs, 1)
    add         x4,x12,x2                   //pu1_src_tmp2_8 = pu1_src + src_strd
    dup         v26.4h, v2.h[2]            //coeffabs_2 = vdup_laneq_s16(coeffabs, 2)
    sub         x20,x10,x2,lsl #1           //2*src_strd - wd
    add         x20, x20,x10                //4*src_strd - 2*wd for hbd
    add         x20, x20,x10, lsl #1                //4*src_strd - 2*wd for hbd
    neg         x9, x20
    dup         v27.4h, v2.h[3]            //coeffabs_3 = vdup_laneq_s16(coeffabs, 3)
    sub         x20,x10,x3,lsl #1           //2*dst_strd - wd
    add         x20, x20,x10
    neg         x8, x20
    dup         v28.4h, v2.h[4]            //coeffabs_4 = vdup_laneq_s16(coeffabs, 4)
    dup         v29.4h, v2.h[5]            //coeffabs_5 = vdup_laneq_s16(coeffabs, 5)
    dup         v30.4h, v2.h[6]            //coeffabs_6 = vdup_laneq_s16(coeffabs, 6)
    dup         v31.4h, v2.h[7]            //coeffabs_7 = vdup_laneq_s16(coeffabs, 7)
    b jump

cond_8x:
    abs         v0.8b, v0.8b                //vabs_s8(coeff)
    sshll       v2.8h, v0.8b, #0            //vmovl_s8(vabs_s8((pi1_coeff)));
    dup         v24.8h, v2.h[0]            //coeffabs_0 = vdup_laneq_s16(coeffabs, 0)
    sub         x12,x0,#6                   //pu2_src - 3
    dup         v25.8h, v2.h[1]            //coeffabs_1 = vdup_laneq_s16(coeffabs, 1)
    add         x4,x12,x2                   //pu1_src_tmp2_8 = pu1_src + src_strd
    dup         v26.8h, v2.h[2]            //coeffabs_2 = vdup_laneq_s16(coeffabs, 2)
    dup         v27.8h, v2.h[3]            //coeffabs_3 = vdup_laneq_s16(coeffabs, 3)
    dup         v28.8h, v2.h[4]            //coeffabs_4 = vdup_laneq_s16(coeffabs, 4)
    dup         v29.8h, v2.h[5]            //coeffabs_5 = vdup_laneq_s16(coeffabs, 5)
    dup         v30.8h, v2.h[6]            //coeffabs_6 = vdup_laneq_s16(coeffabs, 6)
    dup         v31.8h, v2.h[7]            //coeffabs_7 = vdup_laneq_s16(coeffabs, 7)
    add         x8, x2, x2
    lsl         x21, x10, #1
    sub         x8, x8, x21
    add         x9, x3, x3
    sub         x9, x9, x21
    b jump


jump:

    mov         w4, #1023
    dup         v23.8h,w4               // max1023_16x4 = vdup_n_u16(1023);
    cmp         x17, #8
    bge         outer_loop_8

outer_loop_4:

    add         x6,x1,x3                    //pu1_dst + dst_strd
    add         x4,x12,x2                   //pu1_src + src_strd
    subs        x5,x10,#0                   //checks wd

    ble         end_inner_loop_4

inner_loop_4:
    ld1         {v0.4h},[x12],x11           //vector load pu1_src
    ld1         {v1.4h},[x12],x11
    ld1         {v2.4h},[x12],x11
    ld1         {v3.4h},[x12],x11
    ld1         {v4.4h},[x12],x11
    ld1         {v5.4h},[x12],x11
    ld1         {v6.4h},[x12],x11
    ld1         {v7.4h},[x12],x11

    smull       v8.4s, v1.4h, v25.4h        //mul_res = vmull_s16(src[0_1], coeffabs_1)//
    smlal       v8.4s, v0.4h, v24.4h        //mul_res = vmlsl_s16(src[0_0], coeffabs_0)//
    smlal       v8.4s, v3.4h, v27.4h        //mul_res = vmlal_s16(src[0_3], coeffabs_3)//
    smlal       v8.4s, v2.4h, v26.4h        //mul_res = vmlsl_s16(src[0_2], coeffabs_2)//
    smlal       v8.4s, v4.4h, v28.4h        //mul_res = vmlal_s16(src[0_4], coeffabs_4)//
    smlal       v8.4s, v5.4h, v29.4h        //mul_res = vmlsl_s16(src[0_5], coeffabs_5)//
    smlal       v8.4s, v6.4h, v30.4h        //mul_res = vmlal_s16(src[0_6], coeffabs_6)//
    smlal       v8.4s, v7.4h, v31.4h        //mul_res = vmlsl_s16(src[0_7], coeffabs_7)//

    ld1         {v12.4h},[x4],x11           //vector load pu1_src + src_strd
    ld1         {v13.4h},[x4],x11
    ld1         {v14.4h},[x4],x11
    ld1         {v15.4h},[x4],x11
    ld1         {v16.4h},[x4],x11           //vector load pu1_src + src_strd
    ld1         {v17.4h},[x4],x11
    ld1         {v18.4h},[x4],x11
    ld1         {v19.4h},[x4],x11           //vector load pu1_src + src_strd

    smull       v10.4s, v13.4h, v25.4h        //mul_res = vmull_s16(src[0_1], coeffabs_1)//
    smlal       v10.4s, v12.4h, v24.4h        //mul_res = vmlsl_s16(src[0_0], coeffabs_0)//
    smlal       v10.4s, v15.4h, v27.4h        //mul_res = vmlal_s16(src[0_3], coeffabs_3)//
    smlal       v10.4s, v14.4h, v26.4h        //mul_res = vmlsl_s16(src[0_2], coeffabs_2)//
    smlal       v10.4s, v16.4h, v28.4h        //mul_res = vmlal_s16(src[0_4], coeffabs_4)//
    smlal       v10.4s, v17.4h, v29.4h        //mul_res = vmlsl_s16(src[0_5], coeffabs_5)//
    smlal       v10.4s, v18.4h, v30.4h        //mul_res = vmlal_s16(src[0_6], coeffabs_6)//
    smlal       v10.4s, v19.4h, v31.4h        //mul_res = vmlsl_s16(src[0_7], coeffabs_7)//

    shrn    v20.4h, v8.4s,#2            //right shift and saturating narrow result  vqrshrun_n_s32(mul_res_32x4_r0_0, 6);
    shrn    v8.4h, v10.4s,#2            //right shift and saturating narrow result 2

    st1         {v20.4h},[x1],#8            //store the result pu1_dst
    st1         {v8.4h},[x6],#8             //store the result pu1_dst


    subs        x5,x5,#4                    //decrement the wd loop
    cmp         x5,#4
    bgt         inner_loop_4

end_inner_loop_4:
    subs        x14,x14,#2                  //decrement the ht loop
    add         x12,x12,x9                  //increment the src pointer by 2*src_strd-wd
    add         x1,x1,x8                    //increment the dst pointer by 2*dst_strd-wd
    bgt         outer_loop_4

end_loops:

    #pop {x4-x12,x15}                  //reload the registers from sp
    ldp         x19, x20,[sp], #16
    ret

outer_loop_8:

    add         x6,x1,x3                    //pu1_dst + dst_strd
    add         x4,x12,x2                   //pu1_src + src_strd
    subs        x5,x10,#0                   //checks wd

    ble         end_inner_loop_8

inner_loop_8:
    ld1         {v0.8h},[x12],x11           //vector load pu1_src
    ld1         {v1.8h},[x12],x11
    ld1         {v2.8h},[x12],x11
    ld1         {v3.8h},[x12],x11
    ld1         {v4.8h},[x12],x11
    ld1         {v5.8h},[x12],x11
    ld1         {v6.8h},[x12],x11
    ld1         {v7.8h},[x12],x11

    mul         v21.8h, v1.8h, v25.8h
    mul         v20.8h, v0.8h, v24.8h
    mla         v21.8h, v3.8h, v27.8h
    mla         v20.8h, v2.8h, v26.8h
    mul         v22.8h, v4.8h, v28.8h
    mla         v20.8h, v5.8h, v29.8h
    mla         v22.8h, v6.8h, v30.8h
    mla         v20.8h, v7.8h, v31.8h

    uaddl       v12.4s, v21.4h, v22.4h
    uaddl2      v13.4s, v21.8h, v22.8h
    usubw       v14.4s, v12.4s, v20.4h
    usubw2      v15.4s, v13.4s, v20.8h
    shrn    v8.4h,  v14.4s, #2
    shrn    v20.4h, v15.4s, #2
    st1         {v8.4h}, [x1],#8
    st1         {v20.4h},[x1],#8

    ld1         {v12.8h},[x4],x11           //vector load pu1_src + src_strd
    ld1         {v13.8h},[x4],x11
    ld1         {v14.8h},[x4],x11
    ld1         {v15.8h},[x4],x11
    ld1         {v16.8h},[x4],x11           //vector load pu1_src + src_strd
    ld1         {v17.8h},[x4],x11
    ld1         {v18.8h},[x4],x11
    ld1         {v19.8h},[x4],x11           //vector load pu1_src + src_strd

    mul         v1.8h, v13.8h, v25.8h
    mul         v0.8h, v12.8h, v24.8h
    mla         v1.8h, v15.8h, v27.8h
    mla         v0.8h, v14.8h, v26.8h
    mul         v2.8h, v16.8h, v28.8h
    mla         v0.8h, v17.8h, v29.8h
    mla         v2.8h, v18.8h, v30.8h
    mla         v0.8h, v19.8h, v31.8h

    uaddl       v12.4s, v1.4h, v2.4h
    uaddl2      v13.4s, v1.8h, v2.8h
    usubw       v14.4s, v12.4s, v0.4h
    usubw2      v15.4s, v13.4s, v0.8h
    shrn    v8.4h,  v14.4s, #2
    shrn    v20.4h, v15.4s, #2
    st1         {v8.4h}, [x6],#8
    st1         {v20.4h},[x6],#8

    subs        x5,x5,#8                    //decrement the wd loop
    cmp         x5,#8
    bge         inner_loop_8

end_inner_loop_8:
    subs        x14,x14,#2                  //decrement the ht loop
    add         x12,x12,x8
    add         x1,x1,x9
    bgt         outer_loop_8

end_loops_8:
    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
    ldp         x19, x20,[sp], #16
    ret

