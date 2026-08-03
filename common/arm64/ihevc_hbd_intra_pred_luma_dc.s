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
//void ihevc_hbd__intra_pred_luma_dc(uword16 *pu1_ref,
//                              word32 src_strd,
//                              uword8 *pu1_dst,
//                              word32 dst_strd,
//                              word32 nt,
//                              word32 mode)
//
//**************variables vs registers*****************************************
//x0 => *pu1_ref
//x1 => src_strd
//x2 => *pu1_dst
//x3 => dst_strd

//stack contents from #40
//    nt
//    mode
//    pi1_coeff

.text
.align 4
.include "ihevc_neon_macros.s"


.globl ihevc_hbd_intra_pred_luma_dc_av8

.type ihevc_hbd_intra_pred_luma_dc_av8, %function

ihevc_hbd_intra_pred_luma_dc_av8:

    // stmfd sp!, {x4-x12, x14}            //stack stores the values of the arguments

    stp         x19, x20,[sp,#-16]!


//********** testing
    //mov        x6, #128
    //b        prologue_cpy_32
//********** testing

    mov         x11, #2                    //mov #2 to x11 (to be used to add to 2dc_val & 3dc_val)
    mov         x9, #0
    mov         v17.s[0], w11
    mov         v17.s[1], w9            //v17=2

    clz         w5,w4               //w5= clz(nt)

    add         x6, x0, x4, lsl #1                  //&src[nt]
    sub         x20, x5, #32                //clz(nt) - 32 =log2nt
    neg         x5, x20                     //x5 = 32 -  clz(nt)
    add         x7, x0, x4, lsl #2          //x7= src + 2(2nt)= &src[2nt]..extra *2 as 10 bit requires 2B for 1 pixel

    add         x8, x7, #2                  //x8 = x7 + 2=&src[2nt+1]
    mvn         x5, x5                //x5 = 1's compl(32 -  clz(nt))
    add         x5, x5, #1            //x5 = 1 + 1's compl(32 -  clz(nt)) = clz(nt)-32
    dup         v7.2s,w5              //v7.2s=clz(nt)-32

    ldrh        w14, [x8]             //w14 = src[2nt+1]
    sxtw        x14,w14               //x14 = src[2nt+1]
    shl         d7, d7,#32            //d7 = left shift(clz(nt)-32, 32)

    sub         x9, x7, #2                  //&src[2nt-1]
    sshr        d7, d7,#32            //d7 = right shift with sign extension(d7,32)

    mov         x7, x8                      //x7 = &src[2nt+1]     x7 also stores 2nt+1

    ldrh        w12, [x9]            //w12= src[2nt-1]
    sxtw        x12,w12              //x12 = src[2nt-1]
    add         x14, x14, x12               //src[2nt+1] + src[2nt-1]
    add         x14, x14, x11               //src[2nt+1] + src[2nt-1] + 2

    cmp         x4, #4                     //cmp(x4,4)
    beq         dc_4

    mov         x10, x4                     //x10= nt

add_loop:
    ld1         {v0.4h},[x6],#8
    ld1         {v1.4h},[x8],#8
    add         v2.4h, v0.4h, v1.4h
    ld1         {v0.4h},[x6],#8
    ld1         {v1.4h},[x8],#8
    add         v3.4h, v0.4h, v1.4h
    add         v4.4h, v2.4h, v3.4h
    uaddlp      v5.2s, v4.4h

    mov         x5, #0
    mov         v6.s[0], w4
    mov         v6.s[1], w5                 //store nt to accumulate

    uadalp      v6.1d,  v5.2s
    subs        x10, x10,#8
    beq         epil_add_loop

core_loop_add:
    ld1         {v0.4h},[x6],#8
    ld1         {v1.4h},[x8],#8
    add         v2.4h, v0.4h, v1.4h
    ld1         {v0.4h},[x6],#8
    ld1         {v1.4h},[x8],#8
    add         v3.4h, v0.4h, v1.4h
    add         v4.4h, v2.4h, v3.4h
    uaddlp      v5.2s, v4.4h
    uadalp      v6.1d,  v5.2s

    subs        x10, x10,#8
    bne         core_loop_add

epil_add_loop:

    sshl        d18, d6, d7                 //1+log2nt=32-clz(nt) d6=acc_dc  d7 = -[32-clz(nt)] d18 = d6<<d7 =(acc_dc) shr by log2nt+1
                              //d18 = dc_val = acc_dc>>log2nt+1
    cmp         x4, #32

    mov         v28.s[0], w14               //w14=src[2nt+1] + src[2nt-1] + 2
    mov         v28.s[1], w5                //w5=0 .....src[2nt+1]+2+src[2nt-1] moved to d28
    mov         x20,#128
    csel        x6, x20, x6,eq              //if(x4==32) then x6=x20=128 else x6=x6

    dup         v16.8h, v18.4h[0]           //v16.4h=dc_val
    shl         d25, d18,#1                 //d25=2*dc_val

    beq         prologue_cpy_32

    add         d27,  d25 ,  d28            //d27=src[2nt+1]+2+src[2nt-1]+2dc_val
    mov         x20,#0
    csel        x6, x20, x6,ne              //nt

    ushr        v29.4h, v27.4h,#2           //v29=final dst[0]'s value
    csel        x10, x4, x10,ne             //if(x4!=32) then x10=x4=nt else x10=x10=nt-16

    add         d23,  d25 ,  d18            //d23=3*dc_val
    add         d23,  d23 ,  d17            //3*dc_val + 2
    dup         v24.8h, v23.4h[0]           //3*dc_val + 2 (moved to all lanes)

    sub         x12, x3, x3, lsl #3         //-7*strd
    add         x12,x12,x12
    add         x12, x12, #16                //offset after one 8x8 block (-7*strd + 8)

    sub         x0, x3, x4                  //strd - nt
    add x0,x0,x0

prologue_col:
    //0th column and 0-7 rows done here
    //x8 and x9 (2nt+1+col 2nt-1-row)
   add         x0, x0, #16                  //strd - nt + 8

    mov         x8, x7                      //&src[2nt+1]   ISNT IT REDUNDANT INSTRUCTION
    ld1         {v0.8h},[x8],#16             //col 1::7 load (prol)
    add         v0.8h,  v0.8h ,  v24.8h   //col 1::7 add 3dc+2 (prol)
    sshr        v2.8h, v0.8h,#2

    sub         x9, x9, #14                  //&src[2nt-1-row]
    ld1         {v1.8h},[x9]                //row 7::1 (0 also) load (prol)
    add         v1.8h,  v1.8h ,  v24.8h   //row 1::7 add 3dc+2 (prol)
    sshr        v3.8h, v1.8h,#2
    sub         x9, x9, #16

    ld1         {v6.8h},[x8]                //col 8::15 load (prol extra)
    add         v6.8h,  v6.8h ,  v24.8h   //col 8::15 add 3dc+2 (prol extra)

    movi        d19, #0x000000000000ffff
    bsl         v19.16b,v29.16b,v2.16b
    add         x16, x3, x3

    rev64       v3.8h,  v3.8h
    DUP d25,V3.D[0]
    DUP d26,V3.D[1]

    # ins v25.8h[7],v3.8h[0]
    # ins v25.8h[6],v3.8h[1]
    # ins v25.8h[5],v3.8h[2]
    # ins v25.8h[4],v3.8h[3]
    # ins v25.8h[3],v3.8h[4]
    # ins v25.8h[2],v3.8h[5]
    # ins v25.8h[1],v3.8h[6]
    # ins v25.8h[0],v3.8h[7]

    st1         {v19.8h},[x2], x16           //store row 0 (prol)
    sshr        d26, d26,#16                   //row 0 shift (prol) (first value to be ignored)
    movi        d20, #0x000000000000ffff     //byte mask row 1 (prol)

loop_again_col_row:

    bsl         v20.16b,  v26.16b ,  v16.16b    //row 1    (prol)
    st1         {v20.8h},[x2], x16         //store row 1 (prol)
    sshr        d26, d26,#16                   //row 1 shift (prol)

    sshr        v4.8h, v6.8h,#2            //columns shx2 movn (prol extra)

    movi        d21, #0x000000000000ffff    //byte mask row 2 (prol)
    bsl         v21.16b,  v26.16b ,  v16.16b    //row 2 (prol)
    st1         {v21.8h},[x2], x16           //store row 2 (prol)
    sshr        d26, d26,#16                   //row 2 shift (prol)

    movi        d20, #0x000000000000ffff    //byte mask row 3 (prol)
    bsl         v20.16b,  v26.16b ,  v16.16b    //row 3    (prol)
    st1         {v20.8h},[x2], x16           //store row 3 (prol)
    sshr        d26, d26,#16                   //row 3 shift (prol)

    movi        d21, #0x000000000000ffff    //byte mask row 4 (prol)
    bsl         v21.16b,  v25.16b ,  v16.16b    //row 4 (prol)
    st1         {v21.8h},[x2], x16           //store row 4 (prol)
    sshr        d25, d25,#16                   //row 4 shift (prol)

    movi        d20, #0x000000000000ffff    //byte mask row 5 (prol)
    bsl         v20.16b,  v25.16b ,  v16.16b    //row 5 (prol)
    st1         {v20.8h},[x2], x16           //store row 5 (prol)
    sshr        d25, d25,#16                  //row 5 shift (prol)

    movi        d21, #0x000000000000ffff    //byte mask row 6 (prol)
    bsl         v21.16b,  v25.16b ,  v16.16b    //row 6 (prol)
    st1         {v21.8h},[x2], x16           //store row 6 (prol)
    sshr        d25, d25,#16                   //row 6 shift (prol)

    movi        d20, #0x000000000000ffff    //byte mask row 7 (prol)
    bsl         v20.16b,  v25.16b ,  v16.16b    //row 7 (prol)
    st1         {v20.8h},[x2], x12          //store row 7 (prol)
    sshr        d25, d25,#16                   //row 7 shift (prol)

    subs        x10, x10, #8                //counter for cols
    beq         end_func
    blt         copy_16

    ld1         {v1.8h},[x9]                //row 8::15 load (prol extra)
    add         v1.8h,  v1.8h ,  v24.8h   //row 8::15 add 3dc+2 (prol extra)

    movi        d20, #0x000000000000ffff    //byte mask row 9 (prol)
    sshr     v3.8h, v1.8h,#2            //rows shx2 movn (prol)

    rev64       v3.8h,  v3.8h
    DUP d25,V3.D[0]
    DUP d26,V3.D[1]
    st1         {v4.8h},[x2], x16           //store 2nd col (for 16x16)

    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x0          //go to next row for 16


    bsl         v20.16b,  v26.16b ,  v16.16b    //row 9    (prol)
    st1         {v20.8h},[x2], x16        //store row 9 (prol)
    subs        x10, x10, #8

    sshr        d26, d26,#16                   //row 9 shift (prol)

    movi        d20, #0x000000000000ffff    //byte mask row 9 (prol)

    b           loop_again_col_row


copy_16:
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2], x16
    st1         {v16.8h},[x2]

    b           end_func

prologue_cpy_32:
    mov         x9, #128
    lsl         x3, x3, 1
    add         x5, x2, x3
    add         x8, x5, x3
    add         x10, x8, x3
    dup         v20.8h, v16.4h[0]
    lsl         x6, x3, #2    //x6 = 4*dst_strd
    add         x6, x6, #-48  //x6 = 4*dst_strd - 48

    st1         {v20.8h}, [x2],#16   //pu2_dst,pu2_dst[1],...pu2_dst[7]//8 pixels in a row gets copied
    st1         {v20.8h}, [x2],#16   //pu2_dst[8],pu2_dst[9],...pu2_dst[15]//next 8 pixels in a row gets copied
    st1         {v20.8h}, [x5],#16    //pu2_dst[1* dst_strd],pu2_dst[1*dst_strd+1],...pu2_dst[1*dst_strd+7]
    st1         {v20.8h}, [x5],#16    //pu2_dst[1* dst_strd + 8],pu2_dst[1*dst_strd+9],...pu2_dst[1*dst_strd+15]
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x10],#16  //pu2_dst[3*dst_strd],pu2_dst[3*dst_strd+1],...pu2_dst[3*dst_strd+7]
    st1         {v20.8h}, [x10],#16  //pu2_dst[3*dst_strd + 8],pu2_dst[3*dst_strd+9],...pu2_dst[3*dst_strd+15]

    st1         {v20.8h}, [x2], #16   //pu2_dst[16],pu2_dst[17],...pu2_dst[31] //next 16 pixels in 1st row gets copied
    st1         {v20.8h}, [x2], x6   //pu2_dst[16],pu2_dst[17],...pu2_dst[31] //next 16 pixels in 1st row gets copied
    st1         {v20.8h}, [x5], #16
    st1         {v20.8h}, [x5], x6
    st1         {v20.8h}, [x8], #16
    st1         {v20.8h}, [x8], x6
    st1         {v20.8h}, [x10], #16   //pu2_dst[3*dst_strd + 16],pu2_dst[3*dst_strd + 17],...pu2_dst[3*dst_strd + 31]
    st1         {v20.8h}, [x10], x6  //pu2_dst[3*dst_strd + 16],pu2_dst[3*dst_strd + 17],...pu2_dst[3*dst_strd + 31]

    sub         x9, x9, #32                 //32x32 prol/epil counter dec

kernel_copy:
    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x10],#16
    st1         {v20.8h}, [x10],#16

    st1         {v20.8h}, [x2], #16
    st1         {v20.8h}, [x2], x6
    st1         {v20.8h}, [x5], #16
    st1         {v20.8h}, [x5], x6
    st1         {v20.8h}, [x8], #16
    st1         {v20.8h}, [x8], x6
    st1         {v20.8h}, [x10], #16
    st1         {v20.8h}, [x10], x6

    subs        x9, x9, #32

    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x10],#16
    st1         {v20.8h}, [x10],#16

    st1         {v20.8h}, [x2], #16
    st1         {v20.8h}, [x2], x6
    st1         {v20.8h}, [x5], #16
    st1         {v20.8h}, [x5], x6
    st1         {v20.8h}, [x8], #16
    st1         {v20.8h}, [x8], x6
    st1         {v20.8h}, [x10], #16
    st1         {v20.8h}, [x10], x6

    bne         kernel_copy

epilogue_copy:
    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x2],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x5],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x8],#16
    st1         {v20.8h}, [x10],#16
    st1         {v20.8h}, [x10],#16

    st1         {v20.8h}, [x2], #16
    st1         {v20.8h}, [x2]
    st1         {v20.8h}, [x5], #16
    st1         {v20.8h}, [x5]
    st1         {v20.8h}, [x8], #16
    st1         {v20.8h}, [x8]
    st1         {v20.8h}, [x10], #16
    st1         {v20.8h}, [x10]

    b           end_func


  dc_4:
      ld1         {v0.4h},[x6],#8
    ld1         {v1.4h},[x8],#8
    add         v2.4h, v0.4h, v1.4h
    uaddlp      v3.2s,v2.4h

    mov         x5, #0                      //
    mov         v6.s[0], w4
    mov         v6.s[1], w5                 //store nt to accumulate

    uadalp      v6.1d,  v3.2s               //accumulate all inp into d6 =acc_dc
    sshl        d18, d6, d7                 //dc = (acc_val) shr by log2nt+1
    mov         x8, x7                      //&src[2nt+1]

    shl         d25, d18,#1                 //2*dc
    sub         x9, x9, #6                  //&src[2nt-1-row]

    mov         v28.s[0], w14
    mov         v28.s[1], w5               //src[2nt+1]+2+src[2nt-1] moved to d28

    dup         v16.4h, v18.4h[0]           //dc_val
    add         d27,  d25 ,  d28            //src[2nt+1]+2+src[2nt-1]+2dc_val

    ushr        v29.4h, v27.4h,#2           //final dst[0]'s value in v29

    sub         x12, x3, x3, lsl #2         //-3*strd
    add         x12, x12, x12
    add         d23,  d25 ,  d18            //3*dc

    add         d23,  d23 ,  d17            //3*dc + 2
    add         x12, x12, #8                //offset after one 4x4 block (-3*strd + 4)

    dup         v24.8h, v23.4h[0]           //3*dc + 2 (moved to all lanes)

    ld1         {v0.4h},[x8]                //col 1::3 load (prol)
    ld1         {v1.4h},[x9]                //row 3::1 (0 also) load (prol)

    add         v0.4h,  v0.4h ,  v24.4h   //col 1::7 add 3dc+2 (prol)
    add         v1.4h,  v1.4h ,  v24.4h   //row 1::7 add 3dc+2 (prol)

    movi        d19, #0x0ffff    //
    sshr     v2.4h, v0.4h,#2            //columns shx2 movn (prol)

    movi        d20, #0x0ffff    //byte mask row 1 (prol)
    sshr     v3.4h, v1.4h,#2            //rows shx2 movn (prol)

    bsl         v19.8b,  v29.8b ,  v2.8b    //first row with dst[0]

    rev64       v3.4h,  v3.4h

    add      x16, x3, x3
    st1         {v19.1d},[x2], x16         //store row 0 (prol)

    sshr        d3, d3,#16                  //row 0 shift (prol) (first value to be ignored)

    movi        d21, #0x0ffff    //byte mask row 2 (prol)

    bsl         v20.8b,  v3.8b ,  v16.8b    //row 1    (prol)
    sshr        d3, d3,#16                   //row 1 shift (prol)

    st1         {v20.1d},[x2], x16         //store row 1 (prol)

    bsl         v21.8b,  v3.8b ,  v16.8b    //row 2 (prol)

    movi        d20, #0x0ffff    //byte mask row 3 (prol)

    sshr        d3, d3,#16                   //row 2 shift (prol)
    st1         {v21.1d},[x2], x16         //store row 2 (prol)

    bsl         v20.8b,  v3.8b ,  v16.8b    //row 3    (prol)
    st1         {v20.1d},[x2]             //store row 3 (prol)

epilogue_end:
end_func:
    // ldmfd sp!,{x4-x12,x15}                  //reload the registers from sp
    ldp         x19, x20,[sp],#16

    ret
