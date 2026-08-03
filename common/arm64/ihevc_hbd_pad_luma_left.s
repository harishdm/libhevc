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
//*******************************************************************************
//*/
//.if pad_left_luma == c
//void ihevc_hbd_pad_left_luma(UWORD16 *pu2_src,
//                        word32 src_strd,
//                        word32 ht,
//                        word32 pad_size)
//**************variables vs registers*************************
//    x0 => *pu1_src
//    x1 => src_strd
//    x2 => ht
//    x3 => pad_size

.text
.align 4

.globl ihevc_hbd_pad_left_luma_av8

.type ihevc_hbd_pad_left_luma_av8, %function

ihevc_hbd_pad_left_luma_av8:
    lsl x1, x1,#1  //src_strd << 1 (src_strd value coming from main.c is in terms of no of pixels and in HBD 1pixel = 2B)
    lsl x3, x3,#1  //pad_size << 1 (pad_size value coming from main.c is in terms of no of pixels and in HBD 1pixel = 2B)
loop_start_hbd_luma_left:
    // pad size is assumed to be pad_left = 80(80pixels * 2 = 160 Bytes)

    sub         x4,x0,x3

    ldrh        w8,[x0]
    add         x0,x0,x1
    ldrh        w9,[x0]
    add         x0,x0,x1
    ldrh        w10,[x0]
    add         x0,x0,x1
    ldrh        w11,[x0]
    add         x0,x0,x1

    dup         v0.8h,w8
    dup         v2.8h,w9
    dup         v4.8h,w10
    dup         v6.8h,w11

    add         x5,x4,x1

    st1         {v0.8h},[x4],#16        // 8*8*2/8=16 byes store
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4],#16
    st1         {v0.8h},[x4]

    add         x6,x5,x1

    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5],#16
    st1         {v2.8h},[x5]

    add         x7,x6,x1

    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6],#16
    st1         {v4.8h},[x6]

    subs        x2, x2,#4

    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7],#16
    st1         {v6.8h},[x7]

    // total of 4rows*(16*10) = 4 * 160 = 4 * pad_left store

    bne         loop_start_hbd_luma_left

    ret

