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
//.if pad_left_chroma == c
//void ihevc_pad_left_chroma(uword32 *pu1_src,
//                            word32 src_strd,
//                            word32 ht,
//                            word32 pad_size)
//{
//    x0 => *pu1_src
//    x1 => src_strd
//    x2 => ht
//    x3 => pad_size



.globl ihevc_hbd_pad_left_chroma_av8

.type ihevc_hbd_pad_left_chroma_av8, %function

ihevc_hbd_pad_left_chroma_av8:
    lsl x1,x1,#1
    lsl x3, x3,#1

loop_start_chroma_left:
    // pad size is assumed to be pad_left = 80
    sub         x4,x0,x3

    ldr        w8,[x0]
    add         x0,x0,x1
    ldr        w9,[x0]
    add         x0,x0,x1
    ldr        w10,[x0]
    add         x0,x0,x1
    ldr        w11,[x0]
    add         x0,x0,x1

    dup         v0.4s,w8
    dup         v2.4s,w9
    dup         v4.4s,w10
    dup         v6.4s,w11

    add         x5,x4,x1

    st1         {v0.4s},[x4],#16         //128/8 = 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16         //128/8 = 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4],#16           // 16 bytes store
    st1         {v0.4s},[x4]               // 16 bytes store

    add         x6,x5,x1

    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.4s},[x5]               //128/8 = 16 bytes store

    add         x7,x6,x1

    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.4s},[x6]               //128/8 = 16 bytes store

    subs        x2, x2,#4

    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.4s},[x7]               //128/8 = 16 bytes store

    // total of 4rows*(16*5) = 4 * 80 = 4 * pad_left store

    bne         loop_start_chroma_left

    ret













































































































































