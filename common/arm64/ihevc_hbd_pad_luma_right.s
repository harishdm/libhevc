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
.globl ihevc_hbd_pad_right_luma_av8

.type ihevc_hbd_pad_right_luma_av8, %function

ihevc_hbd_pad_right_luma_av8:
lsl x1, x1,#1  //src_strd << 1 (src_strd value coming from main.c is in terms of no of pixels and in HBD 1pixel = 2B)
lsl x3, x3,#1  //pad_size << 1 (pad_size value coming from main.c is in terms of no of pixels and in HBD 1pixel = 2B)

loop_start_luma_right:
    // pad size is assumed to be pad_right = 80
    mov         x4,x0

    ldrh        w8,[x0, #-2]
    add         x0,x0,x1
    ldrh        w9,[x0, #-2]
    add         x0,x0,x1
    ldrh        w10,[x0, #-2]
    add         x0,x0,x1
    ldrh        w11,[x0, #-2]
    add         x0,x0,x1

    add         x5,x4,x1
    add         x6,x5,x1
    add         x7,x6,x1

    dup         v0.8h,w8
    dup         v2.8h,w9
    dup         v4.8h,w10
    dup         v6.8h,w11

    st1         {v0.8h},[x4],#16           //128/8 = 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4],#16           // 16 bytes store
    st1         {v0.8h},[x4]               // 16 bytes store


    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5],#16           //128/8 = 16 bytes store
    st1         {v2.8h},[x5]               //128/8 = 16 bytes store

    subs        x2, x2,#4

    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6],#16           //128/8 = 16 bytes store
    st1         {v4.8h},[x6]               //128/8 = 16 bytes store

    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7],#16           //128/8 = 16 bytes store
    st1         {v6.8h},[x7]               //128/8 = 16 bytes store


    bne         loop_start_luma_right

    ret
