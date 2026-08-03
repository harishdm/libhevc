/******************************************************************************
*
* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
******************************************************************************/
/**
*******************************************************************************
* @file
*  ihevc_debug.c
*
* @brief
*  Functions used for codec debugging
*
* @author
*  Ittiam
*
* @par List of Functions:
*
* @remarks
*  None
*
*******************************************************************************
*/

/*****************************************************************************/
/* File Includes                                                             */
/*****************************************************************************/
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ihevc_typedefs.h"

#include "ihevc_defs.h"
#include "ihevc_debug.h"
#include "ihevc_structs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"

#include "ihevc_common_tables.h"
#include "ihevc_error.h"
#include "ihevc_cabac_tables.h"

/**
*******************************************************************************
*
* @brief
*  Function used for printing various structure sizes
*
* @par Description:
*  Function used for printing various structure sizes
*
* @returns  none
*
* @remarks
*
*******************************************************************************
*/

void ihevc_debug_print_struct_sizes()
{
    printf("sizeof(tu_t) %d\n", sizeof(tu_t));
    printf("sizeof(pu_t) %d\n", sizeof(pu_t));
    printf("sizeof(pu_mv_t) %d\n", sizeof(pu_mv_t));
    printf("sizeof(vps_t) %d\n", sizeof(vps_t));
    printf("sizeof(sps_t) %d\n", sizeof(sps_t));
    printf("sizeof(pps_t) %d\n", sizeof(pps_t));
    printf("sizeof(slice_header_t) %d\n", sizeof(slice_header_t));

    return;
}


/**
*******************************************************************************
*
* @brief
*  Function used for verifying if bitfield
*  accesses are optimized by looking at the disassembly
*
* @par Description:
*  Function used for verifying if bitfield
*  accesses are optimized by looking at the disassembly
*
* @returns  none
*
* @remarks
*
*******************************************************************************
*/


WORD32 ihevc_debug_test_bitfields(void *buf)
{
    tu_t *ps_tu = buf;
    WORD32 intra_mode;

    ps_tu->b4_pos_x = 3;
    ps_tu->b4_pos_y = 4;
    ps_tu->b3_size = 2;
    intra_mode = ps_tu->b6_luma_intra_mode + ps_tu->b3_chroma_intra_mode_idx;

    return intra_mode;
}
