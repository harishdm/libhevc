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
*  ihevcd_func_selector.h
*
* @brief
*  For each function decide whether to use C function,  or Neon intrinsics
* or Cortex A8 intrinsics or Neon  assembly or cortex a8 assembly or intel
* x86  instrinsics
*
* @remarks
*  None
*
*******************************************************************************
*/

#ifndef __ihevcd_func_selector_H__
#define __ihevcd_func_selector_H__

#include "ihevcd_func_types.h"

#define IT_RECON_DC_LUMA                C
#define IT_RECON_DC_CHROMA              C
#define FORMAT_CONV_420SP_TO_420P       C
#define FORMAT_CONV_420SP_TO_420SP      C
#define FORMAT_CONV_420SP_TO_RGBA8888   C
#endif  /* __ihevcd_func_selector_H__ */

