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
*  ihevcd_error.h
*
* @brief
*  Definitions related to error handling
*
* @author
*  Harish
*
* @par List of Functions:
*
* @remarks
*  None
*
*******************************************************************************
*/

#ifndef _IHEVCD_ERROR_H_
#define _IHEVCD_ERROR_H_

/**
 * Enumerations for error codes used in the codec.
 * Not all these are expected to be returned to the application.
 * Only select few will be exported
 */
typedef enum
{
    /**
     *  No error
     */
    IHEVCD_SUCCESS = 0,
    /**
     *  Codec calls done without successful init
     */
    IHEVCD_INIT_NOT_DONE                        = 0x100,
    /**
     *  Unsupported level passed as an argument
     */
    IHEVCD_LEVEL_UNSUPPORTED,
    /**
     *  Unsupported number of reference pictures passed as an argument
     */
    IHEVCD_NUM_REF_UNSUPPORTED,
    /**
     *  Unsupported number of reorder pictures passed as an argument
     */
    IHEVCD_NUM_REORDER_UNSUPPORTED,
    /**
     *  Unsupported number of extra display pictures passed as an argument
     */
    IHEVCD_NUM_EXTRA_DISP_UNSUPPORTED,
    /**
     *  Invalid display stride requested.
     */
    IHEVCD_INVALID_DISP_STRD,
    /**
     *  Buffer size to hold version string is not sufficient
     *  Allocate more to hold version string
     */
    IHEVCD_CXA_VERS_BUF_INSUFFICIENT            = 0x200,
    /**
     * Stream chroma format other than YUV420
     */
    IHEVCD_UNSUPPORTED_CHROMA_FMT_IDC           = 0x300,

    /**
     * VPS id more than MAX_VPS_CNT
     */
    IHEVCD_UNSUPPORTED_VPS_ID,
    /**
     * SPS id more than MAX_SPS_CNT
     */

    IHEVCD_UNSUPPORTED_SPS_ID,
    /**
     * PPS id more than MAX_PPS_CNT
     */

    IHEVCD_UNSUPPORTED_PPS_ID,

    /**
     * Invelid Parameter while decoding
     */
    IHEVCD_INVALID_PARAMETER,

    /**
     * Invalid header
     */
    IHEVCD_INVALID_HEADER,

    /**
     * In sufficient memory allocated for MV Bank
     */
    IHEVCD_INSUFFICIENT_MEM_MVBANK,

    /**
     * In sufficient memory allocated for MV Bank
     */
    IHEVCD_INSUFFICIENT_MEM_PICBUF,

    /**
     * Buffer manager error
     */
    IHEVCD_BUF_MGR_ERROR,

    /**
     * No free MV Bank buffer available to store current pic
     */
    IHEVCD_NO_FREE_MVBANK,

    /**
     * No free picture buffer available to store current pic
     */
    IHEVCD_NO_FREE_PICBUF,
    /**
     * Reached slice header in header mode
     */
    IHEVCD_SLICE_IN_HEADER_MODE,

    /**
     * Ignore current slice and continue
     */
    IHEVCD_IGNORE_SLICE,

    /**
     * Reference Picture not found
     */
    IHEVCD_REF_PIC_NOT_FOUND,

    /**
     * Reached end of sequence
     */
    IHEVCD_END_OF_SEQUENCE,

    /**
     * Width/height greater than max width and max height
     */
    IHEVCD_UNSUPPORTED_DIMENSIONS,

    /**
     * Bit depth is greater than 8
     */
    IHEVCD_UNSUPPORTED_BIT_DEPTH,

    /**
     * Limit on the number of frames decoded
     */
    IHEVCD_NUM_FRAMES_LIMIT_REACHED,

    /**
     * VUI parameters not found
     */
    IHEVCD_VUI_PARAMS_NOT_FOUND,

    /**
     * Profile higher than init profile
     */
     IHEVCD_GEN_PROFILE_HIGHER_THAN_INIT_PROFILE,

     /**
      * Unsupported profile
      */
      IHEVCD_GEN_PROFILE_UNSUPPORTED,

      /**
       * Invalid output chroma format
       */
       IHEVCD_INVALID_OUT_CHROMA_FMT,

    /**
     * Unsupported parameter while decoding
     */
    IHEVCD_UNSUPPORTED_PARAMETER,

    /**
     * Generic failure
     */
    IHEVCD_FAIL                             = 0x7FFFFFFF
}IHEVCD_ERROR_T;
#endif /* _IHEVCD_ERROR_H_ */
