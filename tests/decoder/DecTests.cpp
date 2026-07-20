/******************************************************************************
 *
 * Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
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

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "DecHelper.h"
#include "TestCommon.h"

namespace libhevc {
namespace test {

struct DecodeStreamConfig {
  std::string inputFile;
  Format format;
  std::string refMd5File;
};

void PrintTo(const DecodeStreamConfig& config, ::std::ostream* os) {
  *os << config.inputFile;
}

// Helper to resolve absolute file path for test assets
static std::string getFullPath(const std::string& inputFile) {
  if (const char* envPath = std::getenv("HEVC_TEST_DIR")) {
    return std::string(envPath) + "/" + inputFile;
  }
  return inputFile;
}

class DecTestFixture
    : public ::testing::TestWithParam<std::tuple<DecodeStreamConfig, size_t>> {
 protected:
  DecTestFixture() = default;
  ~DecTestFixture() override = default;
};

TEST_P(DecTestFixture, DecodeVerify) {
  auto [config, cores] = GetParam();

  std::string inputPath = getFullPath(config.inputFile);
  std::string refMd5Path = getFullPath(config.refMd5File);

  auto builder = DecHelper::Builder()
                     .setInputFilePath(inputPath)
                     .setRefMd5Path(refMd5Path)
                     .setFormat(config.format)
                     .setCores(cores);

  std::unique_ptr<DecHelper> helper = builder.build();
  ASSERT_NE(helper, nullptr)
      << "Failed to build DecHelper for input: " << inputPath;
  EXPECT_TRUE(helper->decodeFile())
      << "Decoding failed for input: " << inputPath;
}

static const std::vector<DecodeStreamConfig> kDecodeStreams = {
    {"AMP_A_Samsung_7/AMP_A_Samsung_7.bin", Format::yuv420p, "AMP_A_Samsung_7/AMP_A_Samsung_7_md5.txt"},
    {"AMP_B_Samsung_7/AMP_B_Samsung_7.bin", Format::yuv420p, "AMP_B_Samsung_7/AMP_B_Samsung_7_md5.txt"},
    {"AMP_D_Hisilicon_3/AMP_D_Hisilicon.bit", Format::yuv420p, "AMP_D_Hisilicon_3/AMP_D_Hisilicon_md5.txt"},
    {"AMP_E_Hisilicon_3/AMP_E_Hisilicon.bit", Format::yuv420p, "AMP_E_Hisilicon_3/AMP_E_Hisilicon_md5.txt"},
    {"AMP_F_Hisilicon_3/AMP_F_Hisilicon_3.bit", Format::yuv420p, "AMP_F_Hisilicon_3/AMP_F_Hisilicon_3_md5.txt"},
    {"AMVP_A_MTK_4/AMVP_A_MTK_4/AMVP_A_MTK_4.bit", Format::yuv420p, "AMVP_A_MTK_4/AMVP_A_MTK_4/AMVP_A_MTK_4_md5.txt"},
    {"AMVP_B_MTK_4/AMVP_B_MTK_4/AMVP_B_MTK_4.bit", Format::yuv420p, "AMVP_B_MTK_4/AMVP_B_MTK_4/AMVP_B_MTK_4_md5.txt"},
    {"AMVP_C_Samsung_7/AMVP_C_Samsung_7.bin", Format::yuv420p, "AMVP_C_Samsung_7/AMVP_C_Samsung_7_md5.txt"},
    {"BUMPING_A_ericsson_1/BUMPING_A_ericsson_1.bit", Format::yuv420p, "BUMPING_A_ericsson_1/BUMPING_A_ericsson_1_md5.txt"},
    {"CAINIT_A_SHARP_4/CAINIT_A_SHARP_4.bit", Format::yuv420p, "CAINIT_A_SHARP_4/CAINIT_A_SHARP_4_md5.txt"},
    {"CAINIT_B_SHARP_4/CAINIT_B_SHARP_4.bit", Format::yuv420p, "CAINIT_B_SHARP_4/CAINIT_B_SHARP_4_md5.txt"},
    {"CAINIT_C_SHARP_3/CAINIT_C_SHARP_3.bit", Format::yuv420p, "CAINIT_C_SHARP_3/CAINIT_C_SHARP_3_md5.txt"},
    {"CAINIT_D_SHARP_3/CAINIT_D_SHARP_3.bit", Format::yuv420p, "CAINIT_D_SHARP_3/CAINIT_D_SHARP_3_md5.txt"},
    {"CAINIT_E_SHARP_3/CAINIT_E_SHARP_3.bit", Format::yuv420p, "CAINIT_E_SHARP_3/CAINIT_E_SHARP_3_md5.txt"},
    {"CAINIT_F_SHARP_3/CAINIT_F_SHARP_3.bit", Format::yuv420p, "CAINIT_F_SHARP_3/CAINIT_F_SHARP_3_md5.txt"},
    {"CAINIT_G_SHARP_3/CAINIT_G_SHARP_3.bit", Format::yuv420p, "CAINIT_G_SHARP_3/CAINIT_G_SHARP_3_md5.txt"},
    {"CAINIT_H_SHARP_3/CAINIT_H_SHARP_3.bit", Format::yuv420p, "CAINIT_H_SHARP_3/CAINIT_H_SHARP_3_md5.txt"},
    {"CIP_A_Panasonic_3/CIP_A_Panasonic_3/CIP_A_Panasonic_3.bit", Format::yuv420p, "CIP_A_Panasonic_3/CIP_A_Panasonic_3/CIP_A_Panasonic_3_md5.txt"},
    {"CIP_C_Panasonic_2/CIP_C_Panasonic_2/CIP_C_Panasonic_2.bit", Format::yuv420p, "CIP_C_Panasonic_2/CIP_C_Panasonic_2/CIP_C_Panasonic_2_md5.txt"},
    {"CONFWIN_A_Sony_1/CONFWIN_A_Sony_1/CONFWIN_A_Sony_1.bit", Format::yuv420p, "CONFWIN_A_Sony_1/CONFWIN_A_Sony_1/CONFWIN_A_Sony_1_md5.txt"},
    {"DBLK_A_SONY_3/DBLK_A_SONY_3.bit", Format::yuv420p, "DBLK_A_SONY_3/DBLK_A_SONY_3_md5.txt"},
    {"DBLK_B_SONY_3/DBLK_B_SONY_3.bit", Format::yuv420p, "DBLK_B_SONY_3/DBLK_B_SONY_3_md5.txt"},
    {"DBLK_C_SONY_3/DBLK_C_SONY_3.bit", Format::yuv420p, "DBLK_C_SONY_3/DBLK_C_SONY_3_md5.txt"},
    {"DBLK_D_VIXS_2/DBLK_D_VIXS_2/DBLK_D_VIXS_2.bit", Format::yuv420p, "DBLK_D_VIXS_2/DBLK_D_VIXS_2/DBLK_D_VIXS_2_md5.txt"},
    {"DBLK_E_VIXS_2/DBLK_E_VIXS_2/DBLK_E_VIXS_2.bit", Format::yuv420p, "DBLK_E_VIXS_2/DBLK_E_VIXS_2/DBLK_E_VIXS_2_md5.txt"},
    {"DBLK_F_VIXS_2/DBLK_F_VIXS_2/DBLK_F_VIXS_2.bit", Format::yuv420p, "DBLK_F_VIXS_2/DBLK_F_VIXS_2/DBLK_F_VIXS_2_md5.txt"},
    {"DBLK_G_VIXS_2/DBLK_G_VIXS_2/DBLK_G_VIXS_2.bit", Format::yuv420p, "DBLK_G_VIXS_2/DBLK_G_VIXS_2/DBLK_G_VIXS_2_md5.txt"},
    {"DELTAQP_A_BRCM_4/DELTAQP_A_BRCM_4/DELTAQP_A_BRCM_4.bit", Format::yuv420p, "DELTAQP_A_BRCM_4/DELTAQP_A_BRCM_4/DELTAQP_A_BRCM_4_md5.txt"},
    {"DELTAQP_B_SONY_3/DELTAQP_B_SONY_3.bit", Format::yuv420p, "DELTAQP_B_SONY_3/DELTAQP_B_SONY_3_md5.txt"},
    {"DELTAQP_C_SONY_3/DELTAQP_C_SONY_3.bit", Format::yuv420p, "DELTAQP_C_SONY_3/DELTAQP_C_SONY_3_md5.txt"},
    {"DSLICE_A_HHI_5/DSLICE_A_HHI_5/DSLICE_A_HHI_5.bin", Format::yuv420p, "DSLICE_A_HHI_5/DSLICE_A_HHI_5/DSLICE_A_HHI_5_md5.txt"},
    {"DSLICE_B_HHI_5/DSLICE_B_HHI_5/DSLICE_B_HHI_5.bin", Format::yuv420p, "DSLICE_B_HHI_5/DSLICE_B_HHI_5/DSLICE_B_HHI_5_md5.txt"},
    {"DSLICE_C_HHI_5/DSLICE_C_HHI_5/DSLICE_C_HHI_5.bin", Format::yuv420p, "DSLICE_C_HHI_5/DSLICE_C_HHI_5/DSLICE_C_HHI_5_md5.txt"},
    {"ENTP_A_QUALCOMM_1/ENTP_A_Qualcomm_1.bit", Format::yuv420p, "ENTP_A_QUALCOMM_1/ENTP_A_Qualcomm_1_md5.txt"},
    {"ENTP_B_Qualcomm_1/ENTP_B_Qualcomm_1.bit", Format::yuv420p, "ENTP_B_Qualcomm_1/ENTP_B_Qualcomm_1_md5.txt"},
    {"ENTP_C_Qualcomm_1/ENTP_C_Qualcomm_1.bit", Format::yuv420p, "ENTP_C_Qualcomm_1/ENTP_C_Qualcomm_1_md5.txt"},
    {"EXT_A_ericsson_4/EXT_A_ericsson_4.bit", Format::yuv420p, "EXT_A_ericsson_4/EXT_A_ericsson_4_md5.txt"},
    {"FILLER_A_Sony_1/FILLER_A_Sony_1/FILLER_A_Sony_1.bit", Format::yuv420p, "FILLER_A_Sony_1/FILLER_A_Sony_1/FILLER_A_Sony_1_md5.txt"},
    {"HRD_A_Fujitsu_3/HRD_A_Fujitsu_3.bin", Format::yuv420p, "HRD_A_Fujitsu_3/HRD_A_Fujitsu_3_md5.txt"},
    {"INITQP_A_Sony_1/INITQP_A_Sony_1/INITQP_A_Sony_1.bit", Format::yuv420p, "INITQP_A_Sony_1/INITQP_A_Sony_1/INITQP_A_Sony_1_md5.txt"},
    {"IPRED_A_docomo_2/IPRED_A_docomo_2.bit", Format::yuv420p, "IPRED_A_docomo_2/IPRED_A_docomo_2_md5.txt"},
    {"IPRED_B_Nokia_3/IPRED_B_Nokia_3.bit", Format::yuv420p, "IPRED_B_Nokia_3/IPRED_B_Nokia_3_md5.txt"},
    {"IPRED_C_Mitsubishi_3/IPRED_C_Mitsubishi_3/IPRED_C_Mitsubishi_3.bit", Format::yuv420p, "IPRED_C_Mitsubishi_3/IPRED_C_Mitsubishi_3/IPRED_C_Mitsubishi_3_md5.txt"},
    {"LS_A_Orange_2/LS_A_Orange_2/LS_A_Orange_2.bit", Format::yuv420p, "LS_A_Orange_2/LS_A_Orange_2/LS_A_Orange_2_md5.txt"},
    {"LS_B_Orange_4/LS_B_Orange_4/LS_B_Orange_4.bit", Format::yuv420p, "LS_B_Orange_4/LS_B_Orange_4/LS_B_Orange_4_md5.txt"},
    {"LTRPSPS_A_Qualcomm_1/LTRPSPS_A_Qualcomm_1.bit", Format::yuv420p, "LTRPSPS_A_Qualcomm_1/LTRPSPS_A_Qualcomm_1_md5.txt"},
    {"MAXBINS_A_TI_5/MAXBINS_A_TI_5/MAXBINS_A_TI_5.bit", Format::yuv420p, "MAXBINS_A_TI_5/MAXBINS_A_TI_5/MAXBINS_A_TI_5_md5.txt"},
    {"MAXBINS_B_TI_5/MAXBINS_B_TI_5/MAXBINS_B_TI_5.bit", Format::yuv420p, "MAXBINS_B_TI_5/MAXBINS_B_TI_5/MAXBINS_B_TI_5_md5.txt"},
    {"MAXBINS_C_TI_5/MAXBINS_C_TI_5/MAXBINS_C_TI_5.bit", Format::yuv420p, "MAXBINS_C_TI_5/MAXBINS_C_TI_5/MAXBINS_C_TI_5_md5.txt"},
    {"MERGE_A_TI_3/MERGE_A_TI_3/MERGE_A_TI_3.bit", Format::yuv420p, "MERGE_A_TI_3/MERGE_A_TI_3/MERGE_A_TI_3_md5.txt"},
    {"MERGE_B_TI_3/MERGE_B_TI_3/MERGE_B_TI_3.bit", Format::yuv420p, "MERGE_B_TI_3/MERGE_B_TI_3/MERGE_B_TI_3_md5.txt"},
    {"MERGE_C_TI_3/MERGE_C_TI_3/MERGE_C_TI_3.bit", Format::yuv420p, "MERGE_C_TI_3/MERGE_C_TI_3/MERGE_C_TI_3_md5.txt"},
    {"MERGE_D_TI_3/MERGE_D_TI_3/MERGE_D_TI_3.bit", Format::yuv420p, "MERGE_D_TI_3/MERGE_D_TI_3/MERGE_D_TI_3_md5.txt"},
    {"MERGE_E_TI_3/MERGE_E_TI_3/MERGE_E_TI_3.bit", Format::yuv420p, "MERGE_E_TI_3/MERGE_E_TI_3/MERGE_E_TI_3_md5.txt"},
    {"MERGE_F_MTK_4/MERGE_F_MTK_4/MERGE_F_MTK_4.bit", Format::yuv420p, "MERGE_F_MTK_4/MERGE_F_MTK_4/MERGE_F_MTK_4_md5.txt"},
    {"MERGE_G_HHI_4/MERGE_G_HHI_4/MERGE_G_HHI_4.bit", Format::yuv420p, "MERGE_G_HHI_4/MERGE_G_HHI_4/MERGE_G_HHI_4_md5.txt"},
    {"MVCLIP_A_qualcomm_3/MVCLIP_A_qualcomm_3/MVCLIP_A_qualcomm_3.bit", Format::yuv420p, "MVCLIP_A_qualcomm_3/MVCLIP_A_qualcomm_3/MVCLIP_A_qualcomm_3_md5.txt"},
    {"MVDL1ZERO_A_docomo_4/MVDL1ZERO_A_docomo_4.bit", Format::yuv420p, "MVDL1ZERO_A_docomo_4/MVDL1ZERO_A_docomo_4_md5.txt"},
    {"MVEDGE_A_qualcomm_3/MVEDGE_A_qualcomm_3/MVEDGE_A_qualcomm_3.bit", Format::yuv420p, "MVEDGE_A_qualcomm_3/MVEDGE_A_qualcomm_3/MVEDGE_A_qualcomm_3_md5.txt"},
    {"NUT_A_ericsson_5/NUT_A_ericsson_5.bit", Format::yuv420p, "NUT_A_ericsson_5/NUT_A_ericsson_5_md5.txt"},
    {"NoOutPrior_A_Qualcomm_1/NoOutPrior_A_Qualcomm_1.bit", Format::yuv420p, "NoOutPrior_A_Qualcomm_1/NoOutPrior_A_Qualcomm_1_md5.txt"},
    {"NoOutPrior_B_Qualcomm_1/NoOutPrior_B_Qualcomm_1.bit", Format::yuv420p, "NoOutPrior_B_Qualcomm_1/NoOutPrior_B_Qualcomm_1_md5.txt"},
    {"OPFLAG_A_Qualcomm_1/OPFLAG_A_Qualcomm_1.bit", Format::yuv420p, "OPFLAG_A_Qualcomm_1/OPFLAG_A_Qualcomm_1_md5.txt"},
    {"OPFLAG_B_Qualcomm_1/OPFLAG_B_Qualcomm_1.bit", Format::yuv420p, "OPFLAG_B_Qualcomm_1/OPFLAG_B_Qualcomm_1_md5.txt"},
    {"OPFLAG_C_Qualcomm_1/OPFLAG_C_Qualcomm_1.bit", Format::yuv420p, "OPFLAG_C_Qualcomm_1/OPFLAG_C_Qualcomm_1_md5.txt"},
    {"PICSIZE_A_Bossen_1/PICSIZE_A_Bossen_1/PICSIZE_A_Bossen_1.bin", Format::yuv420p, "PICSIZE_A_Bossen_1/PICSIZE_A_Bossen_1/PICSIZE_A_Bossen_1_md5.txt"},
    {"PICSIZE_B_Bossen_1/PICSIZE_B_Bossen_1/PICSIZE_B_Bossen_1.bin", Format::yuv420p, "PICSIZE_B_Bossen_1/PICSIZE_B_Bossen_1/PICSIZE_B_Bossen_1_md5.txt"},
    {"PICSIZE_C_Bossen_1/PICSIZE_C_Bossen_1/PICSIZE_C_Bossen_1.bin", Format::yuv420p, "PICSIZE_C_Bossen_1/PICSIZE_C_Bossen_1/PICSIZE_C_Bossen_1_md5.txt"},
    {"PICSIZE_D_Bossen_1/PICSIZE_D_Bossen_1/PICSIZE_D_Bossen_1.bin", Format::yuv420p, "PICSIZE_D_Bossen_1/PICSIZE_D_Bossen_1/PICSIZE_D_Bossen_1_md5.txt"},
    {"PMERGE_A_TI_3/PMERGE_A_TI_3/PMERGE_A_TI_3.bit", Format::yuv420p, "PMERGE_A_TI_3/PMERGE_A_TI_3/PMERGE_A_TI_3_md5.txt"},
    {"PMERGE_B_TI_3/PMERGE_B_TI_3/PMERGE_B_TI_3.bit", Format::yuv420p, "PMERGE_B_TI_3/PMERGE_B_TI_3/PMERGE_B_TI_3_md5.txt"},
    {"PMERGE_C_TI_3/PMERGE_C_TI_3/PMERGE_C_TI_3.bit", Format::yuv420p, "PMERGE_C_TI_3/PMERGE_C_TI_3/PMERGE_C_TI_3_md5.txt"},
    {"PMERGE_D_TI_3/PMERGE_D_TI_3/PMERGE_D_TI_3.bit", Format::yuv420p, "PMERGE_D_TI_3/PMERGE_D_TI_3/PMERGE_D_TI_3_md5.txt"},
    {"PMERGE_E_TI_3/PMERGE_E_TI_3/PMERGE_E_TI_3.bit", Format::yuv420p, "PMERGE_E_TI_3/PMERGE_E_TI_3/PMERGE_E_TI_3_md5.txt"},
    {"POC_A_Bossen_3/POC_A_Bossen_3/POC_A_Bossen_3.bin", Format::yuv420p, "POC_A_Bossen_3/POC_A_Bossen_3/POC_A_Bossen_3_md5.txt"},
    {"PPS_A_qualcomm_7/PPS_A_qualcomm_7/PPS_A_qualcomm_7.bit", Format::yuv420p, "PPS_A_qualcomm_7/PPS_A_qualcomm_7/PPS_A_qualcomm_7_md5.txt"},
    {"PS_B_VIDYO_3/PS_B_VIDYO_3/PS_B_VIDYO_3.bit", Format::yuv420p, "PS_B_VIDYO_3/PS_B_VIDYO_3/PS_B_VIDYO_3_md5.txt"},
    {"RAP_A_docomo_6/RAP_A_docomo_6.bit", Format::yuv420p, "RAP_A_docomo_6/RAP_A_docomo_6_md5.txt"},
    {"RAP_B_Bossen_2/RAP_B_Bossen_2.bit", Format::yuv420p, "RAP_B_Bossen_2/RAP_B_Bossen_2_md5.txt"},
    {"RPLM_A_qualcomm_4/RPLM_A_qualcomm_4/RPLM_A_qualcomm_4.bit", Format::yuv420p, "RPLM_A_qualcomm_4/RPLM_A_qualcomm_4/RPLM_A_qualcomm_4_md5.txt"},
    {"RPLM_B_qualcomm_4/RPLM_B_qualcomm_4/RPLM_B_qualcomm_4.bit", Format::yuv420p, "RPLM_B_qualcomm_4/RPLM_B_qualcomm_4/RPLM_B_qualcomm_4_md5.txt"},
    {"RPS_A_docomo_5/RPS_A_docomo_5.bit", Format::yuv420p, "RPS_A_docomo_5/RPS_A_docomo_5_md5.txt"},
    {"RPS_B_qualcomm_5/RPS_B_qualcomm_5/RPS_B_qualcomm_5.bit", Format::yuv420p, "RPS_B_qualcomm_5/RPS_B_qualcomm_5/RPS_B_qualcomm_5_md5.txt"},
    {"RPS_C_ericsson_5/RPS_C_ericsson_5.bit", Format::yuv420p, "RPS_C_ericsson_5/RPS_C_ericsson_5_md5.txt"},
    {"RPS_D_ericsson_6/RPS_D_ericsson_6.bit", Format::yuv420p, "RPS_D_ericsson_6/RPS_D_ericsson_6_md5.txt"},
    {"RPS_E_qualcomm_5/RPS_E_qualcomm_5/RPS_E_qualcomm_5.bit", Format::yuv420p, "RPS_E_qualcomm_5/RPS_E_qualcomm_5/RPS_E_qualcomm_5_md5.txt"},
    {"RPS_F_docomo_2/RPS_F_docomo_2.bit", Format::yuv420p, "RPS_F_docomo_2/RPS_F_docomo_2_md5.txt"},
    {"RQT_A_HHI_4/RQT_A_HHI_4/RQT_A_HHI_4.bit", Format::yuv420p, "RQT_A_HHI_4/RQT_A_HHI_4/RQT_A_HHI_4_md5.txt"},
    {"RQT_B_HHI_4/RQT_B_HHI_4/RQT_B_HHI_4.bit", Format::yuv420p, "RQT_B_HHI_4/RQT_B_HHI_4/RQT_B_HHI_4_md5.txt"},
    {"RQT_C_HHI_4/RQT_C_HHI_4/RQT_C_HHI_4.bit", Format::yuv420p, "RQT_C_HHI_4/RQT_C_HHI_4/RQT_C_HHI_4_md5.txt"},
    {"RQT_D_HHI_4/RQT_D_HHI_4/RQT_D_HHI_4.bit", Format::yuv420p, "RQT_D_HHI_4/RQT_D_HHI_4/RQT_D_HHI_4_md5.txt"},
    {"RQT_E_HHI_4/RQT_E_HHI_4/RQT_E_HHI_4.bit", Format::yuv420p, "RQT_E_HHI_4/RQT_E_HHI_4/RQT_E_HHI_4_md5.txt"},
    {"RQT_F_HHI_4/RQT_F_HHI_4/RQT_F_HHI_4.bit", Format::yuv420p, "RQT_F_HHI_4/RQT_F_HHI_4/RQT_F_HHI_4_md5.txt"},
    {"RQT_G_HHI_4/RQT_G_HHI_4/RQT_G_HHI_4.bit", Format::yuv420p, "RQT_G_HHI_4/RQT_G_HHI_4/RQT_G_HHI_4_md5.txt"},
    {"SAODBLK_A_MainConcept_4/SAODBLK_A_MainConcept_4/SAODBLK_A_MainConcept_4.bin", Format::yuv420p, "SAODBLK_A_MainConcept_4/SAODBLK_A_MainConcept_4/SAODBLK_A_MainConcept_4_md5.txt"},
    {"SAODBLK_B_MainConcept_4/SAODBLK_B_MainConcept_4/SAODBLK_B_MainConcept_4.bin", Format::yuv420p, "SAODBLK_B_MainConcept_4/SAODBLK_B_MainConcept_4/SAODBLK_B_MainConcept_4_md5.txt"},
    {"SAO_A_MediaTek_4/SAO_A_MediaTek_4.bit", Format::yuv420p, "SAO_A_MediaTek_4/SAO_A_MediaTek_4_md5.txt"},
    {"SAO_B_MediaTek_5/SAO_B_MediaTek_5.bit", Format::yuv420p, "SAO_B_MediaTek_5/SAO_B_MediaTek_5_md5.txt"},
    {"SAO_C_Samsung_5/SAO_C_Samsung_5/SAO_C_Samsung_5.bin", Format::yuv420p, "SAO_C_Samsung_5/SAO_C_Samsung_5/SAO_C_Samsung_5_md5.txt"},
    {"SAO_D_Samsung_5/SAO_D_Samsung_5/SAO_D_Samsung_5.bin", Format::yuv420p, "SAO_D_Samsung_5/SAO_D_Samsung_5/SAO_D_Samsung_5_md5.txt"},
    {"SAO_E_Canon_4/SAO_E_Canon_4.bit", Format::yuv420p, "SAO_E_Canon_4/SAO_E_Canon_4_md5.txt"},
    {"SAO_F_Canon_3/SAO_F_Canon_3.bit", Format::yuv420p, "SAO_F_Canon_3/SAO_F_Canon_3_md5.txt"},
    {"SAO_G_Canon_3/SAO_G_Canon_3.bit", Format::yuv420p, "SAO_G_Canon_3/SAO_G_Canon_3_md5.txt"},
    {"SAO_H_Parabola_1/SAO_H_Parabola_1.bit", Format::yuv420p, "SAO_H_Parabola_1/SAO_H_Parabola_1_md5.txt"},
    {"SDH_A_Orange_4/SDH_A_Orange_4/SDH_A_Orange_4.bit", Format::yuv420p, "SDH_A_Orange_4/SDH_A_Orange_4/SDH_A_Orange_4_md5.txt"},
    {"SLICES_A_Rovi_3/SLICES_A_Rovi_3.bin", Format::yuv420p, "SLICES_A_Rovi_3/SLICES_A_Rovi_3_md5.txt"},
    {"SLIST_A_Sony_5/SLIST_A_Sony_5/SLIST_A_Sony_5.bin", Format::yuv420p, "SLIST_A_Sony_5/SLIST_A_Sony_5/SLIST_A_Sony_5_md5.txt"},
    {"SLIST_B_Sony_9/SLIST_B_Sony_9/SLIST_B_Sony_9.bin", Format::yuv420p, "SLIST_B_Sony_9/SLIST_B_Sony_9/SLIST_B_Sony_9_md5.txt"},
    {"SLIST_C_Sony_4/SLIST_C_Sony_4/SLIST_C_Sony_4.bin", Format::yuv420p, "SLIST_C_Sony_4/SLIST_C_Sony_4/SLIST_C_Sony_4_md5.txt"},
    {"SLIST_D_Sony_9/SLIST_D_Sony_9/str.bin", Format::yuv420p, "SLIST_D_Sony_9/SLIST_D_Sony_9/str_md5.txt"},
    {"SLPPLP_A_VIDYO_2/SLPPLP_A_VIDYO_2/SLPPLP_A_VIDYO_2.bit", Format::yuv420p, "SLPPLP_A_VIDYO_2/SLPPLP_A_VIDYO_2/SLPPLP_A_VIDYO_2_md5.txt"},
    {"STRUCT_A_Samsung_7/STRUCT_A_Samsung_7.bin", Format::yuv420p, "STRUCT_A_Samsung_7/STRUCT_A_Samsung_7_md5.txt"},
    {"STRUCT_B_Samsung_7/STRUCT_B_Samsung_7.bin", Format::yuv420p, "STRUCT_B_Samsung_7/STRUCT_B_Samsung_7_md5.txt"},
    {"TILES_A_Cisco_2/TILES_A_Cisco_2/TILES_A_Cisco_2.bin", Format::yuv420p, "TILES_A_Cisco_2/TILES_A_Cisco_2/TILES_A_Cisco_2_md5.txt"},
    {"TILES_B_Cisco_1/TILES_B_Cisco_1/TILES_B_Cisco_1.bin", Format::yuv420p, "TILES_B_Cisco_1/TILES_B_Cisco_1/TILES_B_Cisco_1_md5.txt"},
    {"TMVP_A_MS_3/TMVP_A_MS_3.bit", Format::yuv420p, "TMVP_A_MS_3/TMVP_A_MS_3_md5.txt"},
    {"TSCL_A_VIDYO_5/TSCL_A_VIDYO_5/TSCL_A_VIDYO_5.bit", Format::yuv420p, "TSCL_A_VIDYO_5/TSCL_A_VIDYO_5/TSCL_A_VIDYO_5_md5.txt"},
    {"TSCL_B_VIDYO_4/TSCL_B_VIDYO_4/TSCL_B_VIDYO_4.bit", Format::yuv420p, "TSCL_B_VIDYO_4/TSCL_B_VIDYO_4/TSCL_B_VIDYO_4_md5.txt"},
    {"TSKIP_A_MS_3/TSKIP_A_MS_3.bit", Format::yuv420p, "TSKIP_A_MS_3/TSKIP_A_MS_3_md5.txt"},
    {"TUSIZE_A_Samsung_1/TUSIZE_A_Samsung_1.bin", Format::yuv420p, "TUSIZE_A_Samsung_1/TUSIZE_A_Samsung_1_md5.txt"},
    {"VPSID_A_VIDYO_2/VPSID_A_VIDYO_2/VPSID_A_VIDYO_2.bit", Format::yuv420p, "VPSID_A_VIDYO_2/VPSID_A_VIDYO_2/VPSID_A_VIDYO_2_md5.txt"},
    {"VPSSPSPPS_A_MainConcept_1/VPSSPSPPS_A_MainConcept_1.bin", Format::yuv420p, "VPSSPSPPS_A_MainConcept_1/VPSSPSPPS_A_MainConcept_1_md5.txt"},
    {"WPP_A_ericsson_MAIN_2/WPP_A_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_A_ericsson_MAIN_2/WPP_A_ericsson_MAIN_2_md5.txt"},
    {"WPP_B_ericsson_MAIN_2/WPP_B_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_B_ericsson_MAIN_2/WPP_B_ericsson_MAIN_2_md5.txt"},
    {"WPP_C_ericsson_MAIN_2/WPP_C_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_C_ericsson_MAIN_2/WPP_C_ericsson_MAIN_2_md5.txt"},
    {"WPP_D_ericsson_MAIN_2/WPP_D_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_D_ericsson_MAIN_2/WPP_D_ericsson_MAIN_2_md5.txt"},
    {"WPP_E_ericsson_MAIN_2/WPP_E_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_E_ericsson_MAIN_2/WPP_E_ericsson_MAIN_2_md5.txt"},
    {"WPP_F_ericsson_MAIN_2/WPP_F_ericsson_MAIN_2.bit", Format::yuv420p, "WPP_F_ericsson_MAIN_2/WPP_F_ericsson_MAIN_2_md5.txt"},
    {"WP_A_Toshiba_3/WP_A_Toshiba_3.bit", Format::yuv420p, "WP_A_Toshiba_3/WP_A_Toshiba_3_md5.txt"},
    {"WP_B_Toshiba_3/WP_B_Toshiba_3.bit", Format::yuv420p, "WP_B_Toshiba_3/WP_B_Toshiba_3_md5.txt"},
    {"cip_B_NEC_3/cip_B_NEC_3/cip_B_NEC_3.bit", Format::yuv420p, "cip_B_NEC_3/cip_B_NEC_3/cip_B_NEC_3_md5.txt"},
    {"ipcm_A_NEC_3/ipcm_A_NEC_3/ipcm_A_NEC_3.bit", Format::yuv420p, "ipcm_A_NEC_3/ipcm_A_NEC_3/ipcm_A_NEC_3_md5.txt"},
    {"ipcm_B_NEC_3/ipcm_B_NEC_3/ipcm_B_NEC_3.bit", Format::yuv420p, "ipcm_B_NEC_3/ipcm_B_NEC_3/ipcm_B_NEC_3_md5.txt"},
    {"ipcm_C_NEC_3/ipcm_C_NEC_3/ipcm_C_NEC_3.bit", Format::yuv420p, "ipcm_C_NEC_3/ipcm_C_NEC_3/ipcm_C_NEC_3_md5.txt"},
    {"ipcm_D_NEC_3/ipcm_D_NEC_3/ipcm_D_NEC_3.bit", Format::yuv420p, "ipcm_D_NEC_3/ipcm_D_NEC_3/ipcm_D_NEC_3_md5.txt"},
    {"ipcm_E_NEC_2/ipcm_E_NEC_2/ipcm_E_NEC_2.bit", Format::yuv420p, "ipcm_E_NEC_2/ipcm_E_NEC_2/ipcm_E_NEC_2_md5.txt"},
};

INSTANTIATE_TEST_SUITE_P(DecoderRegression, DecTestFixture,
                         ::testing::Combine(::testing::ValuesIn(kDecodeStreams),
                                            ::testing::Values(1, 2, 3, 4)));

}  // namespace test
}  // namespace libhevc
