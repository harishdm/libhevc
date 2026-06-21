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
#include <vector>

#include "EncHelper.h"
#include "DecHelper.h"
#include "Md5Wrapper.h"
#include "TestCommon.h"

namespace libhevc {
namespace test {

struct EncodeStreamConfig {
  std::string filename;
  size_t width;
  size_t height;
  Format format;
};

void PrintTo(const EncodeStreamConfig& config, ::std::ostream* os) {
  *os << config.filename;
}

// Class EncodeTestParams is no longer needed since we use std::tuple with Combine

// Helper to resolve absolute file path for test assets
static std::string getFullPath(const std::string& filename) {
  if (const char* envPath = std::getenv("HEVC_TEST_DIR")) {
    return std::string(envPath) + "/" + filename;
  }
#ifdef PROJECT_DIR
  return std::string(PROJECT_DIR) + "/" + filename;
#else
  return filename;
#endif
}

// Generate dynamic synthetic 4:2:0 YUV video sequence (10 frames)
static void generateSyntheticYuv(const std::string& path, size_t width, size_t height, size_t numFrames) {
  std::ofstream outFile(path, std::ios::binary);
  ASSERT_TRUE(outFile.is_open()) << "Failed to create synthetic YUV file: " << path;

  size_t ySize = width * height;
  size_t uvSize = (width / 2) * (height / 2);
  std::vector<uint8_t> frameBuf(ySize + 2 * uvSize);

  for (size_t t = 0; t < numFrames; ++t) {
    uint8_t* yData = frameBuf.data();
    uint8_t* uData = yData + ySize;
    uint8_t* vData = uData + uvSize;

    // Fill Y (luma) moving gradient
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        yData[y * width + x] = static_cast<uint8_t>((x + y + t * 4) % 256);
      }
    }

    // Fill U (chroma cb)
    for (size_t y = 0; y < height / 2; ++y) {
      for (size_t x = 0; x < width / 2; ++x) {
        uData[y * (width / 2) + x] = static_cast<uint8_t>((128 + t * 2) % 256);
      }
    }

    // Fill V (chroma cr)
    for (size_t y = 0; y < height / 2; ++y) {
      for (size_t x = 0; x < width / 2; ++x) {
        vData[y * (width / 2) + x] = static_cast<uint8_t>((64 - t * 2) % 256);
      }
    }

    outFile.write(reinterpret_cast<const char*>(frameBuf.data()), frameBuf.size());
  }
}

class EncTestFixture : public ::testing::TestWithParam<std::tuple<EncodeStreamConfig, size_t>> {
 protected:
  EncTestFixture() = default;
  ~EncTestFixture() override = default;
};

// Helper to compute MD5 checksum of a file
static std::string computeFileMd5(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    return "";
  }
  Md5Wrapper md5;
  md5.init();
  char buffer[4096];
  while (file.good()) {
    file.read(buffer, sizeof(buffer));
    std::streamsize bytesRead = file.gcount();
    if (bytesRead > 0) {
      md5.update(reinterpret_cast<const uint8_t*>(buffer), bytesRead);
    }
  }
  return md5.finalize();
}

TEST_P(EncTestFixture, EncodeVerify) {
  auto [config, cores] = GetParam();

  std::string inputPath = getFullPath(config.filename);
  std::string outBitstreamPath = "temp_output.hevc";
  std::string reconMd5Path = "temp_recon_md5.txt";

  // Check if input YUV file exists. If not, generate synthetic YUV file!
  std::ifstream f(inputPath);
  if (!f.good()) {
    GTEST_FAIL() << "Input file not found: " << inputPath;
  }
  f.close();

  // Clean up any stale outputs
  std::remove(outBitstreamPath.c_str());

  auto encBuilder = EncHelper::Builder()
                     .setInputFilePath(inputPath)
                     .setOutFilePath(outBitstreamPath)
                     .setOutputReconMd5Path(reconMd5Path)
                     .setFormat(config.format)
                     .setResolution(config.width, config.height)
                     .setFrameRate(30000, 1000)
                     .setCores(cores);

  std::unique_ptr<EncHelper> helper = encBuilder.build();
  ASSERT_NE(helper, nullptr) << "Failed to build EncHelper for input: " << inputPath;
  EXPECT_TRUE(helper->encodeFile()) << "Encoding failed for input: " << inputPath;

  auto decBuilder = DecHelper::Builder()
                     .setInputFilePath(outBitstreamPath)
                     .setRefMd5Path(reconMd5Path)
                     .setFormat(libhevc::test::Format::yuv420sp_uv)
                     .setCores(cores);
  std::unique_ptr<DecHelper> decHelper = decBuilder.build();
  ASSERT_NE(decHelper, nullptr) << "Failed to build DecHelper for input: " << outBitstreamPath;
  EXPECT_TRUE(decHelper->decodeFile()) << "Decoding failed for input: " << outBitstreamPath;
  // Clean up temporary outputs
  //std::remove(outBitstreamPath.c_str());
}

static std::string getTestName(const ::testing::TestParamInfo<std::tuple<EncodeStreamConfig, size_t>>& info) {
  auto [config, cores] = info.param;
  std::string name = config.filename;
  for (char& c : name) {
    if (c == '.' || c == '/' || c == '-') {
      c = '_';
    }
  }
  name += "_cores_" + std::to_string(cores);
  return name;
}

static const std::vector<EncodeStreamConfig> kEncodeStreams = {
    {"bbb_176x144_yuv420.yuv", 176, 144, Format::yuv420p}
};

INSTANTIATE_TEST_SUITE_P(
    EncoderRegression, EncTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(kEncodeStreams),
        ::testing::Values(1, 2, 3, 4)
    ),
    getTestName
);

}  // namespace test
}  // namespace libhevc
