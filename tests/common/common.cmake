enable_testing()
libhevc_add_gtest_executable(
  ihevc_luma_inter_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_inter_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_luma_intra_pred_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_luma_intra_pred_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_res_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_res_test.cc
)

libhevc_add_gtest_executable(
  ihevc_itrans_recon_test
  SOURCES ${HEVC_ROOT}/tests/common/ihevc_itrans_recon_test.cc
)

include(GoogleTest)
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set(HEVC_GTEST_DISCOVERY_MODE DISCOVERY_MODE PRE_TEST)
else()
  set(HEVC_GTEST_DISCOVERY_MODE)
endif()
include("${HEVC_ROOT}/tests/common/helpers.cmake")



include(FetchContent)
FetchContent_Declare(
  hevc_test_data
  URL "https://storage.googleapis.com/android_media/external/libhevc/tests/HevcTestRes-1.0.zip"
)
FetchContent_MakeAvailable(hevc_test_data)

include("${HEVC_ROOT}/tests/decoder/DecHelper.cmake")
include("${HEVC_ROOT}/tests/encoder/EncHelper.cmake")

