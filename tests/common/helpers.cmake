#******************************************************************************
#
# Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

# Define the common test helper library
add_library(libhevc_test_common STATIC
  ${HEVC_ROOT}/tests/common/BitsBuf.cpp
  ${HEVC_ROOT}/tests/common/RawBuf.cpp
  ${HEVC_ROOT}/tests/common/Md5Wrapper.cpp
  ${HEVC_ROOT}/tests/common/BitsFile.cpp
  ${HEVC_ROOT}/tests/common/RawFile.cpp
)

# Export include directories so other libraries/tests can find headers
target_include_directories(libhevc_test_common PUBLIC
  ${HEVC_ROOT}/tests/common
)

# Detect OpenSSL and disable MD5 validation if not found
find_package(OpenSSL)
if(OPENSSL_FOUND)
  target_link_libraries(libhevc_test_common PRIVATE OpenSSL::Crypto)
else()
  target_compile_definitions(libhevc_test_common PUBLIC DISABLE_MD5)
  message(STATUS "OpenSSL not found. MD5 validation will be disabled.")
endif()
