//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/matmul.hpp"

namespace kai::test {

namespace {
const std::array<UkernelVariant<kai_matmul_clamp_f32_f32_f32p_ukernel>, 2> ukernel_variants = {
    {{
         {kai_get_m_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_n_step_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_nr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_kr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_sr_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_dst_offset_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_get_dst_size_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla,
          kai_run_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla},
         "matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla",
         cpu_has_sme2,
     },
     {{kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
       kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla},
      "matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla",
      cpu_has_sme2}}};

// TODO: Reimplement these helpers in fill.cpp. These methods are currently duplicated here so they can be specialized
// on the Buffer return type.
template <typename T>
Buffer fill_matrix_raw(size_t height, size_t width, std::function<T(size_t, size_t)> gen) {
    const auto size = height * width * size_in_bits<T> / 8;
    KAI_ASSUME(width * size_in_bits<T> % 8 == 0);

    Buffer data(size);
    auto ptr = static_cast<T*>(data.data());

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            write_array<T>(ptr, y * width + x, gen(y, x));
        }
    }

    return data;
}

template <typename T>
Buffer fill_matrix_random_raw(size_t height, size_t width, uint32_t seed) {
    using TDist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<float>, std::uniform_int_distribution<T>>;

    std::mt19937 rnd(seed);
    TDist dist;

    return fill_matrix_raw<T>(height, width, [&](size_t, size_t) { return dist(rnd); });
}

template <typename Value>
Buffer fill_random(size_t length, uint32_t seed) {
    return fill_matrix_random_raw<Value>(1, length, seed);
}
}  // namespace

class MatMulTest_f32_f32_f32p : public ::testing::TestWithParam<MatMulTestParams> {};

TEST_P(MatMulTest_f32_f32_f32p, EndToEnd)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
    const auto& [variant_idx, matmul_shape] = GetParam();
    const auto& ukernel_variant = ukernel_variants.at(variant_idx);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    constexpr uint32_t seed = 0;

    const size_t m = matmul_shape.m;
    const size_t n = matmul_shape.n;
    const size_t k = matmul_shape.k;

    GTEST_ASSERT_EQ(m, 1);

    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const Buffer ref_lhs(fill_random<float>(m * k, seed + 0));
    const Buffer ref_rhs(fill_random<float>(n * k, seed + 1));
    const Buffer ref_bias(fill_random<float>(n, seed + 2));

    // Runs the reference implementation
    const auto ref_dst_no_clamp = matmul(
        ref_lhs.data(), nullptr, nullptr, DataType::FP32, ref_rhs.data(), nullptr, nullptr, DataType::FP32,
        ref_bias.data(), nullptr, nullptr, DataType::FP32, DataType::FP32, m, n, k, false, false);

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_no_clamp.data(), m * n, clamp_ratio);
    const auto ref_dst = clamp<float>(ref_dst_no_clamp.data(), m * n, clamp_min, clamp_max);

    // Run the RHS packing micro-kernel.
    const auto rhs_stride = n * sizeof(float);

    size_t imp_packed_rhs_size = 0;
    std::unique_ptr<Buffer> imp_packed_rhs;

    switch (variant_idx) {
        case 0:  // matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
            imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme(n, k);
            imp_packed_rhs = std::make_unique<Buffer>(imp_packed_rhs_size);
            kai_run_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme(
                1, n, k, nr, kr, sr, rhs_stride, ref_rhs.data(), ref_bias.data(), nullptr, imp_packed_rhs->data(), 0,
                nullptr);
            break;
        case 1:  // matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
            imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k);
            imp_packed_rhs = std::make_unique<Buffer>(imp_packed_rhs_size);
            kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                1, n, k, nr, kr, sr, rhs_stride, ref_rhs.data(), ref_bias.data(), nullptr, imp_packed_rhs->data(), 0,
                nullptr);
            break;
        default:
            KAI_ERROR("Unsupported micro-kernel");
    }

    // Run the MatMul micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(m, n);
    ASSERT_EQ(imp_dst_size, ref_dst.size());

    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        m, n, k, ref_lhs.data(), 1, imp_packed_rhs->data(), static_cast<float*>(imp_dst.data()), 1, 1, clamp_min,
        clamp_max);

    // Compare the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < m; ++y) {
        for (size_t x = 0; x < n; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), y * n + x);
            const auto ref_value = read_array<float>(ref_dst.data(), y * n + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_f32_f32p,
    testing::Combine(
        testing::Range<size_t>(0, ukernel_variants.size()),
        testing::Values(
            MatMulShape{1, 1, 1},     //
            MatMulShape{1, 16, 1},    //
            MatMulShape{1, 32, 64},   //
            MatMulShape{1, 7, 74},    //
            MatMulShape{1, 800, 64},  //
            MatMulShape{1, 512, 130}  //
            )),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{ukernel_variants.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);

        std::stringstream sstream;
        sstream << name << "__M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k;
        return sstream.str();
    });

}  // namespace kai::test
