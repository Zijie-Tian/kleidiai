//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 16;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 4;
static const size_t kai_num_bytes_zp_lhs = 4;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_num_bytes_recip_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
static const size_t kai_num_bytes_rsum_rhs = 4;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_k_multiple_of = 32;
static const size_t kai_bl = 32;

inline static size_t kai_get_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    size_t num_bytes_per_block_rhs = (bl / kai_num_bytes_recip_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);

    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_get_k_roundedup(k);
    size_t lhs_packed_stride = kai_mr * ((k_internal * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs);
    // Since the LHS matrix is asymmetric with per-row quantization, we must include the
    // the number of bytes to hold the zero point value
    lhs_packed_stride += kai_mr * kai_num_bytes_zp_lhs;

    return lhs_packed_stride;
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);

    size_t rhs_packed_stride = kai_nr * (num_bytes_per_block * num_blocks_per_row);
    // Since the LHS matrix is quantized asymmetric with per-row quantization, we also include
    // the number of bytes for the reduction sum
    rhs_packed_stride += kai_nr * kai_num_bytes_rsum_rhs;
    // Since the bias is packed with the RHS matrix, the stride is adjusted with the number of bytes of the bias
    rhs_packed_stride += kai_nr * kai_num_bytes_bias;

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    size_t bl,                        //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));
    KAI_ASSUME((bl % kai_bl) == 0);

    if (m == 0) {
        return;
    }

    const size_t num_subblocks = bl / kai_bl;
    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);
    const float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x13, #0x80\n"
        "mov x12, %x[m]\n"
        "mov x20, #0x20\n"
        "sub SP, SP, #0x100\n"
        "mul x13, %x[num_subblocks], x13\n"
        "cmp x12, #0x10\n"
        "madd x13, %x[num_blocks], x13, x20\n"
        "blt 15f\n"
        "1:"  // Row loop
        "mov x11, %x[rhs_packed]\n"
        "mov x10, %x[n]\n"
        "add x9, %x[dst], %x[dst_stride_row], LSL #4\n"
        "2:"  // Column loop
        "mov x27, %x[lhs_packed]\n"
        "movi v6.4s, #0x0\n"
        "mov x24, %x[num_blocks]\n"
        "str q6, [SP, #0x0]\n"
        "str q6, [SP, #0x10]\n"
        "str q6, [SP, #0x20]\n"
        "add x23, x27, x13\n"
        "add x22, x23, x13\n"
        "str q6, [SP, #0x30]\n"
        "add x21, x22, x13\n"
        "str q6, [SP, #0x40]\n"
        "str q6, [SP, #0x50]\n"
        "str q6, [SP, #0x60]\n"
        "str q6, [SP, #0x70]\n"
        "str q6, [SP, #0x80]\n"
        "str q6, [SP, #0x90]\n"
        "str q6, [SP, #0xa0]\n"
        "str q6, [SP, #0xb0]\n"
        "str q6, [SP, #0xc0]\n"
        "str q6, [SP, #0xd0]\n"
        "str q6, [SP, #0xe0]\n"
        "str q6, [SP, #0xf0]\n"
        "3:"  // Block loop
        "movi v2.4s, #0x0\n"
        "movi v17.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v12.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "movi v11.4s, #0x0\n"
        "movi v13.4s, #0x0\n"
        "movi v15.4s, #0x0\n"
        "movi v23.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "movi v16.4s, #0x0\n"
        "movi v21.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "4:"  // Sub block loop
        "ldr q6, [x11, #0x0]\n"
        "ldr q1, [x27, #0x0]\n"
        "movi v25.16b, #0xf0\n"
        "subs x20, x20, #0x1\n"
        "ldr q5, [x23, #0x0]\n"
        "ldr q30, [x22, #0x0]\n"
        "ldr q24, [x21, #0x0]\n"
        "ldr q18, [x11, #0x10]\n"
        "ldr q27, [x27, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "shl v31.16b, v6.16b, #0x4\n"
        "and v6.16b, v6.16b, v25.16b\n"
        "ldr q19, [x22, #0x10]\n"
        "ldr q26, [x21, #0x10]\n"
        "ldr q7, [x11, #0x20]\n"
        "ldr q8, [x27, #0x20]\n"
        "shl v22.16b, v18.16b, #0x4\n"
        "and v18.16b, v18.16b, v25.16b\n"
        "ldr q28, [x23, #0x20]\n"
        ".inst 0x4f81e3e2  // sdot v2.4s, v31.16b, v1.4b[0]\n"
        ".inst 0x4fa1e3f1  // sdot v17.4s, v31.16b, v1.4b[1]\n"
        ".inst 0x4f81ebec  // sdot v12.4s, v31.16b, v1.4b[2]\n"
        ".inst 0x4fa1ebe9  // sdot v9.4s, v31.16b, v1.4b[3]\n"
        "ldr q1, [x22, #0x20]\n"
        ".inst 0x4f85e3ee  // sdot v14.4s, v31.16b, v5.4b[0]\n"
        ".inst 0x4fa5e3eb  // sdot v11.4s, v31.16b, v5.4b[1]\n"
        ".inst 0x4f85ebed  // sdot v13.4s, v31.16b, v5.4b[2]\n"
        ".inst 0x4fa5ebef  // sdot v15.4s, v31.16b, v5.4b[3]\n"
        "ldr q5, [x21, #0x20]\n"
        ".inst 0x4f9ee3f7  // sdot v23.4s, v31.16b, v30.4b[0]\n"
        ".inst 0x4fbee3fd  // sdot v29.4s, v31.16b, v30.4b[1]\n"
        ".inst 0x4f9eebe0  // sdot v0.4s, v31.16b, v30.4b[2]\n"
        ".inst 0x4fbeebe4  // sdot v4.4s, v31.16b, v30.4b[3]\n"
        "ldr q30, [x11, #0x30]\n"
        "add x11, x11, #0x40\n"
        ".inst 0x4f98e3f0  // sdot v16.4s, v31.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3f5  // sdot v21.4s, v31.16b, v24.4b[1]\n"
        ".inst 0x4f98ebea  // sdot v10.4s, v31.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebe3  // sdot v3.4s, v31.16b, v24.4b[3]\n"
        "ldr q24, [x27, #0x30]\n"
        "ldr q31, [x23, #0x30]\n"
        ".inst 0x4f9be2c2  // sdot v2.4s, v22.16b, v27.4b[0]\n"
        ".inst 0x4fbbe2d1  // sdot v17.4s, v22.16b, v27.4b[1]\n"
        ".inst 0x4f9beacc  // sdot v12.4s, v22.16b, v27.4b[2]\n"
        ".inst 0x4fbbeac9  // sdot v9.4s, v22.16b, v27.4b[3]\n"
        "ldr q27, [x22, #0x30]\n"
        ".inst 0x4f94e2ce  // sdot v14.4s, v22.16b, v20.4b[0]\n"
        ".inst 0x4fb4e2cb  // sdot v11.4s, v22.16b, v20.4b[1]\n"
        ".inst 0x4f94eacd  // sdot v13.4s, v22.16b, v20.4b[2]\n"
        ".inst 0x4fb4eacf  // sdot v15.4s, v22.16b, v20.4b[3]\n"
        "ldr q20, [x21, #0x30]\n"
        ".inst 0x4f93e2d7  // sdot v23.4s, v22.16b, v19.4b[0]\n"
        ".inst 0x4fb3e2dd  // sdot v29.4s, v22.16b, v19.4b[1]\n"
        ".inst 0x4f93eac0  // sdot v0.4s, v22.16b, v19.4b[2]\n"
        ".inst 0x4fb3eac4  // sdot v4.4s, v22.16b, v19.4b[3]\n"
        "ldr q19, [x27, #0x40]\n"
        ".inst 0x4f9ae2d0  // sdot v16.4s, v22.16b, v26.4b[0]\n"
        ".inst 0x4fbae2d5  // sdot v21.4s, v22.16b, v26.4b[1]\n"
        ".inst 0x4f9aeaca  // sdot v10.4s, v22.16b, v26.4b[2]\n"
        ".inst 0x4fbaeac3  // sdot v3.4s, v22.16b, v26.4b[3]\n"
        "ldr q22, [x23, #0x40]\n"
        "shl v26.16b, v7.16b, #0x4\n"
        "and v7.16b, v7.16b, v25.16b\n"
        ".inst 0x4f88e342  // sdot v2.4s, v26.16b, v8.4b[0]\n"
        ".inst 0x4fa8e351  // sdot v17.4s, v26.16b, v8.4b[1]\n"
        ".inst 0x4f88eb4c  // sdot v12.4s, v26.16b, v8.4b[2]\n"
        ".inst 0x4fa8eb49  // sdot v9.4s, v26.16b, v8.4b[3]\n"
        "ldr q8, [x22, #0x40]\n"
        ".inst 0x4f9ce34e  // sdot v14.4s, v26.16b, v28.4b[0]\n"
        ".inst 0x4fbce34b  // sdot v11.4s, v26.16b, v28.4b[1]\n"
        ".inst 0x4f9ceb4d  // sdot v13.4s, v26.16b, v28.4b[2]\n"
        ".inst 0x4fbceb4f  // sdot v15.4s, v26.16b, v28.4b[3]\n"
        "ldr q28, [x21, #0x40]\n"
        ".inst 0x4f81e357  // sdot v23.4s, v26.16b, v1.4b[0]\n"
        ".inst 0x4fa1e35d  // sdot v29.4s, v26.16b, v1.4b[1]\n"
        ".inst 0x4f81eb40  // sdot v0.4s, v26.16b, v1.4b[2]\n"
        ".inst 0x4fa1eb44  // sdot v4.4s, v26.16b, v1.4b[3]\n"
        "ldr q1, [x27, #0x50]\n"
        ".inst 0x4f85e350  // sdot v16.4s, v26.16b, v5.4b[0]\n"
        ".inst 0x4fa5e355  // sdot v21.4s, v26.16b, v5.4b[1]\n"
        ".inst 0x4f85eb4a  // sdot v10.4s, v26.16b, v5.4b[2]\n"
        ".inst 0x4fa5eb43  // sdot v3.4s, v26.16b, v5.4b[3]\n"
        "ldr q5, [x23, #0x50]\n"
        "shl v26.16b, v30.16b, #0x4\n"
        "and v30.16b, v30.16b, v25.16b\n"
        "ldr q25, [x22, #0x50]\n"
        ".inst 0x4f98e342  // sdot v2.4s, v26.16b, v24.4b[0]\n"
        ".inst 0x4fb8e351  // sdot v17.4s, v26.16b, v24.4b[1]\n"
        ".inst 0x4f98eb4c  // sdot v12.4s, v26.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb49  // sdot v9.4s, v26.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x50]\n"
        ".inst 0x4f9fe34e  // sdot v14.4s, v26.16b, v31.4b[0]\n"
        ".inst 0x4fbfe34b  // sdot v11.4s, v26.16b, v31.4b[1]\n"
        ".inst 0x4f9feb4d  // sdot v13.4s, v26.16b, v31.4b[2]\n"
        ".inst 0x4fbfeb4f  // sdot v15.4s, v26.16b, v31.4b[3]\n"
        "ldr q31, [x27, #0x60]\n"
        ".inst 0x4f9be357  // sdot v23.4s, v26.16b, v27.4b[0]\n"
        ".inst 0x4fbbe35d  // sdot v29.4s, v26.16b, v27.4b[1]\n"
        ".inst 0x4f9beb40  // sdot v0.4s, v26.16b, v27.4b[2]\n"
        ".inst 0x4fbbeb44  // sdot v4.4s, v26.16b, v27.4b[3]\n"
        "ldr q27, [x23, #0x60]\n"
        ".inst 0x4f94e350  // sdot v16.4s, v26.16b, v20.4b[0]\n"
        ".inst 0x4fb4e355  // sdot v21.4s, v26.16b, v20.4b[1]\n"
        ".inst 0x4f94eb4a  // sdot v10.4s, v26.16b, v20.4b[2]\n"
        ".inst 0x4fb4eb43  // sdot v3.4s, v26.16b, v20.4b[3]\n"
        "ldr q26, [x22, #0x60]\n"
        "ldr q20, [x21, #0x60]\n"
        ".inst 0x4f93e0c2  // sdot v2.4s, v6.16b, v19.4b[0]\n"
        ".inst 0x4fb3e0d1  // sdot v17.4s, v6.16b, v19.4b[1]\n"
        ".inst 0x4f93e8cc  // sdot v12.4s, v6.16b, v19.4b[2]\n"
        ".inst 0x4fb3e8c9  // sdot v9.4s, v6.16b, v19.4b[3]\n"
        "ldr q19, [x27, #0x70]\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4f96e0ce  // sdot v14.4s, v6.16b, v22.4b[0]\n"
        ".inst 0x4fb6e0cb  // sdot v11.4s, v6.16b, v22.4b[1]\n"
        ".inst 0x4f96e8cd  // sdot v13.4s, v6.16b, v22.4b[2]\n"
        ".inst 0x4fb6e8cf  // sdot v15.4s, v6.16b, v22.4b[3]\n"
        "ldr q22, [x23, #0x70]\n"
        "add x23, x23, #0x80\n"
        ".inst 0x4f88e0d7  // sdot v23.4s, v6.16b, v8.4b[0]\n"
        ".inst 0x4fa8e0dd  // sdot v29.4s, v6.16b, v8.4b[1]\n"
        ".inst 0x4f88e8c0  // sdot v0.4s, v6.16b, v8.4b[2]\n"
        ".inst 0x4fa8e8c4  // sdot v4.4s, v6.16b, v8.4b[3]\n"
        "ldr q8, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4f9ce0d0  // sdot v16.4s, v6.16b, v28.4b[0]\n"
        ".inst 0x4fbce0d5  // sdot v21.4s, v6.16b, v28.4b[1]\n"
        ".inst 0x4f9ce8ca  // sdot v10.4s, v6.16b, v28.4b[2]\n"
        ".inst 0x4fbce8c3  // sdot v3.4s, v6.16b, v28.4b[3]\n"
        "ldr q28, [x21, #0x70]\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4f81e242  // sdot v2.4s, v18.16b, v1.4b[0]\n"
        ".inst 0x4fa1e251  // sdot v17.4s, v18.16b, v1.4b[1]\n"
        ".inst 0x4f81ea4c  // sdot v12.4s, v18.16b, v1.4b[2]\n"
        ".inst 0x4fa1ea49  // sdot v9.4s, v18.16b, v1.4b[3]\n"
        ".inst 0x4f85e24e  // sdot v14.4s, v18.16b, v5.4b[0]\n"
        ".inst 0x4fa5e24b  // sdot v11.4s, v18.16b, v5.4b[1]\n"
        ".inst 0x4f85ea4d  // sdot v13.4s, v18.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea4f  // sdot v15.4s, v18.16b, v5.4b[3]\n"
        ".inst 0x4f99e257  // sdot v23.4s, v18.16b, v25.4b[0]\n"
        ".inst 0x4fb9e25d  // sdot v29.4s, v18.16b, v25.4b[1]\n"
        ".inst 0x4f99ea40  // sdot v0.4s, v18.16b, v25.4b[2]\n"
        ".inst 0x4fb9ea44  // sdot v4.4s, v18.16b, v25.4b[3]\n"
        ".inst 0x4f98e250  // sdot v16.4s, v18.16b, v24.4b[0]\n"
        ".inst 0x4fb8e255  // sdot v21.4s, v18.16b, v24.4b[1]\n"
        ".inst 0x4f98ea4a  // sdot v10.4s, v18.16b, v24.4b[2]\n"
        ".inst 0x4fb8ea43  // sdot v3.4s, v18.16b, v24.4b[3]\n"
        ".inst 0x4f9fe0e2  // sdot v2.4s, v7.16b, v31.4b[0]\n"
        ".inst 0x4fbfe0f1  // sdot v17.4s, v7.16b, v31.4b[1]\n"
        ".inst 0x4f9fe8ec  // sdot v12.4s, v7.16b, v31.4b[2]\n"
        ".inst 0x4fbfe8e9  // sdot v9.4s, v7.16b, v31.4b[3]\n"
        ".inst 0x4f9be0ee  // sdot v14.4s, v7.16b, v27.4b[0]\n"
        ".inst 0x4fbbe0eb  // sdot v11.4s, v7.16b, v27.4b[1]\n"
        ".inst 0x4f9be8ed  // sdot v13.4s, v7.16b, v27.4b[2]\n"
        ".inst 0x4fbbe8ef  // sdot v15.4s, v7.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0f7  // sdot v23.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0fd  // sdot v29.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8e4  // sdot v4.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f94e0f0  // sdot v16.4s, v7.16b, v20.4b[0]\n"
        ".inst 0x4fb4e0f5  // sdot v21.4s, v7.16b, v20.4b[1]\n"
        ".inst 0x4f94e8ea  // sdot v10.4s, v7.16b, v20.4b[2]\n"
        ".inst 0x4fb4e8e3  // sdot v3.4s, v7.16b, v20.4b[3]\n"
        ".inst 0x4f93e3c2  // sdot v2.4s, v30.16b, v19.4b[0]\n"
        ".inst 0x4fb3e3d1  // sdot v17.4s, v30.16b, v19.4b[1]\n"
        ".inst 0x4f93ebcc  // sdot v12.4s, v30.16b, v19.4b[2]\n"
        ".inst 0x4fb3ebc9  // sdot v9.4s, v30.16b, v19.4b[3]\n"
        ".inst 0x4f96e3ce  // sdot v14.4s, v30.16b, v22.4b[0]\n"
        ".inst 0x4fb6e3cb  // sdot v11.4s, v30.16b, v22.4b[1]\n"
        ".inst 0x4f96ebcd  // sdot v13.4s, v30.16b, v22.4b[2]\n"
        ".inst 0x4fb6ebcf  // sdot v15.4s, v30.16b, v22.4b[3]\n"
        ".inst 0x4f88e3d7  // sdot v23.4s, v30.16b, v8.4b[0]\n"
        ".inst 0x4fa8e3dd  // sdot v29.4s, v30.16b, v8.4b[1]\n"
        ".inst 0x4f88ebc0  // sdot v0.4s, v30.16b, v8.4b[2]\n"
        ".inst 0x4fa8ebc4  // sdot v4.4s, v30.16b, v8.4b[3]\n"
        ".inst 0x4f9ce3d0  // sdot v16.4s, v30.16b, v28.4b[0]\n"
        ".inst 0x4fbce3d5  // sdot v21.4s, v30.16b, v28.4b[1]\n"
        ".inst 0x4f9cebca  // sdot v10.4s, v30.16b, v28.4b[2]\n"
        ".inst 0x4fbcebc3  // sdot v3.4s, v30.16b, v28.4b[3]\n"
        "bgt 4b\n"
        "ldr d7, [x11, #0x0]\n"
        "ldr q31, [SP, #0x0]\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v12.4s, v12.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "add x11, x11, #0x8\n"
        "shll v7.4s, v7.4h, #0x10\n"
        "fmla v31.4s, v2.4s, v7.4s\n"
        "str q31, [SP, #0x0]\n"
        "ldr q2, [SP, #0x10]\n"
        "fmla v2.4s, v17.4s, v7.4s\n"
        "str q2, [SP, #0x10]\n"
        "ldr q2, [SP, #0x20]\n"
        "fmla v2.4s, v12.4s, v7.4s\n"
        "str q2, [SP, #0x20]\n"
        "ldr q2, [SP, #0x30]\n"
        "fmla v2.4s, v9.4s, v7.4s\n"
        "str q2, [SP, #0x30]\n"
        "ldr q28, [SP, #0x40]\n"
        "scvtf v14.4s, v14.4s, #0x4\n"
        "scvtf v11.4s, v11.4s, #0x4\n"
        "scvtf v13.4s, v13.4s, #0x4\n"
        "scvtf v15.4s, v15.4s, #0x4\n"
        "fmla v28.4s, v14.4s, v7.4s\n"
        "str q28, [SP, #0x40]\n"
        "ldr q1, [SP, #0x50]\n"
        "fmla v1.4s, v11.4s, v7.4s\n"
        "str q1, [SP, #0x50]\n"
        "ldr q11, [SP, #0x60]\n"
        "fmla v11.4s, v13.4s, v7.4s\n"
        "str q11, [SP, #0x60]\n"
        "ldr q14, [SP, #0x70]\n"
        "fmla v14.4s, v15.4s, v7.4s\n"
        "str q14, [SP, #0x70]\n"
        "ldr q19, [SP, #0x80]\n"
        "scvtf v23.4s, v23.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "fmla v19.4s, v23.4s, v7.4s\n"
        "str q19, [SP, #0x80]\n"
        "ldr q15, [SP, #0x90]\n"
        "fmla v15.4s, v29.4s, v7.4s\n"
        "str q15, [SP, #0x90]\n"
        "ldr q25, [SP, #0xa0]\n"
        "fmla v25.4s, v0.4s, v7.4s\n"
        "str q25, [SP, #0xa0]\n"
        "ldr q12, [SP, #0xb0]\n"
        "fmla v12.4s, v4.4s, v7.4s\n"
        "str q12, [SP, #0xb0]\n"
        "ldr q2, [SP, #0xc0]\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v3.4s, v3.4s, #0x4\n"
        "fmla v2.4s, v16.4s, v7.4s\n"
        "str q2, [SP, #0xc0]\n"
        "ldr q16, [SP, #0xd0]\n"
        "fmla v16.4s, v21.4s, v7.4s\n"
        "str q16, [SP, #0xd0]\n"
        "ldr q16, [SP, #0xe0]\n"
        "fmla v16.4s, v10.4s, v7.4s\n"
        "str q16, [SP, #0xe0]\n"
        "ldr q16, [SP, #0xf0]\n"
        "fmla v16.4s, v3.4s, v7.4s\n"
        "str q16, [SP, #0xf0]\n"
        "subs x24, x24, #0x1\n"
        "bgt 3b\n"
        "ld1 { v11.4s }, [x27]\n"
        "ld1 { v10.4s }, [x23]\n"
        "add x27, x27, #0x10\n"
        "add x23, x23, #0x10\n"
        "ld1 { v9.4s }, [x22]\n"
        "ld1 { v8.4s }, [x21]\n"
        "add x22, x22, #0x10\n"
        "add x21, x21, #0x10\n"
        "ldr q31, [SP, #0x0]\n"
        "ldr q30, [SP, #0x10]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x10, #0x4\n"
        "ldr q29, [SP, #0x20]\n"
        "ldr q28, [SP, #0x30]\n"
        "scvtf v11.4s, v11.4s\n"
        "scvtf v10.4s, v10.4s\n"
        "ldr q27, [SP, #0x40]\n"
        "ldr q26, [SP, #0x50]\n"
        "scvtf v9.4s, v9.4s\n"
        "scvtf v8.4s, v8.4s\n"
        "ldr q25, [SP, #0x60]\n"
        "ldr q24, [SP, #0x70]\n"
        "ldr q23, [SP, #0x80]\n"
        "ldr q22, [SP, #0x90]\n"
        "ldr q21, [SP, #0xa0]\n"
        "ldr q20, [SP, #0xb0]\n"
        "ldr q19, [SP, #0xc0]\n"
        "ldr q18, [SP, #0xd0]\n"
        "ldr q17, [SP, #0xe0]\n"
        "ldr q16, [SP, #0xf0]\n"
        "ldr q7, [x11, #0x0]\n"
        "ldr q6, [x27, #0x0]\n"
        "ldr q5, [x23, #0x0]\n"
        "ldr q4, [x22, #0x0]\n"
        "ldr q3, [x21, #0x0]\n"
        "ldr q2, [x11, #0x10]\n"
        "add x11, x11, #0x20\n"
        "ld1r { v1.4s }, [%x[clamp_vals]]\n"
        "ld1r { v0.4s }, [x20]\n"
        "fmla v31.4s, v7.4s, v11.s[0]\n"
        "fmla v30.4s, v7.4s, v11.s[1]\n"
        "fmla v29.4s, v7.4s, v11.s[2]\n"
        "fmla v28.4s, v7.4s, v11.s[3]\n"
        "fmla v27.4s, v7.4s, v10.s[0]\n"
        "fmla v26.4s, v7.4s, v10.s[1]\n"
        "fmla v25.4s, v7.4s, v10.s[2]\n"
        "fmla v24.4s, v7.4s, v10.s[3]\n"
        "fmla v23.4s, v7.4s, v9.s[0]\n"
        "fmla v22.4s, v7.4s, v9.s[1]\n"
        "fmul v31.4s, v31.4s, v6.s[0]\n"
        "fmla v21.4s, v7.4s, v9.s[2]\n"
        "fmla v20.4s, v7.4s, v9.s[3]\n"
        "fmul v30.4s, v30.4s, v6.s[1]\n"
        "fmla v19.4s, v7.4s, v8.s[0]\n"
        "fmla v18.4s, v7.4s, v8.s[1]\n"
        "fmul v29.4s, v29.4s, v6.s[2]\n"
        "fmla v17.4s, v7.4s, v8.s[2]\n"
        "fmla v16.4s, v7.4s, v8.s[3]\n"
        "fmul v28.4s, v28.4s, v6.s[3]\n"
        "fmul v27.4s, v27.4s, v5.s[0]\n"
        "fmul v26.4s, v26.4s, v5.s[1]\n"
        "fmul v25.4s, v25.4s, v5.s[2]\n"
        "fmul v24.4s, v24.4s, v5.s[3]\n"
        "fmul v23.4s, v23.4s, v4.s[0]\n"
        "fmul v22.4s, v22.4s, v4.s[1]\n"
        "fmul v21.4s, v21.4s, v4.s[2]\n"
        "fmul v20.4s, v20.4s, v4.s[3]\n"
        "fmul v19.4s, v19.4s, v3.s[0]\n"
        "fmul v18.4s, v18.4s, v3.s[1]\n"
        "fmul v17.4s, v17.4s, v3.s[2]\n"
        "fmul v16.4s, v16.4s, v3.s[3]\n"
        "fadd v31.4s, v31.4s, v2.4s\n"
        "fadd v30.4s, v30.4s, v2.4s\n"
        "fadd v29.4s, v29.4s, v2.4s\n"
        "fadd v28.4s, v28.4s, v2.4s\n"
        "fadd v27.4s, v27.4s, v2.4s\n"
        "fadd v26.4s, v26.4s, v2.4s\n"
        "fadd v25.4s, v25.4s, v2.4s\n"
        "fadd v24.4s, v24.4s, v2.4s\n"
        "fadd v23.4s, v23.4s, v2.4s\n"
        "fadd v22.4s, v22.4s, v2.4s\n"
        "fadd v21.4s, v21.4s, v2.4s\n"
        "fadd v20.4s, v20.4s, v2.4s\n"
        "fadd v19.4s, v19.4s, v2.4s\n"
        "fadd v18.4s, v18.4s, v2.4s\n"
        "fadd v17.4s, v17.4s, v2.4s\n"
        "fadd v16.4s, v16.4s, v2.4s\n"
        "fmax v31.4s, v31.4s, v1.4s\n"
        "fmax v30.4s, v30.4s, v1.4s\n"
        "fmax v29.4s, v29.4s, v1.4s\n"
        "fmax v28.4s, v28.4s, v1.4s\n"
        "fmax v27.4s, v27.4s, v1.4s\n"
        "fmax v26.4s, v26.4s, v1.4s\n"
        "fmax v25.4s, v25.4s, v1.4s\n"
        "fmax v24.4s, v24.4s, v1.4s\n"
        "fmax v23.4s, v23.4s, v1.4s\n"
        "fmax v22.4s, v22.4s, v1.4s\n"
        "fmax v21.4s, v21.4s, v1.4s\n"
        "fmax v20.4s, v20.4s, v1.4s\n"
        "fmax v19.4s, v19.4s, v1.4s\n"
        "fmax v18.4s, v18.4s, v1.4s\n"
        "fmax v17.4s, v17.4s, v1.4s\n"
        "fmax v16.4s, v16.4s, v1.4s\n"
        "fmin v31.4s, v31.4s, v0.4s\n"
        "fmin v30.4s, v30.4s, v0.4s\n"
        "fmin v29.4s, v29.4s, v0.4s\n"
        "fmin v28.4s, v28.4s, v0.4s\n"
        "fmin v27.4s, v27.4s, v0.4s\n"
        "fmin v26.4s, v26.4s, v0.4s\n"
        "fmin v25.4s, v25.4s, v0.4s\n"
        "fmin v24.4s, v24.4s, v0.4s\n"
        "fmin v23.4s, v23.4s, v0.4s\n"
        "fmin v22.4s, v22.4s, v0.4s\n"
        "fmin v21.4s, v21.4s, v0.4s\n"
        "fmin v20.4s, v20.4s, v0.4s\n"
        "fmin v19.4s, v19.4s, v0.4s\n"
        "fmin v18.4s, v18.4s, v0.4s\n"
        "fmin v17.4s, v17.4s, v0.4s\n"
        "fmin v16.4s, v16.4s, v0.4s\n"
        "blt 9f\n"
        "mov x20, %x[dst]\n"
        "str q31, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q29, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q27, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q26, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q20, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q17, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q16, [x20, #0x0]\n"
        "b 14f\n"
        "9:"  // Partial output
        "mov x28, %x[dst]\n"
        "add x26, x28, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x28, %x[dst_stride_row], LSL #1\n"
        "add x21, x28, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "add x27, x23, %x[dst_stride_row]\n"
        "tbz x10, #1, 10f\n"
        "st1 { v24.d }[0], [x23], #0x8\n"
        "st1 { v25.d }[0], [x25], #0x8\n"
        "st1 { v26.d }[0], [x24], #0x8\n"
        "st1 { v27.d }[0], [x26], #0x8\n"
        "st1 { v28.d }[0], [x20], #0x8\n"
        "st1 { v29.d }[0], [x22], #0x8\n"
        "st1 { v30.d }[0], [x21], #0x8\n"
        "st1 { v31.d }[0], [x28], #0x8\n"
        "tbz x10, #0, 11f\n"
        "st1 { v24.s }[2], [x23]\n"
        "st1 { v25.s }[2], [x25]\n"
        "st1 { v26.s }[2], [x24]\n"
        "st1 { v27.s }[2], [x26]\n"
        "st1 { v28.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x22]\n"
        "st1 { v30.s }[2], [x21]\n"
        "st1 { v31.s }[2], [x28]\n"
        "b 11f\n"
        "10:"  // Output block 0: partial_1_0
        "st1 { v24.s }[0], [x23]\n"
        "st1 { v25.s }[0], [x25]\n"
        "st1 { v26.s }[0], [x24]\n"
        "st1 { v27.s }[0], [x26]\n"
        "st1 { v28.s }[0], [x20]\n"
        "st1 { v29.s }[0], [x22]\n"
        "st1 { v30.s }[0], [x21]\n"
        "st1 { v31.s }[0], [x28]\n"
        "11:"  // Output block 0: Done
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x27, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row], LSL #1\n"
        "add x23, x27, %x[dst_stride_row]\n"
        "add x22, x25, %x[dst_stride_row]\n"
        "add x21, x26, %x[dst_stride_row]\n"
        "add x20, x24, %x[dst_stride_row]\n"
        "tbz x10, #1, 12f\n"
        "st1 { v16.d }[0], [x20], #0x8\n"
        "st1 { v17.d }[0], [x24], #0x8\n"
        "st1 { v18.d }[0], [x21], #0x8\n"
        "st1 { v19.d }[0], [x26], #0x8\n"
        "st1 { v20.d }[0], [x22], #0x8\n"
        "st1 { v21.d }[0], [x25], #0x8\n"
        "st1 { v22.d }[0], [x23], #0x8\n"
        "st1 { v23.d }[0], [x27], #0x8\n"
        "tbz x10, #0, 13f\n"
        "st1 { v16.s }[2], [x20]\n"
        "st1 { v17.s }[2], [x24]\n"
        "st1 { v18.s }[2], [x21]\n"
        "st1 { v19.s }[2], [x26]\n"
        "st1 { v20.s }[2], [x22]\n"
        "st1 { v21.s }[2], [x25]\n"
        "st1 { v22.s }[2], [x23]\n"
        "st1 { v23.s }[2], [x27]\n"
        "b 13f\n"
        "12:"  // Output block 1: partial_1_0
        "st1 { v16.s }[0], [x20]\n"
        "st1 { v17.s }[0], [x24]\n"
        "st1 { v18.s }[0], [x21]\n"
        "st1 { v19.s }[0], [x26]\n"
        "st1 { v20.s }[0], [x22]\n"
        "st1 { v21.s }[0], [x25]\n"
        "st1 { v22.s }[0], [x23]\n"
        "st1 { v23.s }[0], [x27]\n"
        "13:"  // Output block 1: Done
        "14:"  // Output stage exit
        "subs x10, x10, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "mov x20, #0x4\n"
        "sub x12, x12, #0x10\n"
        "cmp x12, #0x10\n"
        "mov %x[dst], x9\n"
        "madd %x[lhs_packed], x20, x13, %x[lhs_packed]\n"
        "bge 1b\n"
        "15:"  // Row loop skip
        "cbz x12, 25f\n"
        "16:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "17:"  // Row tail: Column loop
        "movi v16.4s, #0x0\n"
        "mov x27, %x[lhs_packed]\n"
        "mov x21, %x[num_blocks]\n"
        "str q16, [SP, #0x0]\n"
        "str q16, [SP, #0x10]\n"
        "str q16, [SP, #0x20]\n"
        "str q16, [SP, #0x30]\n"
        "18:"  // Row tail: Block loop
        "movi v2.4s, #0x0\n"
        "movi v17.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "movi v12.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "19:"  // Row tail: Sub block loop
        "ldr q0, [x26, #0x0]\n"
        "ldr q31, [x27, #0x0]\n"
        "movi v30.16b, #0xf0\n"
        "subs x20, x20, #0x1\n"
        "ldr q29, [x26, #0x10]\n"
        "ldr q28, [x27, #0x10]\n"
        "ldr q27, [x26, #0x20]\n"
        "ldr q26, [x27, #0x20]\n"
        "ldr q25, [x26, #0x30]\n"
        "ldr q24, [x27, #0x30]\n"
        "shl v23.16b, v0.16b, #0x4\n"
        "and v0.16b, v0.16b, v30.16b\n"
        "ldr q22, [x27, #0x40]\n"
        "ldr q21, [x27, #0x50]\n"
        "shl v20.16b, v29.16b, #0x4\n"
        "and v29.16b, v29.16b, v30.16b\n"
        "ldr q7, [x27, #0x60]\n"
        "ldr q18, [x27, #0x70]\n"
        "shl v19.16b, v27.16b, #0x4\n"
        "and v27.16b, v27.16b, v30.16b\n"
        ".inst 0x4f9fe2e2  // sdot v2.4s, v23.16b, v31.4b[0]\n"
        ".inst 0x4fbfe2f1  // sdot v17.4s, v23.16b, v31.4b[1]\n"
        "shl v16.16b, v25.16b, #0x4\n"
        "add x26, x26, #0x40\n"
        ".inst 0x4f9feaec  // sdot v12.4s, v23.16b, v31.4b[2]\n"
        ".inst 0x4fbfeae9  // sdot v9.4s, v23.16b, v31.4b[3]\n"
        "and v25.16b, v25.16b, v30.16b\n"
        "add x27, x27, #0x80\n"
        ".inst 0x4f9ce282  // sdot v2.4s, v20.16b, v28.4b[0]\n"
        ".inst 0x4fbce291  // sdot v17.4s, v20.16b, v28.4b[1]\n"
        ".inst 0x4f9cea8c  // sdot v12.4s, v20.16b, v28.4b[2]\n"
        ".inst 0x4fbcea89  // sdot v9.4s, v20.16b, v28.4b[3]\n"
        ".inst 0x4f9ae262  // sdot v2.4s, v19.16b, v26.4b[0]\n"
        ".inst 0x4fbae271  // sdot v17.4s, v19.16b, v26.4b[1]\n"
        ".inst 0x4f9aea6c  // sdot v12.4s, v19.16b, v26.4b[2]\n"
        ".inst 0x4fbaea69  // sdot v9.4s, v19.16b, v26.4b[3]\n"
        ".inst 0x4f98e202  // sdot v2.4s, v16.16b, v24.4b[0]\n"
        ".inst 0x4fb8e211  // sdot v17.4s, v16.16b, v24.4b[1]\n"
        ".inst 0x4f98ea0c  // sdot v12.4s, v16.16b, v24.4b[2]\n"
        ".inst 0x4fb8ea09  // sdot v9.4s, v16.16b, v24.4b[3]\n"
        ".inst 0x4f96e002  // sdot v2.4s, v0.16b, v22.4b[0]\n"
        ".inst 0x4fb6e011  // sdot v17.4s, v0.16b, v22.4b[1]\n"
        ".inst 0x4f96e80c  // sdot v12.4s, v0.16b, v22.4b[2]\n"
        ".inst 0x4fb6e809  // sdot v9.4s, v0.16b, v22.4b[3]\n"
        ".inst 0x4f95e3a2  // sdot v2.4s, v29.16b, v21.4b[0]\n"
        ".inst 0x4fb5e3b1  // sdot v17.4s, v29.16b, v21.4b[1]\n"
        ".inst 0x4f95ebac  // sdot v12.4s, v29.16b, v21.4b[2]\n"
        ".inst 0x4fb5eba9  // sdot v9.4s, v29.16b, v21.4b[3]\n"
        ".inst 0x4f87e362  // sdot v2.4s, v27.16b, v7.4b[0]\n"
        ".inst 0x4fa7e371  // sdot v17.4s, v27.16b, v7.4b[1]\n"
        ".inst 0x4f87eb6c  // sdot v12.4s, v27.16b, v7.4b[2]\n"
        ".inst 0x4fa7eb69  // sdot v9.4s, v27.16b, v7.4b[3]\n"
        ".inst 0x4f92e322  // sdot v2.4s, v25.16b, v18.4b[0]\n"
        ".inst 0x4fb2e331  // sdot v17.4s, v25.16b, v18.4b[1]\n"
        ".inst 0x4f92eb2c  // sdot v12.4s, v25.16b, v18.4b[2]\n"
        ".inst 0x4fb2eb29  // sdot v9.4s, v25.16b, v18.4b[3]\n"
        "bgt 19b\n"
        "ldr d7, [x26, #0x0]\n"
        "ldr q16, [SP, #0x0]\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v12.4s, v12.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "add x26, x26, #0x8\n"
        "shll v7.4s, v7.4h, #0x10\n"
        "fmla v16.4s, v2.4s, v7.4s\n"
        "str q16, [SP, #0x0]\n"
        "ldr q16, [SP, #0x10]\n"
        "fmla v16.4s, v17.4s, v7.4s\n"
        "str q16, [SP, #0x10]\n"
        "ldr q16, [SP, #0x20]\n"
        "fmla v16.4s, v12.4s, v7.4s\n"
        "str q16, [SP, #0x20]\n"
        "ldr q16, [SP, #0x30]\n"
        "fmla v16.4s, v9.4s, v7.4s\n"
        "str q16, [SP, #0x30]\n"
        "subs x21, x21, #0x1\n"
        "bgt 18b\n"
        "ld1 { v21.4s }, [x27]\n"
        "ldr q31, [SP, #0x0]\n"
        "add x27, x27, #0x10\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ldr q30, [SP, #0x10]\n"
        "ldr q29, [SP, #0x20]\n"
        "cmp x25, #0x4\n"
        "ldr q28, [SP, #0x30]\n"
        "ldr q20, [x26, #0x0]\n"
        "ldr q19, [x27, #0x0]\n"
        "ldr q18, [x26, #0x10]\n"
        "scvtf v21.4s, v21.4s\n"
        "add x26, x26, #0x20\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "ld1r { v16.4s }, [x20]\n"
        "fmla v31.4s, v20.4s, v21.s[0]\n"
        "fmla v30.4s, v20.4s, v21.s[1]\n"
        "fmla v29.4s, v20.4s, v21.s[2]\n"
        "fmla v28.4s, v20.4s, v21.s[3]\n"
        "fmul v31.4s, v31.4s, v19.s[0]\n"
        "fmul v30.4s, v30.4s, v19.s[1]\n"
        "fadd v31.4s, v31.4s, v18.4s\n"
        "fmul v29.4s, v29.4s, v19.s[2]\n"
        "fmul v28.4s, v28.4s, v19.s[3]\n"
        "fadd v30.4s, v30.4s, v18.4s\n"
        "fmax v31.4s, v31.4s, v17.4s\n"
        "fadd v29.4s, v29.4s, v18.4s\n"
        "fadd v28.4s, v28.4s, v18.4s\n"
        "fmax v30.4s, v30.4s, v17.4s\n"
        "fmin v31.4s, v31.4s, v16.4s\n"
        "fmax v29.4s, v29.4s, v17.4s\n"
        "fmax v28.4s, v28.4s, v17.4s\n"
        "fmin v30.4s, v30.4s, v16.4s\n"
        "fmin v29.4s, v29.4s, v16.4s\n"
        "fmin v28.4s, v28.4s, v16.4s\n"
        "blt 21f\n"
        "mov x20, %x[dst]\n"
        "cmp x12, #0x1\n"
        "str q31, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 24f\n"
        "cmp x12, #0x2\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 24f\n"
        "cmp x12, #0x3\n"
        "str q29, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 24f\n"
        "str q28, [x20, #0x0]\n"
        "b 24f\n"
        "21:"  // Row tail: Partial output
        "mov x23, %x[dst]\n"
        "cmp x12, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x12, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x12, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #1, 22f\n"
        "st1 { v28.d }[0], [x20], #0x8\n"
        "st1 { v29.d }[0], [x21], #0x8\n"
        "st1 { v30.d }[0], [x22], #0x8\n"
        "st1 { v31.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 23f\n"
        "st1 { v28.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x21]\n"
        "st1 { v30.s }[2], [x22]\n"
        "st1 { v31.s }[2], [x23]\n"
        "b 23f\n"
        "22:"  // Row tail: Output block 0: partial_1_0
        "st1 { v28.s }[0], [x20]\n"
        "st1 { v29.s }[0], [x21]\n"
        "st1 { v30.s }[0], [x22]\n"
        "st1 { v31.s }[0], [x23]\n"
        "23:"  // Row tail: Output block 0: Done
        "24:"  // Row tail: Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 17b\n"
        "subs x12, x12, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x13\n"
        "mov %x[dst], x24\n"
        "bgt 16b\n"
        "25:"  // Row tail: Row loop skip
        "add SP, SP, #0x100\n"
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [num_subblocks] "r"(num_subblocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v2", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7",
          "v8", "v9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9");
}

#endif  // Architectural features check.
