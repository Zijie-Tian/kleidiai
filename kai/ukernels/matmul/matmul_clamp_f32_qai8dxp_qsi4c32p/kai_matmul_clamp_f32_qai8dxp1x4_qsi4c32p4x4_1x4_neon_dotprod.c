//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 1;
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
    // Since the LHS matrix is asymmetric with per-row quantization, we must include
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod(
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
    KAI_ASSUME((k % bl) == 0);

    if (m == 0) {
        return;
    }

    const size_t num_subblocks = bl / kai_bl;
    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);
    const float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x27, #0x20\n"
        "mov x20, #0x8\n"
        "movi v28.16b, #0xf0\n"
        "mov x26, %x[m]\n"
        "mul x27, %x[num_subblocks], x27\n"
        "madd x27, %x[num_blocks], x27, x20\n"
        "1:"  // Row loop
        "mov x25, %x[rhs_packed]\n"
        "mov x24, %x[n]\n"
        "add x23, %x[dst], %x[dst_stride_row]\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v27.16b, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "3:"  // Block loop
        "movi v26.4s, #0x0\n"
        "mov x20, %x[num_subblocks]\n"
        "4:"  // Sub block loop
        "ldr q25, [x25, #0x0]\n"
        "ldr q24, [x22, #0x0]\n"
        "subs x20, x20, #0x1\n"
        "ldr q23, [x25, #0x10]\n"
        "ldr q22, [x25, #0x20]\n"
        "ldr q21, [x25, #0x30]\n"
        "ldr q20, [x22, #0x10]\n"
        "add x25, x25, #0x40\n"
        "add x22, x22, #0x20\n"
        "shl v19.16b, v25.16b, #0x4\n"
        "and v25.16b, v25.16b, v28.16b\n"
        "shl v18.16b, v23.16b, #0x4\n"
        "shl v17.16b, v22.16b, #0x4\n"
        "shl v16.16b, v21.16b, #0x4\n"
        "and v23.16b, v23.16b, v28.16b\n"
        ".inst 0x4f98e27a  // sdot v26.4s, v19.16b, v24.4b[0]\n"
        "and v22.16b, v22.16b, v28.16b\n"
        "and v21.16b, v21.16b, v28.16b\n"
        ".inst 0x4fb8e25a  // sdot v26.4s, v18.16b, v24.4b[1]\n"
        ".inst 0x4f98ea3a  // sdot v26.4s, v17.16b, v24.4b[2]\n"
        ".inst 0x4fb8ea1a  // sdot v26.4s, v16.16b, v24.4b[3]\n"
        ".inst 0x4f94e33a  // sdot v26.4s, v25.16b, v20.4b[0]\n"
        ".inst 0x4fb4e2fa  // sdot v26.4s, v23.16b, v20.4b[1]\n"
        ".inst 0x4f94eada  // sdot v26.4s, v22.16b, v20.4b[2]\n"
        ".inst 0x4fb4eaba  // sdot v26.4s, v21.16b, v20.4b[3]\n"
        "bgt 4b\n"
        "ldr d16, [x25, #0x0]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "sub x21, x21, #0x1\n"
        "add x25, x25, #0x8\n"
        "shll v16.4s, v16.4h, #0x10\n"
        "fmla v27.4s, v26.4s, v16.4s\n"
        "cbnz x21, 3b\n"
        "ld1r { v21.4s }, [x22]\n"
        "ldr q20, [x25, #0x0]\n"
        "add x22, x22, #0x4\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ld1r { v19.4s }, [x22]\n"
        "ldr q18, [x25, #0x10]\n"
        "cmp x24, #0x4\n"
        "add x25, x25, #0x20\n"
        "ld1r { v17.4s }, [%x[clamp_vals]]\n"
        "ld1r { v16.4s }, [x20]\n"
        "scvtf v21.4s, v21.4s\n"
        "fmla v27.4s, v20.4s, v21.s[0]\n"
        "fmul v27.4s, v27.4s, v19.4s\n"
        "fadd v27.4s, v27.4s, v18.4s\n"
        "fmax v27.4s, v27.4s, v17.4s\n"
        "fmin v27.4s, v27.4s, v16.4s\n"
        "blt 5f\n"
        "str q27, [%x[dst], #0x0]\n"
        "b 8f\n"
        "5:"  // Partial output
        "mov x20, %x[dst]\n"
        "tbz x24, #1, 6f\n"
        "st1 { v27.d }[0], [x20], #0x8\n"
        "tbz x24, #0, 7f\n"
        "st1 { v27.s }[2], [x20]\n"
        "b 7f\n"
        "6:"  // Output block 0: partial_1_0
        "st1 { v27.s }[0], [x20]\n"
        "7:"  // Output block 0: Done
        "8:"  // Stores done
        "subs x24, x24, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "subs x26, x26, #0x1\n"
        "add %x[lhs_packed], %x[lhs_packed], x27\n"
        "mov %x[dst], x23\n"
        "bgt 1b\n"
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [num_subblocks] "r"(num_subblocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27");
}

#endif  // Architectural features check.
