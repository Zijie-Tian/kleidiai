//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_data = 4;
static const size_t kai_num_bytes_bias = 4;

static size_t get_block_height(void) {
    const size_t block_height = kai_nr * kai_get_sme_vector_length_u32() / kai_kr;
    return block_height;
}

size_t kai_get_n_step_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(void) {
    return get_block_height();
}

size_t kai_get_rhs_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx, size_t rhs_stride) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * rhs_stride;
}

size_t kai_get_bias_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(size_t k) {
    return kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_data;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * kai_get_rhs_packed_stride_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(size_t n, size_t k) {
    return kai_roundup(n, get_block_height()) * kai_get_rhs_packed_stride_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(k);
}

void kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == get_block_height());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    const size_t block_height = get_block_height();
    const size_t width = k;
    const size_t row_offset = 0;

    const void* in[block_height];
    uint8_t* rhs_packed_ptr = rhs_packed;
    const uint8_t* rhs_ptr = rhs;
    const uint8_t* bias_ptr = bias;

    for (size_t block_y = 0; block_y < n; block_y += block_height) {
        const size_t height = KAI_MIN(n - block_y, block_height);
        void* out = rhs_packed_ptr + block_y * (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_data);

        for (size_t y = 0; y < height; y++) {
            in[y] = rhs_ptr + (block_y + y) * rhs_stride;
        }

        __asm__ __volatile__(
            ".inst 0xd503477f  // SMSTART ZA\n"
            "ptrue p1.b\n"
            "cbz %x[bias], 1f\n"
            "mov x20, %x[height]\n"
            "whilelt p0.s, XZR, %x[height]\n"
            "decw x20\n"
            "ld1w { z16.s }, p0/Z, [%x[bias]]\n"
            "whilelt p0.s, XZR, x20\n"
            "st1w { z16.s }, p1, [%x[out]]\n"
            "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
            "st1w { z16.s }, p1, [%x[out], #1, MUL VL]\n"
            "addvl %x[out], %x[out], #2\n"
            "1:"  // Bias: Done
            "mov x21, %x[width]\n"
            "cntw x17\n"
            "incw x21\n"
            "mov x20, %x[width]\n"
            "sub x21, x21, #0x1\n"
            "sub x16, x17, #0x1\n"
            "udiv x21, x21, x17\n"  // n_passes = ceildiv(width, VL<T>)
            "ands x16, x20, x16\n"
            "sub x20, x21, #0x1\n"
            "sub x15, x17, #0x2\n"
            "mov x14, #0x0\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x17, LSL #3\n"
            "cntw x9, ALL, MUL #2\n"
            "ldr x28, [x11, #0x0]\n"
            "cntw x27, ALL, MUL #3\n"
            "lsr x20, x20, #0x1\n"  // n_loops = (n_passes - 1) / 2
            "ldr x26, [x10, #0x0]\n"
            "and x25, x21, #0x1\n"  // odd_tail = bool(n_passes & 0x1)
            "csel x16, x16, x17, NE\n"
            "ldr x24, [x11, #0x8]\n"
            "ptrue p12.s\n"
            "whilelt p11.s, XZR, %x[height]\n"
            "ldr x21, [x10, #0x8]\n"
            "whilelt p10.s, x17, %x[height]\n"
            "mov x23, %x[row_offset]\n"
            "mov x22, %x[out]\n"
            "whilelt p9.s, x14, %x[width]\n"
            "whilelt p8.s, x14, %x[width]\n"
            "add x11, x11, #0x10\n"
            "add x10, x10, #0x10\n"
            "mov x12, #0x0\n"
            "cbz x15, 3f\n"
            "2:"  // K loop: Charge: Loop
            ".inst 0x25306163  // psel p3.s, p8.s/Z, p11.s[w12]\n"
            ".inst 0x25306142  // psel p2.s, p8.s/Z, p10.s[w12]\n"
            ".inst 0x25706161  // psel p1.s, p8.s/Z, p11.s[w12, #1]\n"
            ".inst 0x25706140  // psel p0.s, p8.s/Z, p10.s[w12, #1]\n"
            ".inst 0xe0970f80  // ld1w { za0h.s[x12] }, p3/Z, [x28, x23, LSL #2]\n"
            "ldr x28, [x11, #0x0]\n"
            ".inst 0xe0970b44  // ld1w { za1h.s[x12] }, p2/Z, [x26, x23, LSL #2]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0970701  // ld1w { za0h.s[x12, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe09702a5  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x23, LSL #2]\n"
            "add x12, x12, #0x2\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "cmp x12, x15\n"
            "blt 2b\n"
            "3:"  // K loop: Charge: End
            ".inst 0x25306163  // psel p3.s, p8.s/Z, p11.s[w12]\n"
            ".inst 0x25306142  // psel p2.s, p8.s/Z, p10.s[w12]\n"
            ".inst 0x25706161  // psel p1.s, p8.s/Z, p11.s[w12, #1]\n"
            ".inst 0x25706140  // psel p0.s, p8.s/Z, p10.s[w12, #1]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x17, LSL #3\n"
            ".inst 0xe0970f80  // ld1w { za0h.s[x12] }, p3/Z, [x28, x23, LSL #2]\n"
            "ldr x28, [x11, #0x0]\n"
            "incw x14\n"
            ".inst 0xe0970b44  // ld1w { za1h.s[x12] }, p2/Z, [x26, x23, LSL #2]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0970701  // ld1w { za0h.s[x12, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe09702a5  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x23, LSL #2]\n"
            "ldr x21, [x10, #0x8]\n"
            "add x10, x10, #0x10\n"
            "incw x23\n"
            "cbz x20, 9f\n"
            "mov x20, x20\n"
            "4:"  // K loop: Main loop
            "whilelt p8.s, x14, %x[width]\n"
            "mov x13, #0x0\n"
            "cbz x15, 6f\n"
            "5:"  // K loop: Main loop: First: Loop
            ".inst 0x25316160  // psel p0.s, p8.s/Z, p11.s[w13]\n"
            ".inst 0x25316142  // psel p2.s, p8.s/Z, p10.s[w13]\n"
            ".inst 0x25716161  // psel p1.s, p8.s/Z, p11.s[w13, #1]\n"
            ".inst 0x25716143  // psel p3.s, p8.s/Z, p10.s[w13, #1]\n"
            ".inst 0xe0972388  // ld1w { za2h.s[x13] }, p0/Z, [x28, x23, LSL #2]\n"
            ".inst 0x25317120  // psel p0.s, p12.s/Z, p9.s[w13]\n"
            "ldr x28, [x11, #0x0]\n"
            ".inst 0xe0972b4c  // ld1w { za3h.s[x13] }, p2/Z, [x26, x23, LSL #2]\n"
            ".inst 0x25317122  // psel p2.s, p12.s/Z, p9.s[w13]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0972709  // ld1w { za2h.s[x13, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            ".inst 0x25717121  // psel p1.s, p12.s/Z, p9.s[w13, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0972ead  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x23, LSL #2]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bfa2c0  // st1w { za0v.s[x13] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25717120  // psel p0.s, p12.s/Z, p9.s[w13, #1]\n"
            ".inst 0xe0b1aac4  // st1w { za1v.s[x13] }, p2/Z, [x22, x17, LSL #2]\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0a9a6c1  // st1w { za0v.s[x13, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bba2c5  // st1w { za1v.s[x13, #1] }, p0/Z, [x22, x27, LSL #2]\n"
            "add x13, x13, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x13, x15\n"
            "blt 5b\n"
            "6:"  // K loop: Main loop: First: Tail
            ".inst 0x25316160  // psel p0.s, p8.s/Z, p11.s[w13]\n"
            ".inst 0x25316142  // psel p2.s, p8.s/Z, p10.s[w13]\n"
            ".inst 0x25716161  // psel p1.s, p8.s/Z, p11.s[w13, #1]\n"
            ".inst 0x25716143  // psel p3.s, p8.s/Z, p10.s[w13, #1]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x17, LSL #3\n"
            ".inst 0xe0972388  // ld1w { za2h.s[x13] }, p0/Z, [x28, x23, LSL #2]\n"
            ".inst 0x25317120  // psel p0.s, p12.s/Z, p9.s[w13]\n"
            "ldr x28, [x11, #0x0]\n"
            "mov x12, #0x0\n"
            ".inst 0xe0972b4c  // ld1w { za3h.s[x13] }, p2/Z, [x26, x23, LSL #2]\n"
            ".inst 0x25317122  // psel p2.s, p12.s/Z, p9.s[w13]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0972709  // ld1w { za2h.s[x13, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            ".inst 0x25717121  // psel p1.s, p12.s/Z, p9.s[w13, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0972ead  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x23, LSL #2]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bfa2c0  // st1w { za0v.s[x13] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25717120  // psel p0.s, p12.s/Z, p9.s[w13, #1]\n"
            ".inst 0xe0b1aac4  // st1w { za1v.s[x13] }, p2/Z, [x22, x17, LSL #2]\n"
            "whilelt p9.s, x14, %x[width]\n"
            "incw x14\n"
            ".inst 0xe0a9a6c1  // st1w { za0v.s[x13, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "incw x23\n"
            ".inst 0xe0bba2c5  // st1w { za1v.s[x13, #1] }, p0/Z, [x22, x27, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "whilelt p8.s, x14, %x[width]\n"
            "cbz x15, 8f\n"
            "7:"  // K loop: Main loop: Second: Loop
            ".inst 0x25306160  // psel p0.s, p8.s/Z, p11.s[w12]\n"
            ".inst 0x25306142  // psel p2.s, p8.s/Z, p10.s[w12]\n"
            ".inst 0x25706161  // psel p1.s, p8.s/Z, p11.s[w12, #1]\n"
            ".inst 0x25706143  // psel p3.s, p8.s/Z, p10.s[w12, #1]\n"
            ".inst 0xe0970380  // ld1w { za0h.s[x12] }, p0/Z, [x28, x23, LSL #2]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            "ldr x28, [x11, #0x0]\n"
            ".inst 0xe0970b44  // ld1w { za1h.s[x12] }, p2/Z, [x26, x23, LSL #2]\n"
            ".inst 0x25307122  // psel p2.s, p12.s/Z, p9.s[w12]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0970701  // ld1w { za0h.s[x12, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            ".inst 0x25707121  // psel p1.s, p12.s/Z, p9.s[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0970ea5  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x23, LSL #2]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bf82c8  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25707120  // psel p0.s, p12.s/Z, p9.s[w12, #1]\n"
            ".inst 0xe0b18acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x17, LSL #2]\n"
            "add x10, x10, #0x10\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            ".inst 0xe0bb82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x27, LSL #2]\n"
            "add x12, x12, #0x2\n"
            "addvl x22, x22, #4\n"
            "cmp x12, x15\n"
            "blt 7b\n"
            "8:"  // K loop: Main loop: Second: Tail
            ".inst 0x25306160  // psel p0.s, p8.s/Z, p11.s[w12]\n"
            ".inst 0x25306142  // psel p2.s, p8.s/Z, p10.s[w12]\n"
            ".inst 0x25706161  // psel p1.s, p8.s/Z, p11.s[w12, #1]\n"
            ".inst 0x25706143  // psel p3.s, p8.s/Z, p10.s[w12, #1]\n"
            "mov x11, %x[in]\n"
            "add x10, %x[in], x17, LSL #3\n"
            ".inst 0xe0970380  // ld1w { za0h.s[x12] }, p0/Z, [x28, x23, LSL #2]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            "ldr x28, [x11, #0x0]\n"
            ".inst 0xe0970b44  // ld1w { za1h.s[x12] }, p2/Z, [x26, x23, LSL #2]\n"
            ".inst 0x25307122  // psel p2.s, p12.s/Z, p9.s[w12]\n"
            "ldr x26, [x10, #0x0]\n"
            ".inst 0xe0970701  // ld1w { za0h.s[x12, #1] }, p1/Z, [x24, x23, LSL #2]\n"
            ".inst 0x25707121  // psel p1.s, p12.s/Z, p9.s[w12, #1]\n"
            "ldr x24, [x11, #0x8]\n"
            "add x11, x11, #0x10\n"
            ".inst 0xe0970ea5  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x23, LSL #2]\n"
            "ldr x21, [x10, #0x8]\n"
            ".inst 0xe0bf82c8  // st1w { za2v.s[x12] }, p0/Z, [x22, XZR, LSL #2]\n"
            ".inst 0x25707120  // psel p0.s, p12.s/Z, p9.s[w12, #1]\n"
            ".inst 0xe0b18acc  // st1w { za3v.s[x12] }, p2/Z, [x22, x17, LSL #2]\n"
            "whilelt p9.s, x14, %x[width]\n"
            "subs x20, x20, #0x1\n"
            ".inst 0xe0a986c9  // st1w { za2v.s[x12, #1] }, p1/Z, [x22, x9, LSL #2]\n"
            "add x10, x10, #0x10\n"
            "incw x14\n"
            ".inst 0xe0bb82cd  // st1w { za3v.s[x12, #1] }, p0/Z, [x22, x27, LSL #2]\n"
            "addvl x22, x22, #4\n"
            "incw x23\n"
            "bgt 4b\n"
            "9:"  // K loop: Tails
            "cbnz x25, 12f\n"
            "mov x11, %x[in]\n"
            "whilelt p8.s, x14, %x[width]\n"
            "mov x12, #0x0\n"
            "10:"  // K loop: Tails: Even: First
            ".inst 0x25307123  // psel p3.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307122  // psel p2.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25306161  // psel p1.s, p8.s/Z, p11.s[w12]\n"
            ".inst 0x25306140  // psel p0.s, p8.s/Z, p10.s[w12]\n"
            ".inst 0xe0bf8ec0  // st1w { za0v.s[x12] }, p3/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0b18ac4  // st1w { za1v.s[x12] }, p2/Z, [x22, x17, LSL #2]\n"
            "addvl x22, x22, #2\n"
            "ldr x21, [x11, #0x0]\n"
            "ldr x20, [x11, x17, LSL #0x3]\n"
            "add x11, x11, #0x8\n"
            ".inst 0xe09706a8  // ld1w { za2h.s[x12] }, p1/Z, [x21, x23, LSL #2]\n"
            ".inst 0xe097028c  // ld1w { za3h.s[x12] }, p0/Z, [x20, x23, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x17\n"
            "blt 10b\n"
            "whilelt p9.s, x14, %x[width]\n"
            "whilelt p8.s, x14, %x[width]\n"
            "mov x12, #0x0\n"
            "11:"  // K loop: Tails: Even: Second
            ".inst 0x25307121  // psel p1.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0xe0bf86c8  // st1w { za2v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0b182cc  // st1w { za3v.s[x12] }, p0/Z, [x22, x17, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x16\n"
            "blt 11b\n"
            "whilelt p8.s, x14, %x[width]\n"
            "b 14f\n"
            "12:"  // K loop: Tails: Odd
            "mov x12, #0x0\n"
            "13:"  // K loop: Tails: Odd: Loop
            ".inst 0x25307121  // psel p1.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0x25307120  // psel p0.s, p12.s/Z, p9.s[w12]\n"
            ".inst 0xe0bf86c0  // st1w { za0v.s[x12] }, p1/Z, [x22, XZR, LSL #2]\n"
            ".inst 0xe0b182c4  // st1w { za1v.s[x12] }, p0/Z, [x22, x17, LSL #2]\n"
            "add x12, x12, #0x1\n"
            "addvl x22, x22, #2\n"
            "cmp x12, x16\n"
            "blt 13b\n"
            "14:"  // K loop: End
            "mov %x[out], x22\n"
            ".inst 0xd503467f  // SMSTOP\n"
            : [out] "+&r"(out)
            : [bias] "r"(bias), [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset), [width] "r"(width)
            : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13",
              "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23",
              "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
              "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25",
              "z26", "z27", "z28", "z29", "z30", "z31");

        bias_ptr += height * kai_num_bytes_bias;
        bias = bias_ptr;
    }
}

#endif  // Architectural features check.
