/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MLA_PROLOG_UB_BUDGET_H
#define MLA_PROLOG_UB_BUDGET_H

#include <algorithm>
#include <cstdint>

namespace optiling {
namespace mla_ub {

// Mirror the active arch92 buffer layout. Keep this estimate in sync when
// enabling new staging/fusion paths or changing the vector scratch layout.
constexpr uint64_t RESERVED_BYTES = 8192U;
constexpr uint64_t BLOCK_BYTES = 32U;

inline uint64_t AlignBytes(uint64_t value, uint64_t alignment = BLOCK_BYTES)
{
    return (value + alignment - 1U) / alignment * alignment;
}

inline uint64_t DivUp(uint64_t value, uint64_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

struct Params {
    uint64_t step = 0;
    uint64_t aiv = 0;
    uint64_t cv = 1;
    uint64_t hcq = 0;
    uint64_t he = 0;
    uint64_t hckv = 0;
    uint64_t d = 0;
    uint64_t dr = 0;
    uint64_t heads = 0;
    uint64_t headsPerCore = 0;
    uint64_t stepHeads = 0;
    uint64_t mm3SingleN = 0;
    uint64_t k1 = 1;
    uint64_t k2 = 1;
    uint64_t kvBytes = 2;
    uint64_t krBytes = 2;
    uint64_t tileSize = 128;
    bool splitM = false;
    bool mx = false;
    bool inputQuant = false;
    bool cqQuant = false;
    bool dequant = false;
    bool nqWave = false;
    bool smooth = false;
    bool krQuant = false;
    bool pertile = false;
    bool int8Pertile = false;
    bool actualSeq = false;
    bool qnQuant = false;
};

struct Budget {
    uint64_t fixed = 0;
    uint64_t scratch = 0;
    uint64_t Total() const
    {
        return fixed + scratch;
    }
};

inline uint64_t RmsScratch(uint64_t cols, uint64_t groups)
{
    return std::max(4U * groups * cols, 6U * cols);
}

inline uint64_t DynamicQnScratch(uint64_t rows, uint64_t cv, uint64_t cols, uint64_t dr)
{
    const uint64_t slice = std::min(rows, 8U * cv);
    const uint64_t ropeRows = std::min(rows, uint64_t{64});
    // BF16 staging, FP32 quantization, both Brcb tails and the live Qr input.
    return 14U * cols * slice + 2U * rows * dr + 2U * ropeRows * dr + 64U + 36U * AlignBytes(slice, 8U) +
           36U * AlignBytes(rows, 8U);
}

inline uint64_t FixedBytes(const Params& p, uint64_t v)
{
    uint64_t fixed = 0U;
    if (!p.splitM && (p.dequant || p.nqWave)) {
        const uint64_t mm3Slot = p.step * (p.d + p.dr) * (p.cqQuant ? 4U : 2U);
        const uint64_t mm4Slot = p.dequant ? DivUp(p.step, 2U) * p.hckv * 2U : 0U;
        // Include the MX wave buffers even when its runtime gate falls back.
        fixed += 2U * AlignBytes(std::max(mm3Slot, mm4Slot));
        if (p.dequant && p.cqQuant && !p.mx) {
            fixed += AlignBytes(p.mm3SingleN * 4U);
        }
    }
    if (!p.splitM && p.inputQuant && !p.mx) {
        fixed += AlignBytes(p.hcq * 4U) + AlignBytes((p.hckv + p.dr) * 4U);
    }
    fixed += AlignBytes(p.hcq * 2U) + AlignBytes(p.hckv * 2U);
    if (!p.splitM && p.cqQuant && !p.mx && p.smooth) {
        fixed += AlignBytes(p.hcq * 4U);
    }
    if (p.krQuant) {
        fixed += AlignBytes(p.dr * 4U);
    }
    if (p.kvBytes == 1U) {
        fixed += AlignBytes(p.inputQuant ? BLOCK_BYTES : p.hckv * 4U);
    }
    fixed += AlignBytes((p.step + 7U) * BLOCK_BYTES);
    const uint64_t sinRows = (p.dequant || p.nqWave) ? DivUp(p.step, p.cv) : v;
    fixed += AlignBytes(8U * p.dr * sinRows);
    if (p.actualSeq && !p.splitM) {
        fixed += AlignBytes(p.step * 8U);
    }
    return fixed;
}

inline uint64_t CqScratch(const Params& p, uint64_t v)
{
    // MM1/RMS. The GA path is included even when only a tail step enables it.
    const uint64_t cqBytes = p.cqQuant ? 1U : 2U;
    const uint64_t cqScaleBytes = AlignBytes(p.mx ? p.hcq / 32U : 1U);
    const uint64_t xScaleBytes = AlignBytes(p.mx ? p.he / 32U : 1U);
    const uint64_t k1 = std::max(p.k1, uint64_t{1});
    const uint64_t cqBase = p.hcq * cqBytes + v * cqScaleBytes + xScaleBytes;
    uint64_t cqScratch = RmsScratch(p.hcq, k1);
    if (p.cqQuant) {
        cqScratch += 4U * p.hcq;
        if (p.mx) {
            cqScratch = std::max(cqScratch, 8U * p.hcq + AlignBytes(p.hcq / 16U));
        }
    }
    uint64_t scratch = cqBase + cqScratch;
    if ((p.splitM && p.mx) || (p.cv == 1U && v >= 2U && k1 > 1U && k1 <= 4U)) {
        const uint64_t mxSlot = 8U * p.hcq + AlignBytes(p.hcq / 16U);
        const uint64_t gaSlot = std::max((k1 + 1U) * p.hcq * 4U, p.mx ? mxSlot : 0U);
        const uint64_t gaBase = 2U * p.hcq * cqBytes + (p.splitM ? v : 2U) * cqScaleBytes +
                                ((!p.mx && p.inputQuant) ? 2U : 1U) * xScaleBytes;
        scratch = std::max(scratch, gaBase + 2U * gaSlot);
    }
    return scratch;
}

inline uint64_t CacheScratch(const Params& p, uint64_t v)
{
    // MM2/cache, with vectorRow == 1. BATCH2 keeps its output live while
    // computing Kr; per-tile quantization uses a separate scale/clip tail.
    const uint64_t k2 = std::max(p.k2, uint64_t{1});
    const uint64_t cacheBase = BLOCK_BYTES + 8U * p.dr * v;
    const uint64_t kvRow = p.hckv * p.kvBytes;
    const uint64_t writeSpan = AlignBytes(4U * kvRow);
    const uint64_t tileCount = p.pertile && p.tileSize != 0U ? p.hckv / p.tileSize : 0U;
    const uint64_t outputSpan = AlignBytes(kvRow + (p.pertile ? 4U * tileCount : 0U));
    uint64_t pertileScratch = 0U;
    if (p.int8Pertile) {
        const uint64_t tilesAligned = AlignBytes(tileCount, 8U);
        pertileScratch =
            4U * (9U * tilesAligned + std::max({AlignBytes(p.hckv, 8U) + 16U * tileCount, 11U * tilesAligned, p.hckv}));
    }
    const uint64_t ckvScratch =
        cacheBase + std::max(writeSpan, outputSpan) + 4U * p.hckv + std::max(RmsScratch(p.hckv, k2), pertileScratch);
    const uint64_t krScratch =
        cacheBase + 3U * kvRow + p.dr * p.krBytes + 4U * p.dr + std::max(4U * k2 * p.dr, 12U * p.dr);
    const uint64_t gatherScratch = cacheBase + 4U * v * p.dr;
    return std::max({ckvScratch, krScratch, gatherScratch});
}

inline uint64_t Mm3Scratch(const Params& p, uint64_t m)
{
    const uint64_t rows = DivUp(m, p.cv);
    const uint64_t half = DivUp(m, 2U) * p.dr;
    if (p.splitM) {
        return std::max(4U * rows * p.d, 14U * rows * p.dr + 4U * p.dr);
    }
    if (p.dequant || p.nqWave) {
        const uint64_t qc = 4U * m * p.d + 4U * (p.d + 8U);
        const uint64_t rope =
            p.nqWave ? 2U * m * p.d + 14U * half : AlignBytes(2U * m * p.d / p.cv) + 12U * half + 4U * p.dr;
        const uint64_t gmRope = 14U * rows * p.dr + 4U * p.dr;
        return std::max({qc, rope, gmRope});
    }
    const uint64_t count = p.stepHeads * p.d;
    return std::max(p.cqQuant ? 14U * count + 192U : 6U * count + 64U, (p.cqQuant ? 18U : 16U) * p.heads * p.dr);
}

inline uint64_t Mm4PeakBytes(const Params& p, uint64_t m, uint64_t postBase)
{
    const uint64_t rows = DivUp(m, p.cv);
    const uint64_t gmQn = DynamicQnScratch(rows, p.cv, p.hckv, p.dr);
    // Current non-fused MM4 UB path, including all head scales.
    const uint64_t ubQn = AlignBytes(DivUp(m, 2U) * p.hckv) + 2U * m * p.dr + 6U * std::min(m, uint64_t{64}) * p.dr +
                          (p.headsPerCore * 4U + 32U) * AlignBytes(m, 8U);
    return std::max(postBase + gmQn, postBase + ubQn);
}

inline uint64_t TailScratch(const Params& p, uint64_t v)
{
    const uint64_t cacheBase = BLOCK_BYTES + 8U * p.dr * v;
    const uint64_t writeSpan = AlignBytes(4U * p.hckv * p.kvBytes);
    const uint64_t postBase = p.splitM ? 0U : AlignBytes(cacheBase + writeSpan);
    uint64_t scratch = 0U;
    // Check all possible tail sizes. The fixed allocation and deferred-cache
    // preamble retain the full-step size, while ceil-half scratch can differ.
    for (uint64_t m = 1U; m <= p.step; ++m) {
        scratch = std::max(scratch, postBase + Mm3Scratch(p, m));
        if (p.qnQuant) {
            scratch = std::max(scratch, Mm4PeakBytes(p, m, postBase));
        }
    }
    return scratch;
}

inline Budget Estimate(const Params& p)
{
    Budget result;
    if (p.step == 0U) {
        return result;
    }
    const uint64_t v = p.splitM ? DivUp(p.step, p.cv) : DivUp(p.step, std::min(p.step, p.aiv));
    result.fixed = FixedBytes(p, v);
    result.scratch = std::max({CqScratch(p, v), CacheScratch(p, v), TailScratch(p, v)});
    result.scratch = AlignBytes(result.scratch);
    return result;
}

} // namespace mla_ub
} // namespace optiling
#endif
