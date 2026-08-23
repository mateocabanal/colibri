#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../qwen_token_kernel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------- */
/* Tiny-model geometry                                                        */
/* -------------------------------------------------------------------------- */

static constexpr uint32_t H          = 64;
static constexpr uint32_t NLAYERS    = 2;
static constexpr uint32_t MAX_T      = 16;

static constexpr uint32_t NH         = 2;
static constexpr uint32_t NKV        = 2;
static constexpr uint32_t HD         = 16;
static constexpr uint32_t ROT        = 8;

static constexpr uint32_t LKH        = 2;
static constexpr uint32_t LKD        = 16;
static constexpr uint32_t LVH        = 2;
static constexpr uint32_t LVD        = 16;
static constexpr uint32_t CONV_K     = 3;

static constexpr uint32_t KDIM       = LKH * LKD;       /* 32 */
static constexpr uint32_t VDIM       = LVH * LVD;       /* 32 */
static constexpr uint32_t QKV_DIM    = 2 * KDIM + VDIM;/* 96 */

static constexpr uint32_t QDIM       = NH * HD;         /* 32 */
static constexpr uint32_t KVDIM      = NKV * HD;        /* 32 */

static constexpr uint32_t NEXPERTS   = 8;
static constexpr uint32_t TOPK       = 2;
static constexpr uint32_t MOE_INTER  = 32;
static constexpr uint32_t SH_INTER   = 16;

static constexpr uint32_t NTOK       = 10; /* exercises positions 8 and 9 */
static constexpr uint32_t PROBE_VOCAB = 23;

static constexpr float EPS           = 1.0e-6f;
static constexpr float THETA         = 10000.0f;

static constexpr uint32_t STATUS_OK            = 0;
static constexpr uint32_t STATUS_NEED_EXPERTS  = 1;
static constexpr uint32_t RESUME_NONE          = 0;
static constexpr uint32_t RESUME_MOE           = 1;

static constexpr uint32_t EMPTY_EXPERT = 0xffffffffu;

/* -------------------------------------------------------------------------- */
/* Deterministic RNG                                                          */
/* -------------------------------------------------------------------------- */

struct Rng {
    uint64_t s;

    explicit Rng(uint64_t seed) : s(seed) {}

    uint32_t u32()
    {
        uint64_t x = s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s = x;
        return (uint32_t)((x * 2685821657736338717ULL) >> 32);
    }

    float signed_unit()
    {
        return ((float)(u32() & 0x00ffffffu) /
                (float)0x00800000u) - 1.0f;
    }
};

/* -------------------------------------------------------------------------- */
/* BF16                                                                       */
/* -------------------------------------------------------------------------- */

static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));

    /* RNE. */
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;

    return (uint16_t)(u >> 16);
}

static float bf16_to_f32(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* -------------------------------------------------------------------------- */
/* Blob builder                                                               */
/* -------------------------------------------------------------------------- */

static size_t align_up(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}

struct BlobBuilder {
    std::vector<uint8_t> b;

    uint64_t alloc(size_t n, size_t align = 16)
    {
        const size_t off = align_up(b.size(), align);
        if (off > b.size())
            b.resize(off, 0);

        b.resize(off + n, 0);
        return (uint64_t)off;
    }

    uint64_t put_f32(const std::vector<float> &v)
    {
        const uint64_t off = alloc(v.size() * sizeof(float), 16);
        memcpy(b.data() + off, v.data(), v.size() * sizeof(float));
        return off;
    }

    uint64_t put_bf16(const std::vector<float> &v)
    {
        const uint64_t off = alloc(v.size() * sizeof(uint16_t), 16);
        auto *dst = reinterpret_cast<uint16_t *>(b.data() + off);

        for (size_t i = 0; i < v.size(); ++i)
            dst[i] = f32_to_bf16(v[i]);

        return off;
    }
};

static std::vector<float>
random_matrix(Rng &rng, uint32_t rows, uint32_t cols, float scale)
{
    std::vector<float> v((size_t)rows * cols);

    for (float &x : v)
        x = rng.signed_unit() * scale;

    return v;
}

static std::vector<float>
norm_weights(Rng &rng, uint32_t n)
{
    std::vector<float> v(n);

    for (float &x : v)
        x = 1.0f + rng.signed_unit() * 0.04f;

    return v;
}

/* -------------------------------------------------------------------------- */
/* Apple8 MXFP4 test packer                                                   */
/* -------------------------------------------------------------------------- */

static constexpr float MX4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,
     2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f,
    -2.0f, -3.0f, -4.0f, -6.0f
};

static size_t apple8_matrix_bytes(uint32_t rows, uint32_t cols)
{
    const uint32_t rt = (rows + 7u) / 8u;
    const uint32_t kg = (cols + 31u) / 32u;
    return (size_t)rt * kg * 136u;
}

static uint8_t encode_e8m0(float scale)
{
    if (!(scale > 0.0f))
        return 127;

    int e = (int)lrintf(log2f(scale));
    e = std::max(-126, std::min(127, e));

    return (uint8_t)(e + 127);
}

static float decode_e8m0(uint8_t e)
{
    return ldexpf(1.0f, (int)e - 127);
}

static uint32_t nearest_mx4(float x)
{
    uint32_t best = 0;
    float best_d = INFINITY;

    for (uint32_t i = 0; i < 16; ++i) {
        const float d = fabsf(x - MX4[i]);

        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }

    return best;
}

/*
 * Exact physical layout expected by the Stage-5 reader:
 *
 *     8 rows x 32 cols / 136 bytes
 *     bytes 0..127   = 8 * 16 packed nibbles
 *     bytes 128..135 = 8 E8M0 row scales
 */
static void
pack_apple8_matrix(uint8_t *dst,
                   const float *src,
                   uint32_t rows,
                   uint32_t cols)
{
    const uint32_t rt_count = (rows + 7u) / 8u;
    const uint32_t kg_count = (cols + 31u) / 32u;

    for (uint32_t rt = 0; rt < rt_count; ++rt) {
        for (uint32_t kg = 0; kg < kg_count; ++kg) {
            uint8_t *tile =
                dst + ((size_t)rt * kg_count + kg) * 136u;

            memset(tile, 0, 136);

            for (uint32_t rr = 0; rr < 8; ++rr) {
                const uint32_t row = rt * 8u + rr;

                float max_abs = 0.0f;

                if (row < rows) {
                    for (uint32_t k = 0; k < 32; ++k) {
                        const uint32_t col = kg * 32u + k;

                        if (col < cols) {
                            max_abs =
                                fmaxf(max_abs,
                                      fabsf(src[(size_t)row * cols + col]));
                        }
                    }
                }

                float ideal = max_abs > 0.0f
                    ? max_abs / 6.0f
                    : 1.0f;

                /*
                 * Choose a power-of-two scale large enough that the largest
                 * magnitude is representable by MX4's ±6 endpoint.
                 */
                int scale_exp =
                    max_abs > 0.0f
                        ? (int)ceilf(log2f(ideal))
                        : 0;

                scale_exp =
                    std::max(-126, std::min(127, scale_exp));

                const float scale =
                    ldexpf(1.0f, scale_exp);

                tile[128u + rr] =
                    encode_e8m0(scale);

                if (row >= rows)
                    continue;

                for (uint32_t k = 0; k < 32; ++k) {
                    const uint32_t col =
                        kg * 32u + k;

                    float v = 0.0f;

                    if (col < cols)
                        v = src[(size_t)row * cols + col];

                    const uint32_t code =
                        nearest_mx4(v / scale);

                    uint8_t &packed =
                        tile[rr * 16u + (k >> 1)];

                    if ((k & 1u) == 0u)
                        packed = (packed & 0xf0u) | (uint8_t)code;
                    else
                        packed = (packed & 0x0fu) |
                                 (uint8_t)(code << 4);
                }
            }
        }
    }
}

static float
apple8_get(const uint8_t *matrix,
           uint32_t row,
           uint32_t col,
           uint32_t cols)
{
    const uint32_t kg_count =
        (cols + 31u) / 32u;

    const uint32_t rt = row >> 3;
    const uint32_t rr = row & 7u;
    const uint32_t kg = col >> 5;
    const uint32_t k  = col & 31u;

    const uint8_t *tile =
        matrix + ((size_t)rt * kg_count + kg) * 136u;

    const uint8_t packed =
        tile[rr * 16u + (k >> 1)];

    const uint32_t code =
        (k & 1u)
            ? (packed >> 4)
            : (packed & 0x0fu);

    return MX4[code] *
           decode_e8m0(tile[128u + rr]);
}

/* -------------------------------------------------------------------------- */
/* Tiny model                                                                 */
/* -------------------------------------------------------------------------- */

struct TinyModel {
    QwenTokenKernelParams p{};
    QwenTokenDeviceLayout dl{};

    BlobBuilder blob;

    size_t state_bytes = 0;
    size_t work_floats = 0;

    std::array<std::array<float, H>, NTOK> input{};

    std::array<uint16_t, PROBE_VOCAB * H> probe_w{};
    std::array<float, PROBE_VOCAB> probe_bias{};
};

static uint64_t
append_expert_bank(TinyModel &m,
                   QwenTokenLayerBlob &L,
                   Rng &rng)
{
    const size_t gate_bytes =
        apple8_matrix_bytes(MOE_INTER, H);

    const size_t up_bytes =
        apple8_matrix_bytes(MOE_INTER, H);

    const size_t down_bytes =
        apple8_matrix_bytes(H, MOE_INTER);

    m.p.expert_gate_rel = 0;
    m.p.expert_up_rel   = gate_bytes;
    m.p.expert_down_rel = gate_bytes + up_bytes;

    const size_t slot_stride =
        align_up(gate_bytes + up_bytes + down_bytes, 16);

    L.expert_slot_stride = slot_stride;

    const uint64_t bank =
        m.blob.alloc((size_t)NEXPERTS * slot_stride, 16);

    L.expert_bank = bank;

    for (uint32_t e = 0; e < NEXPERTS; ++e) {
        uint8_t *slot =
            m.blob.b.data() +
            bank +
            (size_t)e * slot_stride;

        auto gate =
            random_matrix(rng,
                          MOE_INTER,
                          H,
                          0.10f);

        auto up =
            random_matrix(rng,
                          MOE_INTER,
                          H,
                          0.10f);

        auto down =
            random_matrix(rng,
                          H,
                          MOE_INTER,
                          0.10f);

        pack_apple8_matrix(
            slot + m.p.expert_gate_rel,
            gate.data(),
            MOE_INTER,
            H);

        pack_apple8_matrix(
            slot + m.p.expert_up_rel,
            up.data(),
            MOE_INTER,
            H);

        pack_apple8_matrix(
            slot + m.p.expert_down_rel,
            down.data(),
            H,
            MOE_INTER);
    }

    return bank;
}

static void build_layer_common(TinyModel &m,
                               QwenTokenLayerBlob &L,
                               Rng &rng)
{
    L.in_ln =
        m.blob.put_f32(norm_weights(rng, H));

    L.post_ln =
        m.blob.put_f32(norm_weights(rng, H));

    L.router =
        m.blob.put_bf16(
            random_matrix(rng,
                          NEXPERTS,
                          H,
                          0.10f));

    L.se_gate =
        m.blob.put_bf16(
            random_matrix(rng,
                          SH_INTER,
                          H,
                          0.08f));

    L.se_up =
        m.blob.put_bf16(
            random_matrix(rng,
                          SH_INTER,
                          H,
                          0.08f));

    L.se_down =
        m.blob.put_bf16(
            random_matrix(rng,
                          H,
                          SH_INTER,
                          0.08f));

    L.se_g =
        m.blob.put_bf16(
            random_matrix(rng,
                          1,
                          H,
                          0.05f));

    append_expert_bank(m, L, rng);
}

static void build_gdn_layer(TinyModel &m,
                            QwenTokenLayerBlob &L,
                            Rng &rng)
{
    L.kind = 1;

    build_layer_common(m, L, rng);

    L.gdn_in_qkv =
        m.blob.put_bf16(
            random_matrix(rng,
                          QKV_DIM,
                          H,
                          0.075f));

    L.gdn_in_z =
        m.blob.put_bf16(
            random_matrix(rng,
                          VDIM,
                          H,
                          0.075f));

    L.gdn_in_a =
        m.blob.put_bf16(
            random_matrix(rng,
                          LVH,
                          H,
                          0.025f));

    L.gdn_in_b =
        m.blob.put_bf16(
            random_matrix(rng,
                          LVH,
                          H,
                          0.025f));

    L.gdn_out =
        m.blob.put_bf16(
            random_matrix(rng,
                          H,
                          VDIM,
                          0.075f));

    std::vector<float> A(LVH);
    std::vector<float> dt(LVH);

    for (uint32_t h = 0; h < LVH; ++h) {
        A[h]  = -2.2f + rng.signed_unit() * 0.10f;
        dt[h] = -1.0f + rng.signed_unit() * 0.10f;
    }

    L.gdn_A_log   = m.blob.put_f32(A);
    L.gdn_dt_bias = m.blob.put_f32(dt);

    std::vector<float> conv((size_t)QKV_DIM * CONV_K);

    for (uint32_t c = 0; c < QKV_DIM; ++c) {
        conv[(size_t)c * CONV_K + 0] =
            rng.signed_unit() * 0.04f;

        conv[(size_t)c * CONV_K + 1] =
            rng.signed_unit() * 0.05f;

        conv[(size_t)c * CONV_K + 2] =
            0.70f + rng.signed_unit() * 0.03f;
    }

    L.gdn_conv1d =
        m.blob.put_f32(conv);

    L.gdn_norm =
        m.blob.put_f32(norm_weights(rng, LVD));
}

static void build_attn_layer(TinyModel &m,
                             QwenTokenLayerBlob &L,
                             Rng &rng)
{
    L.kind = 0;

    build_layer_common(m, L, rng);

    L.attn_q =
        m.blob.put_bf16(
            random_matrix(rng,
                          QDIM,
                          H,
                          0.075f));

    L.attn_k =
        m.blob.put_bf16(
            random_matrix(rng,
                          KVDIM,
                          H,
                          0.075f));

    L.attn_v =
        m.blob.put_bf16(
            random_matrix(rng,
                          KVDIM,
                          H,
                          0.075f));

    L.attn_o =
        m.blob.put_bf16(
            random_matrix(rng,
                          H,
                          QDIM,
                          0.075f));

    L.attn_qn =
        m.blob.put_f32(norm_weights(rng, HD));

    L.attn_kn =
        m.blob.put_f32(norm_weights(rng, HD));
}

/* -------------------------------------------------------------------------- */
/* Device layout                                                              */
/* -------------------------------------------------------------------------- */

struct StateLayoutBuilder {
    size_t off = 0;

    uint64_t alloc(size_t n, size_t align = 256)
    {
        off = align_up(off, align);

        const uint64_t ret = off;
        off += n;

        return ret;
    }
};

static void build_state_layout(TinyModel &m)
{
    StateLayoutBuilder s;

    m.dl.residual_off =
        s.alloc(H * sizeof(float));

    m.dl.normed_off =
        s.alloc(H * sizeof(float));

    m.dl.moe_input_off =
        s.alloc(H * sizeof(float));

    m.work_floats =
        std::max({
            (size_t)QKV_DIM,
            (size_t)H,
            (size_t)QDIM,
            (size_t)KVDIM,
            (size_t)MOE_INTER,
            (size_t)SH_INTER,
            (size_t)NEXPERTS
        });

    for (uint32_t i = 0; i < 4; ++i) {
        m.dl.work_off[i] =
            s.alloc(m.work_floats * sizeof(float));
    }

    for (uint32_t l = 0; l < NLAYERS; ++l) {
        m.dl.gdn_conv_layer_off[l] =
            s.alloc((size_t)QKV_DIM *
                    (CONV_K - 1) *
                    sizeof(float));

        m.dl.gdn_s_layer_off[l] =
            s.alloc((size_t)LVH *
                    LKD *
                    LVD *
                    sizeof(float));

        m.dl.kv_k_off[l] =
            s.alloc((size_t)MAX_T *
                    KVDIM *
                    sizeof(float));

        m.dl.kv_v_off[l] =
            s.alloc((size_t)MAX_T *
                    KVDIM *
                    sizeof(float));

        m.dl.expert_map_layer_off[l] =
            s.alloc(NEXPERTS * sizeof(uint32_t));

        m.dl.expert_map_slots[l] =
            NEXPERTS;
    }

    m.dl.miss_off =
        s.alloc(sizeof(QwenTokenMissRecord));

    m.state_bytes =
        align_up(s.off, 4096);
}

/* -------------------------------------------------------------------------- */
/* Build whole synthetic model                                                */
/* -------------------------------------------------------------------------- */

static TinyModel make_model()
{
    TinyModel m;
    Rng rng(0x4d434338514b544bull);

    m.p.abi_version = 1;

    m.p.n_layers = NLAYERS;
    m.p.hidden   = H;
    m.p.max_t    = MAX_T;

    m.p.n_heads    = NH;
    m.p.n_kv_heads = NKV;
    m.p.head_dim   = HD;
    m.p.rotary_dim = ROT;

    m.p.lin_k_heads = LKH;
    m.p.lin_k_dim   = LKD;
    m.p.lin_v_heads = LVH;
    m.p.lin_v_dim   = LVD;

    m.p.conv_kernel = CONV_K;

    m.p.n_experts   = NEXPERTS;
    m.p.topk        = TOPK;
    m.p.moe_inter   = MOE_INTER;
    m.p.shared_inter = SH_INTER;

    m.p.eps   = EPS;
    m.p.theta = THETA;

    build_gdn_layer(
        m,
        m.p.layer[0],
        rng);

    build_attn_layer(
        m,
        m.p.layer[1],
        rng);

    m.p.blob_bytes =
        m.blob.b.size();

    build_state_layout(m);

    /*
     * Token inputs.
     */
    for (uint32_t t = 0; t < NTOK; ++t) {
        for (uint32_t d = 0; d < H; ++d) {
            m.input[t][d] =
                rng.signed_unit() * 0.45f;
        }
    }

    /*
     * Stable synthetic probe head used only for comparing final hidden-state
     * trajectories as logits.
     */
    for (uint32_t v = 0; v < PROBE_VOCAB; ++v) {
        for (uint32_t d = 0; d < H; ++d) {
            const float w =
                rng.signed_unit() * 0.025f;

            m.probe_w[(size_t)v * H + d] =
                f32_to_bf16(w);
        }

        /*
         * Deliberately separated biases prevent meaningless argmax failures
         * caused by a nearly exact tie in a random probe head.
         */
        m.probe_bias[v] =
            (float)v * 0.075f;
    }

    return m;
}

/* ========================================================================== */
/* CPU reference                                                              */
/* ========================================================================== */

static float blob_f32(const TinyModel &m,
                      uint64_t off,
                      uint32_t i)
{
    float f;
    memcpy(&f,
           m.blob.b.data() + off + (size_t)i * sizeof(float),
           sizeof(f));
    return f;
}

static float blob_bf16(const TinyModel &m,
                       uint64_t off,
                       uint32_t i)
{
    uint16_t b;
    memcpy(&b,
           m.blob.b.data() + off + (size_t)i * sizeof(uint16_t),
           sizeof(b));
    return bf16_to_f32(b);
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void cpu_bf16_matvec(const TinyModel &m,
                            uint64_t off,
                            const float *x,
                            float *y,
                            uint32_t rows,
                            uint32_t cols)
{
    for (uint32_t r = 0; r < rows; ++r) {
        float s = 0.0f;

        for (uint32_t c = 0; c < cols; ++c) {
            s += blob_bf16(
                     m,
                     off,
                     r * cols + c) *
                 x[c];
        }

        y[r] = s;
    }
}

static void cpu_rmsnorm(const TinyModel &m,
                        const float *x,
                        uint64_t woff,
                        float *y,
                        uint32_t n)
{
    float ss = 0.0f;

    for (uint32_t i = 0; i < n; ++i)
        ss += x[i] * x[i];

    const float inv =
        1.0f / sqrtf(ss / (float)n + EPS);

    for (uint32_t i = 0; i < n; ++i) {
        y[i] =
            x[i] *
            blob_f32(m, woff, i) *
            inv;
    }
}

struct CPUState {
    std::array<float, H> residual{};

    std::array<
        std::array<float, QKV_DIM * (CONV_K - 1)>,
        NLAYERS> conv{};

    std::array<
        std::array<float, LVH * LKD * LVD>,
        NLAYERS> S{};

    std::array<
        std::array<float, MAX_T * KVDIM>,
        NLAYERS> kvk{};

    std::array<
        std::array<float, MAX_T * KVDIM>,
        NLAYERS> kvv{};
};

static void cpu_rope(float *x,
                     uint32_t base,
                     uint32_t pos)
{
    const uint32_t rot_half =
        ROT >> 1;

    for (uint32_t i = 0; i < rot_half; ++i) {
        const float freq =
            powf(THETA,
                 -2.0f * (float)i /
                 (float)ROT);

        const float a =
            (float)pos * freq;

        const float cs = cosf(a);
        const float sn = sinf(a);

        const uint32_t i0 = base + i;
        const uint32_t i1 = base + rot_half + i;

        const float x0 = x[i0];
        const float x1 = x[i1];

        x[i0] = x0 * cs - x1 * sn;
        x[i1] = x1 * cs + x0 * sn;
    }
}

static void cpu_gdn(const TinyModel &m,
                    const QwenTokenLayerBlob &L,
                    CPUState &s,
                    uint32_t layer,
                    const float *normed,
                    float *attn)
{
    std::array<float, QKV_DIM> qkv{};
    std::array<float, VDIM> z{};
    std::array<float, LVH> a{};
    std::array<float, LVH> b{};
    std::array<float, VDIM> y{};

    cpu_bf16_matvec(
        m,
        L.gdn_in_qkv,
        normed,
        qkv.data(),
        QKV_DIM,
        H);

    cpu_bf16_matvec(
        m,
        L.gdn_in_z,
        normed,
        z.data(),
        VDIM,
        H);

    cpu_bf16_matvec(
        m,
        L.gdn_in_a,
        normed,
        a.data(),
        LVH,
        H);

    cpu_bf16_matvec(
        m,
        L.gdn_in_b,
        normed,
        b.data(),
        LVH,
        H);

    /*
     * Causal depthwise conv.
     */
    for (uint32_t c = 0; c < QKV_DIM; ++c) {
        float *hist =
            &s.conv[layer][(size_t)c * (CONV_K - 1)];

        float v = 0.0f;

        for (uint32_t j = 0; j < CONV_K - 1; ++j) {
            v += blob_f32(
                     m,
                     L.gdn_conv1d,
                     c * CONV_K + j) *
                 hist[j];
        }

        const float cur = qkv[c];

        v += blob_f32(
                 m,
                 L.gdn_conv1d,
                 c * CONV_K + CONV_K - 1) *
             cur;

        for (uint32_t j = 0; j + 1 < CONV_K - 1; ++j)
            hist[j] = hist[j + 1];

        hist[CONV_K - 2] = cur;

        qkv[c] = silu(v);
    }

    /*
     * GDN recurrence, identical fixed loop ordering to qtk_gdn_recurrence.
     */
    for (uint32_t h = 0; h < LVH; ++h) {
        float *q =
            qkv.data() + h * LKD;

        float *k =
            qkv.data() + KDIM + h * LKD;

        float *v =
            qkv.data() + 2 * KDIM + h * LVD;

        float *yh =
            y.data() + h * LVD;

        float *Sh =
            s.S[layer].data() +
            (size_t)h * LKD * LVD;

        float qss = 0.0f;
        float kss = 0.0f;

        for (uint32_t kd = 0; kd < LKD; ++kd) {
            qss += q[kd] * q[kd];
            kss += k[kd] * k[kd];
        }

        const float qinv =
            1.0f / sqrtf(qss + 1.0e-6f);

        const float kinv =
            1.0f / sqrtf(kss + 1.0e-6f);

        for (uint32_t kd = 0; kd < LKD; ++kd) {
            q[kd] *= qinv;
            k[kd] *= kinv;
        }

        const float A =
            blob_f32(m, L.gdn_A_log, h);

        const float dt =
            blob_f32(m, L.gdn_dt_bias, h);

        const float sp =
            log1pf(expf(a[h] + dt));

        const float gamma =
            expf(-expf(A) * sp);

        const float beta =
            sigmoid(b[h]);

        for (uint32_t vd = 0; vd < LVD; ++vd) {
            float kv_mem = 0.0f;

            for (uint32_t kd = 0; kd < LKD; ++kd) {
                const size_t si =
                    (size_t)kd * LVD + vd;

                const float decayed =
                    Sh[si] * gamma;

                Sh[si] = decayed;

                kv_mem +=
                    decayed * k[kd];
            }

            const float delta =
                (v[vd] - kv_mem) * beta;

            float out = 0.0f;

            for (uint32_t kd = 0; kd < LKD; ++kd) {
                const size_t si =
                    (size_t)kd * LVD + vd;

                const float updated =
                    Sh[si] +
                    k[kd] * delta;

                Sh[si] = updated;

                out +=
                    updated * q[kd];
            }

            yh[vd] = out;
        }
    }

    /*
     * GDN output norm + z gate.
     */
    for (uint32_t h = 0; h < LVH; ++h) {
        const uint32_t base =
            h * LVD;

        float ss = 0.0f;

        for (uint32_t d = 0; d < LVD; ++d) {
            const float x = y[base + d];
            ss += x * x;
        }

        const float inv =
            1.0f /
            sqrtf(ss / (float)LVD + EPS);

        for (uint32_t d = 0; d < LVD; ++d) {
            const uint32_t i =
                base + d;

            y[i] =
                y[i] *
                inv *
                blob_f32(m, L.gdn_norm, d) *
                silu(z[i]);
        }
    }

    cpu_bf16_matvec(
        m,
        L.gdn_out,
        y.data(),
        attn,
        H,
        VDIM);
}

static void cpu_attention(const TinyModel &m,
                          const QwenTokenLayerBlob &L,
                          CPUState &s,
                          uint32_t layer,
                          uint32_t pos,
                          const float *normed,
                          float *attn)
{
    std::array<float, QDIM> q{};
    std::array<float, KVDIM> k{};
    std::array<float, KVDIM> v{};
    std::array<float, QDIM> ctx{};

    cpu_bf16_matvec(
        m,
        L.attn_q,
        normed,
        q.data(),
        QDIM,
        H);

    cpu_bf16_matvec(
        m,
        L.attn_k,
        normed,
        k.data(),
        KVDIM,
        H);

    cpu_bf16_matvec(
        m,
        L.attn_v,
        normed,
        v.data(),
        KVDIM,
        H);

    for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t base =
            h * HD;

        float ss = 0.0f;

        for (uint32_t d = 0; d < HD; ++d)
            ss += q[base + d] * q[base + d];

        const float inv =
            1.0f /
            sqrtf(ss / (float)HD + EPS);

        for (uint32_t d = 0; d < HD; ++d) {
            q[base + d] =
                q[base + d] *
                blob_f32(m, L.attn_qn, d) *
                inv;
        }

        cpu_rope(q.data(), base, pos);
    }

    for (uint32_t h = 0; h < NKV; ++h) {
        const uint32_t base =
            h * HD;

        float ss = 0.0f;

        for (uint32_t d = 0; d < HD; ++d)
            ss += k[base + d] * k[base + d];

        const float inv =
            1.0f /
            sqrtf(ss / (float)HD + EPS);

        for (uint32_t d = 0; d < HD; ++d) {
            k[base + d] =
                k[base + d] *
                blob_f32(m, L.attn_kn, d) *
                inv;
        }

        cpu_rope(k.data(), base, pos);
    }

    memcpy(
        s.kvk[layer].data() + (size_t)pos * KVDIM,
        k.data(),
        sizeof(float) * KVDIM);

    memcpy(
        s.kvv[layer].data() + (size_t)pos * KVDIM,
        v.data(),
        sizeof(float) * KVDIM);

    const float scale =
        1.0f / sqrtf((float)HD);

    for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t kvh =
            (h * NKV) / NH;

        const uint32_t qb =
            h * HD;

        float mx =
            -INFINITY;

        for (uint32_t t = 0; t <= pos; ++t) {
            const size_t kb =
                (size_t)t * KVDIM +
                (size_t)kvh * HD;

            float dot = 0.0f;

            for (uint32_t d = 0; d < HD; ++d) {
                dot +=
                    q[qb + d] *
                    s.kvk[layer][kb + d];
            }

            mx =
                fmaxf(mx, dot * scale);
        }

        float den = 0.0f;

        for (uint32_t d = 0; d < HD; ++d)
            ctx[qb + d] = 0.0f;

        for (uint32_t t = 0; t <= pos; ++t) {
            const size_t kb =
                (size_t)t * KVDIM +
                (size_t)kvh * HD;

            float dot = 0.0f;

            for (uint32_t d = 0; d < HD; ++d) {
                dot +=
                    q[qb + d] *
                    s.kvk[layer][kb + d];
            }

            const float e =
                expf(dot * scale - mx);

            den += e;

            for (uint32_t d = 0; d < HD; ++d) {
                ctx[qb + d] +=
                    e *
                    s.kvv[layer][kb + d];
            }
        }

        const float inv =
            1.0f / den;

        for (uint32_t d = 0; d < HD; ++d)
            ctx[qb + d] *= inv;
    }

    cpu_bf16_matvec(
        m,
        L.attn_o,
        ctx.data(),
        attn,
        H,
        QDIM);
}

struct Route {
    std::array<uint32_t, TOPK> expert{};
    std::array<float, TOPK> weight{};
};

static Route cpu_route(const TinyModel &m,
                       const QwenTokenLayerBlob &L,
                       const float *x)
{
    std::array<float, NEXPERTS> p{};

    cpu_bf16_matvec(
        m,
        L.router,
        x,
        p.data(),
        NEXPERTS,
        H);

    float mx =
        -INFINITY;

    for (uint32_t e = 0; e < NEXPERTS; ++e)
        mx = fmaxf(mx, p[e]);

    float sum = 0.0f;

    for (uint32_t e = 0; e < NEXPERTS; ++e) {
        p[e] =
            expf(p[e] - mx);

        sum += p[e];
    }

    const float inv =
        1.0f / sum;

    for (float &v : p)
        v *= inv;

    Route r;

    for (uint32_t rank = 0; rank < TOPK; ++rank) {
        uint32_t best =
            EMPTY_EXPERT;

        float best_p =
            -INFINITY;

        for (uint32_t e = 0; e < NEXPERTS; ++e) {
            bool used = false;

            for (uint32_t j = 0; j < rank; ++j) {
                if (r.expert[j] == e)
                    used = true;
            }

            if (!used &&
                (best == EMPTY_EXPERT ||
                 p[e] > best_p)) {
                best = e;
                best_p = p[e];
            }
        }

        r.expert[rank] = best;
        r.weight[rank] = best_p;
    }

    return r;
}

static void cpu_apple8_matvec(const uint8_t *matrix,
                              const float *x,
                              float *y,
                              uint32_t rows,
                              uint32_t cols)
{
    for (uint32_t r = 0; r < rows; ++r) {
        float s = 0.0f;

        for (uint32_t c = 0; c < cols; ++c) {
            s +=
                apple8_get(
                    matrix,
                    r,
                    c,
                    cols) *
                x[c];
        }

        y[r] = s;
    }
}

static void cpu_moe(const TinyModel &m,
                    const QwenTokenLayerBlob &L,
                    const float *x,
                    float *residual)
{
    const Route route =
        cpu_route(m, L, x);

    std::array<float, H> acc{};
    std::array<float, MOE_INTER> gate{};
    std::array<float, MOE_INTER> up{};
    std::array<float, H> down{};

    for (uint32_t rank = 0; rank < TOPK; ++rank) {
        const uint32_t expert =
            route.expert[rank];

        const uint8_t *slot =
            m.blob.b.data() +
            L.expert_bank +
            (size_t)expert *
            L.expert_slot_stride;

        cpu_apple8_matvec(
            slot + m.p.expert_gate_rel,
            x,
            gate.data(),
            MOE_INTER,
            H);

        cpu_apple8_matvec(
            slot + m.p.expert_up_rel,
            x,
            up.data(),
            MOE_INTER,
            H);

        for (uint32_t j = 0; j < MOE_INTER; ++j)
            gate[j] = silu(gate[j]) * up[j];

        cpu_apple8_matvec(
            slot + m.p.expert_down_rel,
            gate.data(),
            down.data(),
            H,
            MOE_INTER);

        for (uint32_t d = 0; d < H; ++d) {
            acc[d] +=
                route.weight[rank] *
                down[d];
        }
    }

    std::array<float, SH_INTER> sg{};
    std::array<float, SH_INTER> su{};
    std::array<float, H> sy{};

    cpu_bf16_matvec(
        m,
        L.se_gate,
        x,
        sg.data(),
        SH_INTER,
        H);

    cpu_bf16_matvec(
        m,
        L.se_up,
        x,
        su.data(),
        SH_INTER,
        H);

    for (uint32_t j = 0; j < SH_INTER; ++j)
        sg[j] = silu(sg[j]) * su[j];

    cpu_bf16_matvec(
        m,
        L.se_down,
        sg.data(),
        sy.data(),
        H,
        SH_INTER);

    float g = 0.0f;

    for (uint32_t d = 0; d < H; ++d) {
        g +=
            blob_bf16(m, L.se_g, d) *
            x[d];
    }

    g = sigmoid(g);

    for (uint32_t d = 0; d < H; ++d) {
        residual[d] +=
            acc[d] +
            sy[d] * g;
    }
}

static void cpu_forward_token(const TinyModel &m,
                              CPUState &s,
                              uint32_t pos,
                              const float *input,
                              float *output)
{
    memcpy(
        s.residual.data(),
        input,
        sizeof(float) * H);

    std::array<float, H> normed{};
    std::array<float, H> attn{};
    std::array<float, H> moe_input{};

    for (uint32_t layer = 0; layer < NLAYERS; ++layer) {
        const auto &L =
            m.p.layer[layer];

        cpu_rmsnorm(
            m,
            s.residual.data(),
            L.in_ln,
            normed.data(),
            H);

        if (L.kind == 1) {
            cpu_gdn(
                m,
                L,
                s,
                layer,
                normed.data(),
                attn.data());
        } else {
            cpu_attention(
                m,
                L,
                s,
                layer,
                pos,
                normed.data(),
                attn.data());
        }

        for (uint32_t d = 0; d < H; ++d)
            s.residual[d] += attn[d];

        cpu_rmsnorm(
            m,
            s.residual.data(),
            L.post_ln,
            moe_input.data(),
            H);

        cpu_moe(
            m,
            L,
            moe_input.data(),
            s.residual.data());
    }

    memcpy(
        output,
        s.residual.data(),
        sizeof(float) * H);
}

static void probe_logits(const TinyModel &m,
                         const float *hidden,
                         float *logits)
{
    for (uint32_t v = 0; v < PROBE_VOCAB; ++v) {
        float s =
            m.probe_bias[v];

        for (uint32_t d = 0; d < H; ++d) {
            s +=
                bf16_to_f32(
                    m.probe_w[(size_t)v * H + d]) *
                hidden[d];
        }

        logits[v] = s;
    }
}

static std::vector<float>
make_cpu_reference(const TinyModel &m)
{
    CPUState s;

    std::vector<float> out(
        (size_t)NTOK * H);

    for (uint32_t t = 0; t < NTOK; ++t) {
        cpu_forward_token(
            m,
            s,
            t,
            m.input[t].data(),
            out.data() + (size_t)t * H);
    }

    return out;
}

/* ========================================================================== */
/* Extract actual embedded MSL from qwen_token_kernel.mm                      */
/* ========================================================================== */

static std::string read_text_file(const char *path)
{
    std::ifstream f(path, std::ios::binary);

    if (!f)
        return {};

    return std::string(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

static std::string load_qwen_source_file()
{
    std::string s =
        read_text_file("qwen_token_kernel.mm");

    if (!s.empty())
        return s;

    s =
        read_text_file("c/qwen_token_kernel.mm");

    if (!s.empty())
        return s;

    s =
        read_text_file("../qwen_token_kernel.mm");

    return s;
}

/*
 * Locate the raw C++ literal containing qwen_token_whole.
 *
 * This avoids adding a test-only source accessor to production code.
 */
static std::string extract_msl(const std::string &cpp)
{
    size_t kernel =
        cpp.find("qwen_token_whole");

    while (kernel != std::string::npos) {
        const size_t line_begin =
            kernel > 128 ? kernel - 128 : 0;

        const std::string before =
            cpp.substr(line_begin,
                       kernel - line_begin);

        if (before.find("kernel void") !=
            std::string::npos) {
            break;
        }

        kernel =
            cpp.find("qwen_token_whole",
                     kernel + 1);
    }

    if (kernel == std::string::npos)
        return {};

    size_t raw =
        cpp.rfind("R\"", kernel);

    while (raw != std::string::npos) {
        const size_t delim_begin =
            raw + 2;

        const size_t paren =
            cpp.find('(', delim_begin);

        if (paren != std::string::npos &&
            paren < kernel &&
            paren - delim_begin <= 16) {
            const std::string delim =
                cpp.substr(
                    delim_begin,
                    paren - delim_begin);

            const std::string close =
                ")" + delim + "\"";

            const size_t end =
                cpp.find(close, kernel);

            if (end != std::string::npos) {
                return cpp.substr(
                    paren + 1,
                    end - (paren + 1));
            }
        }

        if (raw == 0)
            break;

        raw =
            cpp.rfind("R\"", raw - 1);
    }

    return {};
}

/* ========================================================================== */
/* Metal harness                                                              */
/* ========================================================================== */

struct MetalHarness {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipe = nil;

    id<MTLBuffer> weights = nil;
    id<MTLBuffer> params = nil;
    id<MTLBuffer> state = nil;
    id<MTLBuffer> token_pos = nil;
    id<MTLBuffer> layout = nil;

    uint64_t command_buffers = 0;
    uint64_t synchronizations = 0;
};

static bool metal_init(MetalHarness &h,
                       const TinyModel &m)
{
    h.dev =
        MTLCreateSystemDefaultDevice();

    if (!h.dev) {
        fprintf(stderr,
                "FAIL: no Metal device\n");
        return false;
    }

    h.queue =
        [h.dev newCommandQueue];

    if (!h.queue) {
        fprintf(stderr,
                "FAIL: newCommandQueue\n");
        return false;
    }

    const std::string cpp =
        load_qwen_source_file();

    if (cpp.empty()) {
        fprintf(stderr,
                "FAIL: could not read qwen_token_kernel.mm\n");
        return false;
    }

    const std::string msl =
        extract_msl(cpp);

    if (msl.empty()) {
        fprintf(stderr,
                "FAIL: could not extract embedded MSL\n");
        return false;
    }

    NSString *source =
        [[NSString alloc]
            initWithBytes:msl.data()
                   length:msl.size()
                 encoding:NSUTF8StringEncoding];

    NSError *err = nil;

    id<MTLLibrary> lib =
        [h.dev newLibraryWithSource:source
                            options:nil
                              error:&err];

    if (!lib) {
        fprintf(stderr,
                "FAIL: MSL compile:\n%s\n",
                err.localizedDescription.UTF8String);
        return false;
    }

    id<MTLFunction> fn =
        [lib newFunctionWithName:@"qwen_token_whole"];

    if (!fn) {
        fprintf(stderr,
                "FAIL: qwen_token_whole not found\n");
        return false;
    }

    h.pipe =
        [h.dev newComputePipelineStateWithFunction:fn
                                              error:&err];

    if (!h.pipe) {
        fprintf(stderr,
                "FAIL: pipeline:\n%s\n",
                err.localizedDescription.UTF8String);
        return false;
    }

    if (h.pipe.maxTotalThreadsPerThreadgroup < 256) {
        fprintf(stderr,
                "FAIL: device supports only %lu threads/TG\n",
                (unsigned long)
                    h.pipe.maxTotalThreadsPerThreadgroup);
        return false;
    }

    h.weights =
        [h.dev
            newBufferWithBytes:m.blob.b.data()
                        length:m.blob.b.size()
                       options:MTLResourceStorageModeShared];

    h.params =
        [h.dev
            newBufferWithBytes:&m.p
                        length:sizeof(m.p)
                       options:MTLResourceStorageModeShared];

    h.state =
        [h.dev
            newBufferWithLength:m.state_bytes
                        options:MTLResourceStorageModeShared];

    h.token_pos =
        [h.dev
            newBufferWithLength:sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];

    h.layout =
        [h.dev
            newBufferWithBytes:&m.dl
                        length:sizeof(m.dl)
                       options:MTLResourceStorageModeShared];

    if (!h.weights ||
        !h.params ||
        !h.state ||
        !h.token_pos ||
        !h.layout) {
        fprintf(stderr,
                "FAIL: Metal buffer allocation\n");
        return false;
    }

    return true;
}

static bool dispatch_once(MetalHarness &h,
                          uint32_t token_pos)
{
    memcpy(
        h.token_pos.contents,
        &token_pos,
        sizeof(token_pos));

    id<MTLCommandBuffer> cb =
        [h.queue commandBuffer];

    if (!cb)
        return false;

    ++h.command_buffers;

    id<MTLComputeCommandEncoder> enc =
        [cb computeCommandEncoder];

    [enc setComputePipelineState:h.pipe];

    [enc setBuffer:h.weights
            offset:0
           atIndex:0];

    [enc setBuffer:h.params
            offset:0
           atIndex:1];

    [enc setBuffer:h.state
            offset:0
           atIndex:2];

    [enc setBuffer:h.token_pos
            offset:0
           atIndex:3];

    [enc setBuffer:h.layout
            offset:0
           atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];

    ++h.synchronizations;

    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr,
                "FAIL: command buffer: %s\n",
                cb.error.localizedDescription.UTF8String);
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------- */
/* Host-style expert-map operations                                           */
/* -------------------------------------------------------------------------- */

static uint32_t *state_map(const TinyModel &m,
                           MetalHarness &h,
                           uint32_t layer)
{
    auto *base =
        reinterpret_cast<uint8_t *>(
            h.state.contents);

    return reinterpret_cast<uint32_t *>(
        base +
        m.dl.expert_map_layer_off[layer]);
}

static QwenTokenMissRecord *
state_miss(const TinyModel &m,
           MetalHarness &h)
{
    auto *base =
        reinterpret_cast<uint8_t *>(
            h.state.contents);

    return reinterpret_cast<QwenTokenMissRecord *>(
        base + m.dl.miss_off);
}

static float *state_residual(const TinyModel &m,
                             MetalHarness &h)
{
    auto *base =
        reinterpret_cast<uint8_t *>(
            h.state.contents);

    return reinterpret_cast<float *>(
        base + m.dl.residual_off);
}

static void install_identity_map(const TinyModel &m,
                                 MetalHarness &h,
                                 uint32_t layer)
{
    uint32_t *map =
        state_map(m, h, layer);

    for (uint32_t e = 0; e < NEXPERTS; ++e)
        map[e] = e;
}

static void clear_map(const TinyModel &m,
                      MetalHarness &h,
                      uint32_t layer)
{
    uint32_t *map =
        state_map(m, h, layer);

    for (uint32_t e = 0; e < NEXPERTS; ++e)
        map[e] = EMPTY_EXPERT;
}

/* -------------------------------------------------------------------------- */
/* One complete GPU run                                                       */
/* -------------------------------------------------------------------------- */

struct RunResult {
    std::vector<float> outputs;

    uint64_t command_buffers = 0;
    uint64_t synchronizations = 0;

    uint32_t misses = 0;
    uint32_t first_missing_count = 0;
    uint32_t first_resume_layer = UINT32_MAX;
};

static bool gpu_run(const TinyModel &m,
                    MetalHarness &h,
                    RunResult &r)
{
    memset(
        h.state.contents,
        0,
        m.state_bytes);

    h.command_buffers = 0;
    h.synchronizations = 0;

    /*
     * Layer 0 is intentionally empty.
     * Layer 1 is resident from the start.
     *
     * This guarantees one miss in the GDN layer, then makes the resumed
     * dispatch capable of finishing the entire token.
     */
    clear_map(m, h, 0);
    install_identity_map(m, h, 1);

    auto *miss =
        state_miss(m, h);

    miss->status = STATUS_OK;
    miss->resume_phase = RESUME_NONE;

    r.outputs.resize((size_t)NTOK * H);

    for (uint32_t t = 0; t < NTOK; ++t) {
        memcpy(
            state_residual(m, h),
            m.input[t].data(),
            sizeof(float) * H);

        if (!dispatch_once(h, t))
            return false;

        miss =
            state_miss(m, h);

        if (t == 0) {
            /*
             * Forced Stage-5 miss.
             */
            if (miss->status != STATUS_NEED_EXPERTS) {
                fprintf(stderr,
                        "FAIL: forced token-0 dispatch did not NEED_EXPERTS "
                        "(status=%u)\n",
                        miss->status);
                return false;
            }

            if (miss->resume_phase != RESUME_MOE) {
                fprintf(stderr,
                        "FAIL: wrong resume phase %u\n",
                        miss->resume_phase);
                return false;
            }

            if (miss->layer != 0) {
                fprintf(stderr,
                        "FAIL: expected miss on GDN layer 0, got %u\n",
                        miss->layer);
                return false;
            }

            if (miss->missing_count != TOPK) {
                fprintf(stderr,
                        "FAIL: empty map should miss all top-k experts "
                        "(got %u, expected %u)\n",
                        miss->missing_count,
                        TOPK);
                return false;
            }

            ++r.misses;
            r.first_missing_count =
                miss->missing_count;
            r.first_resume_layer =
                miss->layer;

            /*
             * Host fill/publish.
             *
             * Populate every slot after observing the deliberately forced
             * miss.  This prevents unrelated later-token misses from
             * obscuring the resume test.
             */
            install_identity_map(m, h, 0);

            /*
             * Same token_pos.  Kernel must enter resume_phase==MOE and must
             * NOT replay GDN conv/S or pre-MoE residual work.
             */
            if (!dispatch_once(h, t))
                return false;

            miss =
                state_miss(m, h);

            if (miss->status != STATUS_OK ||
                miss->resume_phase != RESUME_NONE) {
                fprintf(stderr,
                        "FAIL: resume did not complete "
                        "(status=%u phase=%u)\n",
                        miss->status,
                        miss->resume_phase);
                return false;
            }
        } else {
            if (miss->status != STATUS_OK ||
                miss->resume_phase != RESUME_NONE) {
                fprintf(stderr,
                        "FAIL: unexpected miss at token %u "
                        "(status=%u phase=%u layer=%u)\n",
                        t,
                        miss->status,
                        miss->resume_phase,
                        miss->layer);
                return false;
            }
        }

        memcpy(
            r.outputs.data() + (size_t)t * H,
            state_residual(m, h),
            sizeof(float) * H);
    }

    r.command_buffers =
        h.command_buffers;

    r.synchronizations =
        h.synchronizations;

    /*
     * 10 logical tokens + one forced miss/re-dispatch.
     */
    if (r.command_buffers != NTOK + 1) {
        fprintf(stderr,
                "FAIL: expected %u command buffers, got %llu\n",
                NTOK + 1,
                (unsigned long long)r.command_buffers);
        return false;
    }

    if (r.synchronizations != NTOK + 1) {
        fprintf(stderr,
                "FAIL: expected %u waits/syncs, got %llu\n",
                NTOK + 1,
                (unsigned long long)r.synchronizations);
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------- */
/* Structural state checks                                                    */
/* -------------------------------------------------------------------------- */

static bool check_gdn_heads(const TinyModel &m,
                            MetalHarness &h)
{
    const uint8_t *base =
        reinterpret_cast<const uint8_t *>(
            h.state.contents);

    const float *S =
        reinterpret_cast<const float *>(
            base + m.dl.gdn_s_layer_off[0]);

    for (uint32_t head = 0; head < LVH; ++head) {
        float l1 = 0.0f;

        const float *Sh =
            S +
            (size_t)head *
            LKD *
            LVD;

        for (uint32_t i = 0; i < LKD * LVD; ++i)
            l1 += fabsf(Sh[i]);

        if (!(l1 > 1.0e-8f)) {
            fprintf(stderr,
                    "FAIL: GDN head %u never updated\n",
                    head);
            return false;
        }
    }

    return true;
}

static bool check_attention_pos9(const TinyModel &m,
                                 MetalHarness &h)
{
    const uint8_t *base =
        reinterpret_cast<const uint8_t *>(
            h.state.contents);

    const float *k =
        reinterpret_cast<const float *>(
            base + m.dl.kv_k_off[1]);

    const float *v =
        reinterpret_cast<const float *>(
            base + m.dl.kv_v_off[1]);

    for (uint32_t head = 0; head < NKV; ++head) {
        float l1k = 0.0f;
        float l1v = 0.0f;

        const size_t p =
            (size_t)9 * KVDIM +
            (size_t)head * HD;

        for (uint32_t d = 0; d < HD; ++d) {
            l1k += fabsf(k[p + d]);
            l1v += fabsf(v[p + d]);
        }

        if (!(l1k > 1.0e-8f) ||
            !(l1v > 1.0e-8f)) {
            fprintf(stderr,
                    "FAIL: attention head %u position 9 KV absent\n",
                    head);
            return false;
        }
    }

    return true;
}

/* -------------------------------------------------------------------------- */
/* Numerical gate                                                             */
/* -------------------------------------------------------------------------- */

struct Distance {
    float max_abs = 0.0f;
    float rms = 0.0f;
    uint32_t argmax_mismatch = 0;
};

static uint32_t argmax(const float *x,
                       uint32_t n)
{
    uint32_t best = 0;

    for (uint32_t i = 1; i < n; ++i) {
        if (x[i] > x[best])
            best = i;
    }

    return best;
}

static Distance compare_logits(const TinyModel &m,
                               const std::vector<float> &cpu,
                               const std::vector<float> &gpu)
{
    Distance d;

    double ss = 0.0;
    uint64_t n = 0;

    std::array<float, PROBE_VOCAB> a{};
    std::array<float, PROBE_VOCAB> b{};

    for (uint32_t t = 0; t < NTOK; ++t) {
        probe_logits(
            m,
            cpu.data() + (size_t)t * H,
            a.data());

        probe_logits(
            m,
            gpu.data() + (size_t)t * H,
            b.data());

        if (argmax(a.data(), PROBE_VOCAB) !=
            argmax(b.data(), PROBE_VOCAB)) {
            ++d.argmax_mismatch;
        }

        for (uint32_t v = 0; v < PROBE_VOCAB; ++v) {
            if (!std::isfinite(a[v]) ||
                !std::isfinite(b[v])) {
                d.max_abs = INFINITY;
                d.rms = INFINITY;
                return d;
            }

            const float e =
                fabsf(a[v] - b[v]);

            d.max_abs =
                fmaxf(d.max_abs, e);

            ss += (double)e * e;
            ++n;
        }
    }

    d.rms =
        sqrtf((float)(ss / (double)n));

    return d;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */

int main()
{
    @autoreleasepool {
        const TinyModel model =
            make_model();

        const std::vector<float> cpu =
            make_cpu_reference(model);

        MetalHarness metal;

        if (!metal_init(metal, model))
            return 1;

        std::array<RunResult, 3> run;

        for (uint32_t i = 0; i < 3; ++i) {
            if (!gpu_run(
                    model,
                    metal,
                    run[i])) {
                fprintf(stderr,
                        "FAIL: GPU run %u\n",
                        i);
                return 1;
            }
        }

        /*
         * GPU self-consistency is the hard determinism gate.
         *
         * CPU/GPU are intentionally NOT byte-compared.
         */
        const size_t bytes =
            run[0].outputs.size() *
            sizeof(float);

        if (memcmp(
                run[0].outputs.data(),
                run[1].outputs.data(),
                bytes) != 0 ||
            memcmp(
                run[0].outputs.data(),
                run[2].outputs.data(),
                bytes) != 0) {
            fprintf(stderr,
                    "FAIL: three-run GPU determinism mismatch\n");
            return 1;
        }

        /*
         * CPU oracle is a numerical-distance gate, not the byte-parity
         * authority.
         */
        const Distance dist =
            compare_logits(
                model,
                cpu,
                run[0].outputs);

        /*
         * These tolerances are intentionally tight enough to expose a wrong
         * recurrence, route, KV scan, MXFP4 layout, or double-applied resume,
         * while allowing Metal-vs-libm transcendental rounding differences.
         */
        static constexpr float MAX_LOGIT_ABS = 2.0e-2f;
        static constexpr float MAX_LOGIT_RMS = 5.0e-3f;

        if (!(dist.max_abs <= MAX_LOGIT_ABS)) {
            fprintf(stderr,
                    "FAIL: max logit distance %.9g > %.9g\n",
                    dist.max_abs,
                    MAX_LOGIT_ABS);
            return 1;
        }

        if (!(dist.rms <= MAX_LOGIT_RMS)) {
            fprintf(stderr,
                    "FAIL: RMS logit distance %.9g > %.9g\n",
                    dist.rms,
                    MAX_LOGIT_RMS);
            return 1;
        }

        if (dist.argmax_mismatch != 0) {
            fprintf(stderr,
                    "FAIL: %u/%u probe-token argmax mismatches\n",
                    dist.argmax_mismatch,
                    NTOK);
            return 1;
        }

        if (run[0].misses != 1 ||
            run[0].first_missing_count != TOPK ||
            run[0].first_resume_layer != 0) {
            fprintf(stderr,
                    "FAIL: miss/resume gate "
                    "(misses=%u missing=%u layer=%u)\n",
                    run[0].misses,
                    run[0].first_missing_count,
                    run[0].first_resume_layer);
            return 1;
        }

        /*
         * State after run[2] remains in the Metal buffer.
         */
        if (!check_gdn_heads(
                model,
                metal)) {
            return 1;
        }

        if (!check_attention_pos9(
                model,
                metal)) {
            return 1;
        }

        printf(
            "QWEN_TOKEN_KERNEL_GATE_OK "
            "tokens=%u "
            "layers=%u "
            "gdn_heads=%u "
            "attn_pos_max=%u "
            "forced_misses=%u "
            "cmd_buffers=%llu "
            "syncs=%llu "
            "runs=3 "
            "gpu_determinism=BYTE_IDENTICAL "
            "max_logit_abs=%.9g "
            "logit_rms=%.9g "
            "argmax_mismatch=%u\n",
            NTOK,
            NLAYERS,
            LVH,
            NTOK - 1,
            run[0].misses,
            (unsigned long long)run[0].command_buffers,
            (unsigned long long)run[0].synchronizations,
            dist.max_abs,
            dist.rms,
            dist.argmax_mismatch);

        return 0;
    }
}