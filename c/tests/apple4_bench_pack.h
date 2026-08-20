#ifndef COLIBRI_TEST_APPLE4_BENCH_PACK_H
#define COLIBRI_TEST_APPLE4_BENCH_PACK_H

enum class Format { MX_ROW, MX_TILE, A4S64, A4S32, A4S16 };

struct Shape {
  const char *name;
  int O, I, S;
  int activation_outlier;
};

struct Packed {
  Format fmt;
  std::vector<uint8_t> record;
  std::vector<uint8_t> scales; /* MX_ROW only */
  std::vector<float> dequant;
  size_t resident_bytes = 0;
};

struct Metric {
  double mse = 0.0;
  double cosine = 0.0;
};

struct RunResult {
  double kernel_ms = 0.0;
  double wall_ms = 0.0;
  double gbps = 0.0;
  double max_rel = 0.0;
  uint64_t bytes_moved = 0;
  int ok = 0;
};

static uint64_t rng_state = UINT64_C(0x131a4e1b5eed1234);
static uint32_t rng_u32() {
  uint64_t x = rng_state;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  rng_state = x;
  return (uint32_t)((x * UINT64_C(2685821657736338717)) >> 32);
}
static float rng_signed() {
  return ((float)(rng_u32() & 0x00ffffffu) / 8388608.0f) - 1.0f;
}

static uint16_t half_bits(float f) {
  __fp16 h = (__fp16)f;
  uint16_t u;
  memcpy(&u, &h, sizeof(u));
  return u;
}
static float half_value(uint16_t u) {
  __fp16 h;
  memcpy(&h, &u, sizeof(u));
  return (float)h;
}
static void store_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, sizeof(v)); }

static void put_nib(uint8_t *row, int k, uint8_t n) {
  uint8_t &b = row[k >> 1];
  if (k & 1) b = (uint8_t)((b & 0x0fu) | ((n & 0x0fu) << 4));
  else       b = (uint8_t)((b & 0xf0u) | (n & 0x0fu));
}

static const float MX4[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                              -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

static uint8_t mx_nearest(float z) {
  bool neg = std::signbit(z);
  float a = std::fabs(z);
  int m;
  if (a < 0.25f) m = 0;
  else if (a < 0.75f) m = 1;
  else if (a < 1.25f) m = 2;
  else if (a < 1.75f) m = 3;
  else if (a < 2.5f) m = 4;
  else if (a < 3.5f) m = 5;
  else if (a < 5.0f) m = 6;
  else m = 7;
  return (uint8_t)(m + ((neg && m) ? 8 : 0));
}

static float e8_scale(uint8_t s) {
  uint32_t u = (uint32_t)s << 23;
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static void quant_mx_block(const float *src, int n, uint8_t *scale_out, uint8_t *nib_out) {
  float mx = 0.0f;
  for (int i = 0; i < n; ++i) mx = std::max(mx, std::fabs(src[i]));
  if (!(mx > 0.0f)) {
    *scale_out = 127;
    memset(nib_out, 0, (size_t)n);
    return;
  }
  float ideal = mx / 6.0f;
  int ef = (int)std::floor(std::log2(ideal));
  double best = INFINITY;
  uint8_t best_s = 127;
  for (int de = -1; de <= 2; ++de) {
    int e = ef + de;
    int sb = std::max(1, std::min(254, e + 127));
    float sc = e8_scale((uint8_t)sb);
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
      uint8_t qn = mx_nearest(src[i] / sc);
      double d = (double)src[i] - (double)MX4[qn] * sc;
      sse += d * d;
    }
    if (sse < best) { best = sse; best_s = (uint8_t)sb; }
  }
  *scale_out = best_s;
  float best_sc = e8_scale(best_s);
  for (int i = 0; i < n; ++i) nib_out[i] = mx_nearest(src[i] / best_sc);
}

static std::vector<float> make_weights(int O, int I) {
  rng_state = UINT64_C(0x13100000) ^ ((uint64_t)(uint32_t)O << 32) ^ (uint32_t)I;
  std::vector<float> w((size_t)O * I);
  for (int o = 0; o < O; ++o) {
    float rs = 0.015f + 0.003f * (float)(o % 17);
    for (int i = 0; i < I; ++i) {
      float v = rng_signed() * rs;
      if (((o * 131 + i * 17) % 4093) == 0) v *= 12.0f;
      /* Quality reference is explicitly FP16, matching issue #131's rough
       * synthetic quality gate rather than granting any candidate FP32 input. */
      w[(size_t)o * I + i] = half_value(half_bits(v));
    }
  }
  return w;
}

static std::vector<float> make_x(const Shape& sh) {
  rng_state = UINT64_C(0x131abcde) ^ ((uint64_t)(uint32_t)sh.I << 32) ^ (uint32_t)sh.S;
  std::vector<float> x((size_t)sh.S * sh.I);
  for (float &v : x) v = rng_signed();
  if (sh.activation_outlier) {
    for (int s = 0; s < sh.S; ++s) x[(size_t)s * sh.I] = 50.0f;
  }
  return x;
}

static Packed pack_mx_row(const std::vector<float>& w, int O, int I) {
  Packed p; p.fmt = Format::MX_ROW;
  int rb = (I + 1) / 2, ng = (I + 31) / 32;
  p.record.assign((size_t)O * rb, 0);
  p.scales.assign((size_t)O * ng, 127);
  p.dequant.resize((size_t)O * I);
  std::vector<float> tmp(32);
  std::vector<uint8_t> nib(32);
  for (int o = 0; o < O; ++o) {
    uint8_t *row = p.record.data() + (size_t)o * rb;
    for (int g = 0; g < ng; ++g) {
      int k0 = g * 32, n = std::min(32, I - k0);
      for (int j = 0; j < n; ++j) tmp[(size_t)j] = w[(size_t)o * I + k0 + j];
      uint8_t sb = 127;
      quant_mx_block(tmp.data(), n, &sb, nib.data());
      p.scales[(size_t)o * ng + g] = sb;
      float sc = e8_scale(sb);
      for (int j = 0; j < n; ++j) {
        put_nib(row, k0 + j, nib[(size_t)j]);
        p.dequant[(size_t)o * I + k0 + j] = MX4[nib[(size_t)j]] * sc;
      }
    }
  }
  p.resident_bytes = p.record.size() + p.scales.size();
  return p;
}

static Packed pack_mx_tile(const Packed& row, int O, int I) {
  Packed p; p.fmt = Format::MX_TILE; p.dequant = row.dequant;
  int rb = (I + 1) / 2, ng = (I + 31) / 32, no = (O + 7) / 8;
  p.record.assign((size_t)no * ng * 136, 0);
  for (int o = 0; o < O; ++o) {
    const uint8_t *wr = row.record.data() + (size_t)o * rb;
    const uint8_t *sr = row.scales.data() + (size_t)o * ng;
    int ot = o >> 3, orow = o & 7;
    for (int g = 0; g < ng; ++g) {
      uint8_t *tile = p.record.data() + ((size_t)ot * ng + g) * 136;
      int k0 = g * 32, n = std::min(32, I - k0);
      int bytes = (n + 1) / 2;
      memcpy(tile + orow * 16, wr + (k0 >> 1), (size_t)bytes);
      tile[128 + orow] = sr[g];
    }
  }
  p.resident_bytes = p.record.size();
  return p;
}

static Packed pack_a4(const std::vector<float>& w, int O, int I, int block) {
  Packed p;
  p.fmt = block == 16 ? Format::A4S16 : block == 32 ? Format::A4S32 : Format::A4S64;
  int no = (O + 7) / 8;
  int ktile = block == 64 ? 64 : 32;
  int ng = (I + ktile - 1) / ktile;
  int stride = block == 16 ? 160 : block == 32 ? 144 : 272;
  int rowbytes = ktile / 2;
  int scale_base = 8 * rowbytes;
  int scales_per_row = ktile / block;
  p.record.assign((size_t)no * ng * stride, 0);
  p.dequant.resize((size_t)O * I);
  for (int o = 0; o < O; ++o) {
    int ot = o >> 3, orow = o & 7;
    for (int kg = 0; kg < ng; ++kg) {
      int tile_k0 = kg * ktile;
      uint8_t *tile = p.record.data() + ((size_t)ot * ng + kg) * stride;
      uint8_t *prow = tile + orow * rowbytes;
      for (int sub = 0; sub < scales_per_row; ++sub) {
        int k0 = tile_k0 + sub * block;
        int n = std::max(0, std::min(block, I - k0));
        float mx = 0.0f;
        for (int j = 0; j < n; ++j) mx = std::max(mx, std::fabs(w[(size_t)o * I + k0 + j]));
        /* Symmetric zero-point-0 INT4 intentionally uses the balanced [-7,7]
         * code range; nibble 0x8 remains unused rather than giving negatives
         * one extra representable magnitude. */
        float sc0 = mx > 0.0f ? mx / 7.0f : 1.0f;
        uint16_t hb = half_bits(sc0);
        float sc = half_value(hb);
        if (!(sc > 0.0f) || !std::isfinite(sc)) { hb = half_bits(1.0f); sc = 1.0f; }
        store_u16(tile + scale_base + orow * scales_per_row * 2 + sub * 2, hb);
        for (int j = 0; j < n; ++j) {
          float z = w[(size_t)o * I + k0 + j] / sc;
          int q = (int)lrintf(z);
          q = std::max(-7, std::min(7, q));
          put_nib(prow, sub * block + j, (uint8_t)(q & 15));
          p.dequant[(size_t)o * I + k0 + j] = (float)q * sc;
        }
      }
    }
  }
  p.resident_bytes = p.record.size();
  return p;
}

static Metric quality(const std::vector<float>& ref, const std::vector<float>& q) {
  long double se = 0.0, dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    long double a = ref[i], b = q[i], d = a - b;
    se += d*d; dot += a*b; aa += a*a; bb += b*b;
  }
  Metric m;
  m.mse = ref.empty() ? 0.0 : (double)(se / (long double)ref.size());
  m.cosine = (aa > 0.0 && bb > 0.0) ? (double)(dot / sqrtl(aa*bb)) : 1.0;
  return m;
}

static void cpu_ref(const std::vector<float>& dq, const std::vector<float>& x,
                    std::vector<double>& y, std::vector<double>& mag,
                    int S, int I, int O) {
  y.assign((size_t)S*O, 0.0); mag.assign((size_t)S*O, 0.0);
  for (int s = 0; s < S; ++s) {
    const float *xr = x.data() + (size_t)s * I;
    for (int o = 0; o < O; ++o) {
      const float *wr = dq.data() + (size_t)o * I;
      double a = 0.0, m = 0.0;
      for (int i = 0; i < I; ++i) {
        double t = (double)wr[i] * xr[i]; a += t; m += std::fabs(t);
      }
      y[(size_t)s*O+o] = a; mag[(size_t)s*O+o] = m;
    }
  }
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n & 1 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

static double now_ms() {
  static mach_timebase_info_data_t tb = {0,0};
  if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom / 1.0e6;
}

static const char *format_name(Format f) {
  switch (f) {
    case Format::MX_ROW: return "MXFP4-row-fmt7";
    case Format::MX_TILE: return "MXFP4-Apple8-tile";
    case Format::A4S64: return "A4S64";
    case Format::A4S32: return "A4S32";
    case Format::A4S16: return "A4S16";
  }
  return "?";
}

#endif
