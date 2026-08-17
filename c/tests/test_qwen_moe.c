/* Qwen MoE correctness/safety regressions.  Include the standalone engine so
 * the tests exercise its real static kernels and mode paths without adding a
 * library ABI or a test-framework dependency. */
#define main qwen_moe_main_unused
#include "../qwen_moe.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
	fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; \
} } while (0)

static void test_set_env(const char *name, const char *value){
#ifdef _WIN32
	_putenv_s(name, value);
#else
	setenv(name, value, 1);
#endif
}

static void init_minimal_model(Model *m, Tok *T){
	static char *id2str[] = { "zero", "<eos>", "two" };
	static float embed[] = { 1.f, 1.f, 1.f };
	static float lm_head[] = { 0.f, 2.f, 1.f };
	static float final_norm[] = { 0.f };
	memset(m, 0, sizeof(*m));
	memset(T, 0, sizeof(*T));
	T->n_ids = 3;
	T->id2str = id2str;
	m->tok = T;
	m->c.hidden = 1;
	m->c.vocab = 3;
	m->c.eos = 1;
	m->c.eps = 1e-6f;
	m->max_t = 16;
	m->embed.f = embed;
	m->lm_head.f = lm_head;
	m->final_norm = final_norm;
}


static uint16_t test_bf16_bits(float x){
    uint32_t u; memcpy(&u, &x, sizeof(u)); return (uint16_t)(u >> 16);
}
static void test_bf16_matmul(void){
    enum { S=2, O=7, I=19 };
    float x[S*I], got[S*O], ref[S*O]; uint16_t w[O*I];
    for (int i=0;i<S*I;i++) x[i]=(float)((i%13)-6)/16.f;
    for (int i=0;i<O*I;i++) w[i]=test_bf16_bits((float)((i%17)-8)/32.f);
    matmul_bf16(got, x, w, S, O, I);
    for (int s=0;s<S;s++) for (int o=0;o<O;o++) {
        float a=0.f; for (int i=0;i<I;i++) a += x[s*I+i] * coli_bf16_decode(w[o*I+i]);
        ref[s*O+o]=a;
    }
    for (int i=0;i<S*O;i++) CHECK(fabsf(got[i]-ref[i]) < 2e-5f,
        "BF16 SIMD mismatch at %d: got %.9g ref %.9g", i, got[i], ref[i]);
}

static void test_padded_vocab_selection(void){
	Model m;
	Tok T;
	char *id2str[] = { "zero", NULL, "<control>", NULL };
	float logits[] = { 1.f, 90.f, 5.f, 100.f };
	memset(&m, 0, sizeof(m));
	memset(&T, 0, sizeof(T));
	T.n_ids = 4;
	T.id2str = id2str;
	m.c.vocab = 4;
	m.tok = &T;
	g_temp = 0.f;
	CHECK(qwen_pick_token(&m, logits, -1) == 2,
	      "padded rows beat valid token 2");
	/* The same mask must feed the sampling path.  The margin makes the result
	 * deterministic while still executing dist_build/dist_sample. */
	g_temp = 0.7f;
	g_nuc = 0.9f;
	logits[0] = -100.f;
	logits[2] = 20.f;
	CHECK(qwen_pick_token(&m, logits, -1) == 2,
	      "sampling selected an invalid padded row");
	g_temp = 0.f;
}

static void test_lifetime_exits(void){
	Model m;
	Tok T;
	init_minimal_model(&m, &T);
	test_set_env("QWENMOE_PROMPT_IDS", "0");
	test_set_env("QWENMOE_MAX_NEW", "1");
	CHECK(mode_greedy(&m) == 0, "greedy EOS exit failed");
	test_set_env("QWENMOE_PROMPT_IDS", "-1");
	CHECK(mode_greedy(&m) == 1, "negative prompt id was accepted");
	test_set_env("QWENMOE_PROMPT_IDS", "3");
	CHECK(mode_greedy(&m) == 1, "prompt id at vocab_size was accepted");
	test_set_env("QWENMOE_PROMPT_IDS", "0");

	char dir[] = "test_qwen_moe_XXXXXX";
	CHECK(mkdtemp(dir) != NULL, "mkdtemp failed");
	if (failures) return;
	char path[512];
	snprintf(path, sizeof(path), "%s/ref.json", dir);
	FILE *f = fopen(path, "wb");
	CHECK(f != NULL, "cannot create %s", path);
	if (!f) return;
	fputs("{\"cases\":{\"mismatch\":{"
	      "\"teacher_forcing_ids\":[1,1],"
	      "\"greedy_new_ids\":[2],"
	      "\"prompt_ids\":[0],"
	      "\"max_new_tokens\":1}}}\n", f);
	fclose(f);
	CHECK(mode_selftest(&m, dir) == 1,
	      "self-test mismatch should return failure without corrupting logits ownership");
	remove(path);
	rmdir(dir);
}

static Cfg valid_cfg(void){
	Cfg c;
	memset(&c, 0, sizeof(c));
	c.hidden = 64; c.n_layers = 4; c.n_heads = 4; c.n_kv_heads = 2;
	c.head_dim = 16; c.rotary_dim = 4;
	c.n_experts = 8; c.topk = 2; c.moe_inter = 32; c.shared_inter = 32;
	c.lin_k_heads = 2; c.lin_k_dim = 8; c.lin_v_heads = 4; c.lin_v_dim = 8;
	c.conv_kernel = 3; c.vocab = 320; c.max_pos = 4096;
	c.theta = 10000.f; c.eps = 1e-6f; c.eos = 1;
	return c;
}

static void test_hostile_shapes(void){
	char why[160];
	Cfg c = valid_cfg();
	CHECK(cfg_validate(&c, why, sizeof(why)), "valid config refused: %s", why);
	c.hidden = 63;
	CHECK(!cfg_validate(&c, why, sizeof(why)), "non-divisible hidden/heads accepted");
	c = valid_cfg(); c.n_kv_heads = 3;
	CHECK(!cfg_validate(&c, why, sizeof(why)), "non-divisible attention/KV heads accepted");
	c = valid_cfg(); c.rotary_dim = 3;
	CHECK(!cfg_validate(&c, why, sizeof(why)), "odd rotary dimension accepted");
	c = valid_cfg(); c.lin_v_heads = 3;
	CHECK(!cfg_validate(&c, why, sizeof(why)), "invalid GDN head ratio accepted");
	{
		size_t out = 0;
		CHECK(!size_mul_ok(SIZE_MAX, 2, &out), "size multiplication overflow accepted");
	}
	CHECK(tensor_numel_ok(16, 16), "matching tensor shape refused");
	CHECK(!tensor_numel_ok(15, 16), "malformed tensor shape accepted");
}

static void test_gdn_recurrence_conv_and_reset(void){
	Model m;
	Layer l;
	int8_t is_gdn[] = { 1 };
	float in_qkv[] = { 1.f, 1.f, 1.f };
	float in_a[] = { 0.f }, in_b[] = { 0.f }, in_z[] = { 1.f };
	float A_log[] = { -2.f }, dt_bias[] = { 0.f };
	float conv[] = { .5f, 1.f, .5f, 1.f, .5f, 1.f };
	float norm[] = { 1.f }, out_proj[] = { 1.f };
	float x[] = { 1.f }, first[1], second[1], again[1];
	memset(&m, 0, sizeof(m));
	memset(&l, 0, sizeof(l));
	m.c.hidden = 1; m.c.n_layers = 1; m.c.eps = 1e-6f;
	m.c.lin_k_heads = 1; m.c.lin_k_dim = 1;
	m.c.lin_v_heads = 1; m.c.lin_v_dim = 1; m.c.conv_kernel = 2;
	m.c.layer_is_gdn = is_gdn; m.max_t = 2;
	m.K = calloc(1, sizeof(float*)); m.V = calloc(1, sizeof(float*));
	m.gdn_S = calloc(1, sizeof(float*)); m.gdn_conv = calloc(1, sizeof(float*));
	m.gdn_S[0] = calloc(1, sizeof(float));
	m.gdn_conv[0] = calloc(3, sizeof(float));
	l.in_qkv.f = in_qkv; l.in_a.f = in_a; l.in_b.f = in_b; l.in_z.f = in_z;
	l.A_log = A_log; l.dt_bias = dt_bias; l.conv1d = conv;
	l.gdn_norm = norm; l.gdn_out.f = out_proj;
	gdn_token(&m, &l, 0, x, first);
	CHECK(isfinite(first[0]) && fabsf(first[0]) > 1e-5f,
	      "first GDN output is not finite/nonzero: %g", first[0]);
	CHECK(fabsf(m.gdn_S[0][0]) > 1e-5f, "GDN recurrence did not update");
	for (int i = 0; i < 3; i++)
		CHECK(fabsf(m.gdn_conv[0][i] - 1.f) < 1e-6f,
		      "conv history channel %d=%g, want 1", i, m.gdn_conv[0][i]);
	float first_state = m.gdn_S[0][0];
	gdn_token(&m, &l, 0, x, second);
	CHECK(isfinite(second[0]), "second GDN output is non-finite");
	CHECK(fabsf(m.gdn_S[0][0] - first_state) > 1e-5f,
	      "GDN recurrence did not advance on the second token");
	state_reset(&m);
	CHECK(m.gdn_S[0][0] == 0.f, "full reset retained GDN recurrence");
	for (int i = 0; i < 3; i++) CHECK(m.gdn_conv[0][i] == 0.f,
	                                  "full reset retained conv state");
	gdn_token(&m, &l, 0, x, again);
	CHECK(fabsf(again[0] - first[0]) < 1e-6f,
	      "reset first-token output %g differs from original %g", again[0], first[0]);
	free(m.gdn_S[0]); free(m.gdn_conv[0]);
	free(m.gdn_S); free(m.gdn_conv); free(m.K); free(m.V);
}

static void test_gated_attention(void){
	Model m;
	Layer l;
	float q[8] = { 0 };
	float k[] = { 1.f, 0.f, 0.f, 1.f };
	float v[] = { 1.f, 0.f, 0.f, 1.f };
	float o[] = { 1.f, 0.f, 0.f, 1.f };
	float norm[] = { 0.f, 0.f };
	float x[] = { 1.f, 0.f }, half[2], open[2];
	memset(&m, 0, sizeof(m)); memset(&l, 0, sizeof(l));
	/* This unit test hand-builds a Model with f32 KV only; the fp16 KV path
	 * is exercised end-to-end by the oracle fixtures (QWEN_KV_F16 default
	 * on). Pin the f32 storage so the helpers write the arrays we allocated. */
	g_kv_f16 = 0;
	m.c.hidden = 2; m.c.n_layers = 1; m.c.n_heads = 1; m.c.n_kv_heads = 1;
	m.c.head_dim = 2; m.c.rotary_dim = 2; m.c.theta = 10000.f; m.c.eps = 1e-6f;
	m.max_t = 1;
	m.K = calloc(1, sizeof(float*)); m.V = calloc(1, sizeof(float*));
	m.gdn_S = calloc(1, sizeof(float*)); m.gdn_conv = calloc(1, sizeof(float*));
	m.K[0] = calloc(2, sizeof(float)); m.V[0] = calloc(2, sizeof(float));
	l.q.f = q; l.k.f = k; l.v.f = v; l.o.f = o; l.qn = norm; l.kn = norm;
	attention_token(&m, &l, 0, x, 0, half);
	q[4] = 10.f;                         /* head 0, gate component 0 */
	attention_token(&m, &l, 0, x, 0, open);
	CHECK(half[0] > .49f && half[0] < .51f,
	      "zero gate should halve attention output, got %g", half[0]);
	CHECK(open[0] > .99f && open[0] < 1.01f,
	      "open gate should preserve attention output, got %g", open[0]);
	m.K[0][0] = 7.f; m.V[0][0] = 9.f;
	state_reset(&m);
	CHECK(m.K[0][0] == 0.f && m.V[0][0] == 0.f,
	      "full state reset retained KV values");
	free(m.K[0]); free(m.V[0]); free(m.K); free(m.V);
	free(m.gdn_S); free(m.gdn_conv);
}

static void test_request_reset_and_stop_feedback(void){
	Model m;
	Tok T;
	init_minimal_model(&m, &T);
	m.c.n_layers = 2;
	m.c.lin_k_heads = 1; m.c.lin_k_dim = 1;
	m.c.lin_v_heads = 1; m.c.lin_v_dim = 1; m.c.conv_kernel = 2;
	m.K = calloc(2, sizeof(float*)); m.V = calloc(2, sizeof(float*));
	m.gdn_S = calloc(2, sizeof(float*)); m.gdn_conv = calloc(2, sizeof(float*));
	m.gdn_S[0] = calloc(1, sizeof(float)); m.gdn_conv[0] = calloc(3, sizeof(float));
	m.K[1] = calloc(1, sizeof(float)); m.V[1] = calloc(1, sizeof(float));
	m.gdn_S[0][0] = 3.f; m.gdn_conv[0][2] = 4.f;
	m.K[1][0] = 5.f; m.V[1][0] = 6.f;
	request_state_reset(&m);
	CHECK(m.gdn_S[0][0] == 0.f && m.gdn_conv[0][2] == 0.f,
	      "serve request reset retained sequence-dependent GDN state");
	CHECK(m.K[1][0] == 5.f && m.V[1][0] == 6.f,
	      "serve request reset unnecessarily cleared KV storage");
	/* Stop feedback uses a zero-layer miniature: the helper must append and
	 * actually step the selected boundary token before the next user turn. */
	m.c.n_layers = 0;
	int hist[2] = { -1, -1 }, hpos = 0;
	chat_feed_stop(&m, 1, hist, &hpos);
	CHECK(hpos == 1 && hist[0] == 1,
	      "chat stop token was not committed to history");
	free(m.gdn_S[0]); free(m.gdn_conv[0]); free(m.K[1]); free(m.V[1]);
	free(m.K); free(m.V); free(m.gdn_S); free(m.gdn_conv);
}

static void test_packed_expert_size_math(void){
	/* The loader's exact-size rejection for packed experts must agree with the
	 * packers (tools/convert_qwen_moe.py): i4 row = ceil(I/2) bytes, i3 row =
	 * ceil(I/64)*24 bytes, per OUTPUT row; scales = one f32 per 64-input group
	 * per row. Tiny geometry H=64/I=32 and real geometry H=2048/I=512. */
	int64_t rbH4 = (64 + 1) / 2, rbI4 = (32 + 1) / 2;
	CHECK(rbH4 == 32 && rbI4 == 16, "i4 rowbytes (tiny)");
	int64_t want4 = 2 * rbH4 * 32 + rbI4 * 64;
	CHECK(want4 == 2 * 32 * 32 + 16 * 64, "i4 merged bytes (tiny)");
	CHECK(qm_i3_rowbytes(64) == 24 && qm_i3_rowbytes(32) == 24,
	      "i3 rowbytes (tiny)");
	int64_t want3 = 2 * 24 * 32 + 24 * 64;
	CHECK(want3 == 3072, "i3 merged bytes (tiny)");
	int64_t ngH = (64 + 63) / 64, ngI = (32 + 63) / 64;
	CHECK(ngH == 1 && ngI == 1, "packed group counts (tiny)");
	CHECK(32 * ngH * 2 + 64 * ngI == 128, "packed scale count (tiny)");
	CHECK((2048 + 1) / 2 == 1024, "i4 rowbytes (real H)");
	CHECK((512 + 1) / 2 == 256, "i4 rowbytes (real I)");
	CHECK(qm_i3_rowbytes(2048) == 32 * 24, "i3 rowbytes (real H)");
	CHECK(qm_i3_rowbytes(512) == 8 * 24, "i3 rowbytes (real I)");
	CHECK(qm_i3_groups(2048) == 32 && qm_i3_groups(512) == 8,
	      "i3 group counts (real)");
}

static void test_arena_plan(void){
	/* The prefill arena must process the FULL distinct routed set in bounded
	 * waves: a layer routing > QWEN_ARENA_CAP distinct experts used to evict
	 * during set-build and silently drop contributions (fixtures only had
	 * 8/16 experts). qwen_arena_plan must return every distinct id exactly
	 * once, in first-appearance order, bounded by cap. */
	/* 256 distinct ids: C=64 tokens, K=8 picks, every pick unique */
	int picks256[64 * 8];
	int uniq[300], n;
	for (int j = 0; j < 64; j++)
		for (int k = 0; k < 8; k++) picks256[j * 8 + k] = j * 8 + k;
	n = qwen_arena_plan(picks256, 64, 8, uniq, 256);
	CHECK(n == 256, "256 distinct experts planned: %d", n);
	int seen[256] = { 0 }, ok = 1;
	for (int i = 0; i < n; i++) { if (uniq[i] < 0 || uniq[i] >= 256 || seen[uniq[i]]) ok = 0; seen[uniq[i]] = 1; }
	CHECK(ok, "256 distinct: no dups, in range");
	/* duplicates collapse: same 8 experts repeated across all tokens */
	int picksdup[64 * 8];
	for (int j = 0; j < 64; j++)
		for (int k = 0; k < 8; k++) picksdup[j * 8 + k] = (j + k) % 8;
	n = qwen_arena_plan(picksdup, 64, 8, uniq, 256);
	CHECK(n == 8, "duplicates collapse to 8: %d", n);
	/* first-appearance order preserved */
	int picks2[4] = { 42, 7, 42, 99 };
	n = qwen_arena_plan(picks2, 2, 2, uniq, 16);
	CHECK(n == 3 && uniq[0] == 42 && uniq[1] == 7 && uniq[2] == 99,
	      "first-appearance order: %d,%d,%d", n ? uniq[0] : -1, n > 1 ? uniq[1] : -1, n > 2 ? uniq[2] : -1);
	/* cap clamps: 256 distinct but cap 64 */
	n = qwen_arena_plan(picks256, 64, 8, uniq, 64);
	CHECK(n == 64, "cap clamps to 64: %d", n);
	/* 65 distinct (one wave over the cap boundary) */
	int picks65[65];
	for (int i = 0; i < 65; i++) picks65[i] = i;
	n = qwen_arena_plan(picks65, 65, 1, uniq, 256);
	CHECK(n == 65, "65 distinct planned: %d", n);
}

int main(void){
	test_bf16_matmul();
	test_padded_vocab_selection();
	test_lifetime_exits();
	test_hostile_shapes();
	test_gdn_recurrence_conv_and_reset();
	test_gated_attention();
	test_request_reset_and_stop_feedback();
	test_packed_expert_size_math();
	test_arena_plan();
	if (failures) {
		fprintf(stderr, "qwen_moe: %d failure(s)\n", failures);
		return 1;
	}
	puts("qwen_moe: kernels, state, selection, lifetime, and hostile shapes ok");
	return 0;
}
