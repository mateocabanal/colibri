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

int main(void){
	test_padded_vocab_selection();
	test_lifetime_exits();
	test_hostile_shapes();
	test_gdn_recurrence_conv_and_reset();
	test_gated_attention();
	test_request_reset_and_stop_feedback();
	if (failures) {
		fprintf(stderr, "qwen_moe: %d failure(s)\n", failures);
		return 1;
	}
	puts("qwen_moe: kernels, state, selection, lifetime, and hostile shapes ok");
	return 0;
}
