from pathlib import Path

h = Path('c/qwen_prefix_cache.h')
s = h.read_text()
if '#define QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB 256u\n' not in s:
    s = s.replace('#define QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS 256\n',
                  '#define QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS 256\n#define QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB 256u\n')
old = '''static inline size_t qwen_prefix_cache_budget_from_env(void) {
    const char *value = getenv("QWEN_PREFIX_CACHE_MB");
    if (!value || !*value) return 0;
    char *end = NULL;
    double mib = strtod(value, &end);
    if (end == value || mib <= 0.0) return 0;
    long double bytes = (long double)mib * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}
'''
new = '''static inline int qpc_ascii_ieq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static inline size_t qwen_prefix_cache_budget_parse(const char *value,
                                                     size_t fallback_bytes) {
    if (!value || !*value) return fallback_bytes;
    if (qpc_ascii_ieq(value, "off")) return 0;
    char *end = NULL;
    long double mib = strtold(value, &end);
    if (end == value || !(mib > 0.0L)) return 0;
    while (*end == ' ' || *end == '\\t' || *end == '\\r' || *end == '\\n') end++;
    if (*end) return 0;
    long double bytes = mib * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}

static inline size_t qwen_prefix_cache_budget_from_env(void) {
    return qwen_prefix_cache_budget_parse(getenv("QWEN_PREFIX_CACHE_MB"), 0);
}

static inline size_t qwen_prefix_cache_budget_for_serve(void) {
    const size_t fallback = (size_t)QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB * 1024u * 1024u;
    return qwen_prefix_cache_budget_parse(getenv("QWEN_PREFIX_CACHE_MB"), fallback);
}
'''
if old in s:
    s = s.replace(old, new)
elif 'qwen_prefix_cache_budget_for_serve' not in s:
    raise SystemExit('budget parser block not found')
h.write_text(s)

q = Path('c/qwen_moe.c')
s = q.read_text()
s = s.replace('size_t prefix_budget = serving ? qwen_prefix_cache_budget_from_env() : 0;',
              'size_t prefix_budget = serving ? qwen_prefix_cache_budget_for_serve() : 0;', 1)
old = '    Model m; model_init(&m, snap, cap);\n    Tok T;'
new = '''    Model m; model_init(&m, snap, cap);
    if (serving) {
        qwen_prefix_cache_init(&m.prefix_cache, prefix_budget,
                               qwen_prefix_cache_min_tokens_from_env(),
                               getenv("QWEN_PREFIX_LOG") != NULL);
        if (m.prefix_cache.log)
            fprintf(stderr,
                    "[QWEN-PREFIX] budget=%.2fMiB min_tokens=%d mode=process-local-exact-hybrid%s\\n",
                    (double)m.prefix_cache.budget_bytes / (1024.0 * 1024.0),
                    m.prefix_cache.min_tokens,
                    getenv("QWEN_PREFIX_CACHE_MB") ? " explicit" : " default-on");
    }
    Tok T;'''
if old in s:
    s = s.replace(old, new, 1)
elif 'mode=process-local-exact-hybrid%s' not in s:
    raise SystemExit('model init insertion point not found')
q.write_text(s)

t = Path('c/tests/test_qwen_prefix_cache.c')
s = t.read_text()
if 'static void test_budget_policy(void)' not in s:
    insert = '''
static void test_budget_policy(void) {
    const size_t mib = 1024u * 1024u;
    const size_t fallback = 256u * mib;
    assert(qwen_prefix_cache_budget_parse(NULL, fallback) == fallback);
    assert(qwen_prefix_cache_budget_parse("", fallback) == fallback);
    assert(qwen_prefix_cache_budget_parse("0", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("off", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("OFF", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("64", fallback) == 64u * mib);
    assert(qwen_prefix_cache_budget_parse("garbage", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("64garbage", fallback) == 0);
}
'''
    s = s.replace('\nstatic void test_hard_budget(void) {', insert + '\nstatic void test_hard_budget(void) {', 1)
    s = s.replace('    test_ram_cap_reservation();\n    test_hard_budget();',
                  '    test_ram_cap_reservation();\n    test_budget_policy();\n    test_hard_budget();')
t.write_text(s)
