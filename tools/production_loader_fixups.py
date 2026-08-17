#!/usr/bin/env python3
from pathlib import Path


def rep(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count))


rep(
    "c/coli_v4_expert_store.c",
    "static int runtime_target_for_profile(const char *profile, ColiRuntimeTarget *runtime,",
    "int coli_v4_runtime_target_for_profile(const char *profile, ColiRuntimeTarget *runtime,",
)
rep(
    "c/coli_v4_expert_store.c",
    "if(runtime_target_for_profile(o->required_profile,&runtime,e,n))goto bad;",
    "if(coli_v4_runtime_target_for_profile(o->required_profile,&runtime,e,n))goto bad;",
)

p = Path("c/coli_v4_expert_store.h")
t = p.read_text()
if '#include "coli_target.h"' not in t:
    t = t.replace('#include "expert_store.h"', '#include "expert_store.h"\n#include "coli_target.h"', 1)
anchor = "int coli_v4_coli_expert_store_open("
if anchor not in t:
    raise SystemExit("expert-store declaration anchor missing")
decl = '''int coli_v4_runtime_target_for_profile(const char *profile,
                                           ColiRuntimeTarget *runtime,
                                           char *error, size_t error_size);

'''
t = t.replace(anchor, decl + anchor, 1)
p.write_text(t)

old = '''    if (coli_dir && coli_executor_open(&engine->coli_static, coli_dir,
            &(ColiExecutorOpenOptions){"macos-arm64-metal-apple8-v1", coli_checksum_policy, 0}, error, error_size)) goto fail;'''
new = '''    if (coli_dir) {
        ColiRuntimeTarget runtime_target;
        ColiExecutorOpenOptions executor_options = {0};
        if (coli_v4_runtime_target_for_profile("macos-arm64-metal-apple8-v1",
                                               &runtime_target, error, error_size))
            goto fail;
        executor_options.required_profile = "macos-arm64-metal-apple8-v1";
        executor_options.runtime_target = &runtime_target;
        executor_options.checksum_policy = coli_checksum_policy;
        if (coli_executor_open(&engine->coli_static, coli_dir,
                               &executor_options, error, error_size))
            goto fail;
    }'''
rep("c/deepseek_v4.c", old, new)
