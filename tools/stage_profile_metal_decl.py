#!/usr/bin/env python3
from pathlib import Path

p = Path("c/backend_metal.h")
s = p.read_text()
anchor = "int  coli_metal_mem_info(size_t *used_bytes, size_t *total_bytes);\n"
addition = anchor + "\n/* Generic cumulative backend timing hooks. Engines opt in at runtime through\n * their profile adapter; counters stay disabled on the normal hot path. */\nvoid coli_metal_profile_set_on(int on);\nvoid coli_metal_profile_reset(void);\nvoid coli_metal_profile_get(uint64_t *encode_ns, uint64_t *submit_ns,\n                            uint64_t *wait_ns, uint64_t *kernel_ns);\n"
if "void coli_metal_profile_set_on(int on);" not in s:
    if anchor not in s:
        raise SystemExit("missing Metal profiler API anchor")
    s = s.replace(anchor, addition, 1)
p.write_text(s)
Path("tools/stage_profile_metal_decl.py").unlink()
