#!/usr/bin/env python3
"""Apply PR #145's Qwen Apple8 direct slice plus M2 profiling fixes."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]

# Apply the main anchor-checked integration first.
runpy.run_path(str(ROOT / "tools" / "apply_qwen_apple8_direct.py"), run_name="__main__")

qwen = ROOT / "c" / "qwen_moe.c"
text = qwen.read_text()

# Async MetalIO moves the physical read out of load_expert().  Charge only the
# exposed use-time wait to t_expio so expert_io_ms remains an honest critical-
# path metric rather than collapsing to enqueue time.
old = '''static void expert_wait_ready(Model *m, Slot *s){
    (void)m;
#ifdef COLI_METALIO
    if (s->mio && s->mio_event > s->mio_waited) {
        metalio_wait(s->mio_event);
        s->mio_waited = s->mio_event;
    }
#endif
}
'''
new = '''static void expert_wait_ready(Model *m, Slot *s){
#ifdef COLI_METALIO
    if (s->mio && s->mio_event > s->mio_waited) {
        double began = now_s();
        if (metalio_wait(s->mio_event) == 0) {
            m->t_expio += now_s() - began;
            s->mio_waited = s->mio_event;
        }
    }
#else
    (void)m; (void)s;
#endif
}
'''
if "m->t_expio += now_s() - began;" not in text:
    if text.count(old) != 1:
        raise SystemExit("qwen_moe.c: expert_wait_ready anchor changed")
    text = text.replace(old, new, 1)

# Keep the direct-slot size arithmetic overflow-clean even though current Qwen
# dimensions are far below SIZE_MAX.
old = '''            size_t uoff = qwen_align16(gb);
            size_t doff = uoff ? qwen_align16(uoff + ub) : 0;
            size_t total = doff && db <= SIZE_MAX - doff ? doff + db : 0;
'''
new = '''            size_t uoff = qwen_align16(gb);
            size_t gu_end = uoff && ub <= SIZE_MAX - uoff ? uoff + ub : 0;
            size_t doff = gu_end ? qwen_align16(gu_end) : 0;
            size_t total = doff && db <= SIZE_MAX - doff ? doff + db : 0;
'''
if "size_t gu_end = uoff" not in text:
    if text.count(old) != 1:
        raise SystemExit("qwen_moe.c: Apple8 slot arithmetic anchor changed")
    text = text.replace(old, new, 1)

qwen.write_text(text)
print("[apple8-qwen-m2] exposed MetalIO wait accounting enabled")
print("[apple8-qwen-m2] overflow-clean direct slot packing enabled")
