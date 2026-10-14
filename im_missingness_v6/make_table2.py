"""Render Table II TeX rows from a protection JSON (portable paths)."""
import json
import sys
from _path_bootstrap import out_path

src = sys.argv[1] if len(sys.argv) > 1 else str(out_path("v6_nhanes_protection.json"))
P = json.load(open(src))
order = ["raw", "blind", "point", "prodbal", "lp_rect", "lp_box", "blk_box", "known_field", "known_task", "oracle"]
label = {
    "raw": "no erasure",
    "blind": "blind exact, product model",
    "point": "balancing at fitted rates",
    "prodbal": r"robust balancing, Thm~\ref{thm:product}",
    "lp_rect": "exact, product model, fields",
    "lp_box": "exact, joint boxes, fields",
    "blk_box": "exact, joint boxes, four-field task",
    "known_field": r"known law at $E_T$, fields",
    "known_task": r"known law at $E_T$, four-field task",
    "oracle": r"zero-leakage transport, Thm~\ref{thm:transport}",
}
cols = [("m=1,E=0.5"), ("m=1,E=1.0"), ("m=4,E=0.5"), ("m=4,E=1.0")]
lines = []
for name in order:
    cells = []
    for c in cols:
        s = P[c][name]
        cells.append("%d & %.2f & %.2f" % (round(100 * s["violation"]), s["retained"], s["complete"]))
    tpr = P["m=1,E=1.0"][name]["tpr"]
    lines.append("%s & %s & %.3f \\\\" % (label[name], " & ".join(cells), tpr))
tex = "\n".join(lines) + "\n"
_out = out_path("table2.tex")
open(_out, "w", encoding="utf-8").write(tex)
print(tex)
print("wrote", _out)
