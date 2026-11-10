"""Assemble 'ML Algorithm Taxonomy.drawio' (overview + one page per category, original hand-drawn style).
usage: python build.py OUT.drawio
"""
import importlib, sys, os, traceback
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dio import Page, E, HL, BLACK, GREY
from taxdata import CATS, PARADIGMS, header, link


def overview():
    p = Page("0. Overview - All Algorithms", "overview")
    COLW, GAP, PGAP = 210, 24, 56
    x, cols, span = 0, {}, {}
    for par in PARADIGMS:
        cs = [c for c in CATS if c["par"] == par]
        for c in cs:
            cols[c["id"]] = x
            s = span.setdefault(par, [x, x]); s[1] = x + COLW
            x += COLW + GAP
        x += PGAP - GAP
    total = x - PGAP
    nalg = sum(len(a) for c in CATS for _, a in c["groups"])
    p.title("Machine Learning Algorithm Taxonomy", 0, 0, total, 34, align="center")
    p.text(f"{nalg} entries · {len(CATS)} categories · ★ yellow = base (root) algorithm of each category · "
           "click a category title to open its page", 0, 58, total, 20, 13, align="center", color="#555555")
    root = p.rect(total / 2 - 170, 110, 340, 54, "Machine Learning Algorithms", size=20, bold=True, sw=2)
    pids = {}
    for par in PARADIGMS:
        a, b = span[par]
        pids[par] = p.rect(a, 230, b - a, 44, par, size=16, bold=True, sw=1.5)
        p.connect(root, pids[par], arrow=None)
    bottom = 0
    for c in CATS:
        cx = cols[c["id"]]
        hid = p.rect(cx, 330, COLW, 50, c["name"], size=13, bold=True, link=link(c["id"]))
        p.connect(pids[c["par"]], hid, arrow=None)
        y = 392
        for gname, algs in c["groups"]:
            p.text(gname, cx, y, COLW, 18, 10, bold=True, color="#555555")
            y += 20
            for name, _ in algs:
                base = name == c["pick"]
                p.text(("★ " if base else "• ") + name, cx + 8, y, COLW - 8, 18, 11, bold=base,
                       bg=HL if base else None, link=link(c["id"]) if base else None)
                y += 20
            y += 6
        p.rect(cx - 6, 386, COLW + 12, y - 386, stroke="#BBBBBB", dashed=True, sw=0.5)
        bottom = max(bottom, y)
    # base-algorithm summary table
    y0 = bottom + 60
    p.text("Base (root) algorithm of each category", 0, y0, 700, 26, 16, bold=True)
    rows = [["Paradigm", "Category", "★ Base algorithm"]] + \
           [[c["par"], c["name"], f'<span style="background:{HL}">{E(c["pick"])}</span>'] for c in CATS]
    p.table(0, y0 + 32, 760, rows, size=12)
    return p


def category_page(i, c):
    pg = Page(f"{i}. {c['pick'] if c['id'] != 'distributions' else c['name']}", c["id"])
    header(pg, c)
    try:
        mod = importlib.import_module(f"p_{c['id']}")
        mod.build(pg, c)
        import glob
        here = os.path.dirname(os.path.abspath(__file__))
        for f in sorted(glob.glob(os.path.join(here, f"p_{c['id']}_extra*.py"))):   # _extra, _extra2, ...
            importlib.import_module(os.path.splitext(os.path.basename(f))[0]).build(pg, c)
    except ModuleNotFoundError as ex:
        if ex.name != f"p_{c['id']}": raise
        pg.text("(drawing not built yet)", 400, 260, 400, 30, 16, italic=True, color=GREY)
    except Exception:
        if os.environ.get("STRICT"): raise
        print(f"!! page {c['id']} failed:\n" + traceback.format_exc(), file=sys.stderr)
        pg = Page(f"{i}. {c['pick']}", c["id"])
        header(pg, c)
        pg.text("(page failed to build)", 400, 260, 400, 30, 16, italic=True, color=GREY)
    return pg


import re
_CTRL = {"\a": "\\a", "\b": "\\b", "\f": "\\f", "\v": "\\v", "\r": "\\r"}


def sanitize(el):
    """Repair LaTeX written in non-raw Python strings (e.g. "\\alpha" typed as "\alpha" -> BEL char)."""
    n = 0
    for e in el.iter():
        for k, v in e.attrib.items():
            if not any(ch in v for ch in "\a\b\f\v\r\t") and "\n" not in v: continue
            new = "".join(_CTRL.get(ch, ch) for ch in v)
            new = re.sub(r"\t(?=[a-zA-Z])", r"\\t", new)                       # \theta, \tau, \text, \times
            new = re.sub(r"\n(?=(eq|u\b|u[\s\\{_^]|abla|ot\b|ewline|eg\b|exists|i\b))", r"\\n", new)  # \neq, \nu, \nabla
            new = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", new)
            if new != v:
                e.set(k, new); n += 1
    return n


def main(out):
    mx = ET.Element("mxfile", host="app.diagrams.net", pages=str(len(CATS) + 1))
    mx.append(overview().d)
    for i, c in enumerate(CATS, 1):
        mx.append(category_page(i, c).d)
    fixed = sanitize(mx)
    if fixed: print(f"sanitized {fixed} label(s) with stray control characters")
    ET.indent(mx)
    ET.ElementTree(mx).write(out, encoding="utf-8", xml_declaration=False)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
