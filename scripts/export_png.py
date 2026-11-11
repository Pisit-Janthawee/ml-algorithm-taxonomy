"""Export every page of a .drawio file to PNG (one file per page) via headless Chrome + draw.io viewer.
usage: python -I export_png.py file.drawio out_dir [--scale S]
"""
import sys, os, re, json, html, subprocess, tempfile, xml.etree.ElementTree as ET

args = sys.argv[1:]
SCALE = 1.0
if "--scale" in args:
    k = args.index("--scale"); SCALE = float(args[k + 1]); del args[k:k + 2]
src, out_dir = os.path.abspath(args[0]), os.path.abspath(args[1])
os.makedirs(out_dir, exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MARGIN = 40
MAX_PX = 16000  # Chrome screenshot limit per side


def bounds(model):
    cells = {}
    for e in model.find("root"):
        c = e if e.tag == "mxCell" else e.find("mxCell")
        cells[e.get("id") or c.get("id")] = c

    def absxy(i):
        c = cells[i]; g = c.find("mxGeometry")
        x = float(g.get("x", 0)) if g is not None else 0
        y = float(g.get("y", 0)) if g is not None else 0
        p = c.get("parent")
        if p in cells and p not in ("0", "1") and cells[p].get("vertex") == "1":
            px, py = absxy(p); return x + px, y + py
        return x, y

    xs, ys = [], []
    for i, c in cells.items():
        g = c.find("mxGeometry")
        if g is None: continue
        if c.get("vertex") == "1":
            x, y = absxy(i)
            xs += [x, x + float(g.get("width", 0))]; ys += [y, y + float(g.get("height", 0))]
        elif c.get("edge") == "1":
            for p in g.iter("mxPoint"):
                xs.append(float(p.get("x", 0))); ys.append(float(p.get("y", 0)))
    return min(xs), min(ys), max(xs), max(ys)


def slug(name):
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


root = ET.parse(src).getroot()
for n, d in enumerate(root.iter("diagram")):
    model = d.find("mxGraphModel")
    x0, y0, x1, y1 = bounds(model)
    w, h = x1 - x0 + 2 * MARGIN, y1 - y0 + 2 * MARGIN
    s = min(SCALE, MAX_PX / w, MAX_PX / h)
    W, H = int(w * s) + 1, int(h * s) + 1
    xml = ET.tostring(model, encoding="unicode")
    cfg = {"highlight": "#0000ff", "nav": False, "resize": False, "toolbar": "", "xml": xml,
           "lightbox": False, "zoom": s, "border": MARGIN}
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#fff}}</style></head>
<body><div class="mxgraph" style="width:{W}px;height:{H}px" data-mxgraph="{html.escape(json.dumps(cfg))}"></div>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>"""
    out = os.path.join(out_dir, f"{n:02d}_{slug(d.get('name', f'page{n}'))}.png")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(page); tmp = f.name
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                        "--virtual-time-budget=20000", f"--screenshot={out}", "file:///" + tmp.replace("\\", "/")],
                       check=True, capture_output=True, timeout=180)
    finally:
        os.remove(tmp)
    print(f"{out}  ({W}x{H}, scale {s:.2f})")
