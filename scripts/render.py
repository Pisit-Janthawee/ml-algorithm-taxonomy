"""Render a region (or whole page) of a .drawio page to PNG via headless Chrome + draw.io viewer.
usage: render.py file.drawio page_index out.png [x0 y0 x1 y1] [--width W]
"""
import sys, json, html, subprocess, os, copy, xml.etree.ElementTree as ET

args = sys.argv[1:]
W = 1800
if "--width" in args:
    k = args.index("--width"); W = int(args[k + 1]); del args[k:k + 2]
src, pidx, out = args[0], int(args[1]), args[2]
box = list(map(float, args[3:7])) if len(args) >= 7 else None

r = ET.parse(src).getroot()
d = list(r.iter("diagram"))[pidx]
model = d.find("mxGraphModel")
root = model.find("root")
items = list(root)

def cell_of(e):
    return e if e.tag == "mxCell" else e.find("mxCell")

info = {}
for e in items:
    c = cell_of(e)
    i = e.get("id") or c.get("id")
    info[i] = (e, c)

def absxy(i):
    e, c = info[i]
    g = c.find("mxGeometry")
    x = float(g.get("x", 0)) if g is not None else 0
    y = float(g.get("y", 0)) if g is not None else 0
    p = c.get("parent")
    if p in info and p not in ("0", "1") and cell_of(info[p][0]).get("vertex") == "1":
        px, py = absxy(p); return x + px, y + py
    return x, y

if box:
    x0, y0, x1, y1 = box
    keep = set(["0", "1"])
    for i, (e, c) in info.items():
        if i in ("0", "1"): continue
        g = c.find("mxGeometry")
        if c.get("vertex") == "1" and g is not None:
            x, y = absxy(i); w = float(g.get("width", 0)); h = float(g.get("height", 0))
            if x < x1 and x + w > x0 and y < y1 and y + h > y0: keep.add(i)
    changed = True
    while changed:  # keep ancestors
        changed = False
        for i in list(keep):
            p = cell_of(info[i][0]).get("parent") if i in info else None
            if p and p not in keep: keep.add(p); changed = True
    for i, (e, c) in info.items():
        if c.get("edge") == "1":
            s, t = c.get("source"), c.get("target")
            pts = [(float(p.get("x", 0)), float(p.get("y", 0))) for p in c.iter("mxPoint")]
            if (s in keep or t in keep) or any(x0 <= px <= x1 and y0 <= py <= y1 for px, py in pts):
                keep.add(i)
    for e in items:
        i = e.get("id") or cell_of(e).get("id")
        if i not in keep: root.remove(e)

xml = ET.tostring(model, encoding="unicode")
cfg = {"highlight": "#0000ff", "nav": False, "resize": False, "toolbar": "", "xml": xml, "lightbox": False}
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#fff}}</style></head>
<body><div class="mxgraph" style="width:{W}px" data-mxgraph="{html.escape(json.dumps(cfg))}"></div>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>"""
tmp = os.path.splitext(out)[0] + ".html"
open(tmp, "w", encoding="utf-8").write(page)
H = int(sys.argv[sys.argv.index("--height") + 1]) if "--height" in sys.argv else 1400
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                "--virtual-time-budget=15000", f"--screenshot={out}", "file:///" + tmp.replace("\\", "/")],
               check=True, capture_output=True, timeout=120)
print("ok", out)
