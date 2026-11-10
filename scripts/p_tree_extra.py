"""Extra panels for the ID3 page: step-by-step learning, decision tracing, entropy min/max interpretation."""
import math
from dio import *
from p_tree import DATA, COLS, H, split, gain, counts, LEAF_Y, LEAF_N


def node(pg, cx, cy, label, kind="node", w=96, h=34, size=11, sw=1):
    fill = {"node": "#FFFFFF", "yes": LEAF_Y, "no": LEAF_N, "q": LIGHTGREY}[kind]
    ell = kind in ("yes", "no")
    if ell:
        return pg.ellipse(cx - w / 2, cy - h / 2, w, h, label, fill=fill, size=size, bold=True, sw=sw)
    return pg.rect(cx - w / 2, cy - h / 2, w, h, label, fill=fill, size=size, bold=True, sw=sw,
                   dashed=(kind == "q"))


def tree(pg, x, y, stage, dx=150, dy=80, w=96, paths=(), size=11):
    """Draw the Play-Tennis ID3 tree at a given growth stage (1..3). paths = [(edge-key list, colour)]."""
    pos = {"root": (x + 2 * dx, y), "Sunny": (x + dx * 0.9, y + dy), "Overcast": (x + 2 * dx, y + dy),
           "Rain": (x + dx * 3.1, y + dy), "High": (x + dx * 0.35, y + 2 * dy), "Normal": (x + dx * 1.45, y + 2 * dy),
           "Strong": (x + dx * 2.55, y + 2 * dy), "Weak": (x + dx * 3.65, y + 2 * dy)}
    colour = {}
    for keys, col in paths:
        for k in keys: colour[k] = col
    def edge(a, b, lab):
        col = colour.get(b, BLACK)
        (ax, ay), (bx, by) = pos[a], pos[b]
        pg.line([(ax, ay + 17), (bx, by - 17)], color=col, width=3 if b in colour else 1, arrow="classic")
        pg.text(lab, (ax + bx) / 2 - 40, (ay + by) / 2 - 10, 80, 16, size - 1, align="center", color=col,
                bold=b in colour)
    node(pg, *pos["root"], "Outlook?", w=w, size=size, sw=3 if "root" in colour else 1)
    edge("root", "Sunny", "Sunny"); edge("root", "Overcast", "Overcast"); edge("root", "Rain", "Rain")
    node(pg, *pos["Overcast"], "Yes", "yes", w=w * 0.8, size=size)
    if stage >= 2:
        node(pg, *pos["Sunny"], "Humidity?", w=w, size=size)
        edge("Sunny", "High", "High"); edge("Sunny", "Normal", "Normal")
        node(pg, *pos["High"], "No", "no", w=w * 0.8, size=size)
        node(pg, *pos["Normal"], "Yes", "yes", w=w * 0.8, size=size)
    else:
        node(pg, *pos["Sunny"], "? [2+,3−]", "q", w=w, size=size)
    if stage >= 3:
        node(pg, *pos["Rain"], "Wind?", w=w, size=size)
        edge("Rain", "Strong", "Strong"); edge("Rain", "Weak", "Weak")
        node(pg, *pos["Strong"], "No", "no", w=w * 0.8, size=size)
        node(pg, *pos["Weak"], "Yes", "yes", w=w * 0.8, size=size)
    else:
        node(pg, *pos["Rain"], "? [3+,2−]", "q", w=w, size=size)


def fmt_split(rows, a):
    parts = split(rows, a)
    n = len(rows)
    segs, after = [], 0
    for v, sub in parts.items():
        y = sum(r[-1] == "Yes" for r in sub)
        segs.append(f"{v} [{y}+,{len(sub) - y}−] H={H(sub):.3f}")
        after += len(sub) / n * H(sub)
    return segs, after


def build(pg: Page, cat):
    Y0 = pg.max_y() + 60

    # ---------------------------------------------------------------- ⑥ step-by-step learning
    x0, y0 = 400, Y0
    pg.panel(x0, y0, 2000, 680, 6, "How ID3 learns, step by step (categorical data → tree)",
             "At each node: compute entropy of the subset, compute IG of every remaining feature, split on the best, repeat")
    steps = [("Step 1 · Root: all 14 days", DATA, COLS, 1, "Outlook"),
             ("Step 2 · Branch Sunny (5 days)", [r for r in DATA if r[0] == "Sunny"], COLS[1:], 2, "Humidity"),
             ("Step 3 · Branch Rain (5 days)", [r for r in DATA if r[0] == "Rain"], COLS[1:], 3, "Wind")]
    for k, (title, rows, feats, stage, best) in enumerate(steps):
        sx = x0 + 20 + k * 660
        pg.text(title, sx, y0 + 66, 620, 22, 14, bold=True, color=BLUE)
        y = sum(r[-1] == "Yes" for r in rows)
        if k == 0:
            pg.html(f"\\(H(S) = -\\frac{{9}}{{14}}\\log_2\\frac{{9}}{{14}} - \\frac{{5}}{{14}}\\log_2\\frac{{5}}{{14}} = "
                    f"{H(rows):.3f}\\)", sx, y0 + 92, 620, 30, 13, valign="middle")
            tab_y = y0 + 128
        else:
            sub = [["Day", "Temp", "Humidity", "Wind", "Play"]]
            for r in rows:
                sub.append([DATA.index(r) + 1] + r[1:4] + [f'<font color="{"#009900" if r[-1] == "Yes" else "#FF0000"}">{r[-1]}</font>'])
            pg.table(sx, y0 + 92, 330, sub, size=10, h=120)
            pg.html(f"\\(H(S_{{{rows[0][0]}}}) = -\\frac{{{y}}}{{5}}\\log_2\\frac{{{y}}}{{5}} - "
                    f"\\frac{{{5 - y}}}{{5}}\\log_2\\frac{{{5 - y}}}{{5}} = {H(rows):.3f}\\)", sx + 340, y0 + 120, 300, 50,
                    12, valign="middle")
            tab_y = y0 + 222
        # gains per feature
        gy = tab_y
        for f in feats:
            segs, after = fmt_split(rows, f)
            g = H(rows) - after
            hl = f == best
            lab = (f"<b>{f}</b>: " + " · ".join(segs) +
                   f"<br>after = {after:.3f} → <b>IG = {H(rows):.3f} − {after:.3f} = "
                   f"{'<span style=\"background:#FFFF00\">' if hl else ''}{g:.3f}{'</span>' if hl else ''}</b>")
            pg.html(lab, sx, gy, 630, 40, 10, valign="top")
            gy += 44
        pg.text(f"→ split on {best}" + (" (both children pure → leaves)" if k else " (Overcast is pure → leaf Yes)"),
                sx, gy + 2, 630, 18, 11, bold=True, color=RED)
        tree(pg, sx + 10, y0 + 450, stage, dx=150, dy=80, w=100, size=10)
        if k < 2:
            pg.line([(sx + 625, y0 + 530), (sx + 655, y0 + 530)], arrow="classic", width=2, color=GREY)
    pg.text("Tree after each step (dashed grey = impure node still to be split)", x0 + 20, y0 + 405, 900, 18, 12,
            italic=True, color="#555555")
    pg.ask("Which question should the tree ask first?",
           f"the feature with the largest IG at the root: Outlook ({gain(DATA, 'Outlook'):.3f}); then repeat on each impure branch",
           x0 + 20, y0 + 332, x0 + 120, y0 + 324, w=620)

    # ---------------------------------------------------------------- ⑦ making a decision
    x0, y0 = 400, Y0 + 720
    pg.panel(x0, y0, 980, 800, 7, "Making a Decision: new days travel down the tree",
             "Each test sends the sample to one branch until it reaches a leaf = prediction")
    tree(pg, x0 + 60, y0 + 110, 3, dx=210, dy=120, w=120, size=12,
         paths=[(["root", "Sunny", "Normal"], GREEN), (["Rain", "Strong"], RED), (["Overcast"], BLUE)])
    pg.table(x0 + 40, y0 + 400, 900, [
        ["Day", "Outlook", "Temp", "Humidity", "Wind", "Path", "Prediction"],
        ['<font color="#009900"><b>D15</b></font>', "Sunny", "Hot", "Normal", "Strong",
         "Outlook = Sunny → Humidity = Normal", '<font color="#009900"><b>Yes</b></font>'],
        ['<font color="#FF0000"><b>D16</b></font>', "Rain", "Mild", "High", "Strong",
         "Outlook = Rain → Wind = Strong", '<font color="#FF0000"><b>No</b></font>'],
        ['<font color="#3333FF"><b>D17</b></font>', "Overcast", "Cool", "High", "Weak",
         "Outlook = Overcast → leaf", '<font color="#3333FF"><b>Yes</b></font>'],
        ["D18", "<b>Foggy</b> (unseen)", "Mild", "High", "Weak",
         "no branch for 'Foggy' → majority class at the node [9+, 5−]", "<b>Yes</b>"]], size=11, h=130)
    pg.bullets(x0 + 20, y0 + 550, 940, [
        "Only the features on the path are used: D15's Temp = Hot and Wind = Strong are <b>ignored</b>",
        "Prediction cost = depth of the tree (here ≤ 2 tests) → very fast",
        "Unseen category value → fall back to the node's majority class (or C4.5 distributes the sample over branches)",
        "Every root-to-leaf path is a readable IF–THEN rule (see ⑤)"], head="Notes:")
    pg.ask("What does the model predict for a new day, and why?",
           "follow one path: D16 (Rain, Strong wind) → Wind = Strong → No; Temp and Humidity are never looked at",
           x0 + 20, y0 + 680, x0 + 940, y0 + 465, w=940)

    # ---------------------------------------------------------------- ⑧ entropy min / max & interpretation
    x0 = 1420
    pg.panel(x0, y0, 980, 800, 8, "Entropy: minimum, maximum & how to interpret it")
    hfun = lambda p: 0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    p = pg.axes(x0 + 70, y0 + 80, 380, 210, (0, 1), (0, 1.1), "p", "H(p)", label_size=12,
                xticks=(0, 0.25, 0.5, 0.75, 1), yticks=(0.5, 1))
    p.curve(hfun, 0.0005, 0.9995, n=80, color=BLUE, width=2)
    p.curve(lambda q: 2 * q * (1 - q), 0, 1, n=60, color=GREY, dashed=True)
    p.scatter([(0.5, 1)], color=RED, r=5)
    p.scatter([(0, 0), (1, 0)], color=GREEN, r=5)
    p.text("max = 1 (50/50, most uncertain)", 0.5, 1.0, dy=-14, w=200, size=10, color=RED)
    p.text("min = 0 (pure)", 0.0, 0, dx=50, dy=-12, w=100, size=10, color=GREEN)
    p.text("Gini", 0.82, 0.3, size=10, color=GREY)
    rows = [["p (Yes)", "0", "0.1", "0.25", "0.5", "0.75", "0.9", "1"],
            ["H(p)"] + [f"{hfun(q):.3f}" for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)],
            ["Gini"] + [f"{2 * q * (1 - q):.3f}" for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)]]
    pg.table(x0 + 30, y0 + 320, 440, rows, size=10, h=70)
    # max entropy vs K
    q = pg.axes(x0 + 560, y0 + 80, 360, 210, (0, 5), (0, 3.3), "K", "", label_size=12,
                yticks=(1, 2, 3))
    q.bars([(i - 0.17, round(math.log2(K), 2)) for i, K in zip((1, 2, 3, 4), (2, 3, 4, 8))], 0.32, fill=BLUE, labels=True)
    q.bars([(i + 0.17, round(1 - 1 / K, 2)) for i, K in zip((1, 2, 3, 4), (2, 3, 4, 8))], 0.32, fill=GREY, labels=True)
    for i, K in zip((1, 2, 3, 4), (2, 3, 4, 8)):
        q.text(f"K={K}", i, 0, dy=26, size=10)
    pg.html('<font color="#3333FF"><b>■ max entropy = log₂K</b></font> &nbsp; <font color="#808080"><b>■ max Gini = 1 − 1/K</b></font>',
            x0 + 560, y0 + 52, 380, 20, 11)
    pg.math("0 \\le H(S) \\le \\log_2 K \\qquad 0 \\le Gini \\le 1-\\tfrac{1}{K} \\qquad 0 \\le IG \\le H(parent)",
            x0 + 470, y0 + 330, 490, 36, 13)
    pg.table(x0 + 30, y0 + 410, 520, [
        ["Entropy (2 classes)", "Meaning", "At a node"],
        ["0", "pure: all samples one class", "make a leaf"],
        ["0 – 0.3", "almost pure", "leaf or 1 more split"],
        ["0.3 – 0.8", "mixed", "keep splitting"],
        ["0.8 – 1", "maximally mixed (≈ coin flip)", "feature needed"]], size=10, h=120)
    pg.text("Small vs big information gain (root, H = 0.940)", x0 + 580, y0 + 400, 380, 18, 12, bold=True)
    for k, (f, col) in enumerate([("Wind", RED), ("Outlook", GREEN)]):
        g = gain(DATA, f)
        parts = split(DATA, f)
        yy = y0 + 430 + k * 100
        pg.text(f"{f}: IG = {g:.3f}", x0 + 580, yy, 170, 18, 11, bold=True, color=col)
        cx = x0 + 760
        for v, sub in parts.items():
            yv = sum(r[-1] == "Yes" for r in sub)
            pg.rect(cx, yy + 22, 62, 40, f"{v}<br>[{yv}+,{len(sub) - yv}−]<br>H={H(sub):.2f}", size=9, raw=True,
                    fill=LEAF_Y if H(sub) == 0 else "#FFFFFF")
            cx += 68
        pg.text("children still mixed → little learned" if k == 0 else "one child pure, others less mixed → useful split",
                x0 + 580, yy + 64, 380, 16, 10, italic=True, color=col)
    pg.bullets(x0 + 20, y0 + 590, 940, [
        "Entropy is measured in <b>bits</b>: H = 1 means you need one yes/no question to know the class",
        "IG = 0 → the feature tells nothing about the target;  IG = H(parent) → the split makes every child pure",
        "Log base 2 for bits; natural log (nats) only rescales — the chosen split stays the same"], head="Interpretation:")
    pg.ask("Is this node pure enough to stop splitting?",
           "read its entropy: 0 = pure → make a leaf; ≈ 1 (2 classes) = coin flip → keep splitting; the maximum "
           "grows to log₂K for K classes",
           x0 + 20, y0 + 695, x0 + 300, y0 + 531, w=940)
