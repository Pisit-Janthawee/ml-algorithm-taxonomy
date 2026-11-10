"""ID3 page: Play-Tennis data, entropy (probability), information gain worked example, resulting tree
(recursive partitioning, real gains), rules / gain ratio / limitations."""
import math
from collections import Counter
from dio import *

COLS = ["Outlook", "Temp", "Humidity", "Wind"]
DATA = [l.split() for l in """Sunny Hot High Weak No
Sunny Hot High Strong No
Overcast Hot High Weak Yes
Rain Mild High Weak Yes
Rain Cool Normal Weak Yes
Rain Cool Normal Strong No
Overcast Cool Normal Strong Yes
Sunny Mild High Weak No
Sunny Cool Normal Weak Yes
Rain Mild Normal Weak Yes
Sunny Mild Normal Strong Yes
Overcast Mild High Strong Yes
Overcast Hot Normal Weak Yes
Rain Mild High Strong No""".splitlines()]
LEAF_Y, LEAF_N = "#D5E8D4", "#F8CECC"


def H(rows):
    n = len(rows)
    c = Counter(r[-1] for r in rows)
    h = -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0
    return abs(h) if h > 1e-12 else 0.0


def split(rows, a):
    i = COLS.index(a)
    out = {}
    for r in rows:
        out.setdefault(r[i], []).append(r)
    return out


def gain(rows, a):
    return H(rows) - sum(len(s) / len(rows) * H(s) for s in split(rows, a).values())


def counts(rows):
    c = Counter(r[-1] for r in rows)
    return f"[{c['Yes']}+, {c['No']}−]"


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① data
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 720, 1, "Training Data: Play Tennis?", "14 days, 4 categorical features, target = Play")
    rows = [["Day"] + COLS + ["Play"]] + [[i + 1] + r[:-1] + [
        f'<font color="{GREEN if r[-1] == "Yes" else RED}"><b>{r[-1]}</b></font>'] for i, r in enumerate(DATA)]
    pg.table(x0 + 40, y0 + 70, 560, rows, size=11, h=330)
    nY = sum(r[-1] == "Yes" for r in DATA)
    pg.text("Target distribution:", x0 + 40, y0 + 415, 150, 22, 12, bold=True)
    for i in range(14):
        pg.rect(x0 + 190 + i * 26, y0 + 414, 22, 22, "Y" if i < nY else "N", size=10,
                fill=LEAF_Y if i < nY else LEAF_N, stroke=GREEN if i < nY else RED)
    pg.html(f"\\(p_{{yes}} = \\frac{{{nY}}}{{14}} = {nY/14:.3f}\\),&nbsp;&nbsp; \\(p_{{no}} = \\frac{{{14-nY}}}{{14}} = "
            f"{(14-nY)/14:.3f}\\)", x0 + 40, y0 + 448, 560, 30, 13, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 495, 600, [
        "<b>Goal:</b> learn if-then rules that predict <i>Play</i> from the weather",
        "ID3 handles <b>categorical</b> features: each split has one branch per value",
        "Question at every node: <i>which feature separates Yes / No best?</i> → Entropy ② & Information Gain ③",
        "Probability background: \\(p_k = \\frac{\\text{count of class }k}{\\text{total}}\\)"],
        head="Notes:")
    pg.ask("How mixed is the target before any split?",
           f"{nY} Yes vs {14 - nY} No (p = {nY / 14:.3f}): mostly Yes but far from pure → we need rules to separate them",
           x0 + 20, y0 + 625, x0 + 190 + nY * 26 - 2, y0 + 436, w=600)

    # ---------------------------------------------------------------- ② entropy
    x0 = 1080
    pg.panel(x0, y0, 620, 720, 2, "Entropy: measuring impurity", "Background: probability & information theory")
    pg.math("H(S) = -\\sum_{k=1}^{K} p_k \\log_2 p_k", x0 + 60, y0 + 62, 500, 46, 20)
    pg.note("proportion of class k in set S", x0 + 410, y0 + 112, x0 + 400, y0 + 92, w=180)
    p = pg.axes(x0 + 80, y0 + 150, 400, 230, (0, 1), (0, 1.1), "p_{yes}", "H", label_size=12,
                xticks=(0, 0.25, 0.5, 0.75, 1), yticks=(0.5, 1))
    ent = lambda v: 0 if v in (0, 1) else -(v * math.log2(v) + (1 - v) * math.log2(1 - v))
    p.curve(ent, 0.0005, 0.9995, n=60, color=BLUE, width=2.5)
    p.curve(lambda v: 2 * v * (1 - v), 0, 1, color=GREY, width=1.5, dashed=True)
    p.text("Entropy", 0.5, 1.0, dy=-12, size=11, color=BLUE)
    p.text("Gini = 1 − Σp²", 0.5, 0.5, dy=12, size=10, color=GREY)
    ps = nY / 14
    p.scatter([(ps, ent(ps))], color=RED, r=5)
    p.vline(ps, 0, ent(ps), color=RED)
    pg.note(f"our data: p = 9/14 → H = {ent(ps):.3f}", x0 + 420, y0 + 135, *p.P(ps, ent(ps)), w=180, color=RED)
    pg.note("pure node: H = 0", x0 + 20, y0 + 400, *p.P(0.0, 0.0), w=110)
    pg.note("50 / 50: H = 1 (max uncertainty)", x0 + 220, y0 + 128, *p.P(0.5, 1.0), w=180)
    pg.math(f"H(S) = -\\tfrac{{9}}{{14}}\\log_2\\tfrac{{9}}{{14}} - \\tfrac{{5}}{{14}}\\log_2\\tfrac{{5}}{{14}}"
            f" = {H(DATA):.3f}", x0 + 30, y0 + 420, 560, 44, 16)
    pg.bullets(x0 + 20, y0 + 480, 580, [
        "Entropy = average number of bits needed to encode the class → 0 when certain, 1 when 50/50 (2 classes)",
        "Max entropy for K classes = \\(\\log_2 K\\)",
        "CART uses <b>Gini</b> instead (no logarithm, faster, very similar shape)",
        "Regression trees use <b>variance</b> of the target instead of entropy"],
        head="Notes:")
    pg.ask("How uncertain is 'Play?' before looking at any feature?",
           f"H = {H(DATA):.3f} bits: close to the 1-bit maximum (50/50), so the label is still very uncertain",
           x0 + 20, y0 + 615, *p.P(ps, ent(ps)), w=580)

    # ---------------------------------------------------------------- ③ information gain
    x0 = 1740
    pg.panel(x0, y0, 660, 720, 3, "Information Gain (worked example)", "Split S on Outlook")
    pg.math("IG(S, A) = H(S) - \\sum_{v \\in values(A)} \\frac{|S_v|}{|S|}\\,H(S_v)", x0 + 60, y0 + 60, 540, 50, 18)
    pg.note("entropy before", x0 + 120, y0 + 118, x0 + 250, y0 + 92, w=90)
    pg.note("weighted entropy after the split", x0 + 400, y0 + 118, x0 + 470, y0 + 96, w=200)
    root = pg.rect(x0 + 250, y0 + 150, 160, 50, f"<b>S</b> {counts(DATA)}<br>H = {H(DATA):.3f}", size=12, raw=True)
    parts = split(DATA, "Outlook")
    kids = []
    for i, v in enumerate(["Sunny", "Overcast", "Rain"]):
        s = parts[v]
        cx = x0 + 60 + i * 200
        fill = LEAF_Y if H(s) == 0 else "none"
        kids.append(pg.rect(cx, y0 + 260, 150, 50, f"{counts(s)}<br>H = {H(s):.3f}", size=12, raw=True, fill=fill))
        pg.connect(root, kids[-1], orth=False)
        pg.text(v, cx + 40, y0 + 222, 80, 18, 11, bold=True, align="center", bg="#FFFFFF")
    wsum = sum(len(s) / 14 * H(s) for s in parts.values())
    pg.math(f"\\text{{after}} = \\tfrac{{5}}{{14}}({H(parts['Sunny']):.3f}) + \\tfrac{{4}}{{14}}(0) + "
            f"\\tfrac{{5}}{{14}}({H(parts['Rain']):.3f}) = {wsum:.3f}", x0 + 20, y0 + 325, 620, 40, 15)
    pg.html(f'<span style="background:{HL}">\\(IG(S, Outlook) = {H(DATA):.4f} - {wsum:.4f} = '
            f'{gain(DATA, "Outlook"):.4f}\\)</span>', x0 + 120, y0 + 368, 420, 32, 15, align="center", valign="middle")
    gains = sorted(((gain(DATA, a), a) for a in COLS), reverse=True)
    rows = [["Feature", "Information Gain"]] + [[a, f"{g:.3f}"] for g, a in gains]
    pg.table(x0 + 170, y0 + 410, 320, rows, size=12, highlight_rows=(1,))
    pg.text(f"→ {gains[0][1]} has the highest gain → becomes the root node", x0 + 20, y0 + 545, 620, 20, 12,
            bold=True, align="center", color=GREEN)
    pg.text("Overcast is already pure (all Yes) → leaf", x0 + 20, y0 + 570, 620, 18, 11, italic=True,
            align="center")
    pg.ask("Which feature should the root split on?",
           f"{gains[0][1]}: largest information gain ({gains[0][0]:.3f}) — it removes the most uncertainty about Play",
           x0 + 20, y0 + 605, x0 + 490, y0 + 443, w=620)

    # ---------------------------------------------------------------- ④ the tree (recursive partitioning)
    x0, y0 = 400, 1000
    pg.panel(x0, y0, 1320, 780, 4, "Recursive Partitioning → the final ID3 tree",
             "Repeat ③ on every impure branch with the remaining features until all leaves are pure")
    cx = x0 + 440

    def dnode(x, y, lab, sub):
        return pg.rect(x - 80, y, 160, 54, f"<b>{lab}?</b><br><font style='font-size:10px'>{sub}</font>", size=13,
                       raw=True, sw=1.5)

    def leaf(x, y, rows):
        yes = rows[0][-1] == "Yes"
        return pg.ellipse(x - 60, y, 120, 50, f"<b>{'Yes' if yes else 'No'}</b><br>{counts(rows)}", size=12,
                          raw=True, fill=LEAF_Y if yes else LEAF_N, stroke=GREEN if yes else RED)

    def edge(a, b, lab, ax, ay, bx, by):
        pg.connect(a, b, orth=False)
        pg.text(lab, (ax + bx) / 2 - 40, (ay + by) / 2 - 9, 80, 18, 11, align="center", bg="#FFFFFF", bold=True)

    sunny, rain = parts["Sunny"], parts["Rain"]
    hum, wind = split(sunny, "Humidity"), split(rain, "Wind")
    r0 = dnode(cx, y0 + 90, "Outlook", f"IG = {gain(DATA, 'Outlook'):.3f}")
    n1 = dnode(cx - 270, y0 + 230, "Humidity", f"IG = {gain(sunny, 'Humidity'):.3f}")
    n2 = leaf(cx, y0 + 232, parts["Overcast"])
    n3 = dnode(cx + 270, y0 + 230, "Wind", f"IG = {gain(rain, 'Wind'):.3f}")
    edge(r0, n1, "Sunny", cx, y0 + 144, cx - 270, y0 + 230)
    edge(r0, n2, "Overcast", cx, y0 + 144, cx, y0 + 232)
    edge(r0, n3, "Rain", cx, y0 + 144, cx + 270, y0 + 230)
    for par, px, groups in [(n1, cx - 270, [("High", hum["High"]), ("Normal", hum["Normal"])]),
                            (n3, cx + 270, [("Strong", wind["Strong"]), ("Weak", wind["Weak"])])]:
        for j, (lab, rows_) in enumerate(groups):
            lx = px - 100 + j * 200
            lf = leaf(lx, y0 + 380, rows_)
            edge(par, lf, lab, px, y0 + 284, lx, y0 + 380)
    pg.text("Decision node = feature test  ·  Leaf = predicted class [count Yes, count No]", x0 + 40, y0 + 450,
            800, 20, 11, italic=True, align="center", color="#555555")
    # recursion tables
    tx = x0 + 900
    pg.text(f"Recursion on Sunny branch (5 rows, H = {H(sunny):.3f})", tx, y0 + 80, 400, 20, 12, bold=True)
    g = sorted(((gain(sunny, a), a) for a in COLS if a != "Outlook"), reverse=True)
    pg.table(tx, y0 + 104, 380, [["Feature", "IG"]] + [[a, f"{v:.3f}"] for v, a in g], size=11,
             highlight_rows=(1,))
    pg.text(f"Recursion on Rain branch (5 rows, H = {H(rain):.3f})", tx, y0 + 210, 400, 20, 12, bold=True)
    g = sorted(((gain(rain, a), a) for a in COLS if a != "Outlook"), reverse=True)
    pg.table(tx, y0 + 234, 380, [["Feature", "IG"]] + [[a, f"{v:.3f}"] for v, a in g], size=11,
             highlight_rows=(1,))
    pg.rect(tx, y0 + 345, 380, 170,
            "<b>ID3(S, features)</b><br><font face='Courier New'>if S is pure: return leaf(class)<br>"
            "if no features: return leaf(majority)<br>A = argmax IG(S, a)<br>node = test on A<br>"
            "for each value v of A:<br>&nbsp;&nbsp;child = ID3(S_v, features − A)<br>return node</font>",
            fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top", raw=True)
    pg.bullets(x0 + 20, y0 + 500, 840, [
        "Greedy, top-down: the best split <i>now</i> is chosen; earlier choices are never revisited",
        "Each feature is used at most once on a path (categorical); the tree stops when leaves are pure",
        "This tree classifies all 14 training days correctly — risk of <b>overfitting</b> on noisy data"],
        head="Notes:")
    pg.ask("How does the tree classify a rainy day with strong wind?",
           f"follow Outlook = Rain → Wind = Strong → leaf No {counts(wind['Strong'])}: each root-to-leaf path is one IF-THEN rule",
           x0 + 20, y0 + 610, cx + 170, y0 + 430, w=840)

    # ---------------------------------------------------------------- ⑤ rules & limitations
    x0 = 1760
    pg.panel(x0, y0, 640, 780, 5, "Rules, Gain Ratio & Limitations")
    pg.bullets(x0 + 20, y0 + 50, 600, [
        "IF Outlook = Overcast THEN <b>Yes</b>",
        "IF Outlook = Sunny AND Humidity = High THEN <b>No</b>",
        "IF Outlook = Sunny AND Humidity = Normal THEN <b>Yes</b>",
        "IF Outlook = Rain AND Wind = Strong THEN <b>No</b>",
        "IF Outlook = Rain AND Wind = Weak THEN <b>Yes</b>"],
        head="Extracted rules (each path = one rule):", size=12)
    pg.text("Bias toward many-valued features", x0 + 20, y0 + 200, 600, 20, 13, bold=True)
    pg.html(f"Splitting on <i>Day</i> (14 unique values) gives 14 pure leaves → IG = {H(DATA):.3f} (the maximum!) "
            "but it is useless for new days.", x0 + 20, y0 + 222, 600, 36, 11)
    si = -sum(len(s) / 14 * math.log2(len(s) / 14) for s in parts.values())
    pg.math("SplitInfo(S,A) = -\\sum_v \\frac{|S_v|}{|S|}\\log_2\\frac{|S_v|}{|S|}", x0 + 20, y0 + 260, 600, 46, 15)
    pg.math("GainRatio = \\frac{IG(S,A)}{SplitInfo(S,A)}", x0 + 20, y0 + 316, 600, 44, 15)
    pg.html(f'C4.5 fix → Outlook: SplitInfo = {si:.3f}, <span style="background:{HL}">GainRatio = '
            f'{gain(DATA, "Outlook") / si:.3f}</span>;  Day: SplitInfo = {math.log2(14):.3f}, '
            f'GainRatio = {H(DATA) / math.log2(14):.3f}', x0 + 20, y0 + 366, 600, 36, 11, align="center")
    rows = [["", "ID3", "C4.5", "CART"],
            ["Criterion", "Information Gain", "Gain Ratio", "Gini / Variance"],
            ["Features", "categorical", "categorical + numeric", "categorical + numeric"],
            ["Splits", "multi-way", "multi-way", "binary"],
            ["Pruning", "no", "yes (error-based)", "yes (cost-complexity)"],
            ["Missing values", "no", "yes", "yes (surrogates)"]]
    pg.table(x0 + 20, y0 + 410, 600, rows, size=11)
    pg.bullets(x0 + 20, y0 + 580, 600, [
        "Overfitting → limit depth, minimum samples per leaf, or prune",
        "Unstable: small data changes → different tree → fixed by Bagging / Random Forest (Ensemble page)"],
        head="Limitations:")
    pg.ask("Why not simply split on 'Day'?",
           f"its IG ({H(DATA):.3f}) is the maximum, but 14 one-row leaves can't predict new days — Gain Ratio "
           f"drops it to {H(DATA) / math.log2(14):.3f}",
           x0 + 20, y0 + 670, x0 + 320, y0 + 402, w=600)
