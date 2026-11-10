"""Apriori page extras: data illustration (raw baskets → one-hot → itemsets → rules) + choosing min_support /
min_confidence / lift."""
from itertools import combinations
from dio import *

GOODS = ["Bread", "Milk", "Butter", "Eggs", "Beer", "Chips", "Diaper", "Cola"]


SMALL = [{"Bread", "Butter", "Milk"}, {"Bread", "Butter"}, {"Beer", "Chips", "Diaper"}, {"Milk", "Eggs"},
         {"Bread", "Butter", "Eggs"}, {"Beer", "Diaper"}, {"Bread", "Milk", "Eggs"}, {"Beer", "Chips"},
         {"Bread", "Butter", "Cola"}, {"Diaper", "Beer", "Chips"}, {"Milk", "Eggs", "Cola"}, {"Bread", "Chips"}]


def make_baskets(n, seed=4):
    """Synthetic shop with planted associations: Bread→Butter, Beer→Chips, Diaper→Beer, Milk→Eggs."""
    r = seeded(seed)
    out = []
    for _ in range(n):
        b = set()
        if r.random() < 0.55: b.add("Bread")
        if "Bread" in b and r.random() < 0.6: b.add("Butter")
        if r.random() < 0.45: b.add("Milk")
        if "Milk" in b and r.random() < 0.5: b.add("Eggs")
        if r.random() < 0.25: b.add("Diaper")
        if r.random() < (0.65 if "Diaper" in b else 0.2): b.add("Beer")
        if "Beer" in b and r.random() < 0.7: b.add("Chips")
        if r.random() < 0.2: b.add("Cola")
        if r.random() < 0.1: b.add("Chips")
        if not b: b.add(r.choice(GOODS))
        out.append(b)
    return out


def apriori(T, minsup):
    n = len(T)
    sup = lambda s: sum(1 for t in T if s <= t) / n
    L, k_sets = {}, [frozenset([i]) for i in GOODS]
    while k_sets:
        cur = {s: sup(s) for s in k_sets}
        cur = {s: v for s, v in cur.items() if v >= minsup}
        L.update(cur)
        keys = list(cur)
        nxt = set()
        for a, b in combinations(keys, 2):
            u = a | b
            if len(u) == len(a) + 1 and all(frozenset(c) in cur for c in combinations(u, len(u) - 1)):
                nxt.add(u)
        k_sets = list(nxt)
    return L


def rules(L, minconf):
    out = []
    for s, sv in L.items():
        if len(s) < 2: continue
        for k in range(1, len(s)):
            for A in combinations(s, k):
                A = frozenset(A); B = s - A
                conf = sv / L[A]
                if conf >= minconf:
                    out.append((A, B, sv, conf, conf / L[B]))
    return sorted(out, key=lambda r: -r[4])


def fmt(s):
    return "{" + ", ".join(sorted(s)) + "}"


def build(pg: Page, cat):
    small = SMALL
    x0, y0 = 400, pg.max_y() + 60

    # ---------------------------------------------------------------- ⑥ data illustration
    pg.panel(x0, y0, 2000, 600, 6, "Data Illustration: raw baskets → one-hot matrix → frequent itemsets → rules",
             "12 receipts from a small shop · min_support = 0.25 (3 of 12 baskets) · min_confidence = 0.6")
    pg.text("1. Raw transactions (one row per receipt)", x0 + 20, y0 + 64, 400, 20, 12, bold=True)
    rows = [["TID", "items"]] + [[f"T{i + 1:02d}", ", ".join(sorted(b))] for i, b in enumerate(small)]
    pg.table(x0 + 20, y0 + 90, 360, rows, size=11, h=13 * 21 + 4)
    pg.connect(pg.text("", x0 + 380, y0 + 230, 1, 1), pg.text("", x0 + 420, y0 + 230, 1, 1), orth=False)
    # one-hot matrix
    pg.text("2. One-hot (binary) basket matrix", x0 + 430, y0 + 64, 500, 20, 12, bold=True)
    head = ["TID"] + [g[:6] for g in GOODS]
    rows = [head]
    for i, b in enumerate(small):
        rows.append([f"T{i + 1:02d}"] + ['<b>1</b>' if g in b else '<font color="#BBBBBB">0</font>' for g in GOODS])
    cnt = [sum(1 for b in small if g in b) for g in GOODS]
    rows.append(["count"] + [f"<b>{c}</b>" for c in cnt])
    rows.append(["support"] + [f"{c / 12:.2f}" for c in cnt])
    pg.table(x0 + 430, y0 + 90, 520, rows, size=11, h=15 * 21 + 4, highlight_rows=(13, 14))
    pg.text("column sum ÷ 12 = support of each single item", x0 + 430, y0 + 420, 520, 18, 10, italic=True,
            color="#555555", align="center")
    # frequent itemsets
    L = apriori(small, 0.25)
    pg.text("3. Frequent itemsets (support ≥ 0.25)", x0 + 1000, y0 + 64, 420, 20, 12, bold=True)
    rows = [["k", "itemset", "count", "support"]]
    for s, v in sorted(L.items(), key=lambda kv: (len(kv[0]), -kv[1])):
        rows.append([len(s), fmt(s), round(v * 12), f"{v:.2f}"])
    pg.table(x0 + 1000, y0 + 90, 400, rows, size=11, h=len(rows) * 21 + 4,
             highlight_rows=tuple(i + 1 for i, (s, _) in enumerate(sorted(L.items(), key=lambda kv: (len(kv[0]), -kv[1])))
                                  if len(s) >= 2))
    # rules
    R = rules(L, 0.6)
    pg.text("4. Rules A → B (confidence ≥ 0.6), sorted by lift", x0 + 1450, y0 + 64, 520, 20, 12, bold=True)
    rows = [["rule", "supp", "conf", "lift"]]
    hl = ()
    for i, (A, B, sv, cf, lf) in enumerate(R[:12]):
        rows.append([f"{fmt(A)} → {fmt(B)}", f"{sv:.2f}", f"{cf:.2f}", f"<b>{lf:.2f}</b>" if lf > 1 else f"{lf:.2f}"])
        if lf > 1.2: hl += (i + 1,)
    pg.table(x0 + 1450, y0 + 90, 520, rows, size=11, h=len(rows) * 21 + 4, highlight_rows=hl)
    if R:
        A, B, sv, cf, lf = R[0]
        sa = L[A]; sb = L[B]
        pg.math(f"conf = \\frac{{{sv:.2f}}}{{{sa:.2f}}} = {cf:.2f},\\quad lift = \\frac{{{cf:.2f}}}{{{sb:.2f}}} = {lf:.2f}",
                x0 + 1450, y0 + 100 + len(rows) * 21 + 20, 520, 44, 15)
        pg.text(f"top rule worked out: {fmt(A)} → {fmt(B)}", x0 + 1450, y0 + 100 + len(rows) * 21 + 66, 520, 18,
                11, align="center", color=RED)
    pg.bullets(x0 + 20, y0 + 470, 1380, [
        "Apriori never looks at the raw text: it only needs the <b>binary matrix</b> (item present / absent per basket)",
        "Level k = 1 → keep frequent items → join them into pairs → prune pairs with an infrequent item → count again …",
        "Highlighted rules have lift &gt; 1.2: the items appear together more often than if they were independent",
        "Rules with lift ≈ 1 (or &lt; 1) are just popular items, not real associations — e.g. anything → {Bread}"],
        head="How the algorithm uses the data:", size=11)
    if R:
        A, B, sv, cf, lf = R[0]
        pg.ask("Which pair of items gives the strongest rule here?",
               f"{fmt(A)} → {fmt(B)}: confidence {cf:.2f}, lift {lf:.2f} = {lf:.1f}× more often together than by chance",
               x0 + 1040, y0 + 440, x0 + 1622, y0 + 121, w=500)

    # ---------------------------------------------------------------- ⑦ choosing thresholds
    y0 = y0 + 640
    big = make_baskets(300, seed=11)
    pg.panel(x0, y0, 2000, 540, 7, "Finding the optimal hyperparameters: min_support, min_confidence, lift",
             "300 simulated baskets with 4 planted associations (Bread→Butter, Milk→Eggs, Diaper→Beer, Beer→Chips)")
    sups = [0.04 + 0.02 * i for i in range(19)]
    nI, nR, nRL = [], [], []
    for s in sups:
        Ls = apriori(big, s)
        Rs = rules(Ls, 0.5)
        nI.append(len(Ls)); nR.append(len(Rs)); nRL.append(sum(1 for r in Rs if r[4] > 1.2))
    top = max(max(nI), max(nR)) * 1.1
    p = pg.axes(x0 + 80, y0 + 110, 560, 300, (0, 0.42), (0, top), "min\\_support", "count", label_size=12,
                xticks=(0.05, 0.1, 0.2, 0.3, 0.4), yticks=tuple(v for v in (10, 20, 40, 60, 80, 100) if v < top))
    p.poly(list(zip(sups, nI)), color=BLUE, width=2)
    p.poly(list(zip(sups, nR)), color=ORANGE, width=2)
    p.poly(list(zip(sups, nRL)), color=GREEN, width=2.5)
    pg.html('<font color="#3333FF"><b>— frequent itemsets</b></font><br><font color="#FF8000"><b>— rules (conf ≥ 0.5)</b>'
            '</font><br><font color="#009900"><b>— rules with lift &gt; 1.2</b></font>', x0 + 450, y0 + 110, 200, 54, 11)
    choose = 0.08
    p.vline(choose, 0, top * 0.95, color=RED, width=1.5)
    pg.note(f"chosen min_support = {choose} (lift-filtered rules stay, noise rules drop)", x0 + 300, y0 + 70,
            *p.P(choose, top * 0.8), w=330, color=RED)
    pg.text("(a) Counts vs min_support", x0 + 20, y0 + 64, 300, 20, 13, bold=True)
    pg.ask("Which min_support should I choose?",
           f"{choose}: the knee; below it noise rules explode, above it useful lift-filtered rules start to vanish",
           x0 + 320, y0 + 172, p.P(choose, 0)[0] + 3, y0 + 250, w=300)
    pg.bullets(x0 + 20, y0 + 440, 640, [
        "Low support → combinatorial explosion of itemsets and rules (slow, mostly noise)",
        "High support → only the obvious best-sellers survive; interesting niche rules vanish",
        "Pick the knee where the count of <b>useful</b> (lift-filtered) rules is still stable"], size=11)
    # confidence curve at chosen support
    Lc = apriori(big, choose)
    confs = [0.3 + 0.05 * i for i in range(14)]
    cR = [len(rules(Lc, c)) for c in confs]
    cRL = [sum(1 for r in rules(Lc, c) if r[4] > 1.2) for c in confs]
    ex = x0 + 720
    pg.text(f"(b) Rules vs min_confidence (min_support = {choose})", ex, y0 + 64, 520, 20, 13, bold=True)
    topc = max(cR) * 1.15
    q = pg.axes(ex + 60, y0 + 110, 480, 300, (0.3, 0.95), (0, topc), "min\\_confidence", "rules", label_size=12,
                origin=(0.3, 0), xticks=(0.3, 0.5, 0.7, 0.9))
    q.poly(list(zip(confs, cR)), color=ORANGE, width=2)
    q.poly(list(zip(confs, cRL)), color=GREEN, width=2.5)
    q.vline(0.6, 0, topc * 0.95, color=RED, width=1.5)
    q.text("chosen 0.6", 0.6, topc * 0.95, dy=-8, size=10, color=RED)
    pg.bullets(ex, y0 + 440, 600, [
        "Confidence alone favours rules whose consequent is simply popular",
        "Always add a <b>lift</b> (or leverage / conviction) filter: lift &gt; 1 = positive association",
        "Too high confidence keeps only near-certain, often trivial rules"], size=11)
    # final rule list at chosen thresholds
    Rf = [r for r in rules(Lc, 0.6) if r[4] > 1.2][:8]
    fx = x0 + 1380
    pg.text("(c) Final rules: support ≥ 0.08, conf ≥ 0.6, lift > 1.2", fx, y0 + 64, 600, 20, 13, bold=True)
    rows = [["rule", "supp", "conf", "lift", "contains a planted pair?"]]
    planted = [({"Bread"}, {"Butter"}), ({"Milk"}, {"Eggs"}), ({"Diaper"}, {"Beer"}), ({"Beer"}, {"Chips"})]
    for A, B, sv, cf, lf in Rf:
        ok = any(set(A) | set(B) >= a | b for a, b in planted)
        rows.append([f"{fmt(A)} → {fmt(B)}", f"{sv:.2f}", f"{cf:.2f}", f"{lf:.2f}", "✔" if ok else "–"])
    pg.table(fx, y0 + 92, 590, rows, size=11, h=len(rows) * 21 + 4)
    pg.bullets(fx, y0 + 110 + len(rows) * 21, 590, [
        "Check the survivors with the business: actionable? (shelf placement, bundles, cross-sell)",
        "Validate on a later period: good rules keep their lift over time",
        "max_len (itemset size) limits long, hard-to-use rules"], head="Business check:", size=11)
