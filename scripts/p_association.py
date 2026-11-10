"""Apriori page: transactions, itemset lattice with pruning, level-wise counting, support/confidence/lift, rules."""
from itertools import combinations
from dio import *

T = [{"Bread", "Milk"},
     {"Bread", "Diaper", "Beer", "Eggs"},
     {"Milk", "Diaper", "Beer", "Cola"},
     {"Bread", "Milk", "Diaper", "Beer"},
     {"Bread", "Milk", "Diaper", "Cola"}]
ITEMS = ["Bread", "Milk", "Diaper", "Beer", "Eggs", "Cola"]
MAIN = ["Bread", "Milk", "Diaper", "Beer"]
MINSUP = 3


def sup(s):
    return sum(1 for t in T if set(s) <= t)


def _cross(pg, x, y, w, h):
    pg.line([(x, y), (x + w, y + h)], color=RED, width=1.5)
    pg.line([(x, y + h), (x + w, y)], color=RED, width=1.5)


def build(pg: Page, cat):
    N = len(T)
    L1 = [i for i in ITEMS if sup([i]) >= MINSUP]
    C2 = list(combinations(L1, 2))
    L2 = [c for c in C2 if sup(c) >= MINSUP]
    C3 = []
    for c in combinations(L1, 3):
        pruned = any(tuple(s) not in L2 for s in combinations(c, 2))
        C3.append((c, pruned))

    # ---------------------------------------------------------------- ① transactions
    x0, y0 = 400, 240
    pg.panel(x0, y0, 560, 660, 1, "Transactions (market basket)", "Background: itemsets = sets of items")
    pg.table(x0 + 30, y0 + 70, 500, [["TID", "Items bought"]] +
             [[f"T{i}", ", ".join(sorted(t, key=ITEMS.index))] for i, t in enumerate(T, 1)], size=12)
    pg.text("One-hot view (1 = item in basket):", x0 + 30, y0 + 240, 500, 20, 12, bold=True)
    rows = [["TID"] + ITEMS] + [[f"T{i}"] + ["1" if it in t else "·" for it in ITEMS] for i, t in enumerate(T, 1)]
    rows.append(["<b>count</b>"] + [f"<b>{sup([it])}</b>" for it in ITEMS])
    pg.table(x0 + 30, y0 + 265, 500, rows, size=11, highlight_rows=(6,))
    pg.ask("Which items are popular enough to analyse?",
           "count ≥ 3: Bread, Milk, Diaper, Beer; Eggs and Cola are too rare",
           x0 + 400, y0 + 296, x0 + 377, y0 + 413, w=150)
    pg.html(f"<span style='background:{HL}'><b>min support = {MINSUP} transactions ({100 * MINSUP // N}%)</b></span>",
            x0 + 30, y0 + 445, 500, 24, 13, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 480, 520, [
        "Itemset = any set of items, e.g. {Diaper, Beer};  k-itemset has k items",
        "With d items there are \\(2^d - 1\\) possible itemsets → brute force explodes",
        "<b>Goal</b>: find all frequent itemsets (support ≥ min support), then rules A → B",
        "<b>Apriori property</b>: every subset of a frequent itemset is frequent ⇔ if an itemset is "
        "infrequent, all its supersets are infrequent → prune them without counting"], head="Where:", h=150)

    # ---------------------------------------------------------------- ② lattice
    x0 = 1000
    pg.panel(x0, y0, 1400, 660, 2, "Itemset Lattice — level-wise search with pruning",
             f"Count support level by level (min support = {MINSUP}); never generate supersets of infrequent itemsets")
    W, H = 150, 44
    cx = x0 + 640
    pos = {}

    def node(key, label, x, y, state):
        fill = {"freq": HL, "inf": LIGHTRED, "pruned": "#F5F5F5", "root": "none"}[state]
        stroke = {"freq": BLACK, "inf": RED, "pruned": "#AAAAAA", "root": BLACK}[state]
        pg.rect(x, y, W, H, label, fill=fill, stroke=stroke, dashed=state == "pruned", size=11, raw=True,
                color="#999999" if state == "pruned" else None)
        if state == "inf": _cross(pg, x, y, W, H)
        pos[key] = (x + W / 2, y, y + H)

    node((), "∅", cx - W / 2, y0 + 75, "root")
    ly = [y0 + 75, y0 + 175, y0 + 295, y0 + 415, y0 + 530]
    for i, it in enumerate(ITEMS):
        s = sup([it])
        node((it,), f"<b>{it}</b><br>sup = {s}", x0 + 70 + i * 190, ly[1], "freq" if s >= MINSUP else "inf")
    for i, c in enumerate(combinations(MAIN, 2)):
        s = sup(c)
        node(c, f"<b>{', '.join(c)}</b><br>sup = {s}", x0 + 70 + i * 190, ly[2], "freq" if s >= MINSUP else "inf")
    for i, c in enumerate(combinations(MAIN, 3)):
        pruned = any(tuple(x) not in L2 for x in combinations(c, 2))
        lab = f"<b>{', '.join(c)}</b><br>" + ("pruned (not counted)" if pruned else f"sup = {sup(c)}")
        node(c, lab, x0 + 260 + i * 230, ly[3], "pruned" if pruned else ("freq" if sup(c) >= MINSUP else "inf"))
    node(tuple(MAIN), f"<b>{', '.join(MAIN)}</b><br>pruned", cx - W / 2, ly[4], "pruned")
    # edges subset -> superset
    keys = list(pos)
    for a in keys:
        for b in keys:
            if len(b) == len(a) + 1 and set(a) <= set(b) and (len(a) == 0 or all(i in MAIN for i in b)):
                pa, pb = pos[a], pos[b]
                bad = pos and len(a) > 0 and (sup(a) < MINSUP)
                pg.line([(pa[0], pa[2]), (pb[0], pb[1])], color=RED if bad else "#999999", width=0.6,
                        dashed=bad, opacity=70)
    for it in ("Eggs", "Cola"):
        pa = pos[(it,)]
        pg.line([(pa[0], pa[2]), (pa[0], pa[2] + 26)], color=RED, dashed=True, width=0.6)
        pg.text("supersets never generated", pa[0] - 75, pa[2] + 26, 150, 16, 9, align="center", color=RED)
    for lvl, lab in enumerate(["k = 0", "k = 1", "k = 2", "k = 3", "k = 4"]):
        pg.text(lab, x0 + 14, ly[lvl] + 12, 50, 20, 12, bold=True, color="#555555")
    lg = x0 + 1170
    for i, (fill, stroke, dash, txt) in enumerate([(HL, BLACK, False, "frequent (≥ min sup)"),
                                                   (LIGHTRED, RED, False, "infrequent (counted)"),
                                                   ("#F5F5F5", "#AAAAAA", True, "pruned by Apriori property")]):
        pg.rect(lg, y0 + 530 + i * 30, 30, 20, fill=fill, stroke=stroke, dashed=dash)
        pg.text(txt, lg + 38, y0 + 530 + i * 30, 190, 20, 10)
    pg.note("{Bread, Beer} is infrequent → every superset containing it is pruned", x0 + 640, y0 + 585,
            pos[("Bread", "Diaper", "Beer")][0], pos[("Bread", "Diaper", "Beer")][2], w=420, color=RED)
    cnt = len(ITEMS) + len(C2) + sum(1 for _, p in C3 if not p)
    pb = pos[("Bread", "Milk", "Beer")]
    pg.ask("Why don't we count every possible itemset?",
           f"a superset of an infrequent set is infrequent → only {cnt} of {2 ** len(ITEMS) - 1} counted",
           x0 + 70, y0 + 490, pb[0] - W / 2 + 10, pb[2] - 4, w=240)
    pg.html(f"Counted only <b>{len(ITEMS) + len(C2) + sum(1 for _, p in C3 if not p)}</b> candidate itemsets "
            f"instead of all <b>{2 ** len(ITEMS) - 1}</b> → that is the power of pruning",
            x0 + 20, y0 + 620, 900, 24, 12, valign="middle")

    # ---------------------------------------------------------------- ③ level tables
    x0, y0 = 400, 940
    pg.panel(x0, y0, 720, 600, 3, "Level-wise counting: Cₖ → Lₖ", "Join → Prune → Count → Filter")
    r1 = [["C₁ itemset", "sup"]] + [[it if sup([it]) >= MINSUP else f"<s>{it}</s>", sup([it])] for it in ITEMS]
    pg.table(x0 + 20, y0 + 70, 200, r1, size=11, highlight_rows=tuple(i + 1 for i, it in enumerate(ITEMS)
                                                                      if sup([it]) >= MINSUP))
    r2 = [["C₂ itemset", "sup"]] + [[", ".join(c) if sup(c) >= MINSUP else f"<s>{', '.join(c)}</s>", sup(c)]
                                    for c in C2]
    pg.table(x0 + 240, y0 + 70, 230, r2, size=11, highlight_rows=tuple(i + 1 for i, c in enumerate(C2)
                                                                       if sup(c) >= MINSUP))
    r3 = [["C₃ itemset", "sup"]] + [[", ".join(c) if not p else f"<s>{', '.join(c)}</s>",
                                     "pruned" if p else sup(c)] for c, p in C3]
    pg.table(x0 + 490, y0 + 70, 210, r3, size=10)
    for xa in (x0 + 220, x0 + 470):
        pg.line([(xa + 2, y0 + 120), (xa + 18, y0 + 120)], arrow="block", width=1.5)
    pg.text(f"L₁ = {{{', '.join(L1)}}}", x0 + 20, y0 + 265, 220, 30, 10, bold=True)
    pg.text("L₂ = {" + "}, {".join(", ".join(c) for c in L2) + "}", x0 + 240, y0 + 265, 240, 44, 10, bold=True)
    pg.text("L₃ = ∅ → stop", x0 + 490, y0 + 265, 210, 20, 10, bold=True, color=RED)
    pg.ask("Why is {Bread, Milk, Beer} never counted?",
           "it contains {Bread, Beer} (sup 2 < 3), so it cannot be frequent → pruned",
           x0 + 480, y0 + 160, x0 + 572, y0 + 133, w=225)
    pg.rect(x0 + 20, y0 + 320, 680, 160,
            "<b>Pseudo-code</b><br><font face='Courier New'>L1 = {items with support ≥ min_sup}<br>"
            "k = 2<br>while L(k−1) not empty:<br>&nbsp;&nbsp;Ck = join(L(k−1), L(k−1))&nbsp;&nbsp;# candidates of size k<br>"
            "&nbsp;&nbsp;Ck = prune(Ck)&nbsp;&nbsp;# drop if any (k−1)-subset not in L(k−1)<br>"
            "&nbsp;&nbsp;Lk = {c in Ck : support(c) ≥ min_sup}&nbsp;&nbsp;# one scan of the data<br>"
            "&nbsp;&nbsp;k += 1</font>", fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top",
            raw=True)
    pg.bullets(x0 + 20, y0 + 495, 680, [
        "One full database scan per level → slow on big data (FP-Growth needs only 2 scans)",
        "Low min support → many candidates;  high min support → miss rare but useful rules"], head="Cost:")

    # ---------------------------------------------------------------- ④ support / confidence / lift (Venn)
    x0 = 1160
    pg.panel(x0, y0, 620, 600, 4, "Support, Confidence & Lift", "Rule: Diaper → Beer")
    pg.rect(x0 + 40, y0 + 70, 540, 200, stroke=BLACK)
    pg.text(f"All transactions  |T| = {N}", x0 + 50, y0 + 74, 250, 18, 11, bold=True)
    pg.ellipse(x0 + 50, y0 + 100, 250, 150, fill=LIGHTBLUE, stroke=BLUE, sw=1.5, opacity=60)
    pg.ellipse(x0 + 190, y0 + 110, 220, 130, fill=LIGHTRED, stroke=RED, sw=1.5, opacity=60)
    pg.text(f"Diaper  ({sup(['Diaper'])})", x0 + 70, y0 + 165, 110, 20, 12, bold=True, color=BLUE)
    pg.text(f"Beer  ({sup(['Beer'])})", x0 + 320, y0 + 165, 80, 20, 12, bold=True, color=RED)
    pg.html(f"<b>Diaper ∩ Beer<br>({sup(['Diaper', 'Beer'])})</b>", x0 + 195, y0 + 158, 105, 34, 11,
            align="center", valign="middle")
    sd, sb, sdb = sup(["Diaper"]) / N, sup(["Beer"]) / N, sup(["Diaper", "Beer"]) / N
    conf = sdb / sd
    lift = conf / sb
    pg.ask("Is buying diapers really linked to beer?",
           f"yes: lift = {lift:.2f} > 1 → beer is {lift:.2f}× more likely in diaper baskets",
           x0 + 418, y0 + 92, x0 + 290, y0 + 175, w=160)
    pg.math(f"Support(A \\cup B) = \\frac{{|T_{{A \\cup B}}|}}{{|T|}} = \\frac{{{sup(['Diaper', 'Beer'])}}}{{{N}}} = {sdb:.1f}",
            x0 + 20, y0 + 285, 580, 46, 15)
    pg.math(f"Confidence(A \\to B) = \\frac{{Supp(A \\cup B)}}{{Supp(A)}} = \\frac{{{sdb:.1f}}}{{{sd:.1f}}} = {conf:.2f}",
            x0 + 20, y0 + 340, 580, 46, 15)
    pg.math(f"Lift(A \\to B) = \\frac{{Conf(A \\to B)}}{{Supp(B)}} = \\frac{{{conf:.2f}}}{{{sb:.1f}}} = {lift:.2f}",
            x0 + 20, y0 + 395, 580, 46, 15)
    pg.html(f"<span style='background:{HL}'>Diaper → Beer: {conf:.0%} of diaper baskets also have beer, "
            f"{lift:.2f}× more than chance</span>", x0 + 20, y0 + 452, 580, 24, 13, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 490, 580, [
        "A ∪ B in the formula = the itemset containing <b>both</b> A and B (overlap in the Venn)",
        "Lift > 1 positive association · = 1 independent · < 1 negative (substitutes)"], head="Read it:")

    # ---------------------------------------------------------------- ⑤ rules
    x0 = 1820
    pg.panel(x0, y0, 580, 600, 5, "Rule Generation from Lₖ", "Every split of a frequent itemset into A → B")
    rows = [["Rule A → B", "supp", "conf", "lift"]]
    hi = []
    for c in L2:
        for A, B in ((c[0], c[1]), (c[1], c[0])):
            s = sup(c) / N; cf = sup(c) / sup([A]); lf = cf / (sup([B]) / N)
            rows.append([f"{A} → {B}", f"{s:.1f}", f"{cf:.2f}", f"{lf:.2f}"])
            if cf >= 0.7 and lf > 1: hi.append(len(rows) - 1)
    pg.table(x0 + 30, y0 + 65, 520, rows, size=11, highlight_rows=tuple(hi))
    pg.ask("Which rules are worth acting on?",
           "confidence ≥ 0.7 and lift > 1: only Diaper → Beer and Beer → Diaper",
           x0 + 20, y0 + 120, x0 + 206, y0 + 228, w=175)
    pg.html(f"<span style='background:{HL}'>&nbsp;&nbsp;&nbsp;</span> kept: confidence ≥ 0.70 and lift > 1",
            x0 + 30, y0 + 300, 520, 20, 11, valign="middle")
    pg.bullets(x0 + 20, y0 + 335, 540, [
        "Confidence is not symmetric: Beer → Diaper = 1.00 but Diaper → Beer = 0.75",
        "Bread → Milk has conf 0.75 but lift < 1 → popular items, not a real association",
        "Always check lift (or conviction), not confidence alone"], head="Insights:")
    pg.bullets(x0 + 20, y0 + 450, 540, [
        "Shelf placement & bundles (put beer near diapers)",
        "'Customers who bought X also bought Y' recommendations",
        "Co-occurring events, e.g. news keywords that appear before a price drop"], head="Use cases:")
