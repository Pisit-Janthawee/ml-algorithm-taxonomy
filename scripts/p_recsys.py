"""Recommendation Systems page (base: User-based Collaborative Filtering)."""
import math
from dio import *

USERS = ["U1 Ann", "U2 Ben", "U3 Cat", "U4 Dan", "U5 Eve", "U6 Fay"]
ITEMS = ["I1", "I2", "I3", "I4", "I5", "I6"]
N = None
R = [[5, 4, N, 1, N, 2],
     [4, 5, 1, 2, 3, 1],
     [5, 4, 2, 1, 2, N],
     [1, 2, 5, 4, 4, 5],
     [2, N, 4, 5, 5, 4],
     [4, 5, 2, N, 3, 2]]
DESC = ["space action hero battle", "space hero adventure", "romance drama love story",
        "drama family love", "space comedy family fun", "romance comedy love fun"]
TU, TI = 0, 4          # target cell: U1 Ann, item I5


def rating_color(v):
    if v is None: return "#FFFFFF"
    t = (v - 1) / 4
    r = int(235 - 180 * t); g = int(242 - 150 * t); b = int(255 - 40 * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def sim_color(s):
    s = max(-1, min(1, s))
    if s >= 0:
        v = int(255 - 150 * s); return f"#FF{v:02X}{v:02X}"
    v = int(255 - 150 * -s); return f"#{v:02X}{v:02X}FF"


def mean(xs): return sum(xs) / len(xs)


def umean(u): return mean([v for v in R[u] if v is not None])


def pearson(u, v):
    co = [i for i in range(6) if R[u][i] is not None and R[v][i] is not None]
    mu, mv = umean(u), umean(v)
    a = [R[u][i] - mu for i in co]; b = [R[v][i] - mv for i in co]
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0, co


def adj_cos(i, j):
    us = [u for u in range(6) if R[u][i] is not None and R[u][j] is not None]
    a = [R[u][i] - umean(u) for u in us]; b = [R[u][j] - umean(u) for u in us]
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


def rating_grid(pg, x, y, cell=46, hl=None, pred=None, labels=True):
    if labels:
        for j, it in enumerate(ITEMS):
            pg.text(it, x + 70 + j * cell, y - 20, cell, 18, 11, bold=True, align="center")
    for u in range(6):
        if labels: pg.text(USERS[u], x, y + u * cell, 68, cell, 11, bold=True)
        for i in range(6):
            v = R[u][i]
            if v is None and pred and (u, i) in pred:
                pg.rect(x + 70 + i * cell, y + u * cell, cell, cell, f"{pred[(u, i)]:.1f}", fill=HL, size=12,
                        bold=True, color=RED)
            else:
                lab = "?" if v is None else str(v)
                pg.rect(x + 70 + i * cell, y + u * cell, cell, cell, lab, fill=rating_color(v), size=13,
                        bold=v is None, color=RED if v is None else None,
                        sw=3 if hl and (u, i) in hl else 1, stroke=RED if hl and (u, i) in hl else BLACK)
    return x + 70 + 6 * cell, y + 6 * cell


def tfidf():
    docs = [d.split() for d in DESC]
    vocab = sorted({w for d in docs for w in d})
    n = len(docs)
    df = {w: sum(1 for d in docs if w in d) for w in vocab}
    idf = {w: math.log(n / df[w]) for w in vocab}
    tf = [{w: d.count(w) / len(d) for w in vocab} for d in docs]
    vec = [[tf[k][w] * idf[w] for w in vocab] for k in range(n)]
    return docs, vocab, df, idf, vec


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def build(pg: Page, cat):
    # ------------------------------------------------------------------ ① overview
    x0, y0 = 400, 240
    pg.panel(x0, y0, 760, 700, 1, "The User–Item Matrix & the approaches",
             "Rows = users, columns = items, cells = ratings (1–5); '?' = not rated yet")
    gx, gy = x0 + 30, y0 + 100
    ex, ey = rating_grid(pg, gx, gy, hl={(TU, TI)})
    known = sum(1 for row in R for v in row if v is not None)
    pg.text(f"{known} of 36 cells known → sparsity {100 * (36 - known) / 36:.0f} % (real systems: > 99 %)",
            gx, ey + 10, 400, 20, 11, italic=True)
    pg.ask("Will Ann like item I5, which she has never rated?",
           "the whole page predicts this red '?' cell — every method fills it in a different way",
           x0 + 440, y0 + 90, gx + 70 + TI * 46 + 46, gy + TU * 46 + 23, w=290)
    # approach tree
    tx, ty = x0 + 30, y0 + 430
    pg.text("Approaches (what each one uses)", tx, ty - 6, 400, 20, 13, bold=True)
    boxes = [("Popularity / Top-N", "item counts only", "#F5F5F5", tx, ty + 30),
             ("Content-based", "item features (BoW / TF-IDF) ②", "#FFE6CC", tx + 240, ty + 30),
             ("Collaborative filtering", "other users' ratings ③ ④", LIGHTBLUE, tx + 480, ty + 30),
             ("Matrix factorization", "latent factors from R ⑤", "#E1D5E7", tx + 480, ty + 130),
             ("Hybrid / Deep", "content + CF + context", "#D5E8D4", tx + 240, ty + 130)]
    ids = []
    for name, sub, col, bx, by in boxes:
        ids.append(pg.rect(bx, by, 210, 56, f"<b>{name}</b><br><font style='font-size:10px'>{sub}</font>",
                           fill=col, raw=True))
    pg.connect(ids[0], ids[1], orth=False, label="personalize")
    pg.connect(ids[1], ids[2], orth=False, label="use crowd")
    pg.connect(ids[2], ids[3], orth=False, label="compress")
    pg.connect(ids[3], ids[4], orth=False, label="combine")
    pg.text("★ base of CF: User-based (GroupLens, 1994)", tx, ty + 100, 230, 18, 10, italic=True, color=RED)

    # ------------------------------------------------------------------ ② content-based BoW / TF-IDF
    x0 = 1200
    pg.panel(x0, y0, 1200, 700, 2, "Content-based: Bag-of-Words → TF-IDF → cosine similarity",
             "Recommend items whose description looks like what the user already liked")
    docs, vocab, df, idf, vec = tfidf()
    rows = [["item", "description"]] + [[ITEMS[k], DESC[k]] for k in range(6)]
    pg.table(x0 + 20, y0 + 85, 300, rows, size=10, h=170)
    # TF-IDF matrix (top words)
    show = ["space", "hero", "love", "romance", "drama", "family", "comedy", "fun"]
    trows = [["", *show]]
    for k in range(6):
        trows.append([ITEMS[k]] + [f"{vec[k][vocab.index(w)]:.2f}" if vec[k][vocab.index(w)] else "·" for w in show])
    pg.text("TF-IDF matrix (8 of the vocabulary words)", x0 + 340, y0 + 62, 500, 18, 12, bold=True)
    pg.table(x0 + 340, y0 + 85, 500, trows, size=10, h=170)
    pg.math(r"\text{tfidf}(w,d) = \frac{\text{count}(w,d)}{|d|}\cdot \ln\frac{N}{df(w)}", x0 + 860, y0 + 70,
            320, 50, 15)
    pg.note("word frequency in the item", x0 + 860, y0 + 130, x0 + 960, y0 + 108, w=150)
    pg.note("rare word → high weight", x0 + 1030, y0 + 130, x0 + 1110, y0 + 108, w=140)
    pg.html(f"e.g. <b>'love'</b> is in {df['love']} items → idf = ln(6/{df['love']}) = {idf['love']:.2f};"
            f" <b>'space'</b> in {df['space']} → idf = {idf['space']:.2f}", x0 + 860, y0 + 160, 320, 40, 11)
    # cosine similarity heatmap
    sim = [[cos(vec[a], vec[b]) for b in range(6)] for a in range(6)]
    hx, hy, c = x0 + 60, y0 + 300, 40
    pg.text("Item–item cosine similarity", hx - 40, hy - 40, 300, 18, 12, bold=True)
    for j in range(6):
        pg.text(ITEMS[j], hx + j * c, hy - 20, c, 18, 10, bold=True, align="center")
    for a in range(6):
        pg.text(ITEMS[a], hx - 38, hy + a * c, 34, c, 10, bold=True, align="right")
        for b in range(6):
            pg.rect(hx + b * c, hy + a * c, c, c, f"{sim[a][b]:.2f}", fill=sim_color(sim[a][b]), size=9)
    pg.math(r"\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}",
            hx, hy + 6 * c + 14, 240, 46, 14)
    # recommendation for Ann: liked I1 (5) and I2 (4)
    liked = [0, 1]
    prof = [sum(vec[k][t] * R[TU][k] for k in liked) / sum(R[TU][k] for k in liked) for t in range(len(vocab))]
    cand = [k for k in range(6) if R[TU][k] is None]
    allsc = sorted([(cos(prof, vec[k]), k) for k in range(6) if k not in liked], reverse=True)
    rx = x0 + 400
    pg.text("Ann's profile = rating-weighted average of the items she liked (I1 = 5, I2 = 4)", rx, y0 + 280,
            760, 20, 12, bold=True)
    pg.math(r"\mathbf{p}_u = \frac{\sum_{i \in liked} r_{ui}\,\mathbf{v}_i}{\sum r_{ui}}", rx, y0 + 305, 260, 50, 15)
    top = max(s for s, _ in allsc) * 1.35 or 1
    q = pg.axes(rx + 340, y0 + 330, 380, 150, (-0.5, len(allsc) - 0.5), (0, top), "", "cos", label_size=11)
    for n_, (s, k) in enumerate(allsc):
        q.bars([(n_, s)], 0.6, fill=HL if k in cand else LIGHTBLUE, stroke=BLUE)
        q.text(f"{s:.2f}", n_, s, dy=-10, size=10)
        q.text(ITEMS[k] + (" ?" if k in cand else ""), n_, 0, dy=12, size=10, bold=k in cand)
    pg.text("yellow = unrated by Ann (candidates)", rx + 340, y0 + 505, 380, 16, 10, italic=True, align="center")
    best = max((cos(prof, vec[k]), k) for k in cand)
    pg.ask("Which unseen item matches Ann's taste in content?",
           f"{ITEMS[best[1]]} (cos = {best[0]:.2f}) — it shares 'space' with I1/I2; the other candidate "
           f"I3 shares no words → cos = 0",
           rx, y0 + 380, *q.P(0, allsc[0][0]), w=320)
    _content_score = best
    pg.bullets(rx, y0 + 540, 760, [
        "Needs no other users → works for <b>new items</b> (item cold start) as soon as they have a description",
        "Over-specialisation: only recommends 'more of the same' (filter bubble); no serendipity",
        "Bag-of-Words / TF-IDF details: NLP page; embeddings (Word2Vec, BERT) capture synonyms"],
        head="Notes:", size=11)

    # ------------------------------------------------------------------ ③ user-based CF
    x0, y0 = 400, 980
    pg.panel(x0, y0, 1000, 760, 3, "★ User-based Collaborative Filtering (worked)",
             "Find users who rate like Ann, then average their opinion of I5")
    mu = [umean(u) for u in range(6)]
    rows = [["user", "\\(\\bar{r}_u\\)", *ITEMS]]
    for u in range(6):
        rows.append([USERS[u].split()[0], f"{mu[u]:.2f}"] +
                    ["·" if R[u][i] is None else f"{R[u][i] - mu[u]:+.2f}" for i in range(6)])
    pg.text("Step 1 — mean-centre each user (removes 'generous' vs 'strict' raters)", x0 + 20, y0 + 62, 520, 18,
            12, bold=True)
    pg.table(x0 + 20, y0 + 86, 460, rows, size=10, h=170, highlight_rows=(1,))
    sims = []
    for v in range(1, 6):
        s, co = pearson(TU, v)
        sims.append((v, s, co))
    rated = [(v, s) for v, s, _ in sims if R[v][TI] is not None]
    top = sorted([t for t in rated if t[1] > 0], key=lambda t: -t[1])[:2]
    topset = {v for v, _ in top}
    pg.text("Step 2 — Pearson similarity of Ann with each user (on co-rated items)", x0 + 510, y0 + 62, 480, 18,
            12, bold=True)
    srows = [["user", "co-rated", "sim(Ann, v)", "r(v, I5)", "r − r̄", "use?"]]
    for v, s, co in sims:
        srows.append([USERS[v].split()[0], ",".join(ITEMS[i] for i in co), f"{s:+.3f}", R[v][TI],
                      f"{R[v][TI] - mu[v]:+.2f}", "✔ top-2" if v in topset else ""])
    hl = tuple(k + 1 for k, (v, _, _) in enumerate(sims) if v in topset)
    pg.table(x0 + 510, y0 + 86, 470, srows, size=10, h=150, highlight_rows=hl)
    pg.math(r"sim(u,v)=\frac{\sum_{i}(r_{ui}-\bar r_u)(r_{vi}-\bar r_v)}{\sqrt{\sum_i(r_{ui}-\bar r_u)^2}\sqrt{\sum_i(r_{vi}-\bar r_v)^2}}",
            x0 + 510, y0 + 245, 470, 50, 13)
    # neighbour graph
    cx, cy = x0 + 190, y0 + 430
    ann = pg.ellipse(cx - 40, cy - 22, 80, 44, "Ann", fill=HL, sw=2, bold=True)
    for k, (v, s, _) in enumerate(sims):
        ang = -math.pi / 2 + k * 2 * math.pi / 5
        nx, ny = cx + 150 * math.cos(ang), cy + 110 * math.sin(ang)
        nid = pg.ellipse(nx - 34, ny - 18, 68, 36, USERS[v].split()[1], fill=LIGHTBLUE if s > 0 else LIGHTRED,
                         sw=2 if v in topset else 1)
        pg.line([(cx + 40 * math.cos(ang), cy + 22 * math.sin(ang)), (nx - 34 * math.cos(ang), ny - 18 * math.sin(ang))],
                color=GREEN if v in topset else (BLUE if s > 0 else RED), width=3 if v in topset else 1,
                dashed=v not in topset)
        pg.text(f"{s:+.2f}", (cx + nx) / 2 - 20, (cy + ny) / 2 - 9, 40, 16, 10, bold=True, align="center",
                color=GREEN if v in topset else "#555555")
    pg.text("green = k = 2 nearest neighbours who rated I5; red = opposite taste", x0 + 20, y0 + 565, 360, 30, 10,
            italic=True)
    # prediction
    num = sum(s * (R[v][TI] - mu[v]) for v, s in top)
    den = sum(abs(s) for _, s in top)
    pred_u = mu[TU] + num / den
    pg.text("Step 3 — predict Ann's rating for I5", x0 + 420, y0 + 320, 560, 18, 12, bold=True)
    pg.math(r"\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N_k(u)} sim(u,v)\,(r_{vi}-\bar r_v)}{\sum_{v\in N_k(u)} |sim(u,v)|}",
            x0 + 420, y0 + 345, 540, 56, 15)
    pg.note("Ann's own average", x0 + 420, y0 + 405, x0 + 520, y0 + 380, w=110)
    pg.note("neighbours' opinion relative to their own average", x0 + 650, y0 + 405, x0 + 780, y0 + 380, w=250)
    (v1, s1), (v2, s2) = top
    pg.html(f"\\(\\hat r = {mu[TU]:.2f} + \\frac{{{s1:.3f}\\,({R[v1][TI]}-{mu[v1]:.2f}) + {s2:.3f}\\,({R[v2][TI]}-{mu[v2]:.2f})}}"
            f"{{{s1:.3f}+{s2:.3f}}} = {mu[TU]:.2f} {num / den:+.2f} = \\)"
            f' <span style="background:{HL}"><b>{pred_u:.2f}</b></span>', x0 + 420, y0 + 440, 560, 50, 13,
            valign="middle")
    pg.ask("Will Ann like I5 according to people with her taste?",
           f"probably not: predicted {pred_u:.2f} / 5 — her twins {USERS[v1].split()[1]} and {USERS[v2].split()[1]} "
           f"rated I5 below their own average",
           x0 + 640, y0 + 505, x0 + 770, y0 + 482, w=330)
    pg.bullets(x0 + 420, y0 + 615, 560, [
        "Content-based ② slightly favoured I5 (shared word 'space') — CF disagrees: the crowd knows better",
        "Choose k by validation; ignore users with few co-rated items (significance weighting)",
        "Cost: compare the user with all users → O(#users) per request"], head="Notes:", size=11)
    _pred = {"user": pred_u}

    # ------------------------------------------------------------------ ④ item-based CF
    x0 = 1440
    pg.panel(x0, y0, 960, 760, 4, "Item-based Collaborative Filtering",
             "Items are similar if the same users rate them alike (adjusted cosine)")
    S = [[adj_cos(i, j) for j in range(6)] for i in range(6)]
    hx, hy, c = x0 + 70, y0 + 110, 48
    for j in range(6):
        pg.text(ITEMS[j], hx + j * c, hy - 20, c, 18, 11, bold=True, align="center")
    for i in range(6):
        pg.text(ITEMS[i], hx - 40, hy + i * c, 36, c, 11, bold=True, align="right")
        for j in range(6):
            pg.rect(hx + j * c, hy + i * c, c, c, f"{S[i][j]:+.2f}", fill=sim_color(S[i][j]), size=10,
                    sw=3 if i == TI and j in (3, 5) else 1, stroke=GREEN if i == TI and j in (3, 5) else BLACK)
    pg.text("red = similar, blue = opposite (adjusted cosine)", hx - 40, hy + 6 * c + 8, 340, 16, 10, italic=True)
    pg.math(r"sim(i,j)=\frac{\sum_u (r_{ui}-\bar r_u)(r_{uj}-\bar r_u)}{\sqrt{\sum_u (r_{ui}-\bar r_u)^2}\sqrt{\sum_u (r_{uj}-\bar r_u)^2}}",
            x0 + 30, y0 + 440, 440, 50, 12)
    cand_items = sorted([(S[TI][j], j) for j in range(6) if R[TU][j] is not None and S[TI][j] > 0], reverse=True)[:2]
    num = sum(s * R[TU][j] for s, j in cand_items)
    den = sum(abs(s) for s, _ in cand_items)
    pred_i = num / den
    rx = x0 + 400
    pg.text("Predict r(Ann, I5) from items Ann rated that are similar to I5", rx, y0 + 90, 540, 18, 12, bold=True)
    irows = [["item j", "sim(I5, j)", "Ann's r(j)", "use?"]]
    for j in range(6):
        if R[TU][j] is not None:
            irows.append([ITEMS[j], f"{S[TI][j]:+.3f}", R[TU][j],
                          "✔ top-2" if any(j == jj for _, jj in cand_items) else ""])
    hlr = tuple(k + 1 for k, r_ in enumerate(irows[1:]) if r_[3])
    pg.table(rx, y0 + 115, 340, irows, size=11, h=110, highlight_rows=hlr)
    pg.math(r"\hat r_{ui} = \frac{\sum_{j \in N_k(i)} sim(i,j)\, r_{uj}}{\sum_{j\in N_k(i)} |sim(i,j)|}", rx, y0 + 245,
            400, 56, 15)
    (sa, ja), (sb, jb) = cand_items
    pg.html(f"\\(\\hat r = \\frac{{{sa:.3f}\\cdot{R[TU][ja]} + {sb:.3f}\\cdot{R[TU][jb]}}}{{{sa:.3f}+{sb:.3f}}} = \\)"
            f' <span style="background:{HL}"><b>{pred_i:.2f}</b></span>', rx, y0 + 310, 520, 44, 14,
            valign="middle")
    pg.ask("Do the items most like I5 tell us anything about Ann?",
           f"yes: I5's closest items ({ITEMS[ja]}, {ITEMS[jb]}) got only {R[TU][ja]} and {R[TU][jb]} from Ann "
           f"→ predicted {pred_i:.2f}; same verdict as user-based ({_pred['user']:.2f})",
           rx + 140, y0 + 370, hx + 3 * c + c / 2, hy + TI * c + c / 2, w=380)
    pg.table(x0 + 30, y0 + 520, 900, [
        ["", "User-based ★", "Item-based"],
        ["Neighbours", "similar users", "similar items"],
        ["Stability", "user tastes drift, many users", "item similarities change slowly → precompute"],
        ["Scale", "#users usually ≫ #items → costly", "cheaper online lookup (Amazon, 2003)"],
        ["Explanation", "'people like you liked…'", "'because you liked I4…'"]], size=11, h=120)
    pg.text("Both are k-NN on the rating matrix (see k-NN page) — only the axis differs.", x0 + 30, y0 + 655, 900,
            18, 11, italic=True, align="center")

    # ------------------------------------------------------------------ ⑤ matrix factorization
    x0, y0 = 400, 1780
    pg.panel(x0, y0, 1000, 740, 5, "Matrix Factorization (latent factors, trained by SGD)",
             "Explain every rating with k hidden 'taste' factors: R ≈ U · Vᵀ")
    rnd = seeded(5)
    K = 2
    P = [[rnd.gauss(0, 0.3) for _ in range(K)] for _ in range(6)]
    Q = [[rnd.gauss(0, 0.3) for _ in range(K)] for _ in range(6)]
    bu = [0.0] * 6; bi = [0.0] * 6
    known = [(u, i, R[u][i]) for u in range(6) for i in range(6) if R[u][i] is not None]
    gmu = mean([r for _, _, r in known])
    lr, lam, hist_ = 0.03, 0.05, []
    for ep in range(300):
        rnd.shuffle(known)
        for u, i, r in known:
            e = r - (gmu + bu[u] + bi[i] + sum(P[u][f] * Q[i][f] for f in range(K)))
            bu[u] += lr * (e - lam * bu[u]); bi[i] += lr * (e - lam * bi[i])
            for f in range(K):
                pu, qi = P[u][f], Q[i][f]
                P[u][f] += lr * (e * qi - lam * pu); Q[i][f] += lr * (e * pu - lam * qi)
        rmse = math.sqrt(mean([(r - (gmu + bu[u] + bi[i] + sum(P[u][f] * Q[i][f] for f in range(K)))) ** 2
                               for u, i, r in known]))
        hist_.append(rmse)
    pred = lambda u, i: max(1, min(5, gmu + bu[u] + bi[i] + sum(P[u][f] * Q[i][f] for f in range(K))))
    # matrix-shape diagram
    mx, my = x0 + 30, y0 + 80
    pg.rect(mx, my, 120, 120, "R<br>6 users × 6 items", fill=LIGHTBLUE, raw=True, size=11)
    pg.text("≈", mx + 128, my + 45, 30, 30, 24, bold=True, align="center")
    pg.rect(mx + 165, my, 40, 120, "U<br>6×2", fill="#FFE6CC", raw=True, size=10)
    pg.text("×", mx + 210, my + 45, 24, 30, 20, bold=True, align="center")
    pg.rect(mx + 240, my + 40, 120, 40, "Vᵀ  2×6", fill="#E1D5E7", size=10)
    pg.text("each user = 2 numbers (taste), each item = 2 numbers (character)", mx, my + 128, 380, 30, 10,
            italic=True)
    pg.math(r"\hat r_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i", x0 + 420, y0 + 70, 360, 40, 16)
    pg.math(r"\min \sum_{(u,i)\,known}\left(r_{ui}-\hat r_{ui}\right)^2 + \lambda\left(\lVert\mathbf p_u\rVert^2+\lVert\mathbf q_i\rVert^2+b_u^2+b_i^2\right)",
            x0 + 420, y0 + 115, 560, 44, 13)
    pg.math(r"e_{ui}=r_{ui}-\hat r_{ui};\quad \mathbf p_u \mathrel{+}= \eta\,(e_{ui}\mathbf q_i-\lambda\mathbf p_u);\quad "
            r"\mathbf q_i \mathrel{+}= \eta\,(e_{ui}\mathbf p_u-\lambda\mathbf q_i)", x0 + 420, y0 + 165, 560, 40, 12)
    pg.text(f"μ = {gmu:.2f} (global mean), b = user / item bias, k = 2, η = {lr}, λ = {lam}, 300 epochs",
            x0 + 420, y0 + 205, 560, 18, 10, color="#555555")
    # loss curve
    c = pg.axes(x0 + 60, y0 + 280, 300, 150, (0, 300), (0, max(hist_) * 1.1), "epoch", "RMSE", label_size=11,
                xticks=(0, 100, 200, 300))
    c.poly([(k, h) for k, h in enumerate(hist_) if k % 5 == 0 or k < 10], color=BLUE, width=2)
    c.text(f"train RMSE {hist_[-1]:.2f}", 220, hist_[-1], dy=-14, size=10, color=RED, w=120)
    # filled matrix
    pr = {(u, i): pred(u, i) for u in range(6) for i in range(6) if R[u][i] is None}
    pg.text("Filled matrix (yellow = predicted)", x0 + 420, y0 + 240, 330, 18, 12, bold=True)
    rating_grid(pg, x0 + 420, y0 + 285, cell=36, pred=pr)
    pg.ask("What does the model predict for Ann and I5?",
           f"{pr[(TU, TI)]:.2f} — close to user-based ({_pred['user']:.2f}); MF fills every '?' at once",
           x0 + 420, y0 + 520, x0 + 420 + 70 + TI * 36 + 18, y0 + 285 + 18, w=300)
    # latent space
    lx0 = x0 + 760
    allp = [p for p in P + Q]
    lo = min(min(a for a, _ in allp), min(b for _, b in allp)) - 0.3
    hi = max(max(a for a, _ in allp), max(b for _, b in allp)) + 0.3
    l = pg.axes(lx0, y0 + 300, 210, 210, (lo, hi), (lo, hi), "f_1", "f_2", label_size=11, origin=(0, 0))
    for u in range(6):
        l.scatter([tuple(P[u])], color=BLUE, r=5)
        l.text(USERS[u].split()[1], *P[u], dy=-11, size=9, color=BLUE)
    for i in range(6):
        px_, py_ = l.P(*Q[i])
        pg.rect(px_ - 5, py_ - 5, 10, 10, fill=ORANGE, stroke=ORANGE)
        l.text(ITEMS[i], *Q[i], dy=12, size=9, color=ORANGE)
    pg.text("latent space: ● users  ■ items", lx0, y0 + 250, 220, 18, 11, bold=True)
    def near(u, n=2):
        d = sorted((math.dist(P[u], Q[i]), i) for i in range(6))
        return ", ".join(ITEMS[i] for _, i in d[:n])
    pg.ask("Which users and items belong to the same taste group?",
           f"nearby points: Ann sits by {near(0)}, Dan by {near(3)} — opposite corners = opposite tastes; "
           f"train RMSE {hist_[-1]:.2f} is optimistic on 31 ratings",
           lx0 - 30, y0 + 560, *l.P(*P[TU]), w=250)
    pg.bullets(x0 + 20, y0 + 470, 380, [
        "Dot product p·q = how well a user's taste matches an item",
        "Netflix-prize winner family (SVD, SVD++, ALS for implicit data)",
        "Choose k and λ by validation RMSE / NDCG"], head="Notes:", size=11)

    # ------------------------------------------------------------------ ⑥ evaluation & problems
    x0 = 1440
    pg.panel(x0, y0, 960, 740, 6, "Evaluation & problems", "Rank quality, cold start, sparsity, popularity bias")
    rel = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    total_rel, Kk = 5, 5
    for k, r_ in enumerate(rel):
        pg.rect(x0 + 30 + k * 42, y0 + 80, 38, 38, f"#{k + 1}", fill="#D5E8D4" if r_ else "#F5F5F5",
                stroke="#82B366" if r_ else "#999999", size=10, bold=bool(r_))
    pg.text("Top-10 list for one user: green = item the user actually liked (5 relevant in total)", x0 + 30,
            y0 + 122, 440, 30, 10, italic=True)
    pg.rect(x0 + 26, y0 + 74, 5 * 42 + 2, 50, stroke=RED, dashed=True, sw=1.5)
    pg.text("K = 5", x0 + 26, y0 + 56, 5 * 42, 16, 10, bold=True, color=RED, align="center")
    prec = sum(rel[:Kk]) / Kk
    recall = sum(rel[:Kk]) / total_rel
    dcg = sum(r_ / math.log2(k + 2) for k, r_ in enumerate(rel[:Kk]))
    idcg = sum(1 / math.log2(k + 2) for k in range(min(Kk, total_rel)))
    ndcg = dcg / idcg
    pg.table(x0 + 30, y0 + 165, 430, [
        ["metric (K = 5)", "formula", "value"],
        ["Precision@5", "hits in top-K / K", f"{sum(rel[:Kk])}/5 = {prec:.2f}"],
        ["Recall@5", "hits in top-K / all relevant", f"{sum(rel[:Kk])}/5 = {recall:.2f}"],
        ["NDCG@5", "DCG / ideal DCG", f'<span style="background:{HL}">{dcg:.3f}/{idcg:.3f} = {ndcg:.2f}</span>']],
        size=11, h=100)
    pg.math(r"DCG@K = \sum_{k=1}^{K}\frac{rel_k}{\log_2(k+1)}", x0 + 30, y0 + 275, 300, 44, 14)
    pg.ask("Are the good items near the top of the list?",
           f"NDCG@5 = {ndcg:.2f}: hits at #1, #3, #4 — a perfect ranking (all hits first) would score 1.00",
           x0 + 30, y0 + 330, x0 + 30 + 2 * 42 + 19, y0 + 118, w=270)
    # long tail
    pops = [int(1000 / (k + 1) ** 1.1) for k in range(30)]
    head = sum(pops[:6]) / sum(pops)
    t = pg.axes(x0 + 520, y0 + 90, 400, 150, (0, 30), (0, 1100), "rank", "ratings", label_size=10)
    t.bars([(k + 0.5, v) for k, v in enumerate(pops)], 0.8, fill=LIGHTBLUE, stroke=BLUE)
    t.bars([(k + 0.5, v) for k, v in enumerate(pops[:6])], 0.8, fill=ORANGE, stroke=ORANGE)
    t.text(f"top 20 % of items = {100 * head:.0f} % of ratings", 15, 800, size=10, w=220, color=ORANGE)
    t.text("long tail", 24, 160, size=10, w=80, italic=True)
    pg.text("Popularity bias: CF keeps recommending the head → diversity / exploration terms help", x0 + 520,
            y0 + 270, 420, 30, 10, italic=True)
    pg.ask("Is the recommender only pushing blockbusters?",
           f"check coverage: here the head (6 of 30 items) holds {100 * head:.0f} % of the data, so CF will favour it",
           x0 + 650, y0 + 300, *t.P(3, pops[3]), w=270)
    pg.table(x0 + 30, y0 + 430, 900, [
        ["Problem", "Why it hurts", "Fix"],
        ["Cold start — new user", "no ratings → no neighbours", "popularity, onboarding questions, demographics"],
        ["Cold start — new item", "nobody rated it yet", "content-based (TF-IDF / embeddings) ②"],
        [f"Sparsity ({100 * (36 - 31) / 36:.0f} % here, > 99 % real)", "few co-rated items → noisy similarity",
         "matrix factorization ⑤, implicit feedback (clicks)"],
        ["Popularity bias", "rich get richer, filter bubble", "re-ranking for diversity / novelty"],
        ["Hybrid", "each method has blind spots", "weighted / switching / feature-combined models"]],
        size=11, h=150)
    pg.text("Offline metrics ≠ online success: confirm with an A/B test (Hypothesis Tests page).", x0 + 30,
            y0 + 600, 900, 18, 11, italic=True, align="center")
