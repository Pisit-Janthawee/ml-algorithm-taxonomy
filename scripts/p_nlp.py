"""Bag-of-Words page: pipeline, document-term matrix, vector space, sparsity, TF-IDF, limitations."""
import math
from dio import *

DOCS = ["The cat sat on the mat.", "The dog sat on the log!", "The cat chased the dog."]
CLEAN = [["cat", "sat", "mat"], ["dog", "sat", "log"], ["cat", "chased", "dog"]]
VOCAB = sorted({w for d in CLEAN for w in d})          # cat chased dog log mat sat
DTM = [[d.count(w) for w in VOCAB] for d in CLEAN]


def _chip(pg, x, y, word, fill="#FFFFFF", stroke=BLACK, w=None):
    w = w or 9 * len(word) + 18
    pg.rect(x, y, w, 24, word, fill=fill, stroke=stroke, size=11, rounded=True)
    return w


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① pipeline
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 640, 1, "From Raw Text to Tokens", "Background: text cleaning + tokenization + vocabulary")
    rows = [("Raw text (corpus)", None), ("① Clean: lowercase, remove punctuation", None),
            ("② Remove stop-words ('the', 'on')", None), ("③ Tokenize → list of words", None)]
    yy = y0 + 75
    labels = ["Raw corpus", "Lowercase + strip punctuation", "Remove stop-words", "Tokens"]
    stages = [DOCS,
              [d.lower().replace(".", "").replace("!", "") for d in DOCS],
              [" ".join(c) for c in CLEAN],
              None]
    for k, lab in enumerate(labels):
        pg.text(lab, x0 + 20, yy, 200, 20, 12, bold=True)
        for j in range(3):
            if stages[k] is not None:
                txt = stages[k][j]
                pg.text(f"D{j + 1}:  \"{txt}\"" if k < 3 else txt, x0 + 230, yy + j * 22, 390, 20, 11,
                        color=RED if k == 0 else BLACK)
            else:
                pg.text(f"D{j + 1}:", x0 + 230, yy + j * 26 + 2, 30, 20, 11)
                cx = x0 + 262
                for wd in CLEAN[j]:
                    cx += _chip(pg, cx, yy + j * 26, wd, fill=LIGHTBLUE) + 6
        h = 70 if k < 3 else 82
        if k < 3:
            pg.line([(x0 + 110, yy + 22), (x0 + 110, yy + h + 2)], arrow="classic", color=GREY)
        yy += h + 8
    # vocabulary
    pg.text("④ Vocabulary = unique tokens (sorted)", x0 + 20, yy + 6, 400, 20, 12, bold=True)
    cx = x0 + 30
    for i, wd in enumerate(VOCAB):
        w = _chip(pg, cx, yy + 34, wd, fill=HL)
        pg.text(str(i), cx, yy + 60, w, 14, 9, align="center", color="#555555")
        cx += w + 10
    pg.text("|V| = 6  (index under each word)", x0 + 420, yy + 34, 200, 24, 11, italic=True)
    pg.bullets(x0 + 20, yy + 90, 600, [
        "Stop-words carry little meaning; optional extra steps: stemming (chased → chase), lemmatization",
        "Each document becomes a 'bag' — only <b>which</b> words and <b>how many</b>, not their order"],
        head="Notes:")
    pg.ask("Which features will the model actually see?",
           f"the {len(VOCAB)} vocabulary words: every document becomes {len(VOCAB)} numbers (one per word)",
           x0 + 330, yy + 155, x0 + 300, yy + 58, w=290)

    # ---------------------------------------------------------------- ② document-term matrix
    x0 = 1080
    pg.panel(x0, y0, 620, 640, 2, "Document-Term Matrix (count vectors)", "Background: vector space representation")
    head = ["Doc"] + [f"<b>{w}</b>" for w in VOCAB]
    rows = [head] + [[f"D{i + 1}"] + [f"<b>{v}</b>" if v else "0" for v in r] for i, r in enumerate(DTM)]
    pg.table(x0 + 40, y0 + 80, 540, rows, size=13, h=130)
    pg.note("one row = one document vector", x0 + 380, y0 + 225, x0 + 400, y0 + 180, w=200, color=BLUE)
    pg.note("one column = one word (feature)", x0 + 40, y0 + 225, x0 + 160, y0 + 105, w=200)
    pg.math("x_{d,w} = \\text{count}(w, d)", x0 + 40, y0 + 260, 540, 36, 18)
    pg.math("\\vec{x}_{D1} = [\\,1,\\;0,\\;0,\\;0,\\;1,\\;1\\,]", x0 + 40, y0 + 300, 540, 36, 18)
    # bar chart of D1, D3
    for k, (di, col) in enumerate([(0, BLUE), (2, ORANGE)]):
        gx = x0 + 50 + k * 290
        p = pg.axes(gx, y0 + 370, 220, 110, (-0.5, 5.5), (0, 1.5), "", "count", label_size=10, yticks=(1,))
        if k == 0: p1 = p
        p.bars([(i, DTM[di][i]) for i in range(6)], 0.6, fill=col, opacity=70)
        for i, wd in enumerate(VOCAB):
            p.text(wd, i, 0, w=50, dy=12, size=9)
        pg.text(f"D{di + 1} vector", gx, y0 + 345, 220, 18, 12, bold=True, align="center", color=col)
    pg.bullets(x0 + 20, y0 + 520, 340, [
        "Shape = (number of documents × vocabulary size) = 3 × 6",
        "<b>Binary BoW</b>: 1 if the word appears, else 0 (good for short texts)",
        "These vectors can feed any classifier — e.g. Multinomial Naive Bayes, Logistic Regression"],
        head="Notes:", h=110)
    pg.ask("How is one document turned into numbers?",
           f"D1 = {DTM[0]}: one count per vocabulary word; the whole corpus is a {len(DTM)} × {len(VOCAB)} matrix",
           x0 + 375, y0 + 515, *p1.P(5, 1.0), w=230)

    # ---------------------------------------------------------------- ③ vector space
    x0 = 1740
    pg.panel(x0, y0, 660, 640, 3, "Documents as Vectors (similarity)", "Background: vector space + cosine similarity")
    p = pg.axes(x0 + 80, y0 + 110, 300, 300, (0, 1.4), (0, 1.4), "\\text{cat}", "\\text{dog}", label_size=13,
                xticks=(1,), yticks=(1,))
    vecs = [("D1", (1, 0), BLUE), ("D2", (0, 1), GREEN), ("D3", (1, 1), ORANGE)]
    for name, (a, b), col in vecs:
        pg.line([p.P(0, 0), p.P(a, b)], arrow="classic", color=col, width=2)
        p.text(name, a, b, w=40, dx=16, dy=-10, size=13, bold=True, color=col)
    p.vline(1, 0, 1); p.hline(1, 0, 1)
    pg.note("angle θ between vectors", x0 + 300, y0 + 330, *p.P(0.35, 0.18), w=150)
    pg.text("(only the 'cat' and 'dog' dimensions are drawn)", x0 + 70, y0 + 440, 320, 18, 10, italic=True,
            color="#555555")
    pg.math("\\cos\\theta = \\frac{\\vec{a}\\cdot\\vec{b}}{\\lVert\\vec{a}\\rVert\\,\\lVert\\vec{b}\\rVert}", x0 + 400,
            y0 + 120, 240, 60, 18)
    pg.math("\\cos(D1,D3) = \\frac{1}{\\sqrt{3}\\sqrt{3}} = 0.33", x0 + 400, y0 + 200, 240, 50, 13)
    pg.math("\\cos(D1,D1) = 1", x0 + 400, y0 + 255, 240, 30, 13)
    pg.html("1 = same direction (similar)<br>0 = no shared words", x0 + 410, y0 + 295, 230, 40, 11)
    pg.ask("How similar are two documents?",
           "cos(D1, D3) = 0.33: they share only 'cat'; 1 = same words, 0 = nothing in common",
           x0 + 405, y0 + 350, *p.P(0.75, 0.35), w=235)
    pg.bullets(x0 + 20, y0 + 470, 620, [
        "Each word is one axis → a 50,000-word vocabulary = a 50,000-dimensional space",
        "Similar documents point in similar directions; length depends on document size, so use the angle",
        "Used for search, duplicate detection, document clustering (K-Means on BoW vectors)"],
        head="Notes:")

    # ---------------------------------------------------------------- ④ sparsity
    x0, y0 = 400, 920
    pg.panel(x0, y0, 640, 700, 4, "Sparse Vectors", "Real vocabularies are huge, documents are short")
    pg.text("A news headline (8 words) in a 50,000-word vocabulary:", x0 + 20, y0 + 75, 600, 20, 12, bold=True)
    r = seeded(11)
    on = set(r.sample(range(40), 5))
    cw = 13
    for i in range(40):
        pg.rect(x0 + 30 + i * cw, y0 + 110, cw, 30, "1" if i in on else "", fill=BLUE if i in on else "#FFFFFF",
                stroke="#999999", size=8, color="#FFFFFF")
    pg.text("… 49,960 more zeros", x0 + 30 + 40 * cw + 6, y0 + 112, 80, 26, 10, italic=True)
    pg.note("non-zero = word present", x0 + 30, y0 + 160, x0 + 30 + min(on) * cw + 6, y0 + 140, w=160, color=BLUE)
    pg.text("Sparsity ≈ 99.98 % zeros", x0 + 320, y0 + 160, 280, 20, 13, bold=True, color=RED)
    pg.text("Store only (index : count) pairs  →  compressed sparse row (CSR)", x0 + 20, y0 + 210, 600, 20, 12,
            bold=True)
    pairs = ", ".join(f"{i * 1250 + 17}:1" for i in sorted(on))
    pg.rect(x0 + 40, y0 + 240, 560, 36, "{ " + pairs + " }", fill="#F5F5F5", stroke="#999999", size=12)
    pg.text("Why so sparse? Zipf's law of word frequency", x0 + 20, y0 + 300, 400, 20, 12, bold=True)
    p = pg.axes(x0 + 80, y0 + 350, 300, 170, (0, 50), (0, 1100), "\\text{rank}", "\\text{frequency}",
                label_size=11, xticks=(1, 10, 20, 30, 40, 50))
    p.bars([(r_, 1000 / r_) for r_ in range(1, 51)], 0.7, fill=BLUE, stroke="none", opacity=60)
    p.curve(lambda v: 1000 / v, 0.95, 50, color=RED, width=1.5)
    pg.math("f(r)\\propto\\frac{1}{r}", x0 + 250, y0 + 360, 120, 40, 15, color=RED)
    pg.note("'the', 'of', 'and' … (stop-words)", x0 + 140, y0 + 325, *p.P(1.3, 900), w=190)
    pg.note("long tail: thousands of rare words", x0 + 190, y0 + 435, *p.P(35, 40), w=190)
    pg.ask("Why is a BoW vector almost all zeros?",
           "a few words are very common, most are rare (Zipf): one text uses only a handful of the vocabulary",
           x0 + 405, y0 + 380, *p.P(44, 26), w=215)
    pg.bullets(x0 + 20, y0 + 560, 600, [
        "Limit the vocabulary: <b>max_features</b>, min_df (drop rare words), max_df (drop very common words)",
        "Sparse matrices keep memory small — scikit-learn CountVectorizer returns scipy.sparse",
        "Dense models (neural nets) prefer compact <b>embeddings</b> (Word2Vec, BERT) instead"],
        head="Notes:")

    # ---------------------------------------------------------------- ⑤ TF-IDF
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 5, "Next Step: TF-IDF weighting", "Down-weight words that appear everywhere")
    pg.math("\\text{tf-idf}(w,d) = tf(w,d)\\times\\log_{10}\\frac{N}{df(w)}", x0 + 40, y0 + 75, 540, 50, 19)
    pg.note("count of word w in document d", x0 + 30, y0 + 135, x0 + 280, y0 + 108, w=190, color=BLUE)
    pg.note("N = number of documents", x0 + 250, y0 + 150, x0 + 470, y0 + 92, w=160)
    pg.note("df = documents containing w", x0 + 420, y0 + 135, x0 + 500, y0 + 112, w=170, color=RED)
    N = 3
    df = {w: sum(1 for d in CLEAN if w in d) for w in VOCAB}
    idf = {w: math.log10(N / df[w]) for w in VOCAB}
    rows = [["word"] + VOCAB, ["df"] + [str(df[w]) for w in VOCAB],
            ["idf"] + [f"{idf[w]:.3f}" for w in VOCAB],
            ["D3 tf"] + [str(DTM[2][i]) for i in range(6)],
            ["D3 tf-idf"] + [f"<b>{DTM[2][i] * idf[w]:.3f}</b>" for i, w in enumerate(VOCAB)]]
    pg.table(x0 + 30, y0 + 180, 560, rows, size=12, highlight_rows=(4,), h=150)
    p = pg.axes(x0 + 90, y0 + 380, 420, 130, (-0.5, 5.5), (0, 0.55), "", "tf\\text{-}idf", label_size=11,
                yticks=(0.25, 0.5))
    p.bars([(i, DTM[2][i] * idf[w]) for i, w in enumerate(VOCAB)], 0.6, fill=ORANGE, opacity=70)
    for i, wd in enumerate(VOCAB):
        p.text(wd, i, 0, w=60, dy=12, size=10)
    pg.note("'chased' is unique to D3 → highest weight", x0 + 330, y0 + 360, *p.P(1, 0.477), w=230, color=RED)
    pg.bullets(x0 + 20, y0 + 560, 340, [
        "Word in every document → idf = log(1) = 0 → ignored (acts like an automatic stop-word list)",
        "Rare, specific words get high weight → better features for classification & search",
        "Same matrix shape and sparsity as BoW — just re-weighted counts"],
        head="Notes:", h=120)
    best = max(VOCAB, key=lambda w: DTM[2][VOCAB.index(w)] * idf[w])
    pg.ask("Which words best describe document D3?",
           f"'{best}' (tf-idf = {DTM[2][VOCAB.index(best)] * idf[best]:.3f}): only D3 uses it; shared words "
           "like 'cat', 'dog' weigh less",
           x0 + 375, y0 + 555, x0 + 268, y0 + 302, w=230)

    # ---------------------------------------------------------------- ⑥ limitations
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 6, "Limitations → why N-grams & Embeddings exist")
    pg.text("1. Word order is lost", x0 + 20, y0 + 60, 400, 20, 13, bold=True)
    s1 = pg.rect(x0 + 40, y0 + 95, 190, 36, "\"dog bites man\"", size=13)
    s2 = pg.rect(x0 + 40, y0 + 160, 190, 36, "\"man bites dog\"", size=13)
    v = pg.rect(x0 + 330, y0 + 120, 200, 50, "bites:1  dog:1  man:1", fill=LIGHTRED, stroke=RED, size=13)
    pg.connect(s1, v, orth=False)
    pg.connect(s2, v, orth=False)
    pg.html("<b>✘ identical vectors,<br>different meaning!</b>", x0 + 540, y0 + 120, 120, 50, 11, color=RED,
            valign="middle")
    pg.text("Fix: N-grams (bigrams keep local order)", x0 + 40, y0 + 215, 500, 20, 12, bold=True, color=GREEN)
    cx = x0 + 40
    for wd in ["dog bites", "bites man"]:
        cx += _chip(pg, cx, y0 + 245, wd, fill=LIGHTBLUE) + 8
    cx = x0 + 340
    for wd in ["man bites", "bites dog"]:
        cx += _chip(pg, cx, y0 + 245, wd, fill="#D5E8D4") + 8
    pg.text("→ now different features ✔", x0 + 40, y0 + 275, 400, 20, 11, color=GREEN)
    pg.text("2. No meaning / similarity between words", x0 + 20, y0 + 315, 500, 20, 13, bold=True)
    pg.text("'good' and 'great' are as different as 'good' and 'table' (orthogonal axes)", x0 + 40, y0 + 340, 600, 20,
            11)
    p = pg.axes(x0 + 70, y0 + 380, 170, 140, (0, 1.3), (0, 1.3), "", "", label_size=10)
    pg.line([p.P(0, 0), p.P(1.1, 0)], arrow="classic", color=BLUE, width=2)
    pg.line([p.P(0, 0), p.P(0, 1.1)], arrow="classic", color=ORANGE, width=2)
    p.text("good", 1.1, 0, w=50, dy=12, size=11, color=BLUE)
    p.text("great", 0, 1.1, w=50, dx=30, size=11, color=ORANGE)
    pg.text("BoW: cos = 0", x0 + 80, y0 + 530, 150, 18, 11, align="center", color=RED)
    q = pg.axes(x0 + 360, y0 + 380, 170, 140, (0, 1.3), (0, 1.3), "", "", label_size=10)
    pg.line([q.P(0, 0), q.P(1.0, 0.55)], arrow="classic", color=BLUE, width=2)
    pg.line([q.P(0, 0), q.P(0.85, 0.75)], arrow="classic", color=ORANGE, width=2)
    q.text("good", 1.0, 0.55, w=50, dx=24, size=11, color=BLUE)
    q.text("great", 0.85, 0.75, w=50, dx=10, dy=-12, size=11, color=ORANGE)
    pg.text("Embedding: cos ≈ 0.9", x0 + 340, y0 + 530, 200, 18, 11, align="center", color=GREEN)
    pg.bullets(x0 + 20, y0 + 565, 380, [
        "Evolution: <b>BoW</b> → TF-IDF → N-grams → Word2Vec / GloVe (dense meaning) → <b>BERT</b> (context)",
        "BoW + TF-IDF remain fast, strong baselines for spam filtering and topic classification",
        "Your EA pipeline uses BERT for headline sentiment — BoW + Logistic Regression is the baseline to beat"],
        head="Notes:", h=125)
    pg.ask("Can BoW tell that 'good' and 'great' mean the same?",
           "no — each word is its own axis (cos = 0); embeddings put them close (cos ≈ 0.9)",
           x0 + 420, y0 + 565, *q.P(0.92, 0.66), w=225)
