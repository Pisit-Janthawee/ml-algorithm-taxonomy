"""Tiny draw.io drawing library that imitates the hand-made style of 'ML Empiral Diagram.drawio':
white canvas, Helvetica, black thin lines, arrowed axes, blue (#3333FF) data dots, dashed guides,
LaTeX formulas with small annotation lines, 'Where:' bullet lists, yellow highlights.
All coordinates are absolute page coordinates (no containers) so pages stay easy to edit.
"""
import html, math
import xml.etree.ElementTree as ET

E = html.escape
BLUE, RED, GREY, BLACK, GREEN, ORANGE = "#3333FF", "#FF0000", "#808080", "#000000", "#009900", "#FF8000"
HL = "#FFFF00"          # yellow highlight (original uses it for key terms)
LIGHTRED, LIGHTBLUE, LIGHTGREY = "#FFCCCC", "#CCE5FF", "#EEEEEE"
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


class Page:
    def __init__(self, name, pid):
        self.pid, self.n = pid, 0
        self.d = ET.Element("diagram", name=name, id=pid)
        m = ET.SubElement(self.d, "mxGraphModel", dx="1600", dy="900", grid="1", gridSize="10", guides="1",
                          tooltips="1", connect="1", arrows="1", fold="1", page="0", pageScale="1",
                          pageWidth="850", pageHeight="1100", background="#ffffff", math="1", shadow="0")
        self.root = ET.SubElement(m, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")

    def max_y(self, min_x=400):
        """Bottom of the existing drawing in the body area (x >= min_x): place extra panels below this."""
        best = 0
        for g in self.root.iter("mxGeometry"):
            if g.get("relative") == "1" or g.get("y") is None: continue
            if float(g.get("x", 0)) >= min_x:
                best = max(best, float(g.get("y")) + float(g.get("height", 0)))
        return best

    # ------------------------------------------------------------------ low level
    def _id(self):
        self.n += 1
        return f"{self.pid}-{self.n}"

    def vertex(self, label, x, y, w, h, style, link=None):
        i = self._id()
        if link:
            u = ET.SubElement(self.root, "UserObject", label=label, link=link, id=i)
            c = ET.SubElement(u, "mxCell", style=style, vertex="1", parent="1")
        else:
            c = ET.SubElement(self.root, "mxCell", id=i, value=label, style=style, vertex="1", parent="1")
        ET.SubElement(c, "mxGeometry", {"x": f"{x:.1f}", "y": f"{y:.1f}", "width": f"{w:.1f}",
                                        "height": f"{h:.1f}", "as": "geometry"})
        return i

    def line(self, pts, color=BLACK, width=1, dashed=False, arrow=None, start_arrow=None, curved=False,
             label=None, size=10, opacity=None):
        """Free polyline through page points. arrow: None|'classic'|'block'|'open'."""
        st = [f"endArrow={arrow or 'none'}", "html=1", "rounded=0", f"strokeColor={color}",
              f"strokeWidth={width}", f"fontSize={size}", "fontFamily=Helvetica"]
        if arrow: st.append("endFill=1;endSize=6")
        if start_arrow: st.append(f"startArrow={start_arrow};startFill=1;startSize=6")
        if dashed: st.append("dashed=1")
        if curved: st.append("curved=1")
        if opacity: st.append(f"opacity={opacity}")
        c = ET.SubElement(self.root, "mxCell", id=self._id(), value=label or "", style=";".join(st) + ";",
                          edge="1", parent="1")
        g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(g, "mxPoint", {"x": f"{pts[0][0]:.1f}", "y": f"{pts[0][1]:.1f}", "as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", {"x": f"{pts[-1][0]:.1f}", "y": f"{pts[-1][1]:.1f}", "as": "targetPoint"})
        if len(pts) > 2:
            a = ET.SubElement(g, "Array", {"as": "points"})
            for x, y in pts[1:-1]:
                ET.SubElement(a, "mxPoint", {"x": f"{x:.1f}", "y": f"{y:.1f}"})

    def connect(self, src, tgt, color=BLACK, arrow="classic", dashed=False, orth=True, label=None, width=1):
        st = (f"endArrow={arrow or 'none'};endFill=1;html=1;rounded=0;strokeColor={color};strokeWidth={width};"
              f"fontFamily=Helvetica;fontSize=10;" + ("edgeStyle=orthogonalEdgeStyle;" if orth else "")
              + ("dashed=1;" if dashed else ""))
        c = ET.SubElement(self.root, "mxCell", id=self._id(), value=label or "", style=st, edge="1",
                          parent="1", source=src, target=tgt)
        ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})

    # ------------------------------------------------------------------ text
    def text(self, label, x, y, w=200, h=20, size=12, bold=False, italic=False, align="left", valign="middle",
             color=None, bg=None, link=None, raw=False):
        fs = (1 if bold else 0) + (2 if italic else 0)
        st = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={align};"
              f"verticalAlign={valign};fontFamily=Helvetica;fontSize={size};fontStyle={fs};"
              f"fontColor={color or 'default'};")
        if bg: st += f"labelBackgroundColor={bg};"
        return self.vertex(label if raw else E(label), x, y, w, h, st, link)

    def html(self, label, x, y, w=200, h=20, size=12, align="left", valign="top", color=None, link=None):
        """Text cell with raw HTML / LaTeX label."""
        return self.text(label, x, y, w, h, size, align=align, valign=valign, color=color, link=link, raw=True)

    def math(self, tex, x, y, w=300, h=40, size=14, align="center", color=None, inline=False):
        lab = f"\\({tex}\\)" if inline else f"$${tex}$$"
        return self.html(lab, x, y, w, h, size, align=align, valign="middle", color=color)

    def title(self, label, x, y, w=700, size=24, align="left"):
        return self.text(label, x, y, w, size * 1.6, size, bold=True, align=align)

    def heading(self, label, x, y, w=500, size=16):
        return self.text(label, x, y, w, 26, size, bold=True)

    def bullets(self, x, y, w, items, head=None, size=11, h=None):
        """'Where:' style bullet list. Items are raw HTML (may contain \\( \\) math)."""
        body = (f"<b>{head}</b>" if head else "") + \
               '<ul style="margin:2px 0 0 0;padding-left:16px">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
        h = h or (len(items) * (size + 9) + (size + 8 if head else 0) + 6)
        self.html(body, x, y, w, h, size)
        return h

    def note(self, label, tx, ty, px, py, w=160, size=10, color="#333333", align="left", raw=False,
             lcolor="#555555"):
        """Annotation text at (tx,ty) with a thin line pointing to (px,py) - like the original's callouts."""
        h = size + 8
        self.text(label, tx, ty, w, h, size, align=align, color=color, raw=raw)
        ax = tx if px < tx else (tx + w if px > tx + w else px)
        if align == "right": ax = tx + w if px > tx else ax
        ay = ty + h / 2
        self.line([(ax, ay), (px, py)], color=lcolor, width=0.75)

    def ask(self, question, answer, x, y, px=None, py=None, w=260, size=10, targets=()):
        """Interpretation callout: 'What question does this answer?' + how to read it, with an arrow
        to the component it explains. (px, py) = arrow tip; targets = extra arrow tips.
        question / answer are raw HTML (may contain \\( \\) math)."""
        cpl = max(10, int(w / (size * 0.52)))           # characters per line
        lines = 1 + math.ceil(len(question) / cpl) + math.ceil((len(answer) + 2) / cpl)
        h = lines * (size + 4) + 12
        lab = (f'<b><font color="#7030A0">❓ Doing this answers:</font></b><br><i>{question}</i>'
               f'<br><font color="#333333">→ {answer}</font>')
        self.vertex(lab, x, y, w, h,
                    "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5EEFB;strokeColor=#9673A6;"
                    f"strokeWidth=1;align=left;verticalAlign=top;spacing=6;fontFamily=Helvetica;fontSize={size};")
        tips = ([(px, py)] if px is not None else []) + list(targets)
        for tx, ty in tips:
            sx = x if tx < x else (x + w if tx > x + w else tx)
            sy = y if ty < y else (y + h if ty > y + h else y + h / 2)
            if x <= tx <= x + w and not (y <= ty <= y + h):
                sx = tx
            self.line([(sx, sy), (tx, ty)], color="#9673A6", width=1.2, dashed=True, arrow="classic")
        return h

    # ------------------------------------------------------------------ shapes
    def rect(self, x, y, w, h, label="", fill="none", stroke=BLACK, dashed=False, size=12, bold=False,
             rounded=False, sw=1, align="center", valign="middle", link=None, color=None, raw=False, opacity=None):
        st = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
              f"strokeWidth={sw};fontFamily=Helvetica;fontSize={size};fontStyle={1 if bold else 0};"
              f"align={align};verticalAlign={valign};fontColor={color or 'default'};spacing=4;")
        if dashed: st += "dashed=1;"
        if opacity is not None: st += f"opacity={opacity};"
        return self.vertex(label if raw else E(label), x, y, w, h, st, link)

    def ellipse(self, x, y, w, h, label="", fill="none", stroke=BLACK, sw=1, size=12, dashed=False, bold=False,
                color=None, raw=False, opacity=None):
        st = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
              f"fontFamily=Helvetica;fontSize={size};fontStyle={1 if bold else 0};fontColor={color or 'default'};")
        if dashed: st += "dashed=1;"
        if opacity is not None: st += f"opacity={opacity};"
        return self.vertex(label if raw else E(label), x, y, w, h, st)

    def dot(self, cx, cy, r=4, color=BLUE, stroke="none"):
        return self.ellipse(cx - r, cy - r, 2 * r, 2 * r, fill=color, stroke=stroke)

    def bracket(self, x, y, w, h, rotation=0, label="", size=10):
        st = (f"shape=curlyBracket;whiteSpace=wrap;html=1;rounded=1;labelPosition=left;verticalLabelPosition=middle;"
              f"align=right;verticalAlign=middle;rotation={rotation};fontSize={size};strokeColor={BLACK};")
        return self.vertex(E(label), x, y, w, h, st)

    def table(self, x, y, w, rows, header=True, size=11, h=None, col_align=None, highlight_rows=()):
        """HTML table in a text cell (the original draws tables this way)."""
        out = ['<table style="border-collapse:collapse;width:100%%;font-size:%dpx" border="1" cellpadding="3">' % size]
        for ri, r in enumerate(rows):
            tag = "th" if header and ri == 0 else "td"
            bg = ' style="background:#FFFF00"' if ri in highlight_rows else (
                 ' style="background:#EEEEEE"' if header and ri == 0 else "")
            out.append(f"<tr{bg}>" + "".join(f"<{tag}>{c}</{tag}>" for c in r) + "</tr>")
        out.append("</table>")
        h = h or len(rows) * (size + 10) + 4
        self.html("".join(out), x, y, w, h, size, align="center")
        return h

    def panel(self, x, y, w, h, num, title, sub=None):
        """Section frame: thin dashed rectangle + bold numbered heading (original uses dashed 0.5 frames)."""
        self.rect(x, y, w, h, stroke="#999999", dashed=True, sw=0.5)
        self.text(f"{CIRCLED[num - 1] if num else ''} {title}".strip(), x + 14, y + 10, w - 28, 26, 16, bold=True)
        if sub: self.text(sub, x + 14, y + 36, w - 28, 18, 11, color="#555555")

    # ------------------------------------------------------------------ plotting
    def axes(self, x, y, w, h, xr=(0, 1), yr=(0, 1), xlabel="X", ylabel="Y", label_size=14, origin=None,
             arrow=True, xticks=(), yticks=(), tick_fmt="{:g}"):
        """Draw axes in box (x,y,w,h). Returns a Plot mapping data -> page coordinates."""
        p = Plot(self, x, y, w, h, xr, yr)
        ox, oy = p.P(origin[0] if origin else xr[0], origin[1] if origin else yr[0])
        a = "classic" if arrow else None
        self.line([(x - 4, oy), (x + w + 8, oy)], arrow=a)
        self.line([(ox, y + h + 4), (ox, y - 8)], arrow=a)
        if xlabel: self.math(xlabel, x + w + 6, oy - 4, 60, 24, label_size, align="left")
        if ylabel: self.math(ylabel, ox - 30, y - 34, 60, 24, label_size)
        for t in xticks:
            tx, _ = p.P(t, 0)
            self.line([(tx, oy), (tx, oy + 4)])
            self.text(tick_fmt.format(t), tx - 25, oy + 4, 50, 14, 9, align="center", color="#555555")
        for t in yticks:
            _, ty = p.P(0, t)
            self.line([(ox - 4, ty), (ox, ty)])
            self.text(tick_fmt.format(t), ox - 56, ty - 7, 50, 14, 9, align="right", color="#555555")
        return p


class Plot:
    def __init__(self, pg, x, y, w, h, xr, yr):
        self.pg, self.x, self.y, self.w, self.h, self.xr, self.yr = pg, x, y, w, h, xr, yr

    def P(self, xd, yd):
        return (self.x + (xd - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w,
                self.y + self.h - (yd - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h)

    def curve(self, f, x0=None, x1=None, n=48, **kw):
        x0 = self.xr[0] if x0 is None else x0
        x1 = self.xr[1] if x1 is None else x1
        kw.setdefault("curved", True)
        lo, hi = min(self.yr), max(self.yr)
        runs, cur = [], []
        for i in range(n + 1):           # clip to the y-range: split into visible runs
            xv = x0 + (x1 - x0) * i / n
            yv = f(xv)
            if lo <= yv <= hi:
                cur.append(self.P(xv, yv))
            else:
                if len(cur) > 1: runs.append(cur)
                cur = []
        if len(cur) > 1: runs.append(cur)
        for r in runs:
            self.pg.line(r, **kw)
        return [p for r in runs for p in r]

    def poly(self, data_pts, **kw):
        self.pg.line([self.P(a, b) for a, b in data_pts], **kw)

    def scatter(self, data_pts, color=BLUE, r=3.5):
        for a, b in data_pts:
            self.pg.dot(*self.P(a, b), r=r, color=color)

    def vline(self, xd, y0=None, y1=None, **kw):
        kw.setdefault("dashed", True); kw.setdefault("color", GREY)
        self.pg.line([self.P(xd, self.yr[0] if y0 is None else y0), self.P(xd, self.yr[1] if y1 is None else y1)], **kw)

    def hline(self, yd, x0=None, x1=None, **kw):
        kw.setdefault("dashed", True); kw.setdefault("color", GREY)
        self.pg.line([self.P(self.xr[0] if x0 is None else x0, yd), self.P(self.xr[1] if x1 is None else x1, yd)], **kw)

    def bars(self, data, width, fill=BLUE, stroke=BLACK, opacity=None, labels=False, size=9):
        """data = [(x_center, height)] in data units."""
        for xc, hgt in data:
            x0, yt = self.P(xc - width / 2, hgt)
            x1, yb = self.P(xc + width / 2, 0 if self.yr[0] < 0 else self.yr[0])   # bars grow from the zero line
            top, bot = min(yt, yb), max(yt, yb)
            self.pg.rect(x0, top, x1 - x0, bot - top, fill=fill, stroke=stroke, opacity=opacity)
            if labels:
                self.pg.text(f"{hgt:g}", x0 - 10, top - 15, x1 - x0 + 20, 14, size, align="center")

    def fill_under(self, f, x0, x1, n=16, color=RED, opacity=60):
        """Approximate shaded area under curve with thin bars (draw.io has no path fill for free polylines)."""
        step = (x1 - x0) / n
        for i in range(n):
            xa = x0 + i * step
            xm = xa + step / 2
            pa, pt = self.P(xa, f(xm)), self.P(xa + step, self.yr[0])
            self.pg.rect(pa[0], pa[1], pt[0] - pa[0], pt[1] - pa[1], fill=color, stroke="none", opacity=opacity)

    def text(self, label, xd, yd, w=120, h=18, size=10, align="center", dx=0, dy=0, **kw):
        px, py = self.P(xd, yd)
        ax = {"left": 0, "center": w / 2, "right": w}[align]
        return self.pg.text(label, px - ax + dx, py - h / 2 + dy, w, h, size, align=align, **kw)

    def math(self, tex, xd, yd, w=80, h=24, size=12, dx=0, dy=0, **kw):
        px, py = self.P(xd, yd)
        return self.pg.math(tex, px - w / 2 + dx, py - h / 2 + dy, w, h, size, **kw)


# ---------------------------------------------------------------------- helpers
def normal_pdf(x, mu=0.0, s=1.0):
    return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def seeded(seed):
    import random
    r = random.Random(seed)
    return r
