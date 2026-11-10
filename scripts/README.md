# ML Taxonomy diagram generator

Builds `outputs/ML Algorithm Taxonomy.drawio` (overview + one page per category) in the hand-drawn
style of `inputs/ML Empiral Diagram.drawio`.

## Rebuild

```
cd scripts
python -I build.py "../outputs/ML Algorithm Taxonomy.drawio"
```

Set `STRICT=1` to stop on the first page error (otherwise a failing page is replaced by a placeholder).
Rebuilding overwrites any manual edits made in draw.io — copy the file first if you edited it by hand.

## Files

| File | Role |
|---|---|
| `build.py` | Assembles the overview and all pages; repairs stray control characters in labels |
| `dio.py` | Drawing primitives (text, math, axes/plots, notes, tables, panels) |
| `taxdata.py` / `taxdata_raw.py` | Categories, algorithm families, base pick, "built on" chain, assumptions |
| `p_<id>.py` | Main drawing of a category page (`build(pg, cat)`) |
| `p_<id>_extra*.py` | Extra panels appended below a page (loaded automatically, sorted by name) |
| `l_common.py` | Shared numeric helpers used by some extra modules |
| `render.py` | Renders a page (or a region) to PNG with headless Chrome + the draw.io viewer, for checking |

## Add a page

1. Add a category dict to `taxdata.py` (id, name, par, desc, groups, pick) and append it to `CATS`;
   add `BUILT_ON[id]` and `ASSUMPTIONS[id]`.
2. Create `p_<id>.py` with `def build(pg, cat):` drawing panels at x ≥ 400, y ≥ 240.
3. Rebuild. Check with `python -I render.py <abs .drawio> <page index> <abs out.png> [x0 y0 x1 y1]`.

Use raw strings (`r"\alpha"`) for LaTeX in Python code.
