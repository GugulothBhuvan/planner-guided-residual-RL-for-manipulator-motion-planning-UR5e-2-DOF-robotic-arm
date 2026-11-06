"""Convert report.html to a Word document (report.docx): headings, text with sub/superscripts, tables, figures, equations,
code blocks, a contents field and static lists of figures, tables and videos. Videos become a caption line.

    python html_to_docx.py [report.html] [report.docx]        (needs python-docx and beautifulsoup4)
"""
import re
import sys
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.html"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "report.docx"
soup = BeautifulSoup(src.read_text(encoding="utf-8"), "html.parser")
main = soup.find("main")

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)  # A4
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1.0))
TEXT_W = 6.27

BODY = "Times New Roman"
st = doc.styles
st["Normal"].font.name = BODY
st["Normal"].font.size = Pt(11)
st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), BODY)
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.15
for name, size, before, after in (("Heading 1", 18, 18, 10), ("Heading 2", 14, 14, 6), ("Heading 3", 12, 10, 4), ("Title", 22, 0, 12)):
    s = st[name]
    s.font.name = "Arial"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor(0x1D, 0x24, 0x33)
    s.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    s.element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    s.element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True
cap = st["Caption"]
cap.font.name = BODY
cap.font.size = Pt(9.5)
cap.font.italic = False
cap.font.bold = False
cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
cap.paragraph_format.space_after = Pt(10)


def clean(t):
    return re.sub(r"\s+", " ", t.replace(" ", " "))


def add_field(par, instr):
    for typ, txt in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        if typ:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), typ)
            r._r.append(fc)
        elif txt == instr:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = instr
            r._r.append(it)
        else:
            t = OxmlElement("w:t")
            t.text = txt
            r._r.append(t)


# page numbers
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_field(fp, "PAGE")

# update fields (the contents) when the file is opened
settings = doc.settings.element
uf = OxmlElement("w:updateFields")
uf.set(qn("w:val"), "true")
settings.append(uf)


def runs(par, node, fmt=None, size=None):
    """Add the inline content of `node` to paragraph `par`."""
    fmt = dict(fmt or {})
    for ch in node.children:
        if isinstance(ch, NavigableString):
            txt = clean(str(ch))
            if not txt:
                continue
            r = par.add_run(txt)
            r.bold = fmt.get("b") or None
            r.italic = fmt.get("i") or None
            if fmt.get("sub"):
                r.font.subscript = True
            if fmt.get("sup"):
                r.font.superscript = True
            if fmt.get("code"):
                r.font.name = "Consolas"
                r._r.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Consolas")
                r.font.size = Pt((size or 11) - 1)
            elif size:
                r.font.size = Pt(size)
            if fmt.get("hl"):
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            continue
        n = ch.name
        f = dict(fmt)
        if n in ("b", "strong"):
            f["b"] = True
        elif n in ("em", "i", "var"):
            f["i"] = True
        elif n == "sub":
            f["sub"] = True
        elif n == "sup":
            f["sup"] = True
        elif n == "code":
            f["code"] = True
        elif n == "mark":
            f["hl"] = True
        elif n == "span" and ("ch" in ch.get("class", []) or "no" in ch.get("class", [])):
            if "no" in ch.get("class", []):
                par.add_run("    ")
            runs(par, ch, f, size)
            if "ch" in ch.get("class", []):
                par.add_run(" ")
            continue
        elif n == "br":
            par.add_run().add_break()
            continue
        elif n in ("video", "img", "script", "style"):
            continue
        runs(par, ch, f, size)


def para(node, style=None, align=None, size=None, fmt=None, keep=False):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    runs(p, node, fmt, size)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcol)
    tcPr.append(sh)


def table(tbl, caption_node=None):
    rows = tbl.find_all("tr")
    ncol = max(len(r.find_all(["td", "th"])) for r in rows)
    if caption_node is not None:
        para(caption_node, style="Caption", keep=True)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 9 if ncol <= 5 else (8 if ncol <= 7 else 7)
    for i, r in enumerate(rows):
        cells = r.find_all(["td", "th"])
        for j, c in enumerate(cells[:ncol]):
            cell = t.cell(i, j)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            is_h = c.name == "th"
            runs(p, c, {"b": True} if is_h else None, size)
            if "n" in c.get("class", []):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if is_h:
                shade(cell, "E6E9F0")
            elif r.get("class") and ("best" in r["class"] or "hl" in r["class"]):
                shade(cell, "FFF4D6")
        if i == 0:
            trPr = t.rows[0]._tr.get_or_add_trPr()
            h = OxmlElement("w:tblHeader")
            h.set(qn("w:val"), "true")
            trPr.append(h)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def picture(img, par=None):
    path = (HERE / img["src"]).resolve()
    p = par or doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    try:
        from PIL import Image
        w, h = Image.open(path).size
        width = min(TEXT_W, w / 140.0 if w / 140.0 > 2.5 else 3.2)
        p.add_run().add_picture(str(path), width=Inches(max(2.5, width)))
    except Exception as e:
        p.add_run(f"[image {img['src']} not available: {e}]")


ARTIFACT = "https://claude.ai/artifact/UG9fajbsGEdR2KRMkTy3U4"
KEY_VIDEOS = {"vid-6-1", "vid-6-6", "vid-6-7", "vid-6-17", "vid-6-19"}  # linked with a QR code; the rest get a link only


def add_hyperlink(par, url, text, size=9):
    rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "1F4E8C")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    for e in (c, u, sz):
        rpr.append(e)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    r.append(t)
    h.append(r)
    par._p.append(h)


def qr_png(url, name):
    import qrcode
    path = Path(tempfile.gettempdir()) / f"qr_{name}.png"
    qrcode.make(url, box_size=6, border=2).save(path)
    return path


def video_block(fig, vid):
    """A video cannot play in Word: show a link to the hosted page (and a QR code for the key videos) in its place."""
    vid_id = fig.get("id", "")
    url = f"{ARTIFACT}#{vid_id}"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    if vid_id in KEY_VIDEOS:
        p.add_run().add_picture(str(qr_png(url, vid_id)), width=Inches(1.15))
        p.add_run().add_break()
    r = p.add_run("Video, online: ")
    r.font.size = Pt(9)
    r.italic = True
    add_hyperlink(p, url, url.replace("https://", ""))


def note_box(el):
    """Editing notes (Open, Pending, To verify, Optional) as a shaded, bordered, clearly labelled paragraph."""
    tag = el.find("strong", class_="tag")
    label = tag.get_text(strip=True) if tag else "Note"
    if tag:
        tag.extract()
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.1)
    p.paragraph_format.right_indent = Inches(0.1)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "FFF2B3")
    borders = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "dashed")
        b.set(qn("w:sz"), "8")
        b.set(qn("w:space"), "4")
        b.set(qn("w:color"), "B58900")
        borders.append(b)
    pPr.append(borders)
    pPr.append(shd)
    r = p.add_run(f"NOTE ({label}): ")
    r.bold = True
    r.font.size = Pt(10)
    runs(p, el, None, 10)


def set_page_numbers(section, fmt, start=None, title_page=False):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    sectPr.append(pg)
    section.different_first_page_header_footer = title_page


def figure(fig):
    img = fig.find("img")
    vid = fig.find("video")
    cap_node = fig.find("figcaption")
    todo = fig.find(class_="fig-todo")
    if img is not None and img.get("src", "").lower().endswith(".gif"):
        # animated GIF: keep the first frame (Word shows it as a still)
        pass
    if img is not None:
        picture(img)
    elif vid is not None:
        video_block(fig, vid)
    elif todo is not None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        runs(p, todo)
    if cap_node is not None:
        para(cap_node, style="Caption")


def lst(node, ordered):
    for k, li in enumerate(node.find_all("li", recursive=False), 1):
        if ordered:  # numbered by hand so that every list starts at 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35))
            p.add_run(f"{k}.	")
        else:
            p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        runs(p, li)


def block(el):
    if isinstance(el, NavigableString):
        if clean(str(el)).strip():
            doc.add_paragraph(clean(str(el)))
        return
    n = el.name
    cls = el.get("class", [])
    if n == "h1":
        h = doc.add_heading(level=0 if "title" in cls else 1)
        runs(h, el)
    elif n == "h2":
        runs(doc.add_heading(level=2), el)
    elif n == "h3":
        runs(doc.add_heading(level=3), el)
    elif n == "p":
        para(el)
    elif n in ("ul", "ol"):
        lst(el, n == "ol")
    elif n == "figure":
        figure(el)
    elif n == "table":
        table(el)
    elif n == "pre":
        for line in el.get_text().split("\n"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(line)
            r.font.name = "Consolas"
            r._r.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Consolas")
            r.font.size = Pt(8)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    elif "tbl" in cls:
        capn = el.find(class_="cap")
        tb = el.find("table")
        table(tb, capn)
    elif "eq" in cls:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        runs(p, el, {"i": True})
    elif "todo-block" in cls:
        note_box(el)
    elif n in ("div", "section", "span", "blockquote"):
        if n == "span" and el.find_parent(["p", "li"]):
            return
        kids = [c for c in el.children if not (isinstance(c, NavigableString) and not clean(str(c)).strip())]
        if all(isinstance(c, NavigableString) or c.name in ("b", "strong", "em", "i", "var", "sub", "sup", "code", "mark", "span", "br", "a") for c in kids) and kids:
            para(el, align=WD_ALIGN_PARAGRAPH.CENTER if "cover" in "".join(el.find_parent("section").get("class", [])) else None)
        else:
            for c in kids:
                block(c)


# ---- front matter and body ------------------------------------------------------------------------------------
sheets = main.find_all("section", recursive=True)
first = True
caps = {"fig": [], "tab": [], "vid": []}
for fg in main.find_all("figure"):
    c = fg.find("figcaption")
    k = "vid" if (fg.get("id", "").startswith("vid-")) else "fig"
    if c is not None:
        caps[k].append(clean(c.get_text()).strip())
for tb in main.find_all(class_="tbl"):
    c = tb.find(class_="cap")
    if c is not None:
        caps["tab"].append(clean(c.get_text()).strip())

for s in sheets:
    sid = s.get("id", "")
    if "sheet" not in s.get("class", []):
        continue
    if sid == "missing":
        continue  # the list of yellow items is built by script in the web page
    if sid == "ch1":
        # front matter: roman numerals (none on the cover); Chapter 1 starts a new section with arabic numbers from 1
        set_page_numbers(doc.sections[0], "lowerRoman", title_page=True)
        new_sec = doc.add_section(WD_SECTION.NEW_PAGE)
        new_sec.footer.is_linked_to_previous = False
        fp2 = new_sec.footer.paragraphs[0]
        for r_ in list(fp2.runs):
            r_._r.getparent().remove(r_._r)
        fp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(fp2, "PAGE")
        set_page_numbers(new_sec, "decimal", start=1, title_page=False)
    elif not first:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    first = False
    if sid == "contents":
        doc.add_heading("Contents", level=1)
        p = doc.add_paragraph()
        add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
        continue
    if sid in ("list-of-figures", "list-of-tables", "list-of-videos"):
        title = s.find("h1").get_text(strip=True)
        doc.add_heading(title, level=1)
        key = {"list-of-figures": "fig", "list-of-tables": "tab", "list-of-videos": "vid"}[sid]
        for t in caps[key]:
            p = doc.add_paragraph(t)
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.size = Pt(10)
        continue
    for c in s.children:
        block(c)

doc.save(dst)
print("saved", dst, dst.stat().st_size // 1024, "KB")
