"""Section 5: upgrade html_to_docx.py (note boxes, video links with QR codes, roman/arabic page numbers). Idempotent."""
p = "html_to_docx.py"
s = open(p, encoding="utf-8").read()
if "def video_block" in s:
    print("already patched")
    raise SystemExit


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


rep("from docx.enum.section import WD_ORIENT\n", "from docx.enum.section import WD_ORIENT, WD_SECTION\n")
rep("import re\nimport sys\n", "import re\nimport sys\nimport tempfile\n")

helpers = '''
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

'''
rep("\ndef figure(fig):", helpers + "\ndef figure(fig):")

rep('''    elif vid is not None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[Video: {Path(vid.get('src', '')).name}, in the online report]")
        r.italic = True
        r.font.size = Pt(9)
        p.paragraph_format.keep_with_next = True''', '''    elif vid is not None:
        video_block(fig, vid)''')

rep('''    elif "todo-block" in cls:
        p = doc.add_paragraph()
        runs(p, el, {"hl": True}, 10)''', '''    elif "todo-block" in cls:
        note_box(el)''')

rep('''    if not first:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    first = False''', '''    if sid == "ch1":
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
    first = False''')
open(p, "w", encoding="utf-8").write(s)
print("converter upgraded")
