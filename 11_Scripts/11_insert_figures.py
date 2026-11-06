import re
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

root = Path(r"C:\DIP_Project\DIP_Twin_Cities_Project")
src = Path.home() / "Downloads" / "DIP_Twin_Cities_Report.docx"
out = root / "10_Research_Report" / "DIP_Twin_Cities_Report_final.docx"
if not src.exists():
    raise SystemExit("Report not found in Downloads: " + str(src))
files = {p.name: p for p in root.rglob("*.png")}
pat = re.compile(r"\[\[FIG:([^|\]]+)\|([0-9.]+)\]\]")
doc = Document(str(src))

def all_paragraphs(d):
    res = list(d.paragraphs)
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                res.extend(cell.paragraphs)
    return res

done, missing = 0, []
for p in all_paragraphs(doc):
    m = pat.search(p.text)
    if not m:
        continue
    name, w = m.group(1), float(m.group(2))
    if name not in files:
        missing.append(name)
        continue
    for r in p.runs:
        r.text = ""
    run = p.runs[0] if p.runs else p.add_run()
    run.add_picture(str(files[name]), width=Inches(w))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    done += 1
doc.save(str(out))
print("inserted", done, "figures; missing:", missing)
print("saved:", out)
