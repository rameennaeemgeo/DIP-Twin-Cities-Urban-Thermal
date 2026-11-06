from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

root = Path(r"C:\DIP_Project\DIP_Twin_Cities_Project")
path = root / "10_Research_Report" / "DIP_Twin_Cities_Report_final.docx"
if not path.exists():
    raise SystemExit("Run 11_insert_figures.py first, so that the _final.docx file exists.")
doc = Document(str(path))
pngs = sorted((root / "07_Statistics" / "Histograms").glob("*.png"))
doc.add_page_break()
try:
    doc.add_paragraph("Appendix C: Additional Histograms", style="Heading 1")
except KeyError:
    doc.add_paragraph().add_run("Appendix C: Additional Histograms").bold = True
n = 25
for p in pngs:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.add_run().add_picture(str(p), width=Inches(6.0))
    n += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run("Figure %d: " % n).bold = True
    cap.add_run(p.stem.replace("_", " "))
doc.save(str(path))
print("Added", len(pngs), "histograms to the report")
