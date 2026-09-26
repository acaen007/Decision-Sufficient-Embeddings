"""Render a Markdown report with $...$ / $$...$$ LaTeX to a print-ready PDF: math -> MathML (latex2mathml), Markdown
-> HTML (python-markdown), then headless Chromium --print-to-pdf.  Usage:
    python -m leduc_decision_repr.report_pdf REPORT.md [out.pdf]"""
import re, subprocess, sys, tempfile
from pathlib import Path

CHROME = "/opt/pw-browsers/chromium"
CSS = """
@page { size: A4; margin: 13mm 15mm 13mm 15mm; }
body { font-family: 'Liberation Serif', 'FreeSerif', serif; font-size: 9.8pt; line-height: 1.28; color: #111; }
h1 { font-size: 15.5pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 12pt; margin: 11pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
h3 { font-size: 10.5pt; margin: 8pt 0 2pt; }
p { margin: 3pt 0 4pt; }
ul, ol { margin: 2pt 0 4pt; padding-left: 16pt; } li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 5pt auto; font-size: 8.8pt; }
th, td { border-top: 0.5pt solid #aaa; border-bottom: 0.5pt solid #aaa; padding: 2pt 5pt; }
th { background: #f0f0ec; }
math { font-family: 'FreeSerif', 'Liberation Serif', serif; font-size: 1.05em; }
math[display="block"] { margin: 5pt 0; }
img { display: block; width: 62%; margin: 4pt auto 0; }
em { color: #333; }
"""


def render(md_path, pdf_path=None):
    from latex2mathml.converter import convert
    import markdown
    md_path = Path(md_path); src = md_path.read_text(); maths = []
    def keep(m, display):
        maths.append(convert(m.group(1).strip(), display="block" if display else "inline"))
        return f"\n\nMATHBLOCK{len(maths)-1}ZZ\n\n" if display else f"MATHINL{len(maths)-1}ZZ"
    src = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(m, True), src, flags=re.S)
    src = re.sub(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$", lambda m: keep(m, False), src)
    lines, out = src.split("\n"), []                                   # python-markdown needs a blank line before a list
    for i, ln in enumerate(lines):
        if re.match(r"(\* |- |\d+\. )", ln) and i and lines[i - 1].strip() and not re.match(r"\s*(\* |- |\d+\. )", lines[i - 1]) \
                and not lines[i - 1].startswith("  "):
            out.append("")
        out.append(re.sub(r"^  (?=[*-] )", "    ", ln))                      # 2-space nested items -> 4 spaces
    src = "\n".join(out)
    html = markdown.markdown(src, extensions=["tables", "sane_lists"])
    html = re.sub(r"<p>MATHBLOCK(\d+)ZZ</p>", lambda m: maths[int(m.group(1))], html)
    html = re.sub(r"MATH(?:BLOCK|INL)(\d+)ZZ", lambda m: maths[int(m.group(1))], html)
    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    tmp = md_path.parent / (md_path.stem + ".print.html"); tmp.write_text(doc)
    pdf_path = Path(pdf_path or md_path.with_suffix(".pdf"))
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path.resolve()}", tmp.resolve().as_uri()], check=True, capture_output=True)
    tmp.unlink()
    return pdf_path


if __name__ == "__main__":
    print(render(*sys.argv[1:]))
