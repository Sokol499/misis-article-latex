from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
article = (root / "article.tex").read_text(encoding="utf-8")
files = sorted((root / "sections").glob("*.tex"))
text = article + "\n" + "\n".join(p.read_text(encoding="utf-8") for p in files)
refs = (root / "references.tex").read_text(encoding="utf-8")
assert r"\documentclass{misisarticle}" in article
assert len(files) == 4
for heading in ("Введение", "Анализ существующих исследований", "Постановка задачи", "Методы и организация исследования"):
    assert "\\section{" + heading + "}" in text, f"Missing section {heading}"
for p in files:
    assert "\\input{sections/" + p.stem + "}" in article
cited = set()
for match in re.findall(r"\\cite(?:\[[^]]*\])*\{([^}]+)\}", text):
    cited.update(key.strip() for key in match.split(","))
items = set(re.findall(r"\\bibitem\{([^}]+)\}", refs))
assert cited == items, f"Citations mismatch: missing={cited-items}; extra={items-cited}"
assert len(items) >= 8
assert r"\input{references.tex}" in article
assert "\\printbibliography" not in article
print(f"PASS: {len(files)} sections, {len(items)} bibliography items; original MISIS class")
