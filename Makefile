LATEX ?= tectonic

.PHONY: all article example dist clean

all: article example

article:
	mkdir -p build
	$(LATEX) -o build article.tex

example:
	mkdir -p example/build
	cd example && $(LATEX) -o build example.tex

# Export a self-contained copy of the actual article, excluding the template's demo.
dist:
	rm -rf dist
	mkdir -p dist/misis-article/sections dist/misis-article/scripts
	cp misisarticle.cls article.tex references.tex README.md LICENSE dist/misis-article/
	cp sections/*.tex dist/misis-article/sections/
	cp scripts/check_manuscript.py dist/misis-article/scripts/
	cd dist && zip -r misis-article.zip misis-article
	rm -rf dist/misis-article

clean:
	rm -rf build example/build dist
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml *.xdv *.synctex.gz *.fls *.fdb_latexmk
	rm -f article.pdf example/example.pdf
