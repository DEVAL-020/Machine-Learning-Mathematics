.PHONY: install figures pdf clean all

all: figures pdf

install:
	pip install -r requirements.txt

figures:
	python3 scripts/generate_all.py

pdf:
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.toc main.pdf
