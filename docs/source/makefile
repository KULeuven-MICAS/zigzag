# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build -E
PAPER         =
BUILDDIR      = build

LATEX-BW      = $(BUILDDIR)/latex-bw
LATEX-NAME = music-for-geeks-and-nerds

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
SAMPLEOPTS = -D html_theme=sample

PAPEROPTS = -D latex_elements.pointsize=11pt -D latex_elements.preamble=\\usepackage{mfgan-bw} -D pygments_style=bw -D black_and_white=True -D code_example_wrap=67 -D latex_show_pagerefs=True

SCREENOPTS = -D latex_elements.pointsize=12pt -D latex_elements.classoptions=,openany,oneside -D latex_elements.preamble=\\usepackage{mfgan} -D pygments_style=my_pygment_style.BookStyle -D code_example_wrap=67

MOBIOPTS = -D pygments_style=none
MOBI_NAME = MusicforGeeksandNerds.mobi

# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: default view clean html epub mobi latex pdf pdf-bw text

default: html

all: html epub mobi pdf pdf-bw sample

view:
	open build/html/index.html

clean:
	-rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

epub:
	$(SPHINXBUILD) -b epub2 $(ALLSPHINXOPTS) $(BUILDDIR)/epub

mobi:
	$(SPHINXBUILD) -b mobi -t mobi $(MOBIOPTS) $(ALLSPHINXOPTS) $(BUILDDIR)/mobi
#	cd $(BUILDDIR)/mobi && kindlegen content.opf -o $(MOBI_NAME)

kindle-sync:
	cp $(BUILDDIR)/mobi/$(MOBI_NAME) /Volumes/Kindle/documents/
	diskutil eject /Volumes/Kindle/

latex:
	$(SPHINXBUILD) -b latex $(SCREENOPTS) $(ALLSPHINXOPTS) $(BUILDDIR)/latex

pdf:
	$(SPHINXBUILD) -b latex $(SCREENOPTS) $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	sed -i .bak -f process-latex $(BUILDDIR)/latex/$(LATEX-NAME).tex
	rsync -a latex/ $(BUILDDIR)/latex/
	$(MAKE) -C $(BUILDDIR)/latex pdf

pdf-bw:
	$(SPHINXBUILD) -b latex -t black_and_white $(PAPEROPTS) $(ALLSPHINXOPTS) $(LATEX-BW)
	sed -i .bak -f process-latex $(LATEX-BW)/$(LATEX-NAME).tex
	sed -i .bak '/\\setcounter{page}{1}/d' $(LATEX-BW)/sphinxmanual.cls
	rsync -a latex/ $(LATEX-BW)/
	$(MAKE) -C $(LATEX-BW)/ pdf

sample:
	$(SPHINXBUILD) -b latex -t sample $(SCREENOPTS) $(ALLSPHINXOPTS) $(BUILDDIR)/sample
	sed -i .bak -f process-latex $(BUILDDIR)/sample/$(LATEX-NAME).tex
	rsync -a latex/ $(BUILDDIR)/sample/
	cp figs-pdf/stamp*.pdf $(BUILDDIR)/sample/
	$(MAKE) -C $(BUILDDIR)/sample pdf
	# chapter 1, 3, 5
	pdftk A=$(BUILDDIR)/sample/$(LATEX-NAME).pdf cat A1-6 A21-25 A45-48 A84 output $(BUILDDIR)/sample/$(LATEX-NAME)-sample.pdf