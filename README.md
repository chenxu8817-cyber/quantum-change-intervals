# Paper I submission and local-build bundles

This directory separates three different workflows. Do not upload the entire
`delivery` directory to either arXiv or Quantum.

## 1. arXiv source upload

Upload the contents of `arxiv_source`, preferably as the generated
`quantum-change-intervals-arxiv-source.zip` archive.

The archive is deliberately source-only. It contains the TeX sources, the
custom Quantum class and BibTeX style, `references.bib`, the pre-generated
`main.bbl`, and the three PDF figures. Including both the BibTeX inputs and the
generated BBL makes the archive robust to either bibliography workflow. It does
not contain PDF/auxiliary build products, PNG duplicates, numerical data, or
review renders.

Select PDFLaTeX in the arXiv processor step. The source retains `\pdfoutput=1`
because the vendored Quantum class version 6.1 enforces it during compilation.

Important: the current manuscript still contains author and affiliation
placeholders. Replace them in the canonical
`paper1/quantum_submission/main.tex`, rebuild, and regenerate these bundles
before the actual arXiv upload.

## 2. Quantum initial submission

Quantum's initial submission does not require a manuscript source archive or a
separate manuscript PDF. The required manuscript input is the arXiv identifier
of a preprint posted to, or cross-listed with, `quant-ph`. See
`quantum_initial_submission/SUBMISSION_CHECKLIST.md`.

The PDF in that directory is an author-side reference copy only. Do not treat
it as a required Quantum upload. Optional supplementary material may be added
through the submission system, and the public code/data repository should be
provided as a stable URL when available.

## 3. Local LaTeX build

`local_latex_source` is the complete self-contained source directory for
building the paper locally. It includes both the BibTeX inputs and a generated
BBL. Follow `local_latex_source/BUILD.md`.

## 4. Overleaf project upload

Upload `quantum-change-intervals-overleaf-source.zip` using Overleaf's
`New Project` -> `Upload Project` action. The archive keeps `main.tex` at its
root and contains only the eight source inputs needed by Overleaf. Use
`pdfLaTeX`, select `main.tex` as the main document, and pin the project to TeX
Live 2025. TeX Live 2026's `array` package is incompatible with the current
Quantum class version 6.1 at the theorem-comparison `tabularx` table. The
archive omits the
compiled `main.pdf`, auxiliary files, and the generated BBL because Overleaf
can run BibTeX from `references.bib` and `quantum.bst`.

## Generated archives

- `quantum-change-intervals-arxiv-source.zip`: upload candidate for arXiv.
- `quantum-change-intervals-local-latex-source.zip`: portable local source
  package; this is not an arXiv or Quantum upload.
- `quantum-change-intervals-overleaf-source.zip`: clean Overleaf project
  upload with `main.tex` at the ZIP root.

The checksums and exact file inventories are recorded in `SHA256SUMS.txt` and
`FILE_INVENTORY.txt`.
