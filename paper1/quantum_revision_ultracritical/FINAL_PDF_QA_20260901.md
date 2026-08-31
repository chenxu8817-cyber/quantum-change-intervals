# Final PDF quality assurance

Date: 2026-09-01

This report records the final quality checks for the frozen Paper I article and
Supplemental Material. Both files were rebuilt with Tectonic 0.17.0 from the
authoritative sources in `paper1/quantum_revision_ultracritical`.

## Frozen files

| File | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| Main article | 20 | 407192 | `909DBC6AE2C2B8BCC892FBA1707B357077D2E3A4DEE5767C77AE9ABD897682B7` |
| Supplemental Material | 23 | 215233 | `3987A16E57C2DBB52A5F1AD093746EA77325F0D86609B555019E37DB56B57C02` |
| Main bibliography output |  | 7387 | `471D4C1B01BACD530A4C401F48203E4C298619ECCE1B53CBD3934D7EC1EFA524` |

The main article and Supplemental Material use A4 pages and PDF 1.5. They are
unencrypted, and their title and author metadata agree with the manuscript.

## Compilation, labels, and bibliography

- The clean article and Supplemental Material builds completed with exit
  status zero.
- The final logs contain no LaTeX errors, fatal errors, undefined citations,
  undefined references, multiply defined labels, or overfull boxes.
- The article log contains 27 underfull-box notices caused by narrow
  two-column mathematical or bibliographic lines. Visual inspection found no
  corresponding collision or readability problem. The Supplemental Material
  log contains no underfull-box notice.
- Static source checks found 155 unique article labels and 119 unique
  Supplemental Material labels. Every referenced label is defined, and no
  label is multiply defined.
- All 28 bibliography entries are cited. All 28 citation keys resolve, with no
  unused or duplicate entry.

## Fonts and figures

- All 47 font resources used by the article and all 32 used by the
  Supplemental Material are embedded.
- Figures 1--3 are fully vector based in the frozen article PDF. In
  particular, the cells and color scale in Figure 3(c) are vector objects and
  the PDF contains no image XObject.
- The separate PNG derivatives of Figures 1--3 are 300 dpi, and the Figure 1
  TIFF derivative is 600 dpi. The PDF figures are the authoritative article
  inputs.

## Visual inspection

All 20 article pages and all 23 Supplemental Material pages were rendered at
110 dpi and inspected. Additional full-page checks covered the title and
author block, the literature-comparison table, Figures 1--3, the fixed- and
moving-overlap theorem statements, the no-change results, the conclusion and
references, the Supplemental Material proof map, the continuum and
moving-outer arguments, and the final diagnostic table.

No clipped material, overlapping curve labels, broken equations, unreadable
legends, missing glyphs, misplaced floats, column collisions, or unexpected
blank pages were found. The open lower portion of the final Supplemental
Material page follows naturally from the closing table and reproduction
commands and does not indicate omitted content.

Final status: **PASS**.
