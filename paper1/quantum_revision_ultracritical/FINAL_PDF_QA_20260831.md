# Final PDF quality assurance

Date: 2026-08-31

This report records the final PDF checks for the source snapshot identified in
`PAPER1_RELEASE_MANIFEST.md`. The PDFs were rebuilt with Tectonic 0.17.0 from
the authoritative `quantum_revision_ultracritical` sources.

## Files

| File | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| Main article | 20 | 401711 | `A6F339F5C571C897873100807F2C882DC6A4F997D440E0B5C79726CAB395284C` |
| Supplemental Material | 23 | 216817 | `106EF99B7E466D99EC2F6A3958A151C80BEF693F3D2F04F2076D16B318B22775` |
| Main bibliography output |  | 7387 | `471D4C1B01BACD530A4C401F48203E4C298619ECCE1B53CBD3934D7EC1EFA524` |

Both PDFs use A4 pages, PDF 1.5, and contain no encryption, JavaScript, forms,
or suspect PDF objects. Their title and author metadata agree with the
manuscript.

## Compilation and cross-references

- Main and Supplemental Material builds completed with exit status zero.
- The final logs contain no LaTeX errors, fatal errors, undefined citations,
  undefined references, multiply defined labels, or overfull boxes.
- The main log contains 26 underfull-box notices. The Supplemental Material
  log contains 6. These are benign line-breaking notices in the two-column
  layout and title block.
- Static source checks found 159 unique main-text labels and 119 unique
  Supplemental Material labels, with no duplicates or missing reference
  targets. The two label namespaces do not overlap.
- All 28 bibliography entries are cited, all 28 citation keys resolve, and no
  duplicate bibliography key was found.

## Fonts and figures

- All 47 fonts used by the main PDF and all 32 fonts used by the Supplemental
  Material are embedded.
- Figures 1 and 2 are fully vector based in the article PDF.
- Figure 3 retains vector text, axes, curves, and annotations. Its heat-map
  cells are represented by two losslessly compressed raster objects at their
  native grid resolution; no photographic or continuous-tone image is used.
- The separate PNG derivatives of Figures 1--3 are 300 dpi. The Figure 1 TIFF
  derivative is 600 dpi. The PDF versions remain the authoritative article
  inputs.

## Visual inspection

All 20 main-article pages and all 23 Supplemental Material pages were rendered
at 160 dpi and inspected. The title page, literature table, Figures 1--3,
the hull-dominance theorem, the no-change results, the conclusion, references,
the Supplemental Material proof map, the continuum and moving-outer results,
and the final numerical table received additional full-page inspection.

No clipped material, overlapping labels, broken equations, unreadable legends,
missing glyphs, misplaced floats, or unexpected blank pages were found. The
partly open lower area on the final Supplemental Material page is the natural
result of the closing table and reproduction command, not a missing element.

The 43 rendered pages are pixel-identical to the previously inspected final
layout, while the PDFs and bibliography listed above are freshly bound to the
current source snapshot.

Final status: **PASS**.
