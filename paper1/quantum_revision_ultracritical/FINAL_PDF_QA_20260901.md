# Final PDF quality assurance

Date: 2026-09-01

This report records the final quality checks for the Paper I article and
Supplemental Material. Both files were built from the authoritative sources in
`paper1/quantum_revision_ultracritical` with Tectonic 0.17.0 in new output
directories.

## Frozen files

| File | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| Main article | 18 | 389853 | `DD1D752229E32CF57AED4BDC92AB3F389E59B189BBD87870B37AA6A420F97879` |
| Supplemental Material | 27 | 235797 | `A2A7AABE11D49AC6158B99E8DA37827739FCADD4E6772672FF77158ECA9FA3CE` |
| Main bibliography output |  | 7387 | `685F2AB321CD7AD51E4E2034136B4E4E0386C5E1C6F72A1E004C413AD0107E2C` |

Both PDFs use A4 pages and PDF 1.5. They are unencrypted. Their title and
author metadata agree with the manuscript, and text extraction found no
replacement characters.

## Compilation, labels, and bibliography

- Both clean builds completed with exit status zero.
- The final logs contain no LaTeX error, fatal error, undefined citation,
  undefined reference, multiply defined label, or overfull box.
- The article log contains 23 underfull-box notices from narrow two-column
  mathematical and bibliographic lines. Visual inspection found no collision
  or loss of readability. The Supplemental Material log contains no
  underfull-box notice.
- Static checks found 143 article labels and 138 Supplemental Material labels,
  with no duplicate or undefined label.
- All 28 bibliography entries are cited. Every citation key resolves, and
  there is no duplicate or unused entry.

## Fonts and figures

- All 47 effective font resources used by the article and all 31 used by the
  Supplemental Material have embedded font programs. Type 0 wrapper fonts were
  checked through their embedded descendant fonts.
- Figures 1, 2, and 3 are vector based in the article PDF. The article and
  Supplemental Material contain no raster image XObject.
- The PNG derivatives of Figures 1, 2, and 3 are 300 dpi. Their pixel sizes are
  2164 by 769, 2095 by 803, and 2152 by 877, respectively. The Figure 1 TIFF is
  4329 by 1539 pixels at 600 dpi. The PDF figures are the authoritative article
  inputs.

## Visual inspection

All 18 article pages and all 27 Supplemental Material pages were rendered at
120 dpi and inspected. After two final grammatical edits, the affected
Supplemental pages 2, 3, and 15 were rerendered from the frozen build at
150 dpi and inspected again. The 27th Supplemental page contains the closing
finite-size diagnostic tables and discussion; it is sparse but neither blank
nor misplaced.

The inspection covered the title and author block, abstract, literature table,
Figures 1, 2, and 3, all principal theorem statements, the no-change results,
conclusion, references, Supplemental proof map, continuum and moving-outer
arguments, local SRM refinement, and diagnostic table. No clipped material,
overlapping curve label, broken equation, unreadable legend, missing glyph,
misplaced float, column collision, or unexpected blank page was found.

Final PDF status: **PASS**.
