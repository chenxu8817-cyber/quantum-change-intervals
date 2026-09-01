# Final PDF quality assurance

Date: 2026-09-01

This report records the final quality checks for the Paper I article and
Supplemental Material. Both files were built from the authoritative sources in
`paper1/quantum_revision_ultracritical` with Tectonic 0.17.0 in the new
`build-v1.5.2-main-freeze-20260901b` and
`build-v1.5.2-supp-freeze-20260901b` directories.

## Frozen files

| File | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| Main article | 19 | 389343 | `B8BD20B9C716525E3657BC64864052DCE4CEAA6CC5AD06671F480DE2E6C703F2` |
| Supplemental Material | 28 | 242511 | `5BF2E79CDA49A5458ADBD6432E97DA50A9CBC00D278583C6954A58E574BA5D9E` |
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
- Static checks found 143 article labels and 141 Supplemental Material labels,
  with no duplicate or undefined label.
- All 28 bibliography entries are cited. Every citation key resolves, and
  there is no duplicate or unused entry.

## Fonts and figures

- All 47 effective font resources used by the article and all 31 used by the
  Supplemental Material have embedded font programs. Type 0 wrapper fonts were
  checked through their embedded descendant fonts.
- Figures 1, 2, and 3 are vector based in the article PDF. Recursive PDF
  resource inspection found 13 form XObjects and no raster image XObject in
  the article; the Supplemental Material contains no image XObject.
- The PNG derivatives of Figures 1, 2, and 3 are 300 dpi. Their pixel sizes are
  2164 by 769, 2095 by 803, and 2152 by 877, respectively. The Figure 1 TIFF is
  4329 by 1539 pixels at 600 dpi. The PDF figures are the authoritative article
  inputs.

## Visual inspection

All 19 article pages and all 28 Supplemental Material pages were rendered at
96 dpi from the frozen build and inspected in complete contact sheets. The
article's title page and Figure 3 page, the final reference page, and the final
Supplemental page were also inspected individually. Page-by-page rendered
output is identical to the preceding clean build, so the PDF byte differences
are confined to generation metadata. The 28th Supplemental
page contains the closing finite-size diagnostic tables and discussion; it is
sparse but neither blank nor misplaced.

The inspection covered the title and author block, abstract, literature table,
Figures 1, 2, and 3, all principal theorem statements, the no-change results,
conclusion, references, Supplemental proof map, continuum and moving-outer
arguments, local SRM refinement, and diagnostic table. No clipped material,
overlapping curve label, broken equation, unreadable legend, missing glyph,
misplaced float, column collision, or unexpected blank page was found. An
unsupported second-author email present in an intermediate source was removed
before this build; the title page retains only the corresponding-author email
supplied by the authors.

Final PDF status: **PASS**.
