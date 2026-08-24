# GitHub upload instructions

Target repository:
<https://github.com/chenxu8817-cyber/quantum-change-intervals>

## Release decisions

1. The repository retains its existing MIT License; do not replace it with the
   former temporary all-rights-reserved placeholder.
2. Author metadata in `CITATION.cff` and the manuscript sources is synchronized.
3. Xue Ma's ORCID may be added later if one is to be associated with the release.
4. No credentials, private keys, local virtual environments, or temporary
   verification artifacts belong in the public tree.

## Upload with Git

For the existing repository, clone or pull `main`, copy the extracted package
contents so that `README.md`, `CITATION.cff`, and the source files remain at
the repository root, inspect the diff, and run:

```powershell
git add .
git commit -m "Release Paper I reproducibility package"
git push -u origin main
```

Preserve the remote MIT license terms. Do not force-push or discard unrelated
remote history.

## Upload through the GitHub website

Create an empty repository without generating an additional README, license, or
`.gitignore`. Extract this package locally and upload the extracted files and
directories. Do not upload the ZIP as the only repository file if the intended
result is a browsable source repository.

## Verification after upload

On a clean CPython 3.12 environment, follow `REPRODUCING.md` or run:

```powershell
.\paper1\run_reproduction.ps1 -SkipCertifiedGrid
```

After the public repository is verified, create and push the annotated tag
`v1.1.0-paper1`. A Zenodo or institutional deposit of that exact tag is an
optional follow-up; add its version DOI to `CITATION.cff` and the manuscript
only after the DOI resolves.
