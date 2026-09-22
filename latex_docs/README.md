# LaTeX Documentation - Omega Theory

This directory contains LaTeX-formatted mathematical documentation for each volume of the Omega Theory.

## Structure

Each file corresponds to a volume in `../lean_proofs/` and provides:
- Mathematical notation and formulas
- Layman's summaries
- Cross-references to Lean 4 formalizations
- Bibliography and citations

## Volumes Available

Volumes 09-54 are currently documented in LaTeX format:
- Vol09_HolographicPrinciple.tex through Vol54_TheoryOfNothing.tex

## Compiling

```bash
# Compile a single volume
pdflatex Vol09_HolographicPrinciple.tex

# Or use latexmk for automatic runs
latexmk -pdf Vol09_HolographicPrinciple.tex
```

## Companion Files

- **Lean 4 proofs**: `../lean_proofs/` - Formal verification
- **Text versions**: `../txt_proofs/` - Plain text extracts
