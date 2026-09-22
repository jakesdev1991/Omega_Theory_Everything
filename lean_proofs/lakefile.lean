require mathlib4 from git
  "https://github.com/leanprover/mathlib4" @ "v4.8.0"

@[default_target]
lean_lib OmegaTheory

lean_lib OmegaTheory
  /-- The Omega Theory formalization in Lean 4 -/
  where
    srcDir := "."
    rootNamespace := "OmegaProtocol"
