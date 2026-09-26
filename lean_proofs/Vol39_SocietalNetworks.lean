import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation
import Vol26_NetworkTheory

/-!
# Finite social-network counting model

This module reuses the finite simple graphs of Vol26. A minimum connection
bound requires a no-isolated-vertices hypothesis; it is not true for arbitrary
networks, or even for the connected one-vertex graph. We state this local
hypothesis instead of silently assuming connectivity implies positive degree.

No sociological interpretation is derived from these combinatorial counts.
API change: node, edge and degree quantities now depend on a graph, and lower
bounds require explicit degree assumptions.
-/
namespace OmegaProtocol.Vol39
open OmegaProtocol

abbrev SocialNetwork := Vol26.Graph

def NumNodes (G : SocialNetwork) : ℕ := G.vertexCount
noncomputable def NumConnections (G : SocialNetwork) : ℕ := Vol26.NumEdges G
noncomputable def TotalDegree (G : SocialNetwork) : ℕ := Vol26.DegreeSum G

def NoIsolatedVertices (G : SocialNetwork) : Prop := ∀ v, 1 ≤ Vol26.Degree G v

theorem degree_sum_formula (G : SocialNetwork) : TotalDegree G = 2 * NumConnections G :=
  Vol26.handshaking_lemma G

theorem min_degree (G : SocialNetwork) (h : NoIsolatedVertices G) :
    NumNodes G ≤ TotalDegree G := by
  simpa [NumNodes, TotalDegree] using Vol26.degree_sum_lower_bound G 1 h

theorem min_connections (G : SocialNetwork) (h : NoIsolatedVertices G) :
    NumNodes G ≤ 2 * NumConnections G := by
  have hd := min_degree G h
  rwa [degree_sum_formula] at hd

/-- Natural-number rounding makes the edge lower bound explicit. -/
theorem min_connections_rounded (G : SocialNetwork) (h : NoIsolatedVertices G) :
    (NumNodes G + 1) / 2 ≤ NumConnections G := by
  have hbound := min_connections G h
  omega

/-- Legacy export name for a conditional counting bound, not a social law. -/
theorem society_from_omega (G : SocialNetwork) (h : NoIsolatedVertices G) :
    NumNodes G ≤ 2 * NumConnections G := min_connections G h

theorem twoVertexEdge_no_isolated : NoIsolatedVertices Vol26.twoVertexEdge := by
  intro v
  simpa only [Vol26.twoVertexEdge_degree] using (le_refl (1 : ℕ))

/-- The degree assumption cannot be dropped: one vertex and no edges fails it. -/
theorem singleton_has_isolated_vertex : ¬ NoIsolatedVertices (Vol26.emptyGraph 1) := by
  intro h
  have hd := h (0 : Fin 1)
  rw [Vol26.emptyGraph_degree] at hd
  omega

theorem unconditioned_bound_fails :
    ¬ NumNodes (Vol26.emptyGraph 1) ≤ 2 * NumConnections (Vol26.emptyGraph 1) := by
  change ¬ (1 ≤ 2 * Vol26.NumEdges (Vol26.emptyGraph 1))
  rw [Vol26.emptyGraph_numEdges]
  norm_num

/-- Structural consistency of the separate zero-information model only. -/
theorem bridge_vol39_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 :=
  monotonicity_lemma R

/-- Structural consistency of the separate zero-information model only. -/
theorem bridge_vol39_metric_self_zero (R : QRegion) : d R R = 0 :=
  qregion_self_distance_zero R

end OmegaProtocol.Vol39
