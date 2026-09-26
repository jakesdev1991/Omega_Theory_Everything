import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-!
# Finite simple networks with independent incidence and edge counts

Vertices are `Fin vertexCount`, not a list of indistinguishable Q-regions.
The graph carries symmetric, loop-free adjacency through Mathlib's `SimpleGraph`.
Degree sum counts neighbors vertex by vertex; edge count counts unordered edges.
Their relation follows from the finite graph theorem, not from a definition
that sets one quantity equal to twice the other.

This is a combinatorial model, not a derivation of social or physical adjacency
from mutual information. API change: `Graph` is now a bundled finite simple graph.
-/
namespace OmegaProtocol.Vol26
open OmegaProtocol

structure Graph where
  vertexCount : ℕ
  edges : SimpleGraph (Fin vertexCount)

noncomputable def Degree (G : Graph) (v : Fin G.vertexCount) : ℕ := by
  classical
  exact G.edges.degree v

noncomputable def NumEdges (G : Graph) : ℕ := by
  classical
  exact G.edges.edgeFinset.card

noncomputable def DegreeSum (G : Graph) : ℕ :=
  ∑ v : Fin G.vertexCount, Degree G v

/-- The genuine degree-sum theorem for independently counted quantities. -/
theorem handshaking_lemma (G : Graph) : DegreeSum G = 2 * NumEdges G := by
  classical
  exact G.edges.sum_degrees_eq_twice_card_edges

theorem degree_sum_even (G : Graph) : ∃ k, DegreeSum G = 2 * k :=
  ⟨NumEdges G, handshaking_lemma G⟩

noncomputable def oddDegreeVertices (G : Graph) : Finset (Fin G.vertexCount) := by
  classical
  exact Finset.univ.filter fun v => Odd (Degree G v)

/-- The parity version of handshaking: odd-degree vertices occur in even number. -/
theorem odd_degree_vertices_even (G : Graph) : Even (oddDegreeVertices G).card := by
  classical
  exact G.edges.even_card_odd_degree_vertices

theorem another_odd_degree_vertex (G : Graph) (v : Fin G.vertexCount)
    (hv : Odd (Degree G v)) : ∃ w, w ≠ v ∧ Odd (Degree G w) := by
  classical
  exact G.edges.exists_ne_odd_degree_of_exists_odd_degree v hv

/-- A degree floor yields a global incidence lower bound. -/
theorem degree_sum_lower_bound (G : Graph) (k : ℕ)
    (h : ∀ v, k ≤ Degree G v) : k * G.vertexCount ≤ DegreeSum G := by
  calc k * G.vertexCount = ∑ _v : Fin G.vertexCount, k := by simp [Nat.mul_comm]
    _ ≤ DegreeSum G := Finset.sum_le_sum fun v _ => h v

/-- Legacy export name for the finite combinatorial identity only. -/
theorem network_from_omega (G : Graph) : DegreeSum G = 2 * NumEdges G :=
  handshaking_lemma G

theorem network_degree_nonneg (G : Graph) : DegreeSum G ≥ 0 := Nat.zero_le _

def emptyGraph (n : ℕ) : Graph := ⟨n, ⊥⟩
def twoVertexEdge : Graph := ⟨2, ⊤⟩

theorem emptyGraph_degree (n : ℕ) (v : Fin n) : Degree (emptyGraph n) v = 0 := by
  classical
  simp [Degree, emptyGraph]

theorem emptyGraph_numEdges (n : ℕ) : NumEdges (emptyGraph n) = 0 := by
  classical
  simp [NumEdges, emptyGraph]

/-- Every vertex of the two-vertex complete graph has one neighbor. -/
theorem twoVertexEdge_degree (v : Fin 2) : Degree twoVertexEdge v = 1 := by
  classical
  change (⊤ : SimpleGraph (Fin 2)).degree v = 1
  have hpos : 0 < (⊤ : SimpleGraph (Fin 2)).degree v := by
    apply ((⊤ : SimpleGraph (Fin 2)).degree_pos_iff_exists_adj v).mpr
    fin_cases v
    · exact ⟨1, by simp⟩
    · exact ⟨0, by simp⟩
  have hbound : (⊤ : SimpleGraph (Fin 2)).degree v < 2 := by
    simpa only [Fintype.card_fin] using (⊤ : SimpleGraph (Fin 2)).degree_lt_card_verts v
  omega

theorem twoVertexEdge_counts : DegreeSum twoVertexEdge = 2 ∧ NumEdges twoVertexEdge = 1 := by
  have hd : DegreeSum twoVertexEdge = 2 := by
    change (∑ v : Fin 2, Degree twoVertexEdge v) = 2
    simp [twoVertexEdge_degree]
  have hh := handshaking_lemma twoVertexEdge
  exact ⟨hd, by omega⟩

/-- Structural consistency of the separate legacy singleton model only. -/
theorem bridge_vol26_metric_self_zero (R : QRegion) : d R R = 0 :=
  qregion_self_distance_zero R

end OmegaProtocol.Vol26
