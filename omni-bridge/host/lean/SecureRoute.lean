structure RouteHeader where
  magic_validation : UInt32
  payload_length   : UInt32
  route_id         : UInt32
  risk_tier        : UInt32

def HEADER_SIZE : Nat := 16 -- 4 * sizeof(uint32_t)

-- Theorem: under the valid guard, total frame size will not overflow UInt32.
-- Core-only proof (no Mathlib): omega treats 2^k numerals natively.
theorem payload_no_overflow (h : RouteHeader)
    (h_bound : h.payload_length.toNat ≤ 2^32 - 1 - HEADER_SIZE) :
    h.payload_length.toNat + HEADER_SIZE < 2^32 := by
  have hsize : HEADER_SIZE = 16 := rfl
  omega

-- Corollary: the C++ SecureRouteIsolationFrame::validate() precondition.
theorem header_size_positive : 0 < HEADER_SIZE := by
  decide

-- Every UInt32 payload length is representable by construction.
theorem payload_length_bounded (h : RouteHeader) :
    h.payload_length.toNat < 2^32 := by
  exact h.payload_length.toNat_lt
