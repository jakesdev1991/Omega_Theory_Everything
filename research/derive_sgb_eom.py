"""Derive the static spherically-symmetric scalar-Gauss-Bonnet field equations
from the reduced action, symbolically. No hand-derived component formulas.

Ansatz: ds² = -e^{2δ(r)} μ(r) dt² + dr²/μ(r) + r² dΩ²,  μ = 1 - 2m(r)/r.
Action (G=1, Z=1, V=0): S = ∫ √-g [ R/16π - ½(∇φ)² + ½ηφ² 𝒢 ] × (angular ∫ dΩ = 4π, dropped)
Reduced Lagrangian: L = r² e^δ [ R/16π - ½ μ φ'² + ½ η φ² 𝒢 ].
Euler-Lagrange with second derivatives handled by integration by parts.
"""
import sympy as sp

r = sp.symbols('r', positive=True)
eta = sp.symbols('eta')
mu = sp.Function('mu')(r)
de = sp.Function('de')(r)   # δ
phi = sp.Function('phi')(r)

# metric
th = sp.symbols('theta')
g = sp.Matrix([[-sp.exp(2*de)*mu, 0, 0, 0],
               [0, 1/mu, 0, 0],
               [0, 0, r**2, 0],
               [0, 0, 0, r**2*sp.sin(th)**2]])
coords = [sp.Symbol('t'), r, th, sp.Symbol('phi_c')]
inv = g.inv()
n = 4

# Christoffel
gam = [[[sp.simplify(sum(inv[a, d]*(sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b]) - sp.diff(g[b, c], coords[d])) for d in range(n))/2)
         for c in range(n)] for b in range(n)] for a in range(n)]

# Riemann (mixed) and invariants
def rup(a, b, c, d):
    e = sp.diff(gam[a][b][d], coords[c]) - sp.diff(gam[a][b][c], coords[d])
    for k in range(n):
        e += gam[a][c][k]*gam[k][b][d] - gam[a][d][k]*gam[k][b][c]
    return sp.expand(e)

R = [[sp.simplify(sum(rup(k, i, k, j) for k in range(n))) for j in range(n)] for i in range(n)]
Rs = sp.simplify(sum(inv[i, j]*R[i][j] for i in range(n) for j in range(n)))
RR = sp.simplify(sum(R[i][j]*R[a][b]*inv[i, a]*inv[j, b] for i in range(n) for j in range(n) for a in range(n) for b in range(n)))
K = 0
import itertools
for a, b, c, d in itertools.product(range(n), repeat=4):
    comp = sum(g[a, e]*rup(e, b, c, d) for e in range(n))
    if sp.expand(comp) != 0:
        K += sp.expand(comp)**2 * inv[a, a]*inv[b, b]*inv[c, c]*inv[d, d]
K = sp.simplify(K)
GB = sp.simplify(K - 4*RR + Rs**2)
print(f"R = {sp.simplify(Rs)}")
print(f"GB = {GB}")

# Reduced Lagrangian (drop 4π angular factor)
L = r**2 * sp.exp(de) * (Rs/(16*sp.pi) - sp.Rational(1,2)*mu*sp.diff(phi, r)**2 + sp.Rational(1,2)*eta*phi**2*GB)
L = sp.expand(L)
print(f"L has {len(str(L))} chars")

# Generalized Euler-Lagrange for L(q, q', q''):
def euler_lagrange(L, q):
    q1 = sp.diff(q, r); q2 = sp.diff(q, r, 2)
    term = sp.diff(L, q) - sp.diff(sp.diff(L, q1), r) + sp.diff(sp.diff(L, q2), r, 2)
    return sp.simplify(sp.expand(term))

E_mu = euler_lagrange(L, mu)
E_de = euler_lagrange(L, de)
E_phi = euler_lagrange(L, phi)
print("\nE_phi (scalar equation, up to factor):")
print(sp.simplify(E_phi / (r**2*sp.exp(de)) ))
sp.print_latex(sp.simplify(E_phi))
