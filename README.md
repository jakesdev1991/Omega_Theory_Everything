# Omega Theory Everything

Complete implementation of the Omega Theory framework - a unified physics model deriving spacetime, gravity, and cosmology from quantum information principles.

## Overview

This repository contains the full simulation suite for Omega Theory v4.0, implementing:

- **Sim1**: Emergent Geometry from quantum entanglement
- **Sim2**: Cosmological dynamics with dynamic scale factor
- **Sim3**: Renormalization group flow in spacetime
- **Sim4**: System evolution with v14 depletion mechanics
- **Sim5**: Emergent gravity as entropic force
- **Sim6**: v14 depletion dynamics

## Theory Foundation

Based on the principle that spacetime geometry emerges from quantum entanglement entropy:

```
S_ent = A/4G  →  g_μν = δS_ent/δρ
```

Key papers:
- `Omega_Theory_v4.0_Technical.md` - Full technical specification
- `Omega_Theory_Laymans_Guide.md` - Accessible overview

## Structure

```
Omega_Theory_Everything/
├── Sim1_Emergent_Geometry.py    # Entanglement → geometry
├── Sim2_Cosmology.py            # FLRW + dynamic scale
├── Sim3_Dynamic_Scale.py        # RG flow in spacetime
├── Sim4_Evolution.py            # Evolution + depletion
├── Sim5_Emergent_Gravity.py     # Entropic gravity
├── sim6_v14_depletion.py        # v14 mechanics
├── update_discovery.sh          # Discovery updater
├── requirements.txt             # Python deps
└── *.md                         # Theory documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulations
python Sim1_Emergent_Geometry.py
python Sim2_Cosmology.py
# ... etc
```

## Development

```bash
# Lint
ruff check .
mypy .

# Test
pytest -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.