<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Novel — sealed manuscript staging

This directory stages the novel for the token-gated day-one launch described in
[`../launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md).

The manuscript is committed **encrypted**, with a public SHA-256 commitment of the
plaintext. The plaintext is intentionally never published in this repository before the
launch event; the content key is released only through the token gates on day one.

## Layout

```
novel/
  README.md               # this file (committed)
  seal.sh                 # seal/unseal helper (committed)
  manuscript.md.enc       # AES-256-CBC ciphertext of the manuscript (committed, once sealed)
  manuscript.sha256       # SHA-256 commitment of the plaintext (committed, once sealed)
  .release-key.txt        # content key — NEVER COMMITTED; keep an offline backup
  .plaintext/             # working plaintext — never committed
```

## Workflow

1. Place the manuscript (Markdown) at `.plaintext/manuscript.md`.
2. Run `./seal.sh`. This:
   - generates `.release-key.txt` (64-hex-character random key) if absent;
   - encrypts the manuscript to `manuscript.md.enc` (AES-256-CBC, PBKDF2, random salt);
   - writes the plaintext SHA-256 commitment to `manuscript.sha256`;
   - prints verification instructions.
3. Commit `manuscript.md.enc` and `manuscript.sha256` (never the key or plaintext).
4. **Back up `.release-key.txt` and `.plaintext/` outside this repository** (password
   manager and/or offline media). If the key is lost before launch, the manuscript can be
   re-sealed from the plaintext — but then re-announce the new commitment hash.

## Unseal (authorized holders, day one, or testing)

```bash
./seal.sh --unseal   # writes .plaintext/manuscript.decrypted.md
sha256sum .plaintext/manuscript.decrypted.md   # must equal manuscript.sha256
```

## Provenance note

The commitment certifies the author's own manuscript file, not a transcription. When the
plaintext is revealed on day one, anyone can verify:

```bash
sha256sum revealed_manuscript.md   # compare against the committed manuscript.sha256
```

Keep dated copies of the source draft per [`../docs/PROVENANCE.md`](../docs/PROVENANCE.md).

## License

All materials in this directory are product materials, all rights reserved,
`LicenseRef-Omega-Product-Proprietary`. Viewing files in this public repository does not
grant a right to deploy, copy, modify, distribute, or commercially exploit the novel.
Claimant license terms for the day-one release are to be defined with counsel (see the
launch plan, §6).
