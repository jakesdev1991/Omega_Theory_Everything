#!/usr/bin/env bash
# =============================================================================
# Omega Cloud Host — one-shot provisioner for Oracle Cloud (Ubuntu 24.04)
# Target: the Always Free A1 Flex instance (4 OCPU / 24 GB / 200 GB).
# Run as root ON THE VM:   bash omega_cloud_provision.sh <PUBLIC_IP>
#
# Brings up, with systemd:
#   - nginx reverse proxy (443 TLS via Let's Encrypt for the website)
#     · /            -> Next.js website  (127.0.0.1:3460)
#     · /relay       -> Chorus nostr relay (127.0.0.1:8080, wss:// over 443)
#     · /blossom     -> Chorus blossom server (same origin)
#   - Chorus relay built from source (pinned v2.0.2, branch latest)
#   - Node 22 LTS + the Omega repo (git clone from GitHub over SSH)
#   - Website:  npm ci + build + `next start` (port 3460)
#   - nostr-client daemon + lucifer NIP-90 DVM daemon
#   - certbot with the nginx plugin (needs DOMAIN below)
#
# REQUIRED ENV (edit before running):
#   OMEGA_DOMAIN      e.g. omegatheory.org   (empty = self-signed staging cert)
#   OMEGA_EMAIL       for Let's Encrypt
#   OMEGA_REPO_SSH    ssh URL for private repo, or https public URL
#
# Secrets (relay keys, chorus.conf) are NOT baked in: after provisioning,
# copy ~/.config/omega-nostr/{keys,chorus.conf,tls} from Ryzenvoid, or run
# keygen on the VM. See deploy/README.md.
# =============================================================================
set -euo pipefail

PUBLIC_IP="${1:?usage: omega_cloud_provision.sh <PUBLIC_IP>}"
OMEGA_DOMAIN="${OMEGA_DOMAIN:-}"
OMEGA_EMAIL="${OMEGA_EMAIL:-}"
OMEGA_REPO_URL="${OMEGA_REPO_URL:-https://github.com/jakesdev1991/Omega_Theory_Everything.git}"

echo "== [1/8] base packages =="
apt-get update -y
apt-get install -y curl git build-essential pkg-config libssl-dev nginx python3-certbot-nginx ufw fail2ban

echo "== [2/8] Node 22 LTS =="
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi
node --version

echo "== [3/8] Rust (for chorus) =="
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  source "$HOME/.cargo/env"
fi
cargo --version

echo "== [4/8] Chorus relay (v2.0.2) =="
mkdir -p /opt/omega && cd /opt/omega
if [ ! -d chorus ]; then
  git clone --branch latest --depth 1 https://github.com/mikedilger/chorus.git
fi
cd chorus && cargo build --release
ln -sf /opt/omega/chorus/target/release/chorus /usr/local/bin/chorus
ln -sf /opt/omega/chorus/target/release/chorus_cmd /usr/local/bin/chorus_cmd

echo "== [5/8] Omega repo + website =="
mkdir -p /opt/omega/app && cd /opt/omega/app
if [ ! -d Omega_Theory_Everything ]; then
  git clone "$OMEGA_REPO_URL" Omega_Theory_Everything
fi
cd Omega_Theory_Everything/web
npm ci --no-audit --no-fund
npm run build

echo "== [6/8] systemd services =="
cat > /etc/systemd/system/omega-web.service <<'EOF'
[Unit]
Description=Omega website (Next.js production)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/omega/app/Omega_Theory_Everything/web
EnvironmentFile=-/root/.config/omega-nostr/keys/env.sh
Environment=PORT=3460
ExecStart=/usr/bin/npm run start
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/omega-relay.service <<'EOF'
[Unit]
Description=Omega nostr relay (Chorus 2.0.2)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/chorus /root/.config/omega-nostr/chorus.conf
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/omega-nostr-client.service <<'EOF'
[Unit]
Description=Omega nostr client (store directory publisher)
After=omega-relay.service
Wants=omega-relay.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/omega/app/Omega_Theory_Everything/nostr-client
EnvironmentFile=/root/.config/omega-nostr/keys/env.sh
ExecStart=/usr/bin/node client-daemon.mjs
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/omega-lucifer.service <<'EOF'
[Unit]
Description=Omega Lucifer NIP-90 DVM node
After=omega-relay.service
Wants=omega-relay.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/omega/app/Omega_Theory_Everything/mobile-node
EnvironmentFile=/root/.config/omega-nostr/keys/env.sh
ExecStart=/usr/bin/node lucifer-daemon.mjs
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo "== [7/8] nginx =="
cat > /etc/nginx/sites-available/omega <<'EOF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    server_name _;

    # ACME challenge + redirect to https
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    server_name _;

    # Placeholder certs — certbot replaces these in the final step.
    ssl_certificate     /etc/nginx/ssl/omega-selfsigned.crt;
    ssl_certificate_key /etc/nginx/ssl/omega-selfsigned.key;

    client_max_body_size 64m;

    # Website
    location / {
        proxy_pass http://127.0.0.1:3460;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Nostr relay (wss:// over 443)
    location /relay {
        proxy_pass https://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_ssl_verify off;
    }

    # Blossom file server
    location /blossom {
        proxy_pass https://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_ssl_verify off;
    }
}
EOF

mkdir -p /etc/nginx/ssl
if [ ! -f /etc/nginx/ssl/omega-selfsigned.crt ]; then
  openssl req -x509 -nodes -days 90 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/omega-selfsigned.key \
    -out /etc/nginx/ssl/omega-selfsigned.crt \
    -subj "/CN=${OMEGA_DOMAIN:-$PUBLIC_IP}"
fi
ln -sf /etc/nginx/sites-available/omega /etc/nginx/sites-enabled/omega
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl enable --now nginx

echo "== [8/8] Let's Encrypt (if domain set) =="
if [ -n "$OMEGA_DOMAIN" ] && [ -n "$OMEGA_EMAIL" ]; then
  certbot --nginx -d "$OMEGA_DOMAIN" --non-interactive --agree-tos -m "$OMEGA_EMAIL" --redirect
else
  echo "OMEGA_DOMAIN not set — self-signed staging cert in place. Re-run certbot when the domain is ready."
fi

echo
echo "PROVISION COMPLETE."
echo "Next: copy secrets from Ryzenvoid (keys/, chorus.conf) to /root/.config/omega-nostr/,"
echo "then: systemctl enable --now omega-web omega-relay omega-nostr-client omega-lucifer"
