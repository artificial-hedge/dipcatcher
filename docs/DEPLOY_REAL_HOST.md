# Deploy on a fresh Ubuntu host

This is the copy-paste path for one new Ubuntu VM. The stack is simulated
paper and the research API. It does not place a live order and it does not
talk to a broker. Every command is meant to be pasted as-is. Secret values
are read into the shell. They are not written into this file.

Port 22 is not opened. Use the console, or a path that does not add another
inbound port. The only accepted TCP ports are 443, 8443, and 8444.
Prometheus `9090` and Grafana `3000` stay on `127.0.0.1`.

## Packages, Docker, time, and SSH

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git unattended-upgrades
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo timedatectl set-ntp true
sudo tee /etc/apt/apt.conf.d/20auto-upgrades >/dev/null <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
sudo systemctl enable --now unattended-upgrades
sudo tee /etc/ssh/sshd_config.d/99-dipcatcher.conf >/dev/null <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
EOF
sudo systemctl reload ssh || sudo systemctl reload sshd
```

## Firewall

Default deny on input. Do not allow port 22.

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 443/tcp
sudo ufw allow 8443/tcp
sudo ufw allow 8444/tcp
sudo ufw --force enable
sudo ufw status verbose
```

`ufw status verbose` must show `deny (incoming)` and ALLOW TCP only for
443, 8443, and 8444.

## Clone and sync

```bash
sudo install -d -o "$USER" -g "$USER" /opt/dipcatcher
git clone https://github.com/artificial-hedge/dipcatcher.git /opt/dipcatcher
cd /opt/dipcatcher
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
make sync
```

## Directories

```bash
sudo install -d -m 0755 /var/lib/dipcatcher/lab /var/lib/dipcatcher/research
sudo install -d -m 0755 /var/log/dipcatcher/app /var/log/dipcatcher/proxy /var/log/dipcatcher/audit
sudo install -d -m 0755 /var/backups/dipcatcher /etc/dipcatcher/tls
sudo chown -R 10001:10001 /var/lib/dipcatcher /var/log/dipcatcher
sudo chmod 0700 /etc/dipcatcher
```

The lab and research data directories must be owned by uid 10001. The log
directory is mounted at `/logs` in each replica. `prod-data` creates the
same directories when `LOG_DIR` is `/tmp/dipcatcher-*`, `/var/log/dipcatcher*`,
or `/var/lib/dipcatcher*`.

## Certificates

The compose proxy keeps `auto_https` off. Do not turn ACME on in
`deploy/proxy/Caddyfile`. Pick one of the two host setups below.

### Provided certificates

Copy the full chain and the key into the TLS directory before `prod-certs`.
`prod-certs` generates a localhost certificate only when `dev.crt` or
`dev.key` is missing, so an existing pair is left in place. The key stays
mode 600. Do not commit either file.

```bash
sudo install -m 0644 /path/to/fullchain.pem /etc/dipcatcher/tls/dev.crt
sudo install -m 0600 /path/to/privkey.pem /etc/dipcatcher/tls/dev.key
```

Clients then use `https://127.0.0.1:8443` for the lab API and
`https://127.0.0.1:8444` for the research API, on the names in that
certificate.

### Host Caddy with ACME

This Caddy is a second install on the host. It is not the compose proxy.
It answers port 443, obtains the certificate, and forwards to the loopback
listeners. The compose listeners keep the file in `/etc/dipcatcher/tls`.

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
{
	email ops@example.com
}
lab.example.com {
	reverse_proxy https://127.0.0.1:8443 {
		transport http {
			tls
			tls_insecure_skip_verify
		}
	}
}
research.example.com {
	reverse_proxy https://127.0.0.1:8444 {
		transport http {
			tls
			tls_insecure_skip_verify
		}
	}
}
EOF
sudo systemctl reload caddy
```

Replace `ops@example.com`, `lab.example.com`, and `research.example.com`.
`tls_insecure_skip_verify` is only there because the compose listener is
still the local certificate. When the files in `/etc/dipcatcher/tls` are
the public certificate, drop those two `tls` lines and keep the proxy.

If you have not installed a public certificate, let the makefile create
the local pair:

```bash
cd /opt/dipcatcher
export DIPCATCHER_TLS_DIR=/etc/dipcatcher/tls
make -f deploy/prod.mk prod-certs
```

## Secrets

`prod-secrets` reads `QUANT_API_KEY` and `RESEARCH_API_KEY` from the
environment and writes files under `/tmp/dipcatcher-ops7-secrets` (override
with `SECRET_DIR`). It does not print the values. Do not put them in a
shell history file if you can avoid it.

```bash
cd /opt/dipcatcher
export DIPCATCHER_LAB_DATA=/var/lib/dipcatcher/lab
export DIPCATCHER_RESEARCH_DATA=/var/lib/dipcatcher/research
export DIPCATCHER_TLS_DIR=/etc/dipcatcher/tls
export LOG_DIR=/var/log/dipcatcher
export BACKUP_ROOT=/var/backups/dipcatcher
read -rs QUANT_API_KEY
export QUANT_API_KEY
read -rs RESEARCH_API_KEY
export RESEARCH_API_KEY
make -f deploy/prod.mk prod-secrets
```

## First boot

```bash
cd /opt/dipcatcher
make -f deploy/prod.mk prod-certs
make -f deploy/prod.mk prod-secrets
make -f deploy/prod.mk prod-preflight
make -f deploy/prod.mk prod-up
```

`prod-up` runs certificates, data directories, preflight, and the image
build again, then starts the stack. Preflight as the deploying user
records host-hardening warnings and does not fail the boot on them. To
fail closed, run the same check as root. The command names the variables.
It does not print their values.

```bash
cd /opt/dipcatcher
sudo --preserve-env=QUANT_API_KEY,RESEARCH_API_KEY,DIPCATCHER_TLS_DIR,DIPCATCHER_LAB_DATA,DIPCATCHER_RESEARCH_DATA,LAB_TLS_PORT,RESEARCH_TLS_PORT,PROMETHEUS_PUBLISH_PORT,GRAFANA_PUBLISH_PORT uv run python scripts/prod_preflight.py
```

## Smoke

`prod-smoke` is the acceptance drill. It brings the stack up, checks TLS
`/health`, the 64 KiB body cap, the write lock, and one request id, then
rolls back and stops the stack. After a green smoke, start the stack
again so the service stays up.

```bash
cd /opt/dipcatcher
make -f deploy/prod.mk prod-smoke
make -f deploy/prod.mk prod-up
```

## Backup and log timers

The backup unit writes a new stamp under `/var/backups/dipcatcher`,
rehashes it, and copies only after that check. It does not stop the
writer. A torn read exits 1 and skips the copy. The log unit rotates
files under `/var/log/dipcatcher` and prunes rotated Docker json logs.
It does not rotate `entries.jsonl` or `checkpoints.jsonl`.

`/etc/dipcatcher/backup.env` is mode 600. Use `rsync` or `s3`. Do not put
AWS keys or API keys in it. An instance role or `aws configure` supplies
S3 credentials. `none` verifies and does not copy.

```bash
sudo tee /etc/dipcatcher/backup.env >/dev/null <<'EOF'
DIPCATCHER_BACKUP_COPY=rsync
DIPCATCHER_BACKUP_RSYNC_TARGET=backup@backup.example:/var/backups/dipcatcher
EOF
sudo chmod 600 /etc/dipcatcher/backup.env
cd /opt/dipcatcher
sudo cp deploy/systemd/dipcatcher-prod-backup.service /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-prod-backup.timer /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-log-retain.service /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-log-retain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dipcatcher-prod-backup.timer dipcatcher-log-retain.timer
```

For S3, the env file is:

```bash
sudo tee /etc/dipcatcher/backup.env >/dev/null <<'EOF'
DIPCATCHER_BACKUP_COPY=s3
DIPCATCHER_BACKUP_S3_URI=s3://bucket/dipcatcher
EOF
sudo chmod 600 /etc/dipcatcher/backup.env
```

Cron, instead of the timers:

```bash
cd /opt/dipcatcher
sudo cp deploy/cron/dipcatcher-prod-backup /etc/cron.d/dipcatcher-prod-backup
sudo cp deploy/cron/dipcatcher-log-retain /etc/cron.d/dipcatcher-log-retain
sudo chmod 644 /etc/cron.d/dipcatcher-prod-backup /etc/cron.d/dipcatcher-log-retain
```

The backup schedule is 03:15 every day. Log retention is hourly. Both
invoke `/opt/dipcatcher/.venv/bin/python`.

One quiet backup, with the writer stopped:

```bash
sudo docker stop dipcatcher-ops7-lab-a-1
cd /opt/dipcatcher
make -f deploy/prod.mk LAB_DATA=/var/lib/dipcatcher/lab BACKUP_ROOT=/var/backups/dipcatcher prod-backup-ship
sudo docker start dipcatcher-ops7-lab-a-1
```

## Canary

Build first. `prod-build` saves the current tags as the previous images
and then builds. Containers that are already running keep the old image
until they are recreated. `prod-canary` starts one lab reader that is not
in the Caddy pool, probes it from inside the container, and then recreates
the API replicas and the proxy or leaves them running. The canary does
not take the write lock. Exit 0 means the roll happened.

```bash
cd /opt/dipcatcher
make -f deploy/prod.mk prod-build
make -f deploy/prod.mk prod-canary
```

## Log and disk caps

Compose keeps container stdout and stderr with the json-file driver.
`DIPCATCHER_LOG_MAX_SIZE` defaults to `10m` and `DIPCATCHER_LOG_MAX_FILE`
defaults to `5`. The age cap is the log timer: `DIPCATCHER_LOG_MAX_BYTES`
(10485760), `DIPCATCHER_LOG_MAX_AGE_S` (604800), and
`DIPCATCHER_LOG_MAX_FILES` (5). Per-stream overrides are
`DIPCATCHER_APP_LOG_*`, `DIPCATCHER_PROXY_LOG_*`, and
`DIPCATCHER_AUDIT_LOG_*`.

`DipcatcherDiskPressure` pages when a writable filesystem stays under 15
percent free for 5 minutes. The ratio `0.15` is set in
`deploy/observability/alerts.yml`. The series comes from `node-exporter`.
If that container is down, the rule stays silent.

Names, defaults, and the reason for each knob are in
`deploy/prod.env.example`. They are not `ENV_CONTRACT` names.
