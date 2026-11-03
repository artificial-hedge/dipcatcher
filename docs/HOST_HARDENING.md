# Host hardening

`scripts/prod_preflight.py` records four host checks: unattended
upgrades, the firewall, SSH authentication, and time sync. When the
process is not root, each one is a warning (`"warning": "not root"`) and
the exit status does not change. When the process is root, each one is a
normal check. A failure then makes preflight exit 1.

`prod-up` runs preflight as the deploying user. That path warns. It does
not fail the deploy for these four items. To fail closed, run preflight
as root with the same environment the deploy uses. The command names the
variables. It does not print their values.

```bash
sudo --preserve-env=QUANT_API_KEY,RESEARCH_API_KEY,DIPCATCHER_TLS_DIR,DIPCATCHER_LAB_DATA,DIPCATCHER_RESEARCH_DATA,LAB_TLS_PORT,RESEARCH_TLS_PORT,PROMETHEUS_PUBLISH_PORT,GRAFANA_PUBLISH_PORT uv run python scripts/prod_preflight.py
```

The certificate-window clock check is unchanged. It still fails preflight
for a skew whether or not the process is root. The time-sync check below
is the NTP service.

## Unattended upgrades

Install `unattended-upgrades`. `/etc/apt/apt.conf.d` must set
`APT::Periodic::Unattended-Upgrade "1"` on a line that is not a comment.
`systemctl is-enabled unattended-upgrades` must print `enabled`.

## Firewall

Default deny on input. The only accepted TCP ports are 443, 8443, and
8444. No other TCP port is accepted, including 22.

Prometheus `9090` and Grafana `3000` stay bound to `127.0.0.1` and are
not in the allow list. The compose file does not publish 443. 443 is the
public TLS port when a front door is added in front of the loopback
listeners.

The root check reads `ufw status verbose` when the status is active. The
status must include `deny (incoming)`, and the ALLOW TCP ports must be
exactly 443, 8443, and 8444. Otherwise it reads `iptables -S` and
`ip6tables -S`. The INPUT policy must be DROP or REJECT, and the accepted
TCP destination ports must be that same set. An ACCEPT policy, a TCP
accept with no destination port, or any other port fails the check. An
empty `ip6tables` result is not a second policy. A present ip6tables
policy is held to the same ports.

## SSH

In `/etc/ssh/sshd_config` and every file in `/etc/ssh/sshd_config.d/*.conf`,
every uncommented `PasswordAuthentication` value is `no`. The same applies
to `KbdInteractiveAuthentication` and `ChallengeResponseAuthentication`
when those lines are present. `PubkeyAuthentication` is not `no`.
Missing `PasswordAuthentication` fails the root check. A `Match` block
that sets any of those values back to `yes` fails the check.

SSH is key-only, and port 22 is not one of the three firewall ports.
Reach the host from the console or from a path that does not open another
inbound port.

## Time sync

`timedatectl show -p NTPSynchronized --value` prints `yes`.
systemd-timesyncd or chrony can be the service that makes that true.
A value of `no` fails the root check.
