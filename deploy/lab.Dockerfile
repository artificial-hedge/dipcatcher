# syntax=docker/dockerfile:1

# Harden the crown-jewel lab image without editing Dockerfile.
# Build the root image first (deploy/supply.mk does this):
#
#   docker build -t dipcatcher:supply -f Dockerfile .
#   docker build -t dipcatcher:supply-hardened --build-arg BASE=dipcatcher:supply \
#     -f deploy/lab.Dockerfile .
#
# The parent already drops root at the end of its runtime stage. This layer
# repeats the runtime user as a numeric uid (hadolint DL3066) and replaces the
# shell-form HEALTHCHECK with JSON argv (hadolint DL3025). The locked venv is
# unchanged. Default seccomp and AppArmor are left to the runtime.

ARG BASE=dipcatcher:supply
FROM ${BASE}

# Package pins for the current crown base
# python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
# (`apt-get -s upgrade` on 2026-11-02). After the owner step in pr_body.md
# replaces that digest, delete this RUN. The replacement digest already
# contains these updates.
USER 0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        base-files=13.8+deb13u7 \
        bash=5.2.37-2+b10 \
        ca-certificates=20250419+deb13u1 \
        gzip=1.13-1+deb13u1 \
        libc-bin=2.41-12+deb13u4 \
        libc6=2.41-12+deb13u4 \
        libaudit-common=1:4.0.2-2+deb13u1 \
        libaudit1=1:4.0.2-2+deb13u1 \
        libcap2=1:2.75-10+deb13u1+b3 \
        liblzma5=5.8.1-1+deb13u2 \
        libpcre2-8-0=10.46-1~deb13u3 \
        libsqlite3-0=3.46.1-7+deb13u2 \
        libssl3t64=3.5.7-1~deb13u3 \
        openssl=3.5.7-1~deb13u3 \
        openssl-provider-legacy=3.5.7-1~deb13u3 \
        perl-base=5.40.1-6+deb13u1 \
        tzdata=2026f-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/*
# The installed wheel resolves configs next to site-packages' grandparent
# (lib/pythonX.Y/configs), not /app/configs. Link the copied tree so /ready
# can open configs/research.yaml. The same line is an owner step for Dockerfile.
RUN python -c "import sys; from pathlib import Path; dest = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'configs'; dest.symlink_to('/app/configs')"
USER 10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}/health\", timeout=3)"]
