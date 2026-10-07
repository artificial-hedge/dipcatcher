# Homebrew formula for fxi — the fx-1 interactive CLI.
#
# This file is the canonical formula source. It is meant to live in a
# private tap (the distribution is proprietary — Homebrew/core would reject
# it), e.g.:
#
#   brew tap artificial-hedge/tap https://github.com/artificial-hedge/homebrew-tap
#   brew install fxi
#   brew tap artificial-hedge/tap
#   brew install artificial-hedge/tap/fxi
#
# Release checklist (also in docs/FXI.md):
#   1. uv build  →  dist/fx_1-<version>-py3-none-any.whl
#   2. attach the wheel to the GitHub release v<version>
#   3. sha256sum dist/fx_1-<version>-py3-none-any.whl  → fill `sha256` below
#   4. update `url` and FXI_VERSION below, commit to the tap repo
class Fxi < Formula
  desc "Interactive CLI for fx-1: plug fx1/fx1-lite API keys into the dipcatcher harness"
  homepage "https://github.com/artificial-hedra/dipcatcher"
  url "https://github.com/artificial-hedra/dipcatcher/releases/download/v0.4.0/fx_1-0.4.0-py3-none-any.whl"
  sha256 "FILL_AT_RELEASE"
  license :cannot_represent # proprietary; see LICENSE in the repository

  depends_on "python@3.12"

  def install
    # Install the release wheel into a private virtualenv and expose every
    # console script the fx-1 distribution ships.
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install cached_download
    bin.install_symlink [
      libexec/"bin/fxi",
      libexec/"bin/fx1",
      libexec/"bin/dipcatcher",
      libexec/"bin/quant",
      libexec/"bin/verify-ledger",
      libexec/"bin/mc-engine",
    ]
  end

  def caveats
    <<~EOS
      fxi stores API keys in ~/.fx1/credentials.json (mode 0600), presence-only.
      Start here:
        fxi keys set fx1        # hosted fx-1 key (prompted, not echoed)
        fxi doctor              # readiness check
        fxi                     # interactive shell
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fxi --version")
  end
end
