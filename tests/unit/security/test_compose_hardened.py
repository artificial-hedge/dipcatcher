"""The hardened compose file keeps the default seccomp and AppArmor profiles."""

from __future__ import annotations

from pathlib import Path

import yaml

_PATH = Path(__file__).resolve().parents[3] / "deploy" / "compose.hardened.yml"


def _services() -> dict[str, dict[str, object]]:
    loaded = yaml.load(_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    services = loaded["services"]
    assert isinstance(services, dict)
    return services


def test_both_services_drop_privileges_and_keep_the_default_profiles() -> None:
    services = _services()
    assert set(services) == {"api", "research-api"}
    for name, service in services.items():
        assert "privileged" not in service, name
        assert service["read_only"] == "true", name
        assert service["cap_drop"] == ["ALL"], name
        assert service["security_opt"] == ["no-new-privileges:true"], name
        tmpfs = service["tmpfs"]
        assert isinstance(tmpfs, list)
        mounts = [item.split(":", 1)[0] for item in tmpfs]
        assert mounts == ["/tmp", "/run", "/app/data"], name
        ports = service["ports"]
        assert isinstance(ports, list)
        assert all(port.startswith("127.0.0.1:") for port in ports), name
