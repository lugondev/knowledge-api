from kbase.cli import main


def test_doctor_reports_every_problem_and_fails(capsys, monkeypatch):
    monkeypatch.delenv("KB_API_KEYS", raising=False)
    monkeypatch.delenv("KB_EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("KB_EMBED_MODEL", raising=False)
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert code == 1
    assert "KB_API_KEYS" in out
    assert "KB_EMBED_MODEL" in out


def test_doctor_passes_on_a_complete_environment(capsys, monkeypatch):
    monkeypatch.setenv("KB_API_KEYS", "k:acme")
    monkeypatch.setenv("KB_EMBED_BASE_URL", "http://x/v1")
    monkeypatch.setenv("KB_EMBED_MODEL", "m")
    assert main(["doctor"]) == 0
    assert "ok" in capsys.readouterr().out.lower()


def test_serve_refuses_a_configuration_doctor_would_fail(capsys, monkeypatch):
    # It must fail before binding a port: a service that starts and then 401s
    # every request looks healthy to a load balancer.
    monkeypatch.delenv("KB_API_KEYS", raising=False)
    monkeypatch.delenv("KB_EMBED_MODEL", raising=False)
    assert main(["serve"]) == 1
    assert "refusing to start" in capsys.readouterr().out.lower()


def test_unknown_command_is_an_error(capsys):
    assert main(["nonsense"]) != 0
