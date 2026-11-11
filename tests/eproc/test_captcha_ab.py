import json

import pytest

from juscraper.courts._eproc import captcha


@pytest.fixture
def _pasta(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_DIR", str(tmp_path))
    captcha._ab.update(n=0, ultima=None)
    yield tmp_path
    captcha._ab.update(n=0, ultima=None)


def test_sem_ab_usa_a_leitura_atual(_pasta, monkeypatch):
    monkeypatch.delenv("JUSCRAPER_CAPTCHA_AB", raising=False)
    monkeypatch.setattr(captcha, "ler", lambda img, perfil, renorm=False: ("ab12", 0.9))
    assert captcha.resolver(b"x", "eproc") == ("ab12", "ab12", 0.9)
    assert captcha._ab["ultima"] is None


def test_ab_alterna_e_registra_a_variante(_pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "atual,renorm")
    vistos = []
    monkeypatch.setattr(captcha, "ler", lambda img, perfil, renorm=False: (vistos.append(renorm), ("ab12", 0.9))[1])
    captcha.resolver(b"x", "eproc")
    captcha.registrar("eproc", b"x", "ab12", 0.9, aceito=None)
    captcha.resolver(b"y", "eproc")
    captcha.registrar("eproc", b"y", "ab12", 0.9, aceito=True)
    assert vistos == [False, True]
    linhas = [json.loads(x) for x in (_pasta / "eproc" / "registro.jsonl").read_text().splitlines()]
    assert [x["variante"] for x in linhas] == ["atual", "renorm"]


def test_dessat_so_quando_a_atual_nao_passa(_pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "dessat")
    monkeypatch.setattr(captcha, "sem_saturados", lambda img: b"limpa")
    monkeypatch.setattr(captcha, "ler", lambda img, perfil, renorm=False:
                        ("cd34", 0.8) if img == b"limpa" else ("ab1", 0.2))
    assert captcha.resolver(b"suja", "eproc")[0] == "cd34"
    monkeypatch.setattr(captcha, "ler", lambda img, perfil, renorm=False: ("ab12", 0.9))
    assert captcha.resolver(b"suja", "eproc")[0] == "ab12"


def test_renormaliza_divide_pelas_permitidas():
    probs = [[0.2, 0.2, 0.6]]
    assert captcha.renormaliza(probs, ["", "a", "字"], {"a"}) == [[0.5, 0.5, 0.0]]
