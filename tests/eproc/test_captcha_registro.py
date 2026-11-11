"""registrar: teto e amostra dos captchas rotulados; o registro.jsonl guarda todos."""
import json

import pytest

from juscraper.courts._eproc import captcha


@pytest.fixture(autouse=True)
def pasta(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_DIR", str(tmp_path))
    monkeypatch.delenv("JUSCRAPER_CAPTCHA_MAX_ROTULADOS", raising=False)
    monkeypatch.delenv("JUSCRAPER_CAPTCHA_AMOSTRA", raising=False)
    monkeypatch.setattr(captcha, "_contagem", {})
    monkeypatch.setattr(captcha, "_gravados", {})
    return tmp_path


def _rotulados(tmp_path):
    d = tmp_path / "eproc" / "rotulados"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_sem_variaveis_grava_todo_aceito(pasta):
    for i in range(5):
        captcha.registrar("eproc", f"img{i}".encode(), f"ab{i}c", 0.9, aceito=True)
    assert len(_rotulados(pasta)) == 5


def test_teto_para_de_gravar_png_mas_nao_o_registro(pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_MAX_ROTULADOS", "3")
    for i in range(6):
        captcha.registrar("eproc", f"img{i}".encode(), f"ab{i}c", 0.9, aceito=True)
    assert len(_rotulados(pasta)) == 3
    linhas = (pasta / "eproc" / "registro.jsonl").read_text().splitlines()
    assert len(linhas) == 6
    assert all(json.loads(x)["aceito"] for x in linhas)


def test_teto_conta_o_que_ja_existe_no_disco(pasta, monkeypatch):
    d = pasta / "eproc" / "rotulados"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"x{i}_0.png").write_bytes(b"x")
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_MAX_ROTULADOS", "3")
    captcha.registrar("eproc", b"nova", "abcd", 0.9, aceito=True)
    assert len(_rotulados(pasta)) == 3


def test_teto_vale_entre_processos(pasta, monkeypatch):
    """Outro processo gravando na mesma pasta: a recontagem periódica segura o teto."""
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_MAX_ROTULADOS", "5")
    monkeypatch.setattr(captcha, "RECONTA", 2)
    captcha.registrar("eproc", b"a", "aaaa", 0.9, aceito=True)
    d = pasta / "eproc" / "rotulados"
    for i in range(6):  # o outro processo
        (d / f"x{i}_0.png").write_bytes(b"x")
    for i in range(5):
        captcha.registrar("eproc", f"img{i}".encode(), f"ab{i}c", 0.9, aceito=True)
    assert len(_rotulados(pasta)) <= 7 + 1  # 1 próprio + 6 alheios + no máximo RECONTA - 1 antes de recontar


def test_amostra_grava_uma_parte(pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AMOSTRA", "4")
    for i in range(200):
        captcha.registrar("eproc", f"img{i}".encode(), f"ab{i}c", 0.9, aceito=True)
    n = len(_rotulados(pasta))
    assert 20 <= n <= 80  # ~1 em 4, escolhido pelo hash da imagem
