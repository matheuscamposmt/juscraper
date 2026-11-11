"""Ritmo adaptativo por host: o intervalo encolhe na resposta boa e dobra na recusa."""
import pytest
import responses

from juscraper.core import ritmo

URL = "https://eproc1g.tjxx.jus.br/eproc/externo_controlador.php"


@pytest.fixture(autouse=True)
def _pasta_do_ritmo(tmp_path, monkeypatch, mocker):
    monkeypatch.setenv("JUSCRAPER_RITMO_DIR", str(tmp_path))
    mocker.patch("time.sleep")


def test_resposta_boa_encolhe_ate_o_piso():
    assert ritmo.adapta(10.0, ok=True, piso=0.5) == pytest.approx(9.0)
    assert ritmo.adapta(0.5, ok=True, piso=0.5) == 0.5


def test_recusa_dobra_com_minimo_e_teto():
    assert ritmo.adapta(0.5, ok=False, piso=0.5) == ritmo.RECUSA_MINIMA
    assert ritmo.adapta(8.0, ok=False, piso=0.5) == 16.0
    assert ritmo.adapta(800.0, ok=False, piso=0.5) == ritmo.TETO


def test_retry_after_vira_intervalo_minimo():
    assert ritmo.adapta(1.0, ok=False, piso=0.5, retry_after=30.0) == 30.0
    assert ritmo.retry_after("30") == 30.0
    assert ritmo.retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


@responses.activate
def test_sessao_encolhe_o_intervalo_na_resposta_boa():
    responses.add(responses.GET, URL, body="ok", status=200)
    host = "eproc1g.tjxx.jus.br"
    ritmo.SessaoRitmada(piso=0.5).get(URL)
    assert ritmo.estado(host, piso=0.5)[1] == 0.5


@responses.activate
def test_sessao_levanta_sobrecarga_e_dobra_no_429():
    responses.add(responses.GET, URL, status=429, headers={"Retry-After": "30"})
    with pytest.raises(ritmo.Sobrecarga):
        ritmo.SessaoRitmada(piso=0.5).get(URL)
    assert ritmo.estado("eproc1g.tjxx.jus.br", piso=0.5)[1] == 30.0


@responses.activate
def test_aviso_de_excesso_na_pagina_conta_como_recusa():
    # Mensagem que o e-SAJ devolve quando recusa por consultas simultaneas.
    responses.add(responses.GET, URL, body="Foram identificadas multiplas consultas simultâneas", status=200)
    with pytest.raises(ritmo.Sobrecarga):
        ritmo.SessaoRitmada(piso=0.5).get(URL)


@responses.activate
def test_aviso_de_excesso_no_meio_de_pagina_grande_conta_como_recusa():
    # No e-SAJ do TJAL (09/11/2026) o aviso vem no caractere 24.019 de uma
    # página de 46 KB, depois do cabeçalho e dos scripts do portal.
    corpo = "<script>" + "x" * 24000 + "</script> Foram identificadas multiplas consultas simultâneas"
    responses.add(responses.GET, URL, body=corpo, status=200)
    with pytest.raises(ritmo.Sobrecarga):
        ritmo.SessaoRitmada(piso=0.5).get(URL)
    assert ritmo.estado("eproc1g.tjxx.jus.br", piso=0.5)[1] == ritmo.RECUSA_MINIMA


@responses.activate
def test_registro_de_medidas_por_host(tmp_path, monkeypatch):
    import json
    arq = tmp_path / "ritmo.log"
    monkeypatch.setenv("JUSCRAPER_RITMO_LOG", str(arq))
    monkeypatch.setenv("JUSCRAPER_RITMO_JANELA", "0")
    ritmo._MEDIDAS.clear()
    responses.add(responses.GET, URL, body="ok", status=200)
    responses.add(responses.GET, URL, status=503)
    sessao = ritmo.SessaoRitmada(piso=0.5)
    sessao.get(URL)
    with pytest.raises(ritmo.Sobrecarga):
        sessao.get(URL)
    linhas = [json.loads(x) for x in arq.read_text().splitlines()]
    assert [x["recusas"] for x in linhas] == [0, 1]
    assert all(x["host"] == "eproc1g.tjxx.jus.br" and x["n"] == 1 for x in linhas)
    assert linhas[1]["intervalo"] == 0.5


def test_sem_variavel_nao_grava(tmp_path, monkeypatch):
    monkeypatch.delenv("JUSCRAPER_RITMO_LOG", raising=False)
    ritmo._registra("x.jus.br", 0.1, True, 0.5)
    assert not list(tmp_path.glob("*.log"))
