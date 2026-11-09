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
