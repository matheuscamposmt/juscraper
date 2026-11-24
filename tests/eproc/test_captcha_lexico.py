"""Variante ``lexico`` do captcha do Eproc: so os textos do InfraCaptcha, pela posterior. Sem rede e sem ddddocr."""
import json

import pytest

from juscraper.courts._eproc import captcha


def test_gera_infra_e_lexico():
    assert captcha.gera_infra(ord("o"), ord("B")) == "oBYX"
    assert captcha.gera_infra(ord("y"), ord("o")) == "yott"
    lex = captcha.lexico_infra()
    assert len(lex) == 2766
    assert len(set(lex)) == 2766
    assert all(len(w) == 4 and w == w.lower() for w in lex)
    assert {"obyx", "yott"} <= set(lex)


CHARSET = ["", *"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]


def _probs(texto, certeza=0.9):
    """Probabilidades CTC de um texto: cada caractere numa posicao, com um branco depois de cada um."""
    linhas = []
    for c in texto:
        for alvo in (c, ""):
            linha = [(1 - certeza) / (len(CHARSET) - 1)] * len(CHARSET)
            linha[CHARSET.index(alvo)] = certeza
            linhas.append([linha])
    return linhas


def _linha(**ps):
    resto = (1 - sum(ps.values())) / (len(CHARSET) - len(ps))
    linha = [resto] * len(CHARSET)
    for c, v in ps.items():
        linha[CHARSET.index("" if c == "branco" else c)] = v
    return [linha]


def test_decodifica_lexico_completa_o_tamanho_e_soma_a_caixa():
    # a gulosa junta "tt" num "t" so (o branco do meio perde por pouco) e le "yot"; o lexico acha "yott"
    probs = [_linha(y=0.9), _linha(branco=0.9), _linha(o=0.9), _linha(branco=0.9),
             _linha(t=0.9), _linha(t=0.5, branco=0.45), _linha(t=0.9), _linha(branco=0.9)]
    assert captcha.decodifica_sem_caixa(probs, CHARSET, set(captcha.ALNUM))[0] == "yot"
    t, post = captcha.decodifica_lexico(probs, CHARSET, set(captcha.ALNUM))
    assert t == "yott"
    assert post > 0.5
    t, post = captcha.decodifica_lexico(_probs("OBYX"), CHARSET, set(captcha.ALNUM))
    assert t == "obyx"
    assert post >= captcha.LIMIAR_LEXICO


@pytest.fixture
def pasta(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_DIR", str(tmp_path))
    monkeypatch.setattr(captcha, "_ab", {"n": 0, "ultima": None})
    return tmp_path


def test_variante_lexico_envia_pela_posterior_e_registra_a_etapa(pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "lexico")
    monkeypatch.setattr(captcha, "_probabilidades", lambda _img, _modelo="geral": ("p", "c"))
    monkeypatch.setattr(captcha, "decodifica_lexico", lambda *_a: ("yott", 0.95))
    assert captcha.resolver(b"x", "eproc") == ("yott", "yott", 0.95)
    captcha.registrar("eproc", b"x", "yott", 0.95, aceito=True)
    monkeypatch.setattr(captcha, "decodifica_lexico", lambda *_a: ("yott", 0.6))  # passa no 0,40, nao no 0,8
    assert captcha.resolver(b"y", "eproc") == (None, "yott", 0.6)
    captcha.registrar("eproc", b"y", "yott", 0.6, aceito=None)
    linhas = [json.loads(x) for x in (pasta / "eproc" / "registro.jsonl").read_text().splitlines()]
    assert [(x["variante"], x["etapa"]) for x in linhas] == [("lexico", "lexico"), ("lexico", "nenhuma")]
