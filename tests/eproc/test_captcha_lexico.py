"""Variante ``lexico`` do captcha do Eproc: so os textos do InfraCaptcha, pela posterior. Sem rede e sem ddddocr."""
import io
import json
import sys
import threading

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
    monkeypatch.setattr(captcha, "_saida_onnx", lambda _img: None)  # sem a sessao exposta: o caminho das listas
    monkeypatch.setattr(captcha, "_probabilidades", lambda _img, _modelo="geral": ("p", "c"))
    monkeypatch.setattr(captcha, "decodifica_lexico", lambda *_a: ("yott", 0.95))
    assert captcha.resolver(b"x", "eproc") == ("yott", "yott", 0.95)
    captcha.registrar("eproc", b"x", "yott", 0.95, aceito=True)
    monkeypatch.setattr(captcha, "decodifica_lexico", lambda *_a: ("yott", 0.6))  # passa no 0,40, nao no 0,8
    assert captcha.resolver(b"y", "eproc") == (None, "yott", 0.6)
    captcha.registrar("eproc", b"y", "yott", 0.6, aceito=None)
    linhas = [json.loads(x) for x in (pasta / "eproc" / "registro.jsonl").read_text().splitlines()]
    assert [(x["variante"], x["etapa"]) for x in linhas] == [("lexico", "lexico"), ("lexico", "nenhuma")]


def test_decodifica_lexico_em_varias_threads_com_o_cache_vazio(monkeypatch):
    # 24/11/2026: com 7 vagas, duas threads montavam o lexico juntas e cada texto saia em dobro (posterior 0,5).
    # Com a troca de thread a cada microssegundo, o codigo antigo saia com 13.830 e 22.128 textos.
    intervalo = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    captcha.lexico_infra.cache_clear()
    vars(captcha)["_matrizes_lexico"].cache_clear()  # cache privado, zerado como o _ab dos outros testes
    monkeypatch.setattr(captcha, "_COLUNAS_LEXICO", {})
    probs = _probs("OBYX")
    saidas, barreira = [], threading.Barrier(8)

    def roda():
        barreira.wait()
        saidas.append(captcha.decodifica_lexico(probs, CHARSET, set(captcha.ALNUM)))

    ts = [threading.Thread(target=roda) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    sys.setswitchinterval(intervalo)
    assert len(captcha.lexico_infra()) == 2766
    assert len(set(saidas)) == 1
    assert saidas[0][0] == "obyx"
    assert saidas[0][1] >= captcha.LIMIAR_LEXICO


def _logits(texto, extras=3000, semente=1):
    """Saida antes do softmax: o texto em CTC sobre ruido, com milhares de colunas fora do alfabeto (como o ddddocr)."""
    np = pytest.importorskip("numpy")
    charset = CHARSET + [chr(0x4E00 + i) for i in range(extras)]
    o = np.random.default_rng(semente).normal(0, 2, (2 * len(texto) + 2, 1, len(charset))).astype(np.float32)
    for i, c in enumerate(texto):
        o[2 * i, 0, charset.index(c)] += 9
        o[2 * i + 1, 0, 0] += 9
    return o, charset


def test_decodifica_lexico_saida_igual_ao_softmax_inteiro():
    np = pytest.importorskip("numpy")
    for semente, texto in enumerate(["OBYX", "yott", "a1b2"]):
        o, charset = _logits(texto, semente=semente)
        e = np.exp(o - o.max(2, keepdims=True))
        probs = (e / e.sum(2, keepdims=True)).tolist()  # o que o ddddocr devolve com probability=True
        assert captcha.decodifica_lexico_saida(o, charset, set(captcha.ALNUM)) == captcha.decodifica_lexico(
            probs, charset, set(captcha.ALNUM)
        )


@pytest.mark.usefixtures("pasta")
def test_variante_lexico_le_a_saida_do_modelo_sem_as_listas(monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "lexico")
    monkeypatch.setattr(captcha, "_saida_onnx", lambda _img: _logits("OBYX"))
    monkeypatch.setattr(captcha, "_probabilidades", lambda *_a: pytest.fail("nao deve ler as listas"))
    texto, lido, post = captcha.resolver(b"x", "eproc")
    assert (texto, lido) == ("obyx", "obyx")
    assert post >= captcha.LIMIAR_LEXICO


def test_saida_onnx_tem_o_preparo_do_ddddocr():
    """Com o ddddocr instalado, o softmax da saida direta e o que o ddddocr devolve com probability=True."""
    pytest.importorskip("ddddocr")
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    draw = pytest.importorskip("PIL.ImageDraw")
    im = image.new("RGB", (135, 45), "white")
    draw.Draw(im).text((10, 10), "obyx", fill="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    png = buf.getvalue()

    saida = vars(captcha)["_saida_onnx"](png)  # privadas, lidas como o _matrizes_lexico acima
    assert saida is not None
    o = saida[0].astype(np.float64)
    e = np.exp(o - o.max(-1, keepdims=True))
    probs, charset = vars(captcha)["_probabilidades"](png)
    assert list(saida[1]) == list(charset)
    assert np.allclose(e / e.sum(-1, keepdims=True), np.asarray(probs), atol=1e-6)
