"""``consulta_numero``: uma leitura da página de consulta por captcha, com o mesmo payload (sem rede)."""
import base64
from typing import ClassVar

from juscraper.courts._eproc import download

PNG = base64.b64encode(b"png").decode()
FORM = ('<html><body>Consulta Processual<form action="externo_controlador.php?acao=consulta">'
        f'<input type="hidden" name="hdnInfraTipoPagina" value="1"><img src="data:image/png;base64,{PNG}">'
        '<input type="text" name="txtNumProcesso" value=""></form></body></html>')
CAPA = "<html><body>Capa do Processo</body></html>"
NUMERO = "0001234-56.2026.8.16.0001"


class _Resp:
    def __init__(self, texto):
        self.text, self.status_code = texto, 200
        self.url = "https://eproc1g.tjpr.jus.br/eproc/externo_controlador.php"


class _Sessao:
    """Sessão falsa: GET devolve o formulário; POST devolve a capa e guarda o payload."""

    criadas: ClassVar[list["_Sessao"]] = []

    def __init__(self, *_a, **_k):
        self.posts = []
        _Sessao.criadas.append(self)

    def get(self, _url, **_k):
        return _Resp(FORM)

    def post(self, _url, data=None, **_k):
        self.posts.append(data)
        return _Resp(CAPA)


def test_uma_leitura_da_pagina_por_captcha_com_o_mesmo_payload(monkeypatch):
    """O captcha, o form e o payload saem de uma leitura só da página, com os mesmos campos de antes."""
    _Sessao.criadas = []
    monkeypatch.setattr(download, "SessaoRitmada", _Sessao)
    monkeypatch.setattr(download.ocr, "registrar", lambda *_a, **_k: None)
    monkeypatch.setattr(download.ocr, "define_tribunal", lambda *_a, **_k: None)
    monkeypatch.setattr(download.ocr, "resolver", lambda _png, _perfil: ("abcd", "abcd", 0.99))
    original = download.BeautifulSoup
    leituras: list[str] = []

    def conta(html, parser):
        leituras.append(html)
        return original(html, parser)

    monkeypatch.setattr(download, "BeautifulSoup", conta)
    r = download.consulta_numero("tjpr", NUMERO)
    assert r is not None
    assert r["estado"] == "capa"
    assert leituras == [FORM]
    monkeypatch.setattr(download, "BeautifulSoup", original)
    assert _Sessao.criadas[0].posts == [download.build_search_payload(FORM, NUMERO, "abcd")]
    assert download.extrai_captcha(FORM) == b"png"
