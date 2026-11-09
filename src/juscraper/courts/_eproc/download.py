"""Camada HTTP da consulta pública do Eproc.

Uma consulta por número custa duas requisições, mais as repetições do captcha:

1. ``GET`` da página de consulta, que traz o captcha embutido como
   ``data:image/png;base64``. O captcha vale para um envio só e fica preso ao
   ``PHPSESSID`` da sessão. Medido em 09/11/2026: o mesmo captcha, ou nenhum,
   na segunda consulta da sessão volta "Código da imagem (captcha) incorreto".
2. ``POST`` do formulário no host FINAL do GET. O TJMG redireciona
   ``eproc1g.tjmg`` para ``eproc-consulta-publica-1g.tjmg``; o POST ao host
   antigo vira GET no redirecionamento e o formulário se perde.

Rate limit: :class:`SessaoEspacada` espera ``espera`` segundos entre
requisições ao mesmo host, com uma trava em arquivo que vale para todos os
processos da máquina. HTTP 403, 429, 5xx ou aviso de excesso levantam
:class:`Sobrecarga`, e quem chama faz o backoff.

Tipo de captcha por tribunal, medido em 09/11/2026 por GET a partir de um IP
da AWS:

| Captcha                    | Tribunais                                     |
|----------------------------|-----------------------------------------------|
| imagem (este módulo)       | TJPR, TJMG, TJBA, TJAC, TRF6 (ver ``trf6/``)  |
| Cloudflare Turnstile       | TJSP, TJSC, TJMS, TRF2, JFRJ, JFES            |
| página montada em JS       | TRF4, JFPR, JFRS, JFSC (eprocV2)              |
| HTTP 403 já no primeiro GET| TJRS (Cloudflare), TJRJ, TJSE                 |
"""
from __future__ import annotations

import base64
import fcntl
import re
import time
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import captcha as ocr

NADA = "Nenhum registro encontrado"
RECUSA = re.compile(r"captcha\)? incorreto", re.I)
SOBRECARGA = re.compile(r"consultas simult|too many requests|excesso de (acessos|requisi)", re.I)
TENTATIVAS = 12
ESPERA = 6.0
"""Segundos entre duas requisições ao mesmo host."""
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

URLS: dict[tuple[str, int], str] = {
    ("tjmg", 1): "https://eproc-consulta-publica-1g.tjmg.jus.br/eproc/externo_controlador.php"
                 "?acao=processo_consulta_publica",
    ("tjmg", 2): "https://eproc-consulta-publica-2g.tjmg.jus.br/eproc/externo_controlador.php"
                 "?acao=processo_consulta_publica",
}
"""Tribunais cujo host final difere de ``eproc{grau}g.{tribunal}.jus.br``."""


def url_consulta(tribunal: str, grau: int = 1) -> str:
    """URL da página de consulta pública do Eproc do tribunal."""
    return URLS.get((tribunal, int(grau))) or (
        f"https://eproc{grau}g.{tribunal}.jus.br/eproc/externo_controlador.php"
        "?acao=processo_consulta_publica"
    )


class Sobrecarga(Exception):
    """O tribunal pediu para diminuir o ritmo (403, 429, 5xx ou aviso na página)."""


def _arq_ritmo(host: str) -> str:
    return f"/tmp/juscraper_ritmo_{re.sub(r'[^a-z0-9.]', '_', host.lower())}"


def _sua_vez(host: str, espera: float) -> None:
    """Uma requisição por vez em cada host, com ``espera`` segundos entre elas.

    A trava é um arquivo por host em ``/tmp``, então vale entre processos.
    """
    with open(_arq_ritmo(host), "a+", encoding="ascii") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        ultimo = float(fh.read().strip() or 0)
        falta = ultimo + espera - time.time()
        if falta > 0:
            time.sleep(falta)
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def _marca_fim(host: str) -> None:
    """Conta a espera a partir do fim da resposta."""
    with open(_arq_ritmo(host), "a+", encoding="ascii") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


class SessaoEspacada(requests.Session):
    """``requests.Session`` que espera ``espera`` segundos entre requisições ao mesmo host."""

    espera: float = ESPERA

    def __init__(self, espera: float | None = None) -> None:
        super().__init__()
        self.headers["User-Agent"] = USER_AGENT
        if espera is not None:
            self.espera = espera

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        host = urlparse(url).netloc
        _sua_vez(host, self.espera)
        kwargs.setdefault("timeout", 60)
        try:
            resp = super().request(method, url, *args, **kwargs)
        finally:
            _marca_fim(host)
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            raise Sobrecarga(f"HTTP {resp.status_code} em {host}")
        if SOBRECARGA.search(resp.text[:20000]):
            raise Sobrecarga(f"aviso de excesso de consultas em {host}")
        return resp


def estado(html: str) -> str:
    """Classifica a resposta do POST: ``recusa``, ``capa``, ``nada``, ``lista`` ou ``outro``.

    Só ``capa``, ``nada`` e ``lista`` provam que o formulário foi processado.
    """
    if RECUSA.search(html):
        return "recusa"
    if "Capa do Processo" in html:
        return "capa"
    if NADA in html:
        return "nada"
    if "Lista de Partes" in html or "acao=processo_seleciona_publica" in html:
        return "lista"
    return "outro"


def build_search_payload(form_html: str, numero: str, captcha_texto: str) -> dict[str, str]:
    """Campos do formulário: os ocultos e de texto da página, o número e o captcha."""
    form = BeautifulSoup(form_html, "html.parser").find("form")
    dados = {x.get("name"): x.get("value") or "" for x in form.find_all("input")
             if x.get("name") and x.get("type") in ("hidden", "text")}
    dados.update(txtNumProcesso=re.sub(r"\D", "", numero), txtInfraCaptcha=captcha_texto,
                 sbmNovo="Consultar")
    return dados


def extrai_captcha(form_html: str) -> bytes | None:
    """PNG do captcha embutido na página, ou ``None`` se não houver captcha de imagem."""
    so = BeautifulSoup(form_html, "html.parser")
    img = next((i for i in so.find_all("img") if i.get("src", "").startswith("data:image")
                and i.get("id") != "infraImgAudioCaptcha"), None)
    return base64.b64decode(img["src"].split(",", 1)[1]) if img else None


def consulta_numero(
    tribunal: str,
    numero: str,
    *,
    grau: int = 1,
    tentativas: int = TENTATIVAS,
    contagem: dict | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> dict | None:
    """Consulta um número e devolve a resposta processada, ou ``None`` se o OCR esgotar.

    O retorno tem ``estado`` (``capa``, ``nada`` ou ``lista``), ``texto``,
    ``url`` e ``sessao``. ``contagem`` soma captchas pedidos, enviados e
    aceitos.
    """
    url = url_consulta(tribunal, grau)
    c = contagem if contagem is not None else {}
    recusas = 0
    for n in range(tentativas):
        s = SessaoEspacada(espera=ESPERA * 2 ** min(recusas, 4)) if recusas else SessaoEspacada()
        try:
            pagina = s.get(url)
        except Sobrecarga as exc:
            recusas += 1
            log(f"eproc: tentativa {n + 1}: {exc}")
            if recusas > 4:
                raise
            continue
        recusas = 0
        c["captchas"] = c.get("captchas", 0) + 1
        png = extrai_captcha(pagina.text)
        if png is None:
            log(f"eproc: página sem captcha de imagem (HTTP {pagina.status_code})")
            return None
        texto, lido, conf = ocr.resolver(png, "eproc")
        if texto is None:
            log(f"eproc: tentativa {n + 1}: palpite {lido!r} ({conf}) abaixo do limiar")
            continue
        form = BeautifulSoup(pagina.text, "html.parser").find("form")
        resp = s.post(urljoin(pagina.url, form.get("action")),
                      data=build_search_payload(pagina.text, numero, texto))
        c["enviados"] = c.get("enviados", 0) + 1
        e = estado(resp.text)
        if e == "recusa":
            ocr.registrar("eproc", png, texto, conf, False)
            continue
        if e == "outro":
            log(f"eproc: tentativa {n + 1}: resposta inesperada (HTTP {resp.status_code}, {resp.url[:120]})")
            continue
        ocr.registrar("eproc", png, texto, conf, True)
        c["aceitos"] = c.get("aceitos", 0) + 1
        return {"estado": e, "texto": resp.text, "url": resp.url, "sessao": s}
    return None
