"""Camada HTTP da consulta pública do Eproc.

Uma consulta por número custa duas requisições, mais as repetições do captcha:

1. ``GET`` da página de consulta, que traz o captcha embutido como
   ``data:image/png;base64``. O captcha vale para um envio só e fica preso ao
   ``PHPSESSID`` da sessão. Medido em 09/11/2026: o mesmo captcha, ou nenhum,
   na segunda consulta da sessão volta "Código da imagem (captcha) incorreto".
2. ``POST`` do formulário no host FINAL do GET. O TJMG redireciona
   ``eproc1g.tjmg`` para ``eproc-consulta-publica-1g.tjmg``; o POST ao host
   antigo vira GET no redirecionamento e o formulário se perde.

Rate limit: :class:`juscraper.core.ritmo.SessaoRitmada`. Uma requisição por
vez por host, com intervalo adaptativo: dobra a cada recusa (403, 429, 5xx,
aviso de excesso) e encolhe a cada resposta boa. :class:`Sobrecarga` sobe
para quem chama depois de ``RECUSAS_SEGUIDAS`` recusas.

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
import re
from collections.abc import Callable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from juscraper.core.ritmo import SessaoRitmada, Sobrecarga

from . import captcha as ocr

__all__ = ["SessaoRitmada", "Sobrecarga", "consulta_numero", "estado", "url_consulta"]

NADA = "Nenhum registro encontrado"
RECUSA = re.compile(r"captcha\)? incorreto", re.I)
TENTATIVAS = 12
RECUSAS_SEGUIDAS = 6
"""Recusas por volume seguidas antes de a Sobrecarga subir para quem chama."""

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
    for n in range(TENTATIVAS):
        # Sessão nova a cada captcha: o captcha fica preso ao PHPSESSID. O
        # intervalo do host é do ritmo e sobrevive à troca de sessão.
        s = SessaoRitmada()
        try:
            pagina = s.get(url)
        except Sobrecarga as exc:
            recusas += 1
            log(f"eproc: tentativa {n + 1}: {exc}")
            if recusas >= RECUSAS_SEGUIDAS:
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
        try:
            resp = s.post(urljoin(pagina.url, form.get("action")),
                          data=build_search_payload(pagina.text, numero, texto))
        except Sobrecarga as exc:
            recusas += 1
            log(f"eproc: tentativa {n + 1}: {exc} no envio")
            if recusas >= RECUSAS_SEGUIDAS:
                raise
            continue
        c["enviados"] = c.get("enviados", 0) + 1
        e = estado(resp.text)
        if e == "recusa":
            ocr.registrar("eproc", png, texto, conf, aceito=False)
            continue
        if e == "outro":
            log(f"eproc: tentativa {n + 1}: resposta inesperada (HTTP {resp.status_code}, {resp.url[:120]})")
            continue
        ocr.registrar("eproc", png, texto, conf, aceito=True)
        c["aceitos"] = c.get("aceitos", 0) + 1
        return {"estado": e, "texto": resp.text, "url": resp.url, "sessao": s}
    return None
