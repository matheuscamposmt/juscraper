"""
Functions for downloading process data from TJRJ via PJe public consultation.
"""
import re
import requests
from bs4 import BeautifulSoup

PJE_1G_URL = "https://tjrj.pje.jus.br/1g/ConsultaPublica/listView.seam"
PJE_2G_URL = "https://tjrj.pje.jus.br/2g/ConsultaPublica/listView.seam"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

HEADERS_AJAX = {
    **HEADERS,
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Faces-Request": "partial/ajax",
    "X-Requested-With": "XMLHttpRequest",
}


def _extract_viewstate(html: str) -> str:
    """Extracts javax.faces.ViewState from a JSF page."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("input", {"name": "javax.faces.ViewState"})
    if not tag:
        raise RuntimeError("javax.faces.ViewState não encontrado na página PJe.")
    return tag["value"]


def _extract_process_id(html: str) -> str | None:
    """Extracts the internal process ID from the search results list."""
    soup = BeautifulSoup(html, "html.parser")
    # PJe renders process links with id_processo query parameter
    for a in soup.find_all("a", href=True):
        m = re.search(r"id_processo=(\d+)", a["href"])
        if m:
            return m.group(1)
    # Fallback: look for onclick with process id
    for tag in soup.find_all(attrs={"onclick": True}):
        m = re.search(r"id_processo[=,](\d+)", tag["onclick"])
        if m:
            return m.group(1)
    return None


def cpopg_download(session: requests.Session, numero_processo: str) -> dict:
    """
    Downloads raw process data from the TJRJ PJe 1st-degree public consultation.

    Performs the multi-step JSF/Seam form flow:
      1. GET the search page → capture ViewState + cookies
      2. POST process number → get results list
      3. POST to open process details → get parties, case value, movements

    Args:
        session: A requests.Session (reused across calls for cookie persistence).
        numero_processo: CNJ-formatted process number.

    Returns:
        dict with keys 'lista_html' (search results) and 'detalhes_html'
        (process details page), both as raw HTML strings.
    """
    # Step 1: initial GET to capture session cookies and ViewState
    resp = session.get(PJE_1G_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    viewstate = _extract_viewstate(resp.text)

    # Step 2: POST to search by process number (Seam AJAX partial render)
    search_data = {
        "AJAXREQUEST": "_viewRoot",
        "consulta:numProcesso": numero_processo,
        "consulta:consultar": "consulta:consultar",
        "consulta": "consulta",
        "javax.faces.ViewState": viewstate,
    }
    resp = session.post(PJE_1G_URL, data=search_data, headers=HEADERS_AJAX, timeout=15)
    resp.raise_for_status()
    lista_html = resp.text
    viewstate = _extract_viewstate(lista_html)

    # Step 3: follow the first result to get process detail page
    process_id = _extract_process_id(lista_html)
    detalhes_html = None
    if process_id:
        detail_url = (
            f"https://tjrj.pje.jus.br/1g/ConsultaPublica/detalhesProcesso.seam"
            f"?id_processo={process_id}"
        )
        resp = session.get(detail_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        detalhes_html = resp.text

    return {"lista_html": lista_html, "detalhes_html": detalhes_html}


def cposg_download(session: requests.Session, numero_processo: str) -> dict:
    """
    Downloads raw process data from the TJRJ PJe 2nd-degree public consultation.
    Same flow as cpopg_download but targets the 2g instance.
    """
    resp = session.get(PJE_2G_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    viewstate = _extract_viewstate(resp.text)

    search_data = {
        "AJAXREQUEST": "_viewRoot",
        "consulta:numProcesso": numero_processo,
        "consulta:consultar": "consulta:consultar",
        "consulta": "consulta",
        "javax.faces.ViewState": viewstate,
    }
    resp = session.post(PJE_2G_URL, data=search_data, headers=HEADERS_AJAX, timeout=15)
    resp.raise_for_status()
    lista_html = resp.text
    viewstate = _extract_viewstate(lista_html)

    process_id = _extract_process_id(lista_html)
    detalhes_html = None
    if process_id:
        detail_url = (
            f"https://tjrj.pje.jus.br/2g/ConsultaPublica/detalhesProcesso.seam"
            f"?id_processo={process_id}"
        )
        resp = session.get(detail_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        detalhes_html = resp.text

    return {"lista_html": lista_html, "detalhes_html": detalhes_html}
