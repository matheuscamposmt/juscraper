"""
Functions for parsing process data from TJRJ PJe HTML pages.

Priority fields extracted:
  1. partes       — polo ativo, polo passivo, advogados
  2. valor_causa  — monetary value of the case
  3. movimentacoes — docket entries (date + description)
"""
import re
from bs4 import BeautifulSoup


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ---------------------------------------------------------------------------
# Partes
# ---------------------------------------------------------------------------

def _parse_partes(soup: BeautifulSoup) -> list[dict]:
    """
    Extracts parties from the PJe details page.

    PJe renders parties inside a <div id="partes"> (or similar) with rows
    that contain the pole (ativo/passivo), party name, and lawyers.
    Returns a list of dicts: {polo, nome, tipo, advogados}.
    """
    partes = []

    # Primary selector: table rows inside the partes panel
    partes_section = (
        soup.find("div", id=re.compile(r"partes", re.I))
        or soup.find("fieldset", string=re.compile(r"partes", re.I))
        or soup.find("div", class_=re.compile(r"partes", re.I))
    )

    if partes_section:
        rows = partes_section.find_all("tr")
    else:
        # Fallback: look for any table row mentioning polo ativo/passivo
        rows = soup.find_all("tr", string=re.compile(r"polo|ativo|passivo", re.I))

    polo_atual = None
    for row in rows:
        cells = [_clean(td.get_text()) for td in row.find_all(["td", "th"])]
        if not cells:
            continue

        # Detect polo header rows
        joined = " ".join(cells).lower()
        if "polo ativo" in joined:
            polo_atual = "ativo"
            continue
        if "polo passivo" in joined:
            polo_atual = "passivo"
            continue
        if "terceiro" in joined or "interveniente" in joined:
            polo_atual = "terceiro"
            continue

        # A data row typically has at least 2 non-empty cells
        non_empty = [c for c in cells if c]
        if len(non_empty) < 1:
            continue

        # Attempt to separate name from OAB/adv info
        nome = non_empty[0] if non_empty else ""
        advogados = [c for c in non_empty[1:] if re.search(r"OAB|adv", c, re.I)]

        if nome and polo_atual:
            partes.append({
                "polo": polo_atual,
                "nome": nome,
                "advogados": advogados,
            })

    return partes


# ---------------------------------------------------------------------------
# Valor da causa
# ---------------------------------------------------------------------------

def _parse_valor_causa(soup: BeautifulSoup) -> str | None:
    """
    Extracts the case value (valor da causa) from the PJe details page.

    PJe typically shows it in a labeled field like:
      <label>Valor da causa:</label> <span>R$ 1.000,00</span>
    or inside a table cell adjacent to that label.
    """
    # Look for a label/span/td that contains "valor da causa"
    for tag in soup.find_all(string=re.compile(r"valor da causa", re.I)):
        parent = tag.parent
        # Try next sibling text
        sibling = parent.find_next_sibling()
        if sibling:
            text = _clean(sibling.get_text())
            if text:
                return text
        # Try the cell to the right in a table
        td = parent.find_parent("td")
        if td:
            next_td = td.find_next_sibling("td")
            if next_td:
                text = _clean(next_td.get_text())
                if text:
                    return text

    # Fallback: regex for a Brazilian currency pattern near "valor"
    full_text = soup.get_text(" ")
    m = re.search(
        r"valor da causa[^R\n]{0,30}R\$\s*([\d.,]+)",
        full_text,
        re.I,
    )
    if m:
        return "R$ " + m.group(1)

    return None


# ---------------------------------------------------------------------------
# Movimentações
# ---------------------------------------------------------------------------

def _parse_movimentacoes(soup: BeautifulSoup) -> list[dict]:
    """
    Extracts docket movements from the PJe details page.

    PJe renders movements in a table with columns: data, movimento (description).
    Returns a list of dicts: {data, titulo, descricao}.
    """
    movimentacoes = []

    mov_section = (
        soup.find("div", id=re.compile(r"moviment", re.I))
        or soup.find("fieldset", string=re.compile(r"moviment", re.I))
        or soup.find("div", class_=re.compile(r"moviment", re.I))
    )

    if mov_section:
        rows = mov_section.find_all("tr")
    else:
        rows = []

    # Skip header row
    for row in rows[1:]:
        cells = [_clean(td.get_text()) for td in row.find_all("td")]
        if len(cells) < 2:
            continue
        data = cells[0] if cells else ""
        titulo = cells[1] if len(cells) > 1 else ""
        descricao = cells[2] if len(cells) > 2 else ""

        if data or titulo:
            movimentacoes.append({
                "data": data,
                "titulo": titulo,
                "descricao": descricao,
            })

    return movimentacoes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cpopg_parse(raw: dict) -> dict:
    """
    Parses the raw HTML output from cpopg_download / cposg_download.

    Args:
        raw: dict with 'lista_html' and 'detalhes_html' keys.

    Returns:
        dict with keys:
          - partes        (list[dict])
          - valor_causa   (str | None)
          - movimentacoes (list[dict])
          - _raw          (original raw dict, for debugging)
    """
    detalhes_html = raw.get("detalhes_html") or raw.get("lista_html", "")
    soup = BeautifulSoup(detalhes_html, "html.parser")

    return {
        "partes": _parse_partes(soup),
        "valor_causa": _parse_valor_causa(soup),
        "movimentacoes": _parse_movimentacoes(soup),
        "_raw": raw,
    }


# Alias for cposg (same parsing logic)
cposg_parse = cpopg_parse
