"""Leitura da capa do processo na consulta pública do Eproc.

Devolve o texto literal da página, sem interpretação. O documento da parte vem
mascarado pelo próprio Eproc (CPF "754.**********", CNPJ "60.7**********").
A capa traz os 10 eventos mais recentes; a lista completa está no link
``link_todos_eventos``.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

from bs4 import BeautifulSoup, Tag


class Parte(TypedDict):
    """Parte da tabela "Partes e Representantes", com os advogados dela."""

    polo: str | None
    nome: str
    documento: str | None
    advogados: list[str]


def limpo(s: str | None) -> str:
    """Texto com os espaços em sequência reduzidos a um e sem espaço nas pontas; ``None`` vira ``""``."""
    return re.sub(r"\s+", " ", s or "").strip()


DATA_HORA = re.compile(r"(\d{2})/(\d{2})/(\d{4})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?")


def data_iso(s: str | None) -> str | None:
    """'12/05/2026 16:20:09' -> '2026-05-12T16:20:09'. None se não houver data."""
    m = DATA_HORA.search(s or "")
    if not m:
        return None
    d, mo, a, h, mi, se = m.groups()
    return f"{a}-{mo}-{d}T{h or '00'}:{mi or '00'}:{se or '00'}"


def valor_br(s: str | None) -> float | None:
    """'1.234.567,89' -> 1234567.89. None se não for número."""
    m = re.search(r"-?\d[\d.]*,\d{2}|-?\d+", s or "")
    if not m:
        return None
    return float(m.group(0).replace(".", "").replace(",", "."))


def _parte(linha: str, polo: str | None) -> Parte:
    """Parte de uma linha "- NOME (documento mascarado)"."""
    texto = linha.lstrip("- ").strip()
    m = re.match(r"(.*?)\s*\(([^()]*)\)\s*$", texto)
    nome, doc = (m.group(1), m.group(2)) if m else (texto, None)
    return {"polo": polo, "nome": nome, "documento": doc, "advogados": []}


def _partes_da_celula(td: Tag, polo: str | None) -> list[Parte]:
    """Partes de uma célula do polo, cada uma seguida dos advogados dela."""
    out: list[Parte] = []
    atual: Parte | None = None
    for ln in (limpo(x) for x in td.get_text("\n").split("\n")):
        if not ln:
            continue
        if ln.startswith("-"):
            atual = _parte(ln, polo)
            out.append(atual)
        elif atual is not None:
            atual["advogados"].append(ln)
    return out


def _partes(tabela: Tag | None) -> list[Parte]:
    """Partes e advogados da tabela "Partes e Representantes".

    Cada coluna é um polo. Na célula, a linha que começa com "-" é uma parte,
    "NOME (documento mascarado)". As linhas seguintes, até a próxima parte,
    são os advogados dela, "NOME OAB".
    """
    if tabela is None:
        return []
    polos = [limpo(th.get_text(" ")) for th in tabela.find_all("th")]
    out: list[Parte] = []
    for tr in tabela.find_all("tr"):
        for i, td in enumerate(tr.find_all("td")):
            out.extend(_partes_da_celula(td, polos[i] if i < len(polos) else None))
    return out


def _tabela(s: BeautifulSoup, titulo: str) -> Tag | None:
    for t in s.find_all("table"):
        if (t.get("summary") or "").strip().lower() == titulo.lower():
            return t
    return None


def _evento(tr: Tag) -> dict[str, Any] | None:
    """Evento de uma linha da tabela de eventos; ``None`` se a linha não for de evento."""
    td = [limpo(x.get_text(" ")) for x in tr.find_all("td")]
    if len(td) >= 3 and td[0].isdigit():
        return {"evento": int(td[0]), "data": data_iso(td[1]), "descricao": td[2],
                "documentos": td[4] if len(td) > 4 else None}
    return None


def _eventos(s: BeautifulSoup) -> list[dict[str, Any]]:
    for t in s.find_all("table"):
        th = [limpo(x.get_text(" ")) for x in t.find_all("th")]
        if th[:2] == ["Evento", "Data/Hora"]:
            return [e for e in map(_evento, t.find_all("tr")) if e is not None]
    return []


def _texto(s: BeautifulSoup, id_: str) -> str | None:
    e = s.find(id=id_)
    return limpo(e.get_text(" ")) if e else None


def _info(s: BeautifulSoup) -> dict[str, str | None]:
    """Pares rótulo e valor do quadro "Informações Adicionais"."""
    info: dict[str, str | None] = {}
    fld = s.find(id="fldInformacoesAdicionais")
    if fld:
        for st in fld.find_all("strong"):
            k = limpo(st.get_text(" ")).rstrip(":")
            v = st.find_next_sibling()
            info[k] = limpo(v.get_text(" ")) if v else None
    return info


def _assuntos(s: BeautifulSoup) -> list[str]:
    """Assuntos da tabela "Assuntos", com o principal marcado."""
    t = _tabela(s, "Assuntos")
    if not t:
        return []
    assuntos = []
    for tr in t.find_all("tr"):
        td = [limpo(x.get_text(" ")) for x in tr.find_all("td")]
        if len(td) >= 2:
            assuntos.append(td[1] + (" (principal)" if len(td) > 2 and td[2].lower() == "sim" else ""))
    return assuntos


def ler_capa(html: str) -> dict[str, Any]:
    """Campos da capa do processo. Texto literal da página, sem interpretação."""
    s = BeautifulSoup(html, "html.parser")
    info = _info(s)
    assuntos = _assuntos(s)
    link = next((a["href"] for a in s.find_all("a", href=True) if "eventos=true" in a["href"]), None)
    return {
        "numero": re.sub(r"\D", "", _texto(s, "txtNumProcesso") or ""),
        "data_autuacao": data_iso(_texto(s, "txtAutuacao")),
        "situacao": _texto(s, "txtSituacao"),
        "orgao_julgador": _texto(s, "txtOrgaoJulgador"),
        "juiz": _texto(s, "txtMagistrado"),
        "classe": _texto(s, "txtClasse"),
        "assuntos": assuntos,
        "valor_causa": valor_br(info.get("Valor da Causa")),
        "partes": _partes(_tabela(s, "Partes e Representantes")),
        "info": info,
        "eventos": _eventos(s),
        "link_todos_eventos": link,
    }


def ler_eventos(html: str) -> list[dict[str, Any]]:
    """Eventos da página com a lista completa (``link_todos_eventos`` da capa)."""
    return _eventos(BeautifulSoup(html, "html.parser"))
