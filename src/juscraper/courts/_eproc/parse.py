"""Leitura da capa do processo na consulta pública do Eproc.

Devolve o texto literal da página, sem interpretação. O documento da parte vem
mascarado pelo próprio Eproc (CPF "754.**********", CNPJ "60.7**********").
A capa traz os 10 eventos mais recentes; a lista completa está no link
``link_todos_eventos``.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup


def limpo(s):
    return re.sub(r"\s+", " ", s or "").strip()


DATA_HORA = re.compile(r"(\d{2})/(\d{2})/(\d{4})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?")


def data_iso(s):
    """'12/05/2026 16:20:09' -> '2026-05-12T16:20:09'. None se não houver data."""
    m = DATA_HORA.search(s or "")
    if not m:
        return None
    d, mo, a, h, mi, se = m.groups()
    return f"{a}-{mo}-{d}T{h or '00'}:{mi or '00'}:{se or '00'}"


def valor_br(s):
    """'1.234.567,89' -> 1234567.89. None se não for número."""
    m = re.search(r"-?\d[\d.]*,\d{2}|-?\d+", s or "")
    if not m:
        return None
    return float(m.group(0).replace(".", "").replace(",", "."))


def _partes(tabela):
    """Partes e advogados da tabela "Partes e Representantes".

    Cada coluna é um polo. Na célula, a linha que começa com "-" é uma parte,
    "NOME (documento mascarado)". As linhas seguintes, até a próxima parte,
    são os advogados dela, "NOME OAB".
    """
    if tabela is None:
        return []
    polos = [limpo(th.get_text(" ")) for th in tabela.find_all("th")]
    out = []
    for tr in tabela.find_all("tr"):
        for i, td in enumerate(tr.find_all("td")):
            polo = polos[i] if i < len(polos) else None
            linhas = [limpo(x) for x in td.get_text("\n").split("\n")]
            atual = None
            for ln in linhas:
                if not ln:
                    continue
                if ln.startswith("-"):
                    ln = ln.lstrip("- ").strip()
                    m = re.match(r"(.*?)\s*\(([^()]*)\)\s*$", ln)
                    nome, doc = (m.group(1), m.group(2)) if m else (ln, None)
                    atual = {"polo": polo, "nome": nome, "documento": doc, "advogados": []}
                    out.append(atual)
                elif atual is not None:
                    atual["advogados"].append(ln)
    return out


def _tabela(s, titulo):
    for t in s.find_all("table"):
        if (t.get("summary") or "").strip().lower() == titulo.lower():
            return t
    return None


def _eventos(s):
    for t in s.find_all("table"):
        th = [limpo(x.get_text(" ")) for x in t.find_all("th")]
        if th[:2] == ["Evento", "Data/Hora"]:
            out = []
            for tr in t.find_all("tr"):
                td = [limpo(x.get_text(" ")) for x in tr.find_all("td")]
                if len(td) >= 3 and td[0].isdigit():
                    out.append({"evento": int(td[0]), "data": data_iso(td[1]), "descricao": td[2],
                                "documentos": td[4] if len(td) > 4 else None})
            return out
    return []


def ler_capa(html):
    """Campos da capa do processo. Texto literal da página, sem interpretação."""
    s = BeautifulSoup(html, "html.parser")

    def txt(id_):
        e = s.find(id=id_)
        return limpo(e.get_text(" ")) if e else None

    info = {}
    fld = s.find(id="fldInformacoesAdicionais")
    if fld:
        for st in fld.find_all("strong"):
            k = limpo(st.get_text(" ")).rstrip(":")
            v = st.find_next_sibling()
            info[k] = limpo(v.get_text(" ")) if v else None
    assuntos = []
    t = _tabela(s, "Assuntos")
    if t:
        for tr in t.find_all("tr"):
            td = [limpo(x.get_text(" ")) for x in tr.find_all("td")]
            if len(td) >= 2:
                assuntos.append(td[1] + (" (principal)" if len(td) > 2 and td[2].lower() == "sim" else ""))
    link = next((a["href"] for a in s.find_all("a", href=True) if "eventos=true" in a["href"]), None)
    return {
        "numero": re.sub(r"\D", "", txt("txtNumProcesso") or ""),
        "data_autuacao": data_iso(txt("txtAutuacao")),
        "situacao": txt("txtSituacao"),
        "orgao_julgador": txt("txtOrgaoJulgador"),
        "juiz": txt("txtMagistrado"),
        "classe": txt("txtClasse"),
        "assuntos": assuntos,
        "valor_causa": valor_br(info.get("Valor da Causa")),
        "partes": _partes(_tabela(s, "Partes e Representantes")),
        "info": info,
        "eventos": _eventos(s),
        "link_todos_eventos": link,
    }


def ler_eventos(html):
    return _eventos(BeautifulSoup(html, "html.parser"))
