"""
Scraper for the Court of Justice of Rio de Janeiro (TJRJ) via PJe.

Public consultation endpoints:
  1g: https://tjrj.pje.jus.br/1g/ConsultaPublica/listView.seam
  2g: https://tjrj.pje.jus.br/2g/ConsultaPublica/listView.seam

Priority data extracted:
  1. partes       — parties (polo ativo / passivo / terceiros + advogados)
  2. valor_causa  — monetary value of the case
  3. movimentacoes — docket entries (date + description)
"""
from typing import Union, List

import pandas as pd
import requests

from juscraper.core.base import BaseScraper
from .download import cpopg_download, cposg_download
from .parse import cpopg_parse, cposg_parse


class TJRJScraper(BaseScraper):
    """Scraper for the Court of Justice of Rio de Janeiro (TJRJ) via PJe."""

    PJE_1G_URL = "https://tjrj.pje.jus.br/1g/ConsultaPublica/listView.seam"
    PJE_2G_URL = "https://tjrj.pje.jus.br/2g/ConsultaPublica/listView.seam"

    def __init__(self):
        super().__init__("TJRJ")
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def cpopg_download(self, id_cnj: Union[str, List[str]]) -> list:
        """
        Downloads raw HTML for one or more 1st-degree processes from TJRJ PJe.

        Args:
            id_cnj: CNJ process number or list of numbers.

        Returns:
            List of raw dicts (one per process) with 'lista_html' and
            'detalhes_html' keys.
        """
        if isinstance(id_cnj, str):
            id_cnj = [id_cnj]
        return [cpopg_download(self.session, num) for num in id_cnj]

    def cposg_download(self, id_cnj: Union[str, List[str]]) -> list:
        """
        Downloads raw HTML for one or more 2nd-degree processes from TJRJ PJe.
        """
        if isinstance(id_cnj, str):
            id_cnj = [id_cnj]
        return [cposg_download(self.session, num) for num in id_cnj]

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def cpopg_parse(self, resultados_brutos: list) -> list:
        """
        Parses the raw output of cpopg_download.

        Returns a list of dicts, each with:
          - partes        (list[dict])
          - valor_causa   (str | None)
          - movimentacoes (list[dict])
        """
        return [cpopg_parse(r) for r in resultados_brutos]

    def cposg_parse(self, resultados_brutos: list) -> list:
        """Parses the raw output of cposg_download."""
        return [cposg_parse(r) for r in resultados_brutos]

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def cpopg(self, id_cnj: Union[str, List[str]]) -> pd.DataFrame:
        """
        Fetches and parses 1st-degree process data from TJRJ PJe.

        Returns a DataFrame where each row is one process, with columns:
          numero_processo, partes, valor_causa, movimentacoes
        """
        if isinstance(id_cnj, str):
            numeros = [id_cnj]
        else:
            numeros = list(id_cnj)

        brutos = self.cpopg_download(numeros)
        parsed = self.cpopg_parse(brutos)

        rows = []
        for numero, dados in zip(numeros, parsed):
            rows.append({
                "numero_processo": numero,
                "partes": dados["partes"],
                "valor_causa": dados["valor_causa"],
                "movimentacoes": dados["movimentacoes"],
            })
        return pd.DataFrame(rows)

    def cposg(self, id_cnj: Union[str, List[str]]) -> pd.DataFrame:
        """
        Fetches and parses 2nd-degree process data from TJRJ PJe.
        Same structure as cpopg.
        """
        if isinstance(id_cnj, str):
            numeros = [id_cnj]
        else:
            numeros = list(id_cnj)

        brutos = self.cposg_download(numeros)
        parsed = self.cposg_parse(brutos)

        rows = []
        for numero, dados in zip(numeros, parsed):
            rows.append({
                "numero_processo": numero,
                "partes": dados["partes"],
                "valor_causa": dados["valor_causa"],
                "movimentacoes": dados["movimentacoes"],
            })
        return pd.DataFrame(rows)

    def cjsg(self, *args, **kwargs):
        """Jurisprudence search not implemented for TJRJ in this scraper."""
        raise NotImplementedError(
            "Consulta de jurisprudência não implementada para TJRJ. "
            "Use jus.scraper('datajud') para acessar acórdãos do TJRJ."
        )
