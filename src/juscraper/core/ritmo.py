"""Ritmo adaptativo por host, com backoff exponencial, valendo entre processos.

Cada tribunal tem o próprio limite de requisições, e nenhum o publica. Um
intervalo fixo é lento demais para o tribunal que aguenta mais e rápido demais
para o que aguenta menos. O ritmo descobre o limite pelo comportamento do
servidor:

* uma requisição por vez por host, garantida por uma trava em arquivo que
  vale entre processos da mesma máquina;
* a cada resposta boa, o intervalo encolhe ``ALIVIO`` (10%), até o ``piso``;
* a cada recusa (HTTP 403, 429 ou 5xx, aviso de excesso na página, erro de
  rede), o intervalo dobra, até o ``TETO``. Um ``Retry-After`` numérico vira o
  intervalo mínimo;
* o intervalo conta a partir do fim da resposta anterior.

O estado de cada host (fim da última resposta e intervalo atual) fica num
arquivo em ``JUSCRAPER_RITMO_DIR`` (padrão: o diretório temporário do sistema). Ler o arquivo mostra o
intervalo que o tribunal está aceitando agora.

Com ``JUSCRAPER_RITMO_LOG`` apontando para um arquivo, cada processo grava ali,
a cada ``JUSCRAPER_RITMO_JANELA`` segundos (padrão 300) e por host, uma linha
JSON com o número de requisições, as recusas, a latência mediana e a do
percentil 90 e o intervalo em uso. Serve para medir o efeito de mudar o piso.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PISO = float(os.environ.get("JUSCRAPER_RITMO_PISO", "0.5"))
"""Menor intervalo, em segundos, entre duas requisições ao mesmo host."""
TETO = 900.0
"""Maior intervalo, em segundos, depois de recusas seguidas."""
ALIVIO = 0.9
"""Fator aplicado ao intervalo a cada resposta boa."""
RECUSA_MINIMA = 2.0
"""Intervalo mínimo depois de uma recusa, em segundos."""

AVISO_DE_EXCESSO = re.compile(
    r"consultas simult|too many requests|excesso de (acessos|requisi)", re.I
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


PARAR = threading.Event()
"""Quem usa o ritmo seta no sinal de parada: a espera da vez acorda na hora e
levanta ``Interrompido``. Sem isso, um intervalo alto (até ``TETO``) segurava o
processo dormindo e o gerenciador de serviços o matava sem gravar nada."""


class Interrompido(Exception):
    """Parada pedida (``PARAR``) enquanto se esperava a vez do host."""


class Sobrecarga(Exception):
    """O servidor recusou pelo volume (403, 429, 5xx, aviso na página ou erro de rede)."""


def adapta(intervalo: float, *, ok: bool, piso: float = PISO, retry_after: float | None = None) -> float:
    """Novo intervalo depois de uma resposta: encolhe se ``ok``, dobra se recusada."""
    if ok:
        return max(piso, intervalo * ALIVIO)
    return min(TETO, max(piso, intervalo * 2, RECUSA_MINIMA, retry_after or 0.0))


def retry_after(header: str | None) -> float | None:
    """``Retry-After`` em segundos numéricos; ``None`` se ausente ou em outro formato."""
    try:
        return max(0.0, float(header)) if header is not None else None
    except (TypeError, ValueError):
        return None


def _arquivo(host: str) -> Path:
    pasta = Path(os.environ.get("JUSCRAPER_RITMO_DIR") or tempfile.gettempdir())
    return pasta / f"juscraper_ritmo_{re.sub(r'[^a-z0-9.]', '_', host.lower())}"


def estado(host: str, piso: float = PISO) -> tuple[float, float]:
    """(fim da última resposta, intervalo atual) do host."""
    try:
        with _arquivo(host).open(encoding="ascii") as fh:
            ultimo, intervalo = (float(x) for x in fh.read().split())
            return ultimo, max(piso, intervalo)
    except (OSError, ValueError):
        return 0.0, piso


@contextmanager
def vez(host: str, piso: float = PISO) -> Iterator[dict]:
    """Espera a vez do host, segura a trava durante a requisição e adapta o intervalo.

    Quem usa preenche o dicionário entregue: ``ok`` (bool) e, se houver,
    ``retry_after`` (segundos).
    """
    with _arquivo(host).open("a+", encoding="ascii") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            try:
                ultimo, intervalo = (float(x) for x in fh.read().split())
            except ValueError:
                ultimo, intervalo = 0.0, piso
            intervalo = max(piso, intervalo)
            falta = ultimo + intervalo - time.time()
            if falta > 0 and PARAR.wait(falta):
                raise Interrompido(f"parada pedida durante a espera de {host}")
            resultado: dict = {"ok": False, "retry_after": None, "intervalo": intervalo}
            try:
                yield resultado
            finally:
                novo = adapta(intervalo, ok=resultado["ok"], piso=piso, retry_after=resultado["retry_after"])
                fh.seek(0)
                fh.truncate()
                fh.write(f"{time.time()} {novo}")
                fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


_MEDIDAS: dict[str, dict] = {}
_TRAVA_MEDIDAS = threading.Lock()


def _registra(host: str, latencia: float, ok: bool, intervalo: float) -> None:
    """Acumula a medida do host e grava a linha da janela em ``JUSCRAPER_RITMO_LOG``, se definido."""
    arq = os.environ.get("JUSCRAPER_RITMO_LOG")
    if not arq:
        return
    janela = float(os.environ.get("JUSCRAPER_RITMO_JANELA", "300"))
    agora = time.time()
    with _TRAVA_MEDIDAS:
        m = _MEDIDAS.setdefault(host, {"inicio": agora, "lat": [], "recusas": 0})
        m["lat"].append(latencia)
        m["recusas"] += not ok
        if agora - m["inicio"] < janela:
            return
        lat = sorted(m["lat"])
        linha = {"t": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(agora)), "host": host,
                 "pid": os.getpid(), "segundos": round(agora - m["inicio"]), "n": len(lat),
                 "recusas": m["recusas"], "lat_mediana": round(lat[len(lat) // 2], 3),
                 "lat_p90": round(lat[min(len(lat) - 1, int(len(lat) * 0.9))], 3),
                 "intervalo": round(intervalo, 3)}
        del _MEDIDAS[host]
    try:
        with open(arq, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(linha) + "\n")
    except OSError:
        pass


JANELA_DO_AVISO = 200_000
"""Caracteres da página em que se procura o aviso de excesso. No e-SAJ do TJAL
(09/11/2026) o aviso vem no caractere 24.019, depois do cabeçalho do portal."""


def _tem_aviso(resp: requests.Response) -> bool:
    """Aviso de excesso numa resposta de texto (HTML, JSON); binário não é lido."""
    tipo = resp.headers.get("Content-Type", "").lower()
    if tipo and not any(t in tipo for t in ("text", "html", "json", "xml")):
        return False
    return bool(AVISO_DE_EXCESSO.search(resp.text[:JANELA_DO_AVISO]))


class SessaoRitmada(requests.Session):
    """``requests.Session`` que segue :func:`vez` em toda requisição.

    Levanta :class:`Sobrecarga` quando o servidor recusa pelo volume, depois de
    dobrar o intervalo do host. Quem chama decide se tenta de novo: a próxima
    requisição já espera o intervalo maior.
    """

    def __init__(self, piso: float = PISO) -> None:
        """Sessão com o ``User-Agent`` de navegador e o ``piso`` do intervalo, em segundos."""
        super().__init__()
        self.headers["User-Agent"] = USER_AGENT
        self.piso = piso

    def request(  # type: ignore[override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        """Faz a requisição na vez do host e adapta o intervalo pela resposta."""
        host = urlparse(url).netloc
        kwargs.setdefault("timeout", 60)
        with vez(host, self.piso) as r:
            inicio = time.monotonic()
            try:
                resp = super().request(method, url, *args, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                _registra(host, time.monotonic() - inicio, False, r["intervalo"])
                raise Sobrecarga(f"{type(exc).__name__} em {host}") from exc
            motivo = None
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                motivo = f"HTTP {resp.status_code} em {host}"
                r["retry_after"] = retry_after(resp.headers.get("Retry-After"))
            elif _tem_aviso(resp):
                motivo = f"aviso de excesso de consultas em {host}"
            r["ok"] = motivo is None
            _registra(host, time.monotonic() - inicio, r["ok"], r["intervalo"])
        if motivo:
            raise Sobrecarga(motivo)
        return resp
