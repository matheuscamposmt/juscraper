"""
Captcha de imagem por OCR local, no modelo do CapMonster.

O captcha e resolvido por OCR local, respeitando o rate limit do tribunal. Modelos de
linguagem (Claude, Codex) recusam transcrever captcha, entao o motor e o
ddddocr, um OCR local para captcha (onnx, sem GPU).

Tres ideias do CapMonster:

| CapMonster              | Aqui                                                      |
|-------------------------|-----------------------------------------------------------|
| modulo por tipo de site | PERFIS: alfabeto e tamanho de cada site                   |
| recognizingThreshold    | `limiar`: abaixo dele, o palpite nao e enviado (None)      |
| reportIncorrect         | `registrar`: acerto ou erro, com a imagem, por perfil     |

O palpite que nao passa no limiar nao vai ao site: pede-se um captcha novo.
Isso poupa requisicoes, que e o que o captcha protege.

Medido em 09/11/2026 no Eproc do TJPR, com o proprio site como gabarito
("Codigo da imagem (captcha) incorreto"): de 30 captchas, 10 passaram no
filtro e 8 foram aceitos. Sao 80% de acerto no que e enviado e 27% de
sucesso por captcha obtido. Cada acerto grava a imagem com o texto certo:
sao exemplos rotulados para treinar um modelo proprio depois.
"""
import hashlib
import json
import os
import string
import time
from pathlib import Path

ALNUM = string.ascii_letters + string.digits

PERFIS = {
    # Eproc (TJPR, TJSC, TJRS, TRF4...): 4 caracteres, letras e numeros, com linhas de ruido.
    "eproc": {"alfabeto": ALNUM, "tamanho": (4, 4), "limiar": 0.40},
    # Generico: sem saber o site, aceita de 3 a 8 caracteres.
    "generico": {"alfabeto": ALNUM, "tamanho": (3, 8), "limiar": 0.50},
}

_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        import ddddocr
        _OCR = ddddocr.DdddOcr(show_ad=False)
    return _OCR


def decodifica(probabilidades, charset, permitidos):
    """Decodificacao CTC restrita a um alfabeto.

    Em cada posicao escolhe o caractere permitido (ou o branco) mais provavel,
    junta repeticoes seguidas e tira os brancos. A confianca e a menor
    probabilidade entre os caracteres escolhidos.
    """
    idx = [i for i, c in enumerate(charset) if c == "" or c in permitidos]
    texto, anterior, conf = "", None, 1.0
    for linha in probabilidades:
        while linha and isinstance(linha[0], list):  # o ddddocr embrulha cada posicao: [[p0, p1, ...]]
            linha = linha[0]
        i = max(idx, key=lambda k: linha[k])
        if charset[i] and i != anterior:
            texto += charset[i]
            conf = min(conf, linha[i])
        anterior = i
    return texto, round(conf, 3)


def ler(imagem: bytes, perfil="generico"):
    """(texto, confianca) do OCR, restrito ao alfabeto do perfil."""
    p = PERFIS[perfil]
    r = _ocr().classification(imagem, probability=True)
    # ddddocr 1.6 devolve "charset"/"probabilities"; o 1.5.6, último para o
    # Python 3.9 da VM, devolve "charsets"/"probability".
    probs = r["probabilities"] if "probabilities" in r else r["probability"]
    charset = r["charset"] if "charset" in r else r["charsets"]
    return decodifica(probs, charset, set(p["alfabeto"]))


def resolver(imagem: bytes, perfil="generico"):
    """O texto, se o palpite passa no tamanho e no limiar do perfil; senao None.

    None quer dizer "nao envie, peca outro captcha", como o
    ERROR_CAPTCHA_UNSOLVABLE do CapMonster abaixo do recognizingThreshold.
    """
    texto, conf = ler(imagem, perfil)
    p = PERFIS[perfil]
    if not (p["tamanho"][0] <= len(texto) <= p["tamanho"][1]) or conf < p["limiar"]:
        return None, texto, conf
    return texto, texto, conf


# --- registro, como o reportIncorrect -----------------------------------------

def _dir(perfil) -> Path:
    base = Path(os.environ.get("JUSCRAPER_CAPTCHA_DIR") or Path.home() / ".cache" / "juscraper" / "captcha")
    d = base / perfil
    d.mkdir(parents=True, exist_ok=True)
    return d


_contagem: dict = {}
"""Arquivos em cada pasta de rotulados, contados no disco e somados a cada gravação."""
_gravados: dict = {}
"""Gravações deste processo desde a última contagem no disco, por pasta."""
RECONTA = 100
"""A cada tantas gravações, reconta a pasta: outros processos podem gravar nela."""


def _grava_rotulado(pasta: Path, h: str) -> bool:
    """Se o exemplo rotulado vai para o disco.

    ``JUSCRAPER_CAPTCHA_MAX_ROTULADOS``: teto de arquivos na pasta (0 ou ausente:
    sem teto). ``JUSCRAPER_CAPTCHA_AMOSTRA=N``: grava 1 em N, escolhido pelo
    hash da imagem. O ``registro.jsonl`` guarda todos os resultados de qualquer
    forma; o teto só limita as imagens, que crescem com a coleta.
    """
    n = int(os.environ.get("JUSCRAPER_CAPTCHA_AMOSTRA") or 1)
    if n > 1 and int(h, 16) % n:
        return False
    teto = int(os.environ.get("JUSCRAPER_CAPTCHA_MAX_ROTULADOS") or 0)
    if not teto:
        return True
    if pasta not in _contagem or _gravados.get(pasta, 0) >= RECONTA:
        _contagem[pasta] = sum(1 for _ in pasta.iterdir()) if pasta.exists() else 0
        _gravados[pasta] = 0
    return _contagem[pasta] < teto


def registrar(perfil, imagem: bytes, texto, conf, aceito):
    """Grava o resultado. Captcha aceito vira exemplo rotulado (imagem + texto certo)."""
    d = _dir(perfil)
    h = hashlib.sha1(imagem).hexdigest()[:12]
    pasta = d / "rotulados"
    if aceito and _grava_rotulado(pasta, h):
        pasta.mkdir(exist_ok=True)
        arq = pasta / f"{texto}_{h}.png"
        if not arq.exists():
            arq.write_bytes(imagem)
            _contagem[pasta] = _contagem.get(pasta, 0) + 1
            _gravados[pasta] = _gravados.get(pasta, 0) + 1
    with open(d / "registro.jsonl", "a") as fh:
        fh.write(json.dumps({"t": round(time.time(), 1), "img": h, "texto": texto, "conf": conf,
                             "aceito": aceito}, ensure_ascii=False) + "\n")


def estatisticas(perfil) -> dict:
    arq = _dir(perfil) / "registro.jsonl"
    linhas = [json.loads(x) for x in arq.read_text().splitlines()] if arq.exists() else []
    enviados = [x for x in linhas if x["aceito"] is not None]
    aceitos = sum(1 for x in enviados if x["aceito"])
    return {"perfil": perfil, "enviados": len(enviados), "aceitos": aceitos,
            "acerto": round(aceitos / len(enviados), 3) if enviados else None,
            "rotulados": len(list((_dir(perfil) / "rotulados").glob("*.png"))) if (_dir(perfil) / "rotulados").exists() else 0}
