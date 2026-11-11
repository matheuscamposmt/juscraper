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
    # O site nao diferencia caixa: medido em 11/11/2026 no TJPR, 8 de 8 palpites
    # enviados com a caixa trocada ("b99b" como "B99B") foram aceitos. Por isso
    # a leitura soma "c" e "C" numa classe so ("caixa": False).
    "eproc": {"alfabeto": ALNUM, "tamanho": (4, 4), "limiar": 0.40, "caixa": False},
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


def decodifica_sem_caixa(probabilidades, charset, permitidos):
    """Como :func:`decodifica`, mas maiuscula e minuscula sao uma classe so.

    Em cada posicao soma a probabilidade das duas caixas de cada letra. O
    texto sai em minusculas; a confianca e a menor probabilidade (da classe)
    entre os caracteres escolhidos. Medido em 3.000 rotulados do Eproc: a
    confianca mediana sobe de 0,62 para 0,85.
    """
    classes: dict = {}
    for i, c in enumerate(charset):
        if c == "" or c in permitidos:
            classes.setdefault(c.lower(), []).append(i)
    texto, anterior, conf = "", None, 1.0
    for linha in probabilidades:
        while linha and isinstance(linha[0], list):
            linha = linha[0]
        soma = {k: sum(linha[i] for i in ix) for k, ix in classes.items()}
        k = max(soma, key=soma.get)
        if k and k != anterior:
            texto += k
            conf = min(conf, soma[k])
        anterior = k
    return texto, round(conf, 3)


def sem_saturados(png: bytes, limiar: int = 100, brilho: int = 200) -> bytes:
    """Pinta de branco os pixels saturados e claros (HSV do OpenCV, 0 a 255).

    No Eproc o texto vem escuro e de cor solida e as linhas de ruido em cores
    saturadas e claras. Pixel escuro tem saturacao instavel no HSV, por isso
    o corte exige tambem brilho alto. A imagem continua colorida.
    """
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[(hsv[..., 1] > limiar) & (hsv[..., 2] > brilho)] = 255
    return cv2.imencode(".png", img)[1].tobytes()


def renormaliza(probabilidades, charset, permitidos):
    """Probabilidade de cada posicao dividida pela soma das classes permitidas e do branco."""
    idx = [i for i, c in enumerate(charset) if c == "" or c in permitidos]
    out = []
    for linha in probabilidades:
        while linha and isinstance(linha[0], list):
            linha = linha[0]
        soma = sum(linha[i] for i in idx) or 1.0
        nova = [0.0] * len(linha)
        for i in idx:
            nova[i] = linha[i] / soma
        out.append(nova)
    return out


_OCR_FAIXA = None


def _ocr_faixa(alfabeto):
    """Outra instancia do ddddocr com o charset_range do proprio pacote (descarta, depois do argmax, o resto)."""
    global _OCR_FAIXA
    if _OCR_FAIXA is None:
        import ddddocr
        _OCR_FAIXA = ddddocr.DdddOcr(show_ad=False)
        _OCR_FAIXA.set_ranges(alfabeto)
    return _OCR_FAIXA


def ler(imagem: bytes, perfil="generico", renorm=False):
    """(texto, confianca) do OCR, restrito ao alfabeto do perfil."""
    p = PERFIS[perfil]
    r = _ocr().classification(imagem, probability=True)
    # ddddocr 1.6 devolve "charset"/"probabilities"; o 1.5.6, último para o
    # Python 3.9 da VM, devolve "charsets"/"probability".
    probs = r["probabilities"] if "probabilities" in r else r["probability"]
    charset = r["charset"] if "charset" in r else r["charsets"]
    dec = decodifica if p.get("caixa", True) else decodifica_sem_caixa
    if renorm:
        probs = renormaliza(probs, charset, set(p["alfabeto"]))
    return dec(probs, charset, set(p["alfabeto"]))


VARIANTES = ("atual", "dessat", "renorm", "range")
"""Leituras do teste A/B (JUSCRAPER_CAPTCHA_AB="atual,dessat,renorm,range"), alternadas por captcha:

* atual: a leitura do perfil e o limiar;
* dessat: a atual; se nao passa, le de novo sem as linhas saturadas (sem requisicao a mais);
* renorm: confianca renormalizada sobre o alfabeto, mesmo limiar;
* range: o texto do ddddocr com charset_range; envia se tiver o tamanho do perfil.
"""
_ab = {"n": 0, "ultima": None}


def _passa(p, texto, conf):
    return p["tamanho"][0] <= len(texto) <= p["tamanho"][1] and conf >= p["limiar"]


def _le_variante(imagem, perfil, v):
    p = PERFIS[perfil]
    if v == "renorm":
        return ler(imagem, perfil, renorm=True)
    if v == "range":
        texto = _ocr_faixa(p["alfabeto"]).classification(imagem)
        if not p.get("caixa", True):
            texto = texto.lower()
        return texto, 1.0
    texto, conf = ler(imagem, perfil)
    if v == "dessat" and not _passa(p, texto, conf):
        t2, c2 = ler(sem_saturados(imagem), perfil)
        if _passa(p, t2, c2):
            return t2, c2
    return texto, conf


def resolver(imagem: bytes, perfil="generico"):
    """O texto, se o palpite passa no tamanho e no limiar do perfil; senao None.

    None quer dizer "nao envie, peca outro captcha", como o
    ERROR_CAPTCHA_UNSOLVABLE do CapMonster abaixo do recognizingThreshold.
    """
    ab = [x for x in (os.environ.get("JUSCRAPER_CAPTCHA_AB") or "").split(",") if x in VARIANTES]
    if ab:
        v = ab[_ab["n"] % len(ab)]
        _ab["n"] += 1
        _ab["ultima"] = v
        texto, conf = _le_variante(imagem, perfil, v)
    else:
        _ab["ultima"] = None
        texto, conf = ler(imagem, perfil)
    p = PERFIS[perfil]
    if not _passa(p, texto, conf):
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
        linha = {"t": round(time.time(), 1), "img": h, "texto": texto, "conf": conf, "aceito": aceito}
        if _ab["ultima"]:
            linha["variante"] = _ab["ultima"]
        fh.write(json.dumps(linha, ensure_ascii=False) + "\n")


def estatisticas(perfil) -> dict:
    arq = _dir(perfil) / "registro.jsonl"
    linhas = [json.loads(x) for x in arq.read_text().splitlines()] if arq.exists() else []
    enviados = [x for x in linhas if x["aceito"] is not None]
    aceitos = sum(1 for x in enviados if x["aceito"])
    return {"perfil": perfil, "enviados": len(enviados), "aceitos": aceitos,
            "acerto": round(aceitos / len(enviados), 3) if enviados else None,
            "rotulados": len(list((_dir(perfil) / "rotulados").glob("*.png"))) if (_dir(perfil) / "rotulados").exists() else 0}
