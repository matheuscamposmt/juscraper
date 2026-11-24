"""Captcha de imagem por OCR local, no modelo do CapMonster.

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

O ddddocr e o OpenCV sao dependencias opcionais: so sao importados na
primeira leitura, para que o modulo carregue sem eles.
"""
import functools
import hashlib
import json
import os
import string
import time
from collections.abc import Container, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

ALNUM = string.ascii_letters + string.digits


class Perfil(TypedDict):
    """Alfabeto, tamanho aceito e limiar de confianca do captcha de um site."""

    alfabeto: str
    tamanho: tuple[int, int]
    limiar: float
    caixa: NotRequired[bool]


PERFIS: dict[str, Perfil] = {
    # Eproc (TJPR, TJSC, TJRS, TRF4...): 4 caracteres, letras e numeros, com linhas de ruido.
    # O site nao diferencia caixa: medido em 11/11/2026 no TJPR, 8 de 8 palpites
    # enviados com a caixa trocada ("b99b" como "B99B") foram aceitos. Por isso
    # a leitura soma "c" e "C" numa classe so ("caixa": False).
    "eproc": {"alfabeto": ALNUM, "tamanho": (4, 4), "limiar": 0.40, "caixa": False},
    # Generico: sem saber o site, aceita de 3 a 8 caracteres.
    "generico": {"alfabeto": ALNUM, "tamanho": (3, 8), "limiar": 0.50},
}


class _Motor(Protocol):
    """A parte do ``ddddocr.DdddOcr`` que este modulo usa."""

    def classification(self, img: bytes, *, probability: bool = False) -> object:
        """Texto lido ou, com ``probability``, as probabilidades de cada posicao."""

    def set_ranges(self, charset_range: str) -> None:
        """Restringe o alfabeto da leitura."""


_MOTORES: dict[str, _Motor] = {}
"""Instancias do ddddocr, criadas na primeira leitura e reaproveitadas."""


def _novo_motor() -> _Motor:
    motor: _Motor = import_module("ddddocr").DdddOcr(show_ad=False)
    return motor


def _ocr(modelo: str = "geral") -> _Motor:
    """Instancia do ddddocr usada nas leituras com probabilidade.

    ``modelo="beta"`` e o outro modelo do pacote (``beta=True``), criado so na
    primeira leitura que o pede (cerca de 80 MB a mais de memoria).
    """
    if modelo not in _MOTORES:
        if modelo == "beta":
            _MOTORES[modelo] = import_module("ddddocr").DdddOcr(show_ad=False, beta=True)
        else:
            _MOTORES[modelo] = _novo_motor()
    return _MOTORES[modelo]


def _indices(charset: Sequence[str], permitidos: Container[str]) -> list[int]:
    """Posicoes do branco e dos caracteres permitidos no charset."""
    return [i for i, c in enumerate(charset) if c == "" or c in permitidos]


def _posicao(linha: Sequence[object]) -> Sequence[float]:
    """Probabilidades de uma posicao; o ddddocr embrulha cada posicao: [[p0, p1, ...]]."""
    while linha and isinstance(linha[0], list):
        linha = linha[0]
    return cast("Sequence[float]", linha)


def _mais_provavel(linha: Sequence[float], indices: list[int]) -> int:
    """Indice de maior probabilidade entre ``indices``; no empate, o primeiro."""
    return max(indices, key=lambda k: linha[k])


def decodifica(
    probabilidades: Sequence[Sequence[object]], charset: Sequence[str], permitidos: Container[str]
) -> tuple[str, float]:
    """Decodificacao CTC restrita a um alfabeto.

    Em cada posicao escolhe o caractere permitido (ou o branco) mais provavel,
    junta repeticoes seguidas e tira os brancos. A confianca e a menor
    probabilidade entre os caracteres escolhidos.

    Args:
        probabilidades: probabilidades de cada posicao, na ordem do ``charset``.
        charset: caracteres do modelo; ``""`` e o branco.
        permitidos: caracteres que podem sair no texto.

    Returns:
        O texto e a confianca, arredondada em 3 casas.
    """
    idx = _indices(charset, permitidos)
    texto, anterior, conf = "", None, 1.0
    for linha in probabilidades:
        p = _posicao(linha)
        i = _mais_provavel(p, idx)
        if charset[i] and i != anterior:
            texto += charset[i]
            conf = min(conf, p[i])
        anterior = i
    return texto, round(conf, 3)


def _classes(charset: Sequence[str], permitidos: Container[str]) -> dict[str, list[int]]:
    """Indices de cada caractere permitido em minuscula, juntando as duas caixas."""
    classes: dict[str, list[int]] = {}
    for i in _indices(charset, permitidos):
        classes.setdefault(charset[i].lower(), []).append(i)
    return classes


def _classe_mais_provavel(linha: Sequence[float], classes: dict[str, list[int]]) -> tuple[str, float]:
    """Classe de maior probabilidade somada e essa soma; no empate, a primeira."""
    soma = {k: sum(linha[i] for i in ix) for k, ix in classes.items()}
    k = max(soma, key=lambda c: soma[c])
    return k, soma[k]


def decodifica_sem_caixa(
    probabilidades: Sequence[Sequence[object]], charset: Sequence[str], permitidos: Container[str]
) -> tuple[str, float]:
    """Como :func:`decodifica`, mas maiuscula e minuscula sao uma classe so.

    Em cada posicao soma a probabilidade das duas caixas de cada letra. O
    texto sai em minusculas; a confianca e a menor probabilidade (da classe)
    entre os caracteres escolhidos. Medido em 3.000 rotulados do Eproc: a
    confianca mediana sobe de 0,62 para 0,85.

    Args:
        probabilidades: probabilidades de cada posicao, na ordem do ``charset``.
        charset: caracteres do modelo; ``""`` e o branco.
        permitidos: caracteres que podem sair no texto, em qualquer caixa.

    Returns:
        O texto em minusculas e a confianca, arredondada em 3 casas.
    """
    classes = _classes(charset, permitidos)
    texto, anterior, conf = "", None, 1.0
    for linha in probabilidades:
        k, prob = _classe_mais_provavel(_posicao(linha), classes)
        if k and k != anterior:
            texto += k
            conf = min(conf, prob)
        anterior = k
    return texto, round(conf, 3)


NAO_INFRA = frozenset(range(58, 65)) | frozenset(range(91, 97))
"""Codigos ASCII entre os digitos e as letras, que o InfraCaptcha pula."""

LIMIAR_LEXICO = 0.8
"""Posterior minima do melhor texto do lexico para enviar o palpite (variante ``lexico``)."""


def gera_infra(a: int, b: int) -> str:
    """Texto do InfraCaptcha (framework do TRF4, usado pelo Eproc) a partir dos codigos dos dois primeiros caracteres.

    O terceiro e o quarto caracteres saem dos dois primeiros: com ``x, y`` os codigos em ordem e ``m`` a metade
    arredondada de ``y - x``, o terceiro e ``x + m`` e o quarto e ``y - m``; se o codigo cai entre os digitos e as
    letras ou sai da faixa, fica ``x`` (ou ``y``). Medido em 24/11/2026 nos 50.078 captchas rotulados do Eproc
    (TJMG, TJPR, TJAC e TJBA): a regra vale em todos, sem caixa (2% num controle com os dois ultimos embaralhados).

    Args:
        a: codigo do primeiro caractere.
        b: codigo do segundo caractere.

    Returns:
        O texto de 4 caracteres.
    """
    x, y = sorted((a, b))
    m = (y - x + 1) // 2
    c3 = x if (x + m in NAO_INFRA or x + m > 122) else x + m
    c4 = y if (y - m in NAO_INFRA or y - m < 48) else y - m
    return chr(a) + chr(b) + chr(c3) + chr(c4)


_COLUNAS_LEXICO: dict[tuple[int, tuple[str, ...]], "tuple[list[int], NDArray]"] = {}
"""Por charset do ddddocr: as colunas do branco e do alfabeto e a matriz que soma as duas caixas."""


@functools.cache
def lexico_infra() -> tuple[str, ...]:
    """Os textos que o InfraCaptcha gera, em minusculas (2.766 de 36**4), montados na primeira chamada.

    Montados de uma vez e devolvidos inteiros: as consultas do Eproc rodam em varias threads, e uma lista global
    preenchida aos poucos saia com cada texto em dobro quando duas threads a montavam juntas (24/11/2026: posterior
    presa em 0,5 e nenhum envio).
    """
    cods = [*range(48, 58), *range(65, 91), *range(97, 123)]
    return tuple(sorted({gera_infra(a, b).lower() for a in cods for b in cods}))


@functools.cache
def _matrizes_lexico() -> "tuple[list[str], NDArray, NDArray]":
    """(classes, estados estendidos, saltos) do CTC para cada texto do lexico (branco c1 branco ... c4 branco)."""
    np = import_module("numpy")
    lexico = lexico_infra()
    classes = sorted(set("".join(lexico)))
    rot = np.array([[classes.index(c) + 1 for c in w] for w in lexico])  # 0 e o branco
    ext = np.zeros((len(lexico), 9), dtype=int)
    ext[:, 1::2] = rot
    pula = np.zeros((len(lexico), 9), dtype=bool)
    pula[:, 3::2] = rot[:, 1:] != rot[:, :-1]
    return classes, ext, pula


def _colunas_lexico(
    charset: Sequence[str], permitidos: Container[str], classes: list[str]
) -> "tuple[list[int], NDArray]":
    """As colunas do branco e do alfabeto no charset do ddddocr (que tem milhares) e a matriz que as soma por classe."""
    chave = (len(charset), tuple(charset[:64]))
    if chave not in _COLUNAS_LEXICO:
        np = import_module("numpy")
        cols, grupo = [], []
        for i, c in enumerate(charset):
            k = 0 if c == "" else (classes.index(c.lower()) + 1 if c in permitidos and c.lower() in classes else -1)
            if k >= 0:
                cols.append(i)
                grupo.append(k)
        soma = np.zeros((len(cols), len(classes) + 1))
        soma[np.arange(len(cols)), grupo] = 1.0
        _COLUNAS_LEXICO[chave] = (cols, soma)
    return _COLUNAS_LEXICO[chave]


def _forward_ctc(pr: "NDArray", ext: "NDArray", pula: "NDArray") -> "NDArray":
    """Log-verossimilhanca de cada texto do lexico pelo forward do CTC.

    No dominio linear, normalizado a cada posicao (a escala vai em log): o mesmo resultado do forward em log,
    sem logaddexp.
    """
    np = import_module("numpy")
    a = np.zeros(ext.shape)
    a[:, 0] = pr[0, 0]
    a[:, 1] = pr[0, ext[:, 1]]
    escala = 0.0
    for t in range(1, pr.shape[0]):
        b = a.copy()
        b[:, 1:] += a[:, :-1]
        b[:, 2:] += np.where(pula[:, 2:], a[:, :-2], 0.0)
        a = b * pr[t][ext]
        topo = a.max()
        a /= topo
        escala += float(np.log(topo))
    return np.log(np.maximum(a[:, -1] + a[:, -2], 1e-300)) + escala


def decodifica_lexico(
    probabilidades: Sequence[Sequence[object]], charset: Sequence[str], permitidos: Container[str]
) -> tuple[str, float]:
    """O texto do lexico do InfraCaptcha mais provavel e a probabilidade dele entre os textos do lexico.

    Soma as duas caixas de cada letra (como :func:`decodifica_sem_caixa`) e calcula, pelo forward do CTC, a
    verossimilhanca de cada texto de :func:`lexico_infra` sob as mesmas probabilidades; a posterior e a
    verossimilhanca do melhor normalizada sobre o lexico. Medido em 24/11/2026 em 300 rotulados e 300 resgatados
    do Eproc, so com a leitura da imagem original: acerto de 98,7% e 96,0%, contra 93,0% e 24,7% da leitura
    gulosa; com posterior >= 0,8, 98% e 94% dos casos, com 100% e 98,6% de acerto. Cerca de 11 ms por captcha,
    contra ~30-60 ms de uma leitura do ddddocr.

    Args:
        probabilidades: probabilidades de cada posicao, na ordem do ``charset``.
        charset: caracteres do modelo; ``""`` e o branco.
        permitidos: caracteres que podem sair no texto, em qualquer caixa.

    Returns:
        O texto em minusculas e a posterior, arredondada em 3 casas.
    """
    np = import_module("numpy")
    classes, ext, pula = _matrizes_lexico()
    cols, soma = _colunas_lexico(charset, permitidos, classes)
    p = np.array([[q[i] for i in cols] for q in (_posicao(linha) for linha in probabilidades)], dtype=np.float64)
    m = p @ soma  # caixas somadas: (posicoes, classes + branco)
    pr = np.maximum(m / np.maximum(m.sum(1, keepdims=True), 1e-12), 1e-30)
    ll = _forward_ctc(pr, ext, pula)
    k = int(np.argmax(ll))
    return lexico_infra()[k], round(float(np.exp(ll[k] - np.logaddexp.reduce(ll))), 3)


def sem_saturados(png: bytes, limiar: int = 100, brilho: int = 200) -> bytes:
    """Pinta de branco os pixels saturados e claros (HSV do OpenCV, 0 a 255).

    No Eproc o texto vem escuro e de cor solida e as linhas de ruido em cores
    saturadas e claras. Pixel escuro tem saturacao instavel no HSV, por isso
    o corte exige tambem brilho alto. A imagem continua colorida.
    """
    cv2 = import_module("cv2")
    np = import_module("numpy")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[(hsv[..., 1] > limiar) & (hsv[..., 2] > brilho)] = 255
    saida: bytes = cv2.imencode(".png", img)[1].tobytes()
    return saida


def binariza_otsu(png: bytes) -> bytes:
    """Cinza binarizado pelo limiar de Otsu, devolvido como PNG de tres canais."""
    cv2 = import_module("cv2")
    np = import_module("numpy")
    cinza = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    _, preto_e_branco = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saida: bytes = cv2.imencode(".png", cv2.cvtColor(preto_e_branco, cv2.COLOR_GRAY2BGR))[1].tobytes()
    return saida


def renormaliza(
    probabilidades: Sequence[Sequence[object]], charset: Sequence[str], permitidos: Container[str]
) -> list[list[float]]:
    """Probabilidade de cada posicao dividida pela soma das classes permitidas e do branco."""
    idx = _indices(charset, permitidos)
    out = []
    for linha in probabilidades:
        p = _posicao(linha)
        soma = sum(p[i] for i in idx) or 1.0
        nova = [0.0] * len(p)
        for i in idx:
            nova[i] = p[i] / soma
        out.append(nova)
    return out


def _ocr_faixa(alfabeto: str) -> _Motor:
    """Outra instancia do ddddocr com o charset_range do proprio pacote (descarta, depois do argmax, o resto).

    A instancia e criada uma vez, com o alfabeto da primeira chamada.
    """
    if "faixa" not in _MOTORES:
        motor = _novo_motor()
        motor.set_ranges(alfabeto)
        _MOTORES["faixa"] = motor
    return _MOTORES["faixa"]


def _probabilidades(imagem: bytes, modelo: str = "geral") -> tuple[list, list]:
    """(probabilidades por posicao, charset) do ddddocr."""
    r = cast("dict[str, list]", _ocr(modelo).classification(imagem, probability=True))
    # ddddocr 1.6 devolve "charset"/"probabilities"; o 1.5.6, ultimo para o
    # Python 3.9, devolve "charsets"/"probability".
    probs = r["probabilities"] if "probabilities" in r else r["probability"]
    charset = r["charset"] if "charset" in r else r["charsets"]
    return probs, charset


def ler(
    imagem: bytes, perfil: str = "generico", *, renorm: bool = False, modelo: str = "geral"
) -> tuple[str, float]:
    """(texto, confianca) do OCR, restrito ao alfabeto do perfil."""
    p = PERFIS[perfil]
    probs, charset = _probabilidades(imagem, modelo)
    dec = decodifica if p.get("caixa", True) else decodifica_sem_caixa
    if renorm:
        probs = renormaliza(probs, charset, set(p["alfabeto"]))
    return dec(probs, charset, set(p["alfabeto"]))


VARIANTES = ("atual", "dessat", "renorm", "range", "dessat_otsu", "dessat_beta", "lexico")
"""Leituras do teste A/B (JUSCRAPER_CAPTCHA_AB="atual,dessat,renorm,range"), alternadas por captcha:

* atual: a leitura do perfil e o limiar;
* dessat: a atual; se nao passa, le de novo sem as linhas saturadas (sem requisicao a mais);
* dessat_otsu: a dessat; se ainda nao passa, le a imagem original binarizada por Otsu;
* dessat_beta: a dessat; se ainda nao passa, le a imagem original com o modelo beta;
* renorm: confianca renormalizada sobre o alfabeto, mesmo limiar;
* range: o texto do ddddocr com charset_range; envia se tiver o tamanho do perfil.
* lexico: uma leitura da imagem original, decodificada so entre os textos do InfraCaptcha
  (:func:`decodifica_lexico`); envia se a posterior passa de :data:`LIMIAR_LEXICO`. Substitui o limiar de
  confianca, o filtro de tamanho e a cascata.
"""


class _EstadoAB(TypedDict):
    n: int
    ultima: str | None


_ab: _EstadoAB = {"n": 0, "ultima": None}


def _passa(p: Perfil, texto: str, conf: float) -> bool:
    return p["tamanho"][0] <= len(texto) <= p["tamanho"][1] and conf >= p["limiar"]


_leitura: dict[str, object] = {"etapa": None, "leituras": {}}
"""A leitura mais recente de :func:`resolver`: a etapa da cascata que decidiu (``atual``,
``dessat``, ``otsu``, ``beta``; ``nenhuma`` se nenhuma passou) e o texto e a confianca de cada
etapa lida. :func:`registrar` grava a etapa no registro e as leituras nos indices."""

_TERCEIRA = {"dessat_otsu": "otsu", "dessat_beta": "beta"}
"""Variante em cascata -> nome da terceira leitura."""


def _terceira(imagem: bytes, perfil: str, etapa: str) -> tuple[str, float]:
    if etapa == "otsu":
        return ler(binariza_otsu(imagem), perfil)
    return ler(imagem, perfil, modelo="beta")


def _anota(nome: str, texto: str, conf: float, p: Perfil) -> bool:
    """Guarda a leitura da etapa; se passa no limiar, ela decide."""
    cast("dict[str, list]", _leitura["leituras"])[nome] = [texto, conf]
    if _passa(p, texto, conf):
        _leitura["etapa"] = nome
        return True
    return False


def _cascata(imagem: bytes, perfil: str, v: str) -> tuple[str, float]:
    """atual, depois dessat, depois a terceira leitura da variante; cada uma so se a anterior nao passou."""
    p = PERFIS[perfil]
    texto, conf = ler(imagem, perfil)
    _leitura.update(etapa="atual", leituras={"atual": [texto, conf]})
    if _passa(p, texto, conf):
        return texto, conf
    if not v.startswith("dessat"):
        _leitura["etapa"] = "nenhuma"
        return texto, conf
    t, c = ler(sem_saturados(imagem), perfil)
    if _anota("dessat", t, c, p):
        return t, c
    nome = _TERCEIRA.get(v)
    if nome:
        t, c = _terceira(imagem, perfil, nome)
        if _anota(nome, t, c, p):
            return t, c
    _leitura["etapa"] = "nenhuma"
    return texto, conf


def _le_variante(imagem: bytes, perfil: str, v: str) -> tuple[str, float]:
    p = PERFIS[perfil]
    if v == "lexico":
        probs, charset = _probabilidades(imagem)
        texto, post = decodifica_lexico(probs, charset, set(p["alfabeto"]))
        _leitura.update(etapa="lexico" if post >= LIMIAR_LEXICO else "nenhuma", leituras={"lexico": [texto, post]})
        return texto, post
    if v == "renorm":
        return ler(imagem, perfil, renorm=True)
    if v == "range":
        texto = cast("str", _ocr_faixa(p["alfabeto"]).classification(imagem))
        if not p.get("caixa", True):
            texto = texto.lower()
        return texto, 1.0
    return _cascata(imagem, perfil, v)


def resolver(imagem: bytes, perfil: str = "generico") -> tuple[str | None, str, float]:
    """O texto, se o palpite passa no tamanho e no limiar do perfil; senao None.

    None quer dizer "nao envie, peca outro captcha", como o
    ERROR_CAPTCHA_UNSOLVABLE do CapMonster abaixo do recognizingThreshold.

    Returns:
        ``(texto ou None, texto lido, confianca)``.
    """
    ab = [x for x in (os.environ.get("JUSCRAPER_CAPTCHA_AB") or "").split(",") if x in VARIANTES]
    _leitura.update(etapa=None, leituras={})
    if ab:
        v = ab[_ab["n"] % len(ab)]
        _ab["n"] += 1
        _ab["ultima"] = v
        texto, conf = _le_variante(imagem, perfil, v)
    else:
        _ab["ultima"] = None
        texto, conf = _cascata(imagem, perfil, "atual")
    p = PERFIS[perfil]
    if _ab["ultima"] == "lexico":
        p = cast("Perfil", {**p, "limiar": LIMIAR_LEXICO})
    if not _passa(p, texto, conf):
        return None, texto, conf
    return texto, texto, conf


# --- registro, como o reportIncorrect -----------------------------------------

def _dir(perfil: str) -> Path:
    base = Path(os.environ.get("JUSCRAPER_CAPTCHA_DIR") or Path.home() / ".cache" / "juscraper" / "captcha")
    d = base / perfil
    d.mkdir(parents=True, exist_ok=True)
    return d


_contagem: dict[Path, int] = {}
"""Arquivos em cada pasta de rotulados, contados no disco e somados a cada gravação."""
_gravados: dict[Path, int] = {}
"""Gravações deste processo desde a última contagem no disco, por pasta."""
RECONTA = 100
"""A cada tantas gravações, reconta a pasta: outros processos podem gravar nela."""


def _grava_rotulado(
    pasta: Path, h: str, amostra: str = "AMOSTRA", teto_var: str = "MAX_ROTULADOS", teto_padrao: int = 0
) -> bool:
    """Se o exemplo rotulado vai para o disco.

    ``JUSCRAPER_CAPTCHA_MAX_ROTULADOS``: teto de arquivos na pasta (0 ou ausente:
    sem teto). ``JUSCRAPER_CAPTCHA_AMOSTRA=N``: grava 1 em N, escolhido pelo
    hash da imagem. O ``registro.jsonl`` guarda todos os resultados de qualquer
    forma; o teto só limita as imagens, que crescem com a coleta. ``amostra`` e
    ``teto_var`` trocam o sufixo das duas variáveis (a pasta dos não aceitos usa
    as suas); ``teto_padrao`` vale quando a variável do teto não está definida.
    """
    n = int(os.environ.get(f"JUSCRAPER_CAPTCHA_{amostra}") or 1)
    if n > 1 and int(h, 16) % n:
        return False
    teto = int(os.environ.get(f"JUSCRAPER_CAPTCHA_{teto_var}") or teto_padrao)
    if not teto:
        return True
    if pasta not in _contagem or _gravados.get(pasta, 0) >= RECONTA:
        _contagem[pasta] = sum(1 for _ in pasta.iterdir()) if pasta.exists() else 0
        _gravados[pasta] = 0
    return _contagem[pasta] < teto


def amostra_nao_aceitos() -> bool:
    """Se ``JUSCRAPER_CAPTCHA_NAO_ACEITOS_AMOSTRA`` pede a amostra dos captchas nao aceitos."""
    return int(os.environ.get("JUSCRAPER_CAPTCHA_NAO_ACEITOS_AMOSTRA") or 0) > 0


def _grava_imagem(pasta: Path, nome: str, imagem: bytes) -> bool:
    pasta.mkdir(exist_ok=True)
    arq = pasta / nome
    if arq.exists():
        return False
    arq.write_bytes(imagem)
    _contagem[pasta] = _contagem.get(pasta, 0) + 1
    _gravados[pasta] = _gravados.get(pasta, 0) + 1
    return True


def _grava_nao_aceito(d: Path, h: str, imagem: bytes, linha: dict) -> None:
    """Amostra dos captchas descartados abaixo do limiar ou recusados pelo site, sem rotulo.

    ``JUSCRAPER_CAPTCHA_NAO_ACEITOS_AMOSTRA=N`` grava 1 em N (pelo hash da imagem) em
    ``nao_aceitos/``, com teto ``JUSCRAPER_CAPTCHA_MAX_NAO_ACEITOS`` (padrao 2.000). Ao lado,
    ``nao_aceitos.jsonl`` traz tribunal, texto lido, confianca e motivo de cada imagem. Serve
    para rotular a mao a cauda dificil, que os rotulados (so aceitos) nao mostram.
    """
    if not amostra_nao_aceitos():
        return
    pasta = d / "nao_aceitos"
    if not _grava_rotulado(pasta, h, "NAO_ACEITOS_AMOSTRA", "MAX_NAO_ACEITOS", 2000):
        return
    motivo = "recusado" if linha["aceito"] is False else "abaixo_do_limiar"
    trib = linha.get("tribunal") or "sem_tribunal"
    if _grava_imagem(pasta, f"{trib}_{motivo}_{h}.png", imagem):
        with (d / "nao_aceitos.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**linha, "motivo": motivo, "leituras": _leitura["leituras"]},
                                ensure_ascii=False) + "\n")


def _grava_resgatado(d: Path, h: str, imagem: bytes, linha: dict) -> None:
    """Captcha que so a terceira leitura da cascata (otsu, beta) resolveu e o site aceitou.

    Todos vao para ``resgatados/<texto>_<hash>.png``, sem amostra e fora do teto dos
    rotulados, com teto proprio ``JUSCRAPER_CAPTCHA_MAX_RESGATADOS`` (padrao 5.000). O rotulo
    vem do site. ``resgatados.jsonl`` traz o tribunal e a leitura de cada etapa.
    """
    pasta = d / "resgatados"
    if not _grava_rotulado(pasta, h, "AMOSTRA_RESGATADOS", "MAX_RESGATADOS", 5000):
        return
    if _grava_imagem(pasta, f"{linha['texto']}_{h}.png", imagem):
        with (d / "resgatados.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**linha, "leituras": _leitura["leituras"]}, ensure_ascii=False) + "\n")


_site: dict[str, str | None] = {"tribunal": None}
"""Tribunal da consulta em curso, gravado em cada linha do registro (varios sites dividem um perfil)."""


def define_tribunal(tribunal: str | None) -> None:
    """Tribunal que :func:`registrar` grava nas linhas seguintes (``None`` para nenhum)."""
    _site["tribunal"] = tribunal


def registrar(perfil: str, imagem: bytes, texto: str, conf: float, *, aceito: bool | None) -> None:
    """Grava o resultado. Captcha aceito vira exemplo rotulado (imagem + texto certo).

    Args:
        perfil: nome da pasta do registro (um perfil de :data:`PERFIS` ou outro).
        imagem: PNG do captcha.
        texto: palpite lido.
        conf: confianca do palpite.
        aceito: ``True`` ou ``False`` conforme a resposta do site; ``None`` se
            o palpite nao foi enviado. O tribunal vem de :func:`define_tribunal`.
    """
    d = _dir(perfil)
    # O hash so nomeia o arquivo e escolhe a amostra; nao protege nada.
    h = hashlib.sha1(imagem, usedforsecurity=False).hexdigest()[:12]
    pasta = d / "rotulados"
    if aceito and _grava_rotulado(pasta, h):
        _grava_imagem(pasta, f"{texto}_{h}.png", imagem)
    linha: dict[str, object] = {"t": round(time.time(), 1), "img": h, "texto": texto, "conf": conf, "aceito": aceito}
    if _site["tribunal"]:
        linha["tribunal"] = _site["tribunal"]
    if _ab["ultima"]:
        linha["variante"] = _ab["ultima"]
    if _leitura["etapa"]:
        linha["etapa"] = _leitura["etapa"]
    if not aceito:
        _grava_nao_aceito(d, h, imagem, linha)
    elif _leitura["etapa"] in _TERCEIRA.values():
        _grava_resgatado(d, h, imagem, linha)
    with (d / "registro.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linha, ensure_ascii=False) + "\n")


def estatisticas(perfil: str) -> dict[str, str | int | float | None]:
    """Resumo do registro do perfil.

    Args:
        perfil: nome da pasta do registro.

    Returns:
        ``perfil``, ``enviados`` (palpites enviados ao site), ``aceitos``,
        ``acerto`` (aceitos sobre enviados, ou ``None`` sem envios) e
        ``rotulados`` (imagens gravadas).
    """
    d = _dir(perfil)
    arq = d / "registro.jsonl"
    linhas = [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines()] if arq.exists() else []
    enviados = [x for x in linhas if x["aceito"] is not None]
    aceitos = sum(1 for x in enviados if x["aceito"])
    rotulados = d / "rotulados"
    return {"perfil": perfil, "enviados": len(enviados), "aceitos": aceitos,
            "acerto": round(aceitos / len(enviados), 3) if enviados else None,
            "rotulados": len(list(rotulados.glob("*.png"))) if rotulados.exists() else 0}
