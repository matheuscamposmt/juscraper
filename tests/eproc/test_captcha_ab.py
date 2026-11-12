import json

import pytest

from juscraper.courts._eproc import captcha


@pytest.fixture
def pasta(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def ab(monkeypatch):
    """Estado do A/B zerado, devolvido para o teste conferir a variante da vez."""
    estado = {"n": 0, "ultima": None}
    monkeypatch.setattr(captcha, "_ab", estado)
    return estado


@pytest.mark.usefixtures("pasta")
def test_sem_ab_usa_a_leitura_atual(ab, monkeypatch):
    monkeypatch.delenv("JUSCRAPER_CAPTCHA_AB", raising=False)
    ab["ultima"] = "atual"  # sobra de uma leitura anterior com A/B
    monkeypatch.setattr(captcha, "ler", lambda *_args, **_kwargs: ("ab12", 0.9))
    assert captcha.resolver(b"x", "eproc") == ("ab12", "ab12", 0.9)
    assert ab["ultima"] is None


@pytest.mark.usefixtures("ab")
def test_ab_alterna_e_registra_a_variante(pasta, monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "atual,renorm")
    vistos = []

    def ler(_imagem, _perfil="generico", *, renorm=False):
        vistos.append(renorm)
        return "ab12", 0.9

    monkeypatch.setattr(captcha, "ler", ler)
    captcha.resolver(b"x", "eproc")
    captcha.registrar("eproc", b"x", "ab12", 0.9, aceito=None)
    captcha.resolver(b"y", "eproc")
    captcha.registrar("eproc", b"y", "ab12", 0.9, aceito=True)
    assert vistos == [False, True]
    linhas = [json.loads(x) for x in (pasta / "eproc" / "registro.jsonl").read_text().splitlines()]
    assert [x["variante"] for x in linhas] == ["atual", "renorm"]


@pytest.mark.usefixtures("pasta", "ab")
def test_dessat_so_quando_a_atual_nao_passa(monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "dessat")
    monkeypatch.setattr(captcha, "sem_saturados", lambda _img: b"limpa")
    monkeypatch.setattr(captcha, "ler", lambda img, *_args, **_kwargs:
                        ("cd34", 0.8) if img == b"limpa" else ("ab1", 0.2))
    assert captcha.resolver(b"suja", "eproc")[0] == "cd34"
    monkeypatch.setattr(captcha, "ler", lambda *_args, **_kwargs: ("ab12", 0.9))
    assert captcha.resolver(b"suja", "eproc")[0] == "ab12"


@pytest.mark.usefixtures("pasta", "ab")
@pytest.mark.parametrize(("variante", "terceira"), [("dessat_otsu", "otsu"), ("dessat_beta", "beta")])
def test_cascata_le_a_terceira_so_quando_a_dessat_nao_passa(monkeypatch, variante, terceira):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", variante)
    monkeypatch.setattr(captcha, "sem_saturados", lambda _img: b"limpa")
    monkeypatch.setattr(captcha, "binariza_otsu", lambda _img: b"otsu")
    lidas = []

    def ler(img, *_args, modelo="geral", **_kwargs):
        lidas.append(modelo if modelo == "beta" else img.decode())
        return ("ef56", 0.9) if lidas[-1] == terceira else ("ab1", 0.2)

    monkeypatch.setattr(captcha, "ler", ler)
    assert captcha.resolver(b"suja", "eproc")[0] == "ef56"
    assert lidas == ["suja", "limpa", terceira]
    lidas.clear()

    def ler_passa(img, *_args, **_kwargs):
        lidas.append(img)
        return "cd34", 0.8

    monkeypatch.setattr(captcha, "ler", ler_passa)
    assert captcha.resolver(b"suja", "eproc")[0] == "cd34"
    assert lidas == [b"suja"]


def test_renormaliza_divide_pelas_permitidas():
    probs = [[0.2, 0.2, 0.6]]
    assert captcha.renormaliza(probs, ["", "a", "字"], {"a"}) == [[0.5, 0.5, 0.0]]


def _cascata_otsu(monkeypatch):
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_AB", "dessat_otsu")
    monkeypatch.setattr(captcha, "sem_saturados", lambda _img: b"limpa")
    monkeypatch.setattr(captcha, "binariza_otsu", lambda _img: b"otsu")
    monkeypatch.setattr(captcha, "ler", lambda img, *_a, **_k:
                        ("ef56", 0.9) if img == b"otsu" else ("ab1", 0.2))


@pytest.mark.usefixtures("ab")
def test_etapa_vai_no_registro_e_o_resgatado_aceito_vai_para_resgatados(pasta, monkeypatch):
    _cascata_otsu(monkeypatch)
    captcha.define_tribunal("tjpr")
    assert captcha.resolver(b"suja", "eproc")[0] == "ef56"
    captcha.registrar("eproc", b"suja", "ef56", 0.9, aceito=True)
    linha = json.loads((pasta / "eproc" / "registro.jsonl").read_text().splitlines()[-1])
    assert (linha["etapa"], linha["tribunal"]) == ("otsu", "tjpr")
    assert [p.name.split("_")[0] for p in (pasta / "eproc" / "resgatados").iterdir()] == ["ef56"]
    idx = json.loads((pasta / "eproc" / "resgatados.jsonl").read_text())
    assert idx["leituras"] == {"atual": ["ab1", 0.2], "dessat": ["ab1", 0.2], "otsu": ["ef56", 0.9]}


@pytest.mark.usefixtures("ab")
def test_resgatado_recusado_nao_vai_para_resgatados_e_o_nao_aceito_leva_as_leituras(pasta, monkeypatch):
    _cascata_otsu(monkeypatch)
    monkeypatch.setenv("JUSCRAPER_CAPTCHA_NAO_ACEITOS_AMOSTRA", "1")
    captcha.resolver(b"suja", "eproc")
    captcha.registrar("eproc", b"suja", "ef56", 0.9, aceito=False)
    assert not (pasta / "eproc" / "resgatados").exists()
    idx = json.loads((pasta / "eproc" / "nao_aceitos.jsonl").read_text())
    assert idx["etapa"] == "otsu"
    assert set(idx["leituras"]) == {"atual", "dessat", "otsu"}


@pytest.mark.usefixtures("ab")
def test_sem_ab_so_a_leitura_atual(pasta, monkeypatch):
    monkeypatch.delenv("JUSCRAPER_CAPTCHA_AB", raising=False)
    lidas = []

    def ler(img, *_args, **_kwargs):
        lidas.append(img)
        return "ab1", 0.2

    monkeypatch.setattr(captcha, "ler", ler)
    assert captcha.resolver(b"suja", "eproc")[0] is None
    captcha.registrar("eproc", b"suja", "ab1", 0.2, aceito=None)
    assert lidas == [b"suja"]
    assert json.loads((pasta / "eproc" / "registro.jsonl").read_text())["etapa"] == "nenhuma"
