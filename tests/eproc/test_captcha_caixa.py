from juscraper.courts._eproc import captcha

CHARSET = ["", "c", "C", "0", "o"]


def _linha(**p):
    return [p.get("branco", 0.0), p.get("c", 0.0), p.get("C", 0.0), p.get("z", 0.0), p.get("o", 0.0)]


def test_soma_as_duas_caixas():
    probs = [_linha(c=0.35, C=0.35, z=0.30), _linha(branco=1.0)]
    assert captcha.decodifica(probs, CHARSET, set("cC0o")) == ("c", 0.35)
    assert captcha.decodifica_sem_caixa(probs, CHARSET, set("cC0o")) == ("c", 0.7)


def test_classe_somada_pode_vencer_outro_caractere():
    # "0" sozinho vence "o", mas a soma das caixas de "c" vence os dois
    probs = [_linha(c=0.25, C=0.25, z=0.4, o=0.1)]
    assert captcha.decodifica_sem_caixa(probs, CHARSET, set("cC0o"))[0] == "c"


def test_repeticao_de_caixas_diferentes_junta():
    probs = [_linha(c=0.9), _linha(C=0.9), _linha(branco=1.0), _linha(C=0.9)]
    assert captcha.decodifica_sem_caixa(probs, CHARSET, set("cC0o"))[0] == "cc"


def test_perfil_eproc_sem_caixa():
    assert captcha.PERFIS["eproc"]["caixa"] is False
