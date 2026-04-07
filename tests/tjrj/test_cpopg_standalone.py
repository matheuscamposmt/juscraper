import pytest
from juscraper.courts.tjrj.client import TJRJScraper


@pytest.mark.integration
def test_cpopg_standalone():
    scraper = TJRJScraper()
    # Replace with a real TJRJ 1g process number for manual integration testing
    numero = "0001234-56.2020.8.19.0001"
    df = scraper.cpopg(numero)
    assert not df.empty
    assert "numero_processo" in df.columns
    assert "partes" in df.columns
    assert "valor_causa" in df.columns
    assert "movimentacoes" in df.columns


@pytest.mark.integration
def test_cposg_standalone():
    scraper = TJRJScraper()
    numero = "0001234-56.2020.8.19.0000"
    df = scraper.cposg(numero)
    assert not df.empty
    assert "partes" in df.columns


if __name__ == "__main__":
    test_cpopg_standalone()
