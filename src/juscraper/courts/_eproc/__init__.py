"""Família Eproc: consulta pública de processo (capa) com captcha de imagem.

Experimental. Ainda não segue o contrato de ``CONTRIBUTING.md`` (samples
capturados, testes de contrato, schema, CHANGELOG): é a base que este branch
vai formalizar.

Módulos:

* :mod:`.download` — URL por tribunal, sessão espaçada (rate limit entre
  processos), classificação da resposta e envio do formulário com o captcha.
* :mod:`.parse` — leitura da capa: classe, assuntos, autuação, valor da
  causa, partes com advogados, informações adicionais e eventos.
* :mod:`.captcha` — OCR local do captcha (ddddocr), com limiar de confiança
  e registro dos acertos.
"""
