# Reconhecimento assincrono com aprendizado ligado, tentativa descartada

Teste feito em 06/10/2026 com o commit `71b384e`, o mesmo video de carga e o
mesmo protocolo das rodadas do P5, mas com `FACE_ASYNC_RECOGNITION=1` e
`FACE_LEARN_FROM_STREAM=1`. Ele verifica a convivencia dos dois recursos no
modo de uso; nao substitui a r3, que foi combinada com o aprendizado desligado.

O unico resultado ficou em `descartadas/uma-pessoa-r1-subtensao.json`: 5
frames de aquecimento e 30 medidos, media de 1768,386 ms, mediana de 1850,235
ms e p95 de 2187,474 ms. Pessoa, rosto e gesto apareceram em 30/30; o nome
ficou confirmado em 20/30, pendente em 10/30 e desconhecido em nenhum, os
mesmos estados da r1 assincrona valida. A API nao registrou erro.

O tempo nao e comparavel: `descartadas/vcgencmd-r1.txt` registrou `0x50005` as
17:24:51, com o processador em 600 MHz. Por isso, o arquivo esta preservado
somente como diagnostico de funcionamento e de alimentacao.

O aprendizado guardou referencias tiradas do video de carga. Antes da proxima
serie controlada, elas precisam ser apagadas com `tools/limpar_aprendidos.py`,
ou a serie deve comecar de outro estado explicitamente registrado.
