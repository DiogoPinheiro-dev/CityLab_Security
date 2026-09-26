# Rodada descartada

`duas-pessoas-r1-cena-instavel.json` foi coletada em 26/09/2026 como a primeira
rodada de duas pessoas do commit `bc0a441`, mas a cena mudou no meio:

- So uma pessoa detectada em 16 de 30 frames, e tres em outros 3.
- Nenhum rosto em 18 de 30 frames: a pessoa de frente virou ou saiu.
- Sem rosto o estagio facial cai para cerca de 2 s, entao a mediana de 5655 ms
  mede uma cena mais leve, nao o cenario de duas pessoas.

A pose rodou com 3 threads nos 30 frames; nada indica defeito do codigo. A
rodada foi repetida com as duas pessoas paradas e o resultado valido e
`../duas-pessoas-r1.json`.
