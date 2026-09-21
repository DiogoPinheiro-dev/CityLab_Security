# Rodadas descartadas - cenario de uma pessoa (970a384)

Descartadas como medicao de latencia em 20/09/2026, a pedido do responsavel.

Motivo: as duas rodadas nao mediram a mesma cena. A caixa da pessoa real caiu de
mediana 0,757 (r1) para 0,599 (r2) e os frames com mais de uma caixa subiram de
8/30 para 21/30. Com cenas diferentes, a regra dos 5% repetidos nas tres rodadas
perde base de comparacao. O cenario foi remedido do zero.

Nao usar para comparar latencia. Continuam validas como evidencia da caixa
fantasma (item 5 do backlog):

- r1 (cena forte): 40 caixas em 30 frames, 10 extras, maior caixa 0,539 a 0,897.
- r2 (cena fraca): 53 caixas em 30 frames, 23 extras, maior caixa 0,408 a 0,746.
- Limiar de 0,50 perde 0/30 frames na r1 e 4/30 na r2. Limiar de 0,60 perde
  15/30 na r2. Com estes dados, nenhum limiar acima de 0,40 e seguro.
- Recall preservado nas duas: rosto em 30/30, gesto em 30/30, zero alertas.
