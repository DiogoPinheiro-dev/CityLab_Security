# Rodada descartada: a pose ainda estava com 3 threads

`uma-pessoa-r1-pose-com-3-threads.json` foi rodada em 01/10/2026 as 22:09 pelo
relogio do PC, para a serie de `TORCH_NUM_THREADS=2`. A metrica
`gesture_torch_threads` deu 3 em todos os frames: a linha nao estava valendo no
`.env` do Pi quando a API subiu. Os tempos sairam iguais aos da base
`../../pi3-419ca7f-video-reuso15/`, media de 3703,8 ms, e servem so como mais
uma rodada daquela configuracao. A rodada valida foi refeita depois de
conferir o valor com `--show-config`.
