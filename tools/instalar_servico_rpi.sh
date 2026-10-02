#!/usr/bin/env bash
# Instala a API como servico do usuario no Raspberry Pi: sobe quando o Pi
# liga, volta sozinha 10 s depois de cair e nao depende de sessao SSH.
# Uso, na raiz do projeto no Pi:
#   bash tools/instalar_servico_rpi.sh CAMINHO/certificado.pem CAMINHO/chave.pem
set -euo pipefail

cert="$(realpath "${1:?Informe o certificado TLS}")"
key="$(realpath "${2:?Informe a chave TLS}")"
for file in "$cert" "$key"; do
    [ -f "$file" ] || { echo "Arquivo nao encontrado: $file" >&2; exit 1; }
done
repo="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$repo/.venv/bin/python" ] || { echo "Falta o ambiente $repo/.venv" >&2; exit 1; }

# Uma API iniciada a mao disputaria a porta 8000 com o servico.
if ! systemctl --user is-active --quiet citylab-api.service; then
    if pid="$(pgrep -f tools/run_rpi.py)"; then
        echo "Ha uma API rodando fora do servico (PID $pid). Pare com: pkill -f run_rpi.py" >&2
        exit 1
    fi
fi

unit_dir="$HOME/.config/systemd/user"
mkdir -p "$unit_dir"
sed -e "s|@REPO@|$repo|g" -e "s|@CERT@|$cert|g" -e "s|@KEY@|$key|g" \
    "$repo/tools/citylab-api.service" > "$unit_dir/citylab-api.service"

# Sem linger, os servicos do usuario so rodam enquanto ele esta logado.
linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo no)"
if [ "$linger" != "yes" ]; then
    sudo loginctl enable-linger "$USER"
fi

systemctl --user daemon-reload
systemctl --user enable --now citylab-api.service
# So o estado: o log fica no journalctl.
systemctl --user --no-pager --lines=0 status citylab-api.service || true
