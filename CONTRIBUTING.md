# Fluxo de Branches

Este repositorio usa a `main` como branch estavel do app.

## Papel de cada branch

- `main`: versao estavel, pronta para demonstracao, validacao externa e deploy.
- `server`: branch principal de desenvolvimento da integracao do backend, pipeline e cliente.
- outras branches de desenvolvimento: usadas para features, refactors, correcao de bugs e experimentos antes da promocao para a `main`.

## Regras do fluxo

- O fluxo recomendado e desenvolver em branches de desenvolvimento.
- A `main` deve continuar representando uma versao estavel e validada do app.
- A promocao para a `main` deve preservar contexto, por isso o padrao recomendado e `merge commit`.

## Fluxo recomendado

1. Atualize a branch de desenvolvimento a partir da base mais recente.
2. Desenvolva e teste localmente.
3. Confirme que o working tree esta limpo.
4. Promova a branch para a `main`.
5. Valide a `main` como versao estavel.
6. Publique a `main` no remoto.

## Convencao de nomes

- `feature/<tema>`
- `fix/<tema>`
- `refactor/<tema>`
- `experiment/<tema>`

Exemplos:

- `feature/mobile-client`
- `fix/websocket-reconnect`
- `refactor/pipeline-split`

## Promocao para a main

Use o script:

```powershell
.\scripts\promote-to-main.ps1 -SourceBranch server
```

Se quiser publicar no remoto no mesmo fluxo:

```powershell
.\scripts\promote-to-main.ps1 -SourceBranch server -Push
```

## Recomendacoes para o GitHub

Se quiser refletir esse fluxo no remoto, o ideal e configurar:

- branch protection na `main`
- exigir pull request ou merge controlado para a `main`
- usar tags para marcar snapshots estaveis

O repositorio inclui um script para aplicar a protection na `main` via GitHub CLI:

```powershell
.\scripts\configure-github-main-protection.ps1
```

Configuracao aplicada por esse script:

- exige pull request para mudar a `main`
- exige 1 aprovacao
- bloqueia force push
- bloqueia delete da branch
- exige resolucao de conversa
- aplica protecao tambem para admins

Antes de rodar esse script, faca login na GitHub CLI:

```powershell
gh auth login
```

## Tags estaveis

Quando quiser marcar uma versao estavel validada, use algo como:

```powershell
git tag -a stable-2026-04-18 -m "Stable main snapshot"
git push CityLab_Security stable-2026-04-18
```
