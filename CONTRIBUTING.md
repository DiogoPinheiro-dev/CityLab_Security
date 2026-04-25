# Fluxo de Branches

Este repositorio usa um fluxo simples, pensado para uma equipe pequena.

## Papel de cada branch

- `main`: versao estavel do app, pronta para demonstracao, validacao externa e deploy.
- `server`: branch de desenvolvimento principal para backend, pipeline e cliente.
- outras branches: usadas quando fizer sentido separar uma feature, correcao ou experimento.

## Como trabalhar

Voce pode commitar direto na `main` quando a mudanca ja estiver validada e fizer sentido publica-la como versao estavel.

Para mudancas maiores, ainda e recomendado usar uma branch separada:

```powershell
git switch -c feature/nome-da-feature
```

Depois de validar, faca merge para a `main`:

```powershell
git switch main
git merge feature/nome-da-feature
git push CityLab_Security main
```

## Boas praticas

- mantenha a `main` funcionando antes de publicar
- rode os testes ou uma validacao manual antes do push
- use nomes claros para branches temporarias, como `feature/<tema>`, `fix/<tema>` ou `experiment/<tema>`
- use tags quando quiser marcar uma versao estavel importante

Exemplo de tag:

```powershell
git tag -a stable-2026-04-25 -m "Stable app snapshot"
git push CityLab_Security stable-2026-04-25
```
