param(
    [string]$SourceBranch = "server",
    [string]$Remote = "CityLab_Security",
    [switch]$Push
)

$status = git status --porcelain
if ($status) {
    Write-Error "Working tree nao esta limpo. Commit ou stash as mudancas antes de promover para a main."
    exit 1
}

git fetch $Remote
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

git checkout main
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

git merge --ff-only "$Remote/main"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Nao foi possivel alinhar a main local com a main remota."
    exit $LASTEXITCODE
}

git merge --no-ff $SourceBranch -m "Promote $SourceBranch to stable main"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao promover $SourceBranch para a main."
    exit $LASTEXITCODE
}

if ($Push) {
    git push $Remote main
    exit $LASTEXITCODE
}

Write-Host "Promocao concluida na main local."
