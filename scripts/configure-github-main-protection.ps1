param(
    [string]$Repo = "DiogoPinheiro-dev/CityLab_Security",
    [string]$Branch = "main",
    [int]$RequiredApprovals = 1
)

$ghPath = $null

try {
    $ghPath = (Get-Command gh -ErrorAction Stop).Source
} catch {
    $defaultGhPath = "C:\Program Files\GitHub CLI\gh.exe"
    if (Test-Path $defaultGhPath) {
        $ghPath = $defaultGhPath
    }
}

if (-not $ghPath) {
    Write-Error "GitHub CLI nao encontrada. Instale a gh CLI antes de configurar a branch protection."
    exit 1
}

& $ghPath auth status 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "GitHub CLI sem autenticacao. Rode 'gh auth login' antes de configurar a branch protection."
    exit 1
}

$payload = @{
    required_status_checks = $null
    enforce_admins = $true
    required_pull_request_reviews = @{
        dismiss_stale_reviews = $true
        require_code_owner_reviews = $false
        required_approving_review_count = $RequiredApprovals
        require_last_push_approval = $false
    }
    restrictions = $null
    required_linear_history = $false
    allow_force_pushes = $false
    allow_deletions = $false
    block_creations = $false
    required_conversation_resolution = $true
    lock_branch = $false
    allow_fork_syncing = $true
}

$jsonPayload = $payload | ConvertTo-Json -Depth 10 -Compress
$tempFile = Join-Path $env:TEMP "github-branch-protection.json"
Set-Content -LiteralPath $tempFile -Value $jsonPayload -Encoding utf8

try {
    & $ghPath api `
        --method PUT `
        -H "Accept: application/vnd.github+json" `
        "repos/$Repo/branches/$Branch/protection" `
        --input $tempFile

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "Branch protection aplicada em $Repo::$Branch"
} finally {
    Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
}
