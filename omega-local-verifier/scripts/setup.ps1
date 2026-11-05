$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "Docker is required. Install Docker Desktop first."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv is required: https://docs.astral.sh/uv/"
}

docker build --tag local-lean-verifier:4.27.0 .
uv sync
uv run lean-verify health

Write-Host "Setup complete."
Write-Host "uv run lean-verify verify --spec examples/specs/add_comm.lean --candidate examples/candidates/add_comm_good.lean"
