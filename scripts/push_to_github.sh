#!/usr/bin/env bash
# Push to krishnadhara15 GitHub using the krishnadhara SSH key.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO="${1:-ai-stock-trend-forecast}"
REMOTE="git@github-krishna:krishnadhara15/${REPO}.git"

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "$REMOTE"
else
  git remote set-url origin "$REMOTE"
fi

export GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_krishnadhara -o IdentitiesOnly=yes'
git push -u origin main
echo "Pushed to https://github.com/krishnadhara15/${REPO}"
