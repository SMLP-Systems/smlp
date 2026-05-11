#!/usr/bin/bash

if [[ $# -eq 0 ]]; then
    echo ""
    echo "Usage: $0 <L>"
    echo "L - maximum number of the latest runs"
    echo ""
    exit 1
fi

L="${1}"
gh run list --repo SMLP-Systems/smlp --limit $L --json databaseId,displayTitle \
  | jq -c '.[]' \
  | while read -r run; do
      id=$(echo "$run" | jq -r '.databaseId')
      title=$(echo "$run" | jq -r '.displayTitle')
      counter=$((counter + 1))
      echo "=== $counter. Run $id: $title ==="
      attempt_count=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id" --jq '.run_attempt')
      for i in $(seq 1 $attempt_count); do
          actor=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id/attempts/$i" \
              --jq '.triggering_actor.login')
          artifacts=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id/artifacts" \
              --jq '.artifacts[].name')
          echo "  attempt $i [by: $actor]:"
          echo "$artifacts" | while read -r artifact; do
              echo "    $artifact"
          done
      done
    done
