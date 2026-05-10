#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_ROOT="${PROJECT_ROOT}/results"

MODELS=(
  "qwen/qwen3.5-9b"
  "openai/gpt-oss-20b"
  "deepseek/deepseek-v4-flash"
  "openrouter/owl-alpha"
)

DATASETS=(
  "CSTR.csv|Classify scientific paper abstracts by research topic."
  "Dmoz-Computers.csv|Classify web pages in the computers domain into subcategories."
  "Dmoz-Health.csv|Classify web pages in the health domain into subcategories."
  "Dmoz-Science.csv|Classify web pages in the science domain into subcategories."
  "Dmoz-Sports.csv|Classify web pages in the sports domain into subcategories."
  "NSF.csv|Classify NSF project descriptions by research area."
  "SyskillWebert.csv|Classify web pages by a user's preference rating."
  "classic4.csv|Classify documents into one of the four Classic benchmark collections: CACM, CISI, CRAN, or MED."
  "re8.csv|Classify Reuters news articles into one of the available categories."
  "review_polarity.csv|Classify review texts by sentiment polarity."
  "sms_spam.csv|Classify SMS messages as spam or ham."
)

run_experiment() {
  local dataset_name="$1"
  local task="$2"
  local model="$3"
  local reasoning="$4"

  local -a cmd=(
    python
    "${PROJECT_ROOT}/src/scripts/run.py"
    --task "$task"
    --dataset-name "$dataset_name"
    --output-root "$RESULTS_ROOT"
    --llm-provider openrouter
    --llm-model "$model"
  )

  if [[ "$reasoning" != "none" ]]; then
    cmd+=(--thinking-effort "$reasoning")
  fi

  echo
  echo "============================================================"
  echo "dataset:   $dataset_name"
  echo "task:      $task"
  echo "model:     $model"
  echo "reasoning: $reasoning"
  echo "============================================================"
  echo

  "${cmd[@]}"
}

main() {
  cd "$PROJECT_ROOT" || exit 1

  local dataset_name task model
  for entry in "${DATASETS[@]}"; do
    dataset_name="${entry%%|*}"
    task="${entry#*|}"

    for model in "${MODELS[@]}"; do
      run_experiment "$dataset_name" "$task" "$model" "high"
      run_experiment "$dataset_name" "$task" "$model" "none"
    done
  done
}

main "$@"
