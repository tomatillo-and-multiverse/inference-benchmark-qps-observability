#!/bin/bash

# Copyright 2024 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o xtrace

export IP=$IP

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

if [[ "$PROMPT_DATASET" = "sharegpt" ]]; then
  PROMPT_DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
fi

PYTHON="python3"
BASE_PYTHON_OPTS=(
  "benchmark_serving.py"
  "--save-json-results"
  "--host=$IP"
  "--port=$PORT"
  "--dataset=$PROMPT_DATASET_FILE"
  "--tokenizer=$TOKENIZER"
  "--backend=$BACKEND"
  "--max-input-length=$INPUT_LENGTH"
  "--max-output-length=$OUTPUT_LENGTH"
  "--file-prefix=$FILE_PREFIX"
  "--models=$MODELS"
)

[[ "$TRAFFIC_SPLIT" ]] && BASE_PYTHON_OPTS+=("--traffic-split=$TRAFFIC_SPLIT")
[[ "$OUTPUT_BUCKET" ]] && BASE_PYTHON_OPTS+=("--output-bucket=$OUTPUT_BUCKET")
[[ "$MIN_INPUT_LENGTH" ]] && BASE_PYTHON_OPTS+=("--min-input-length=$MIN_INPUT_LENGTH")
[[ "$MIN_OUPUT_LENGTH" ]] && BASE_PYTHON_OPTS+=("--min-output-length=$MIN_OUPUT_LENGTH")
[[ "$SCRAPE_SERVER_METRICS" = "true" ]] && BASE_PYTHON_OPTS+=("--scrape-server-metrics")
[[ "$SAVE_AGGREGATED_RESULT" = "true" ]] && BASE_PYTHON_OPTS+=("--save-aggregated-result")
[[ "$STREAM_REQUEST" = "true" ]] && BASE_PYTHON_OPTS+=("--stream-request")
[[ "$IGNORE_EOS" = "true" ]] && BASE_PYTHON_OPTS+=("--ignore-eos")
[[ "$OUTPUT_BUCKET_FILEPATH" ]] && BASE_PYTHON_OPTS+=("--output-bucket-filepath" "$OUTPUT_BUCKET_FILEPATH")
[[ "$PM_JOB" ]] && BASE_PYTHON_OPTS+=("--pm-job=$PM_JOB")
[[ "$PM_NAMESPACE" ]] && BASE_PYTHON_OPTS+=("--pm-namespace=$PM_NAMESPACE")



SLEEP_TIME=${SLEEP_TIME:-0}

for request_rate in $(echo $REQUEST_RATES | tr ',' ' '); do
  echo "Benchmarking request rate: ${request_rate}"
  # TODO: Check if profile already exists, if so then skip
  timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
  output_file="latency-profile-${timestamp}.txt"
  
  if [ "$request_rate" == "0" ]; then
    request_rate="inf"
    num_prompts=$MAX_NUM_PROMPTS
  else
    num_prompts=$(awk "BEGIN {print int($request_rate * $BENCHMARK_TIME_SECONDS)}")
  fi

  echo "TOTAL prompts: $num_prompts"
  PYTHON_OPTS=("${BASE_PYTHON_OPTS[@]}" "--request-rate=$request_rate" "--num-prompts=$num_prompts")
  
  $PYTHON "${PYTHON_OPTS[@]}" > "$output_file"
  cat "$output_file"
  echo "Sleeping for $SLEEP_TIME seconds..."
  sleep $SLEEP_TIME
done

export LPG_FINISHED="true"
sleep infinity