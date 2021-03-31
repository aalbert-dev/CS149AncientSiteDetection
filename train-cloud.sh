#!/bin/bash
# Copyright 2019 Google LLC
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
# ==============================================================================
# This script performs cloud training for a PyTorch model.

echo "Submitting AI Platform PyTorch job"

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=project_1_upload

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-xla.1-7

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME="cosi149_$(date +%Y%m%d_%H%M%S)"

PACKAGE_PATH=./model_simple_project # this can be a GCS location to a zipped and uploaded package

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

# Datasets are set by datasets/download-taxi.sh script
TRAIN_PATH=gs://cosi149p1

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC \
    --job-dir ${JOB_DIR} \
    --module-name model_simple_project.train \
    --package-path ${PACKAGE_PATH} \
    -- \
    --root-dir ${TRAIN_PATH} \
    --num-epochs 10 \
    --batch-size 128 \
    --lr 0.01 \
    --seed 420

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*

