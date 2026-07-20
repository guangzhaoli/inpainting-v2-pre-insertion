#!/bin/bash

# 自动获取 access_token
TOKEN=$(curl -s --location --request POST 'https://qz.sii.edu.cn/auth/token' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "username": "250041225002",
    "password": "LGZ@sii2026"
  }' | jq -r '.data.access_token')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "获取 access_token 失败"
  exit 1
fi

echo "==============================="
echo "Token 获取成功: ${TOKEN}"
echo "==============================="

# curl --location --request POST 'https://qz.sii.edu.cn/openapi/v1/train_job/detail' \
#     --header "Authorization: Bearer ${TOKEN}" \
#     --data-raw '{
#       "job_id": "job-4d6860aa-32cc-4173-a65f-feeba6a3b4b3"
#     }'
curl --location --request POST 'https://qz.sii.edu.cn/openapi/v1/train_job/create' \
  --header "Authorization: Bearer ${TOKEN}" \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "name": "all_train",
    "logic_compute_group_id": "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
    "project_id": "project-44e6816d-6abd-48b2-9e0b-942379839755",
    "auto_fault_tolerance": false,
    "framework": "pytorch",
    "command": "source /inspire/hdd/global_user/250041225002/miniconda3/bin/activate && conda activate qwen-inpainting-pre-insertion && cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-pre-insertion && TOTAL_PROCESSES=8 NPROC_PER_NODE=8 NUM_MACHINES=1 MACHINE_RANK=0 RESUME=/inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-pre-insertion/logs/inpainting_v2_teacher_logs/teacher_all_data_native_qwen_image_inpainting/v0/checkpoint-0-12000 bash run.sh configs/teacher/all.yaml",
    "task_priority": 6,
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "framework_config": [{
      "image": "docker.sii.shaipower.online/inspire-studio/guangzhao:1",
      "image_type": "SOURCE_PRIVATE",
      "instance_count": "1",
      "shm_gi": "1800",
      "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
    }]
  }'
