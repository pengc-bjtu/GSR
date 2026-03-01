#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import base64
import re
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



BENCHMARK_DIR = ""
IMAGES_DIR = ""
MAIN_IMAGE_DIR = os.path.join(IMAGES_DIR, "main")
SIDE_IMAGE_DIR = os.path.join(IMAGES_DIR, "side")

USE_API = True
USE_API_EVAL = True
NUM_WORKERS = 40  

if USE_API:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:50001/v1", timeout=3600)


if USE_API_EVAL:
    client_eval = OpenAI(api_key="EMPTY", base_url="http://localhost:50002/v1", timeout=3600)    


def file_to_data_url(path: str, mime: str = "image/jpeg") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def inference_with_openai_api(main_image_path, side_image_path, user_prompt):
    main_image_b64 = file_to_data_url(main_image_path)
    side_image_b64 = file_to_data_url(side_image_path)
    user_content = [
        {"type": "image_url", "image_url": {"url": main_image_b64}},
        {"type": "image_url", "image_url": {"url": side_image_b64}},
        {"type": "text", "text": user_prompt}
    ]
    messages = [{"role": "user", "content": user_content}]
    completion = client.chat.completions.create(model="qwenvl", messages=messages,
                 max_tokens=2048, 
                 )
    return completion.choices[0].message.content.strip()

def extract_answer(model_output):
    try:
        model_output = model_output.replace("```json", "").replace("```", "").strip()
        return json.loads(model_output)
    except Exception:
        return {"Answer_id": "", "Answer_text": "", "Predicted_Boxes": {}}

def extract_option_texts(question_text):
    pattern = r"([A-D])[\s\.:）)]\s*(.+?)(?=\n[A-D][\s\.:）)]|$)"
    matches = re.findall(pattern, question_text, flags=re.S)
    return {k.strip().upper(): v.strip() for k, v in matches}



def compute_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def normalize_name(name):
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = name.replace("the ", "").strip()
    name = name.replace("_", " ").strip()
    return re.sub(r"\s+", " ", name)

def class_match(g, p):
    g, p = normalize_name(g), normalize_name(p)
    return g == p or g in p or p in g

def compute_mean_iou(gt_boxes, pred_boxes):
    if not gt_boxes and not pred_boxes:
        return 1.0
    if not gt_boxes or not pred_boxes:
        return 0.0

    total_ious = []
    norm_gt = {normalize_name(k): v if isinstance(v[0], list) else [v] for k, v in gt_boxes.items()}
    norm_pred = {normalize_name(k): v if isinstance(v[0], list) else [v] for k, v in pred_boxes.items()}

    for gt_cls, gt_list in norm_gt.items():
        matched_cls = next((p for p in norm_pred if class_match(gt_cls, p)), None)
        if not matched_cls:
            continue
        pred_list = norm_pred[matched_cls]
        iou_matrix = np.zeros((len(gt_list), len(pred_list)))
        for i, g in enumerate(gt_list):
            for j, p in enumerate(pred_list):
                iou_matrix[i, j] = compute_iou(g, p)
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        total_ious.extend([iou_matrix[r, c] for r, c in zip(row_idx, col_idx)])
    return np.mean(total_ious) if total_ious else 0.0

def text_f1(pred_text, gt_text):
    pred_tokens = re.findall(r"\w+", pred_text.lower())
    gt_tokens = re.findall(r"\w+", gt_text.lower())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = len(set(pred_tokens) & set(gt_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate_prediction(gt_answer, gt_boxes, gt_answer_text, pred):
    pred_answer = pred.get("Answer_id", "").strip().upper()
    pred_text = pred.get("Answer_text", "").strip()
    pred_boxes = pred.get("Predicted_Boxes", {})
    if isinstance(gt_boxes, list):
        merged = {}
        for i in gt_boxes:
            if isinstance(i, dict): merged.update(i)
        gt_boxes = merged
    acc = 1.0 if pred_answer == gt_answer else 0.0
    f1 = text_f1(pred_text, gt_answer_text)
    miou = compute_mean_iou(gt_boxes, pred_boxes)
    return {"acc": acc, "f1": f1, "miou": miou}

def evaluate_prediction_api(question, gt_answer, gt_answer_text, gt_boxes, pred):
    gt_summary = {"Answer_id": gt_answer, "Answer_text": gt_answer_text, "Target_Boxes": gt_boxes}
    eval_prompt = f"""
You are an evaluation assistant for a multimodal reasoning benchmark.

You will receive:
1. The question text.
2. The ground truth (GT) including answer and bounding boxes.
3. The model's prediction.

Your task:
- Compare the model's output with the GT and evaluate **three metrics**:
  - **Accuracy (acc)**: 1.0 if `Answer_id` exactly matches (A/B/C/D), else 0.0.
  - **F1-score (f1)**: Semantic similarity between `Answer_text` (model) and `Answer_text` (GT), based on meaning overlap, not just string match.
    - Use 1.0 if perfectly same meaning.
    - 0.5–0.9 if partially same meaning (e.g., "phone" vs. "mobile phone").
    - 0.0 if different meaning.
  - **mIoU (miou)**: 
    - 1.0 if both GT and Pred have no boxes (empty).
    - 0.0 if one has boxes but the other not.
    - Else, estimate intersection-over-union between each GT and predicted bounding box (use reasonable approximate judgment).

Finally, output your judgment **strictly in JSON** format:
{{
  "acc": float,
  "f1": float,
  "miou": float
}}

Do not include any explanation outside JSON.
Only return JSON.

Here is the information:

### Question:
{question}

### Ground Truth:
{json.dumps(gt_summary, indent=2, ensure_ascii=False)}

### Model Prediction:
{pred}
"""
    try:
        completion = client_eval.chat.completions.create(
            model="qwenvl", messages=[{"role": "user", "content": eval_prompt}], temperature=0,frequency_penalty=0.1
            # ,max_tokens=2048, 
            )
        content = completion.choices[0].message.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # 如果包含 “</think>”，则取它后面的内容
        if "</think>" in content:
            content = content.split("</think>")[-1]
        content = re.sub(r"```json|```", "", content).strip()
        metrics = json.loads(content)
        for k in ["acc", "f1", "miou"]:
            metrics[k] = float(metrics.get(k, 0.0))
        return metrics
    except Exception as e:
        print("⚠️ API评估失败:", e)
        print("原始输出：",content)
        return {"acc": 0.0, "f1": 0.0, "miou": 0.0}



def process_sample(task_name, item):
    question = item.get("Question", "").strip()
    gt_answer = item.get("Answer", "").strip().upper()
    image_name = item.get("image_name", "").strip()
    gt_boxes = item.get("Target Instances", "")
    option_map = extract_option_texts(question)
    gt_answer_text = option_map.get(gt_answer, "").strip()
    if not question or not gt_answer or not image_name:
        print(f"⚠️ 选中样本字段缺失：{item}")
        return
    main_image_path = os.path.join(MAIN_IMAGE_DIR, image_name)
    side_image_path = os.path.join(SIDE_IMAGE_DIR, image_name)
    if not (os.path.exists(main_image_path) and os.path.exists(side_image_path)):
        print(f"⚠️ 图片缺失：{image_name}")        
        return None

    user_prompt = f"""
{question}

Each option describes a region or condition in the image.

You should answer the question based on the given image, and output your result **strictly** in the following JSON format:

{{
  "Answer_id": "A/B/C/D",
  "Answer_text": "<the full text of the chosen option>",
  "Predicted_Boxes": {{
      "ObjectName1": [x1, y1, x2, y2],
      "ObjectName2": [x1, y1, x2, y2]
  }}
}}

Notes:
- "Answer_id" must be one of A, B, C, or D.
- "Answer_text" must be copied exactly from the question options.
- Bounding boxes correspond to the key objects or areas that support your answer.
- Do NOT include any explanation or reasoning.
- Ensure valid JSON output.
"""

    try:
        response = inference_with_openai_api(main_image_path, side_image_path, user_prompt)
        pred_json = extract_answer(response)
        local_metrics = evaluate_prediction(gt_answer, gt_boxes, gt_answer_text, pred_json)
        if USE_API_EVAL:
            api_metrics = evaluate_prediction_api(question, gt_answer, gt_answer_text, gt_boxes, response)
        else:
            api_metrics = local_metrics  
        return {"task": task_name, "local": local_metrics, "api": api_metrics}
    except Exception as e:
        print(f"❌ 样本出错: {e}")
        return None



def main():
    all_json_files = []
    for task_name in sorted(os.listdir(BENCHMARK_DIR)):
        task_path = os.path.join(BENCHMARK_DIR, task_name)
        if os.path.isdir(task_path) and task_name.startswith("task_"):
            for f in os.listdir(task_path):
                if f.endswith(".json"):
                    all_json_files.append((task_name, os.path.join(task_path, f)))

    print(f"共发现 {len(all_json_files)} 个任务文件，开始多线程评测...")

    results_all = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for task_name, json_path in all_json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                futures.append(executor.submit(process_sample, task_name, item))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            r = future.result()
            if r: results_all.append(r)

    # 聚合
    stats_local = defaultdict(lambda: {"acc": [], "f1": [], "miou": []})
    stats_api = defaultdict(lambda: {"acc": [], "f1": [], "miou": []})

    for r in results_all:
        t = r["task"]
        for k in ["acc", "f1", "miou"]:
            stats_local[t][k].append(r["local"][k])
            stats_api[t][k].append(r["api"][k])

    print("\n====== Benchmark Summary ======")
    total_local = {"acc": [], "f1": [], "miou": []}
    total_api = {"acc": [], "f1": [], "miou": []}

    for task, vals in stats_local.items():
        acc_l, f1_l, miou_l = map(np.mean, (vals["acc"], vals["f1"], vals["miou"]))
        acc_a, f1_a, miou_a = map(np.mean, (stats_api[task]["acc"], stats_api[task]["f1"], stats_api[task]["miou"]))
        print(f"{task:<35} | Local (Acc/F1/mIoU): {acc_l:.3f}/{f1_l:.3f}/{miou_l:.3f} "
              f"| API (Acc/F1/mIoU): {acc_a:.3f}/{f1_a:.3f}/{miou_a:.3f}")
        for k in ["acc", "f1", "miou"]:
            total_local[k].extend(vals[k])
            total_api[k].extend(stats_api[task][k])
    if not USE_API_EVAL:
        print(" 使用的本地推理，即api指标和本地指标相同")
    print("=========================================")
    print(f"Overall Local: Acc={np.mean(total_local['acc']):.3f}, F1={np.mean(total_local['f1']):.3f}, mIoU={np.mean(total_local['miou']):.3f}")
    print(f"Overall API:   Acc={np.mean(total_api['acc']):.3f}, F1={np.mean(total_api['f1']):.3f}, mIoU={np.mean(total_api['miou']):.3f}")
    print("=========================================")

if __name__ == "__main__":
    main()
