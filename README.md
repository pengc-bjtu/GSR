# [CVPR 2026] Can a Second-View Image Be a Language? Geometric and Semantic Cross-Modal Reasoning for X-ray Prohibited Item Detection



## 🛠️ Installation

### 1. Environment Setup
We recommend using Conda to manage the environment.

```bash
conda create -n gsr python=3.10
conda activate gsr
```

### 2. Install Dependencies
Please install the required packages using `pip`. We rely on `LlamaFactory` for training and `vLLM` for high-throughput inference.

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have CUDA installed (tested on CUDA 12.1) compatible with PyTorch and vLLM.

---

## 📦 Model Preparation

### Download Pre-trained Weights
Please download the **[Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)** weights (or the base model used in the paper) from Hugging Face or ModelScope.

1.  Download the model weights.
2.  Place them in a directory (e.g., `./models/Qwen3-VL-8B`).

> **📌 Notice:** Our fine-tuned checkpoints (GSR-8B) will be uploaded to HuggingFace shortly. Once released, you can download them directly to replace the base model for evaluation.

---

## 🚀 Training

We utilize the **LLaMA-Factory** framework for efficient fine-tuning.

To start the full-parameter Supervised Fine-Tuning (SFT), run the following script:

```bash
bash ./train/sft_full.sh
```

**Note:** Before running, please ensure the model path and data path inside `sft_full.sh` are correctly pointed to your local directories.

---

## 📊 Evaluation

We employ a **Server-Client** architecture for evaluation to ensure maximum inference throughput. The model is deployed using **vLLM**, and the evaluation script sends requests to the local server.

### Step 1: Deploy the Model (Server Side)
First, launch the vLLM server. You need to point to your **trained checkpoint** path (the output from the training step).

**Command:**
```bash
# Replace '/path/to/your/trained/checkpoint' with your actual output path
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/your/trained/checkpoint \
    --served-model-name qwenvl \
    --gpu-memory-utilization 0.60 \
    --tensor-parallel-size 4 \
    --port 50001
```

*   `--tensor-parallel-size 4`: Distributes the model across 4 GPUs.
*   `--port 50001`: The port used for API communication.

Wait until you see the message: `Uvicorn running on http://0.0.0.0:50001`.

### Step 2: Run Evaluation (Client Side)
Open a new terminal window and run the evaluation script. This script will send images and prompts to the local vLLM server and calculate metrics.

```bash
# Example: Evaluation script
python ./eval/eval.py \
```
**Note:** Before running, please ensure that the data path in `eval.py` points correctly to your local directory.

## 📂 Data and Models (Coming Soon)

Please note that the **DualXrayBench dataset**, the **GSXray training corpus**, and our **fine-tuned model weights (GSR-8B)** are currently undergoing final organization and formatting. 

We will release the full datasets and checkpoints in this repository shortly. Please stay tuned for future updates!