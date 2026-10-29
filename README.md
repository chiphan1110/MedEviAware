<h1 align="center">MedEviAware: Evidence-Aware Reinforcement Learning for Clinical Reasoning</h1>

## 🔬 Overview

Clinical diagnosis often involves gathering information through patient interviews, physical examinations, and tests. Many medical question-answering datasets provide all of this information at once, offering little guidance on when a model should ask for more evidence or defer to a clinician. When information is missing, the model needs to consider whether it can obtain more evidence and whether that evidence would help it answer correctly.

This project introduces:

- **An evidence-aware clinical reasoning task** based on MediQ. Each case has several versions, with examination findings and test results either *provided*, *available on request*, or *unavailable*.
- **MedEviAware**, a Qwen3-4B-Instruct model trained with Group Relative Policy Optimization (GRPO) to answer, request additional evidence, or defer to a clinician. The reward for a request reflects its expected benefit, estimated from the base model's accuracy with different combinations of evidence.

## 🧩 Method

**Case construction.** We group the facts in each MediQ case into patient history, physical examination findings, and test results. Patient history is always provided. Examination findings and test results can each be provided, available on request, or unavailable, creating up to nine versions of each case. We exclude two combinations from training: examination findings available on request with test results unavailable, and the reverse. These held-out combinations test whether the model can generalize to evidence availability patterns it has not seen during training.

**Evidence Value Estimation.** Before training, we measure the base model's accuracy with patient history alone, then with examination findings, test results, or both added. If the probability of a correct answer is $p$, the expected reward for answering is $2p - 1$. Deferring receives a reward of $0$. A request is valued according to the best expected reward after the requested findings are revealed, minus a small request cost.

**GRPO training.** For each version of a case, the model generates several responses, each containing reasoning and a single action. Correct answers receive a reward of $+1$, incorrect answers $-1$, and deferrals $0$. Valid requests receive their estimated value as a reward. Requests for findings that are unavailable or already provided are penalized.

**Multi-turn inference.** During evaluation, a valid request reveals the corresponding findings from the case, and the model continues with that information in the next turn. The episode ends when the model answers or defers.

## 🏆 Results

On unseen test cases, MedEviAware achieves higher mean scores for both evidence availability patterns seen during training and the two held-out patterns. It answers more selectively, achieves higher accuracy on the questions it answers, and makes fewer wasted requests.

| Method | Mean score (Seen patterns) ↑ | Mean score (Held-out patterns) ↑ | Answer accuracy ↑ | Deferral rate | Wasted requests ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Always answer | 0.25 | 0.18 | 62% | 0% | — |
| Base model (Qwen3-4B-Instruct) | 0.28 | 0.22 | 68% | 13% | 48% |
| **MedEviAware (ours)** | **0.33** | **0.32** | **70%** | 21% | **6%** |

---

## 📁 Repository Structure

```text
MedEviAware/
├── assets/              # Figures for this README
├── preprocess/          # Data preparation scripts and prompts
├── eval/                # Accuracy scoring and multi-turn episode evaluation
├── script/              # Launch scripts for data preparation, training, and evaluation
├── processed_data/      # Processed cases, labels, episodes, and RL datasets in Parquet format
└── verl/                # RL training framework with the MedEviAware reward function
```

## 🚀 Getting Started

### Environment Setup

```bash
git clone https://github.com/chiphan1110/MedEviAware.git
cd MedEviAware

cp .env.example .env
# Edit .env with your paths (ROOT, BASE_MODEL_DIR, RESULTS_DIR, LOG_DIR) and API keys
```

The project uses two conda environments: `verl` for data preparation and training, and `vllm` for model serving.

```bash
conda create -n verl python=3.10 -y && conda activate verl
pip install -e verl/

conda create -n vllm python=3.10 -y && conda activate vllm
pip install vllm
```

### Data

The task uses cases from the MediQ development set. Processed data is included in `processed_data/`, so you can proceed directly to training. To regenerate the data, run:

```bash
STEPS="1 2 3" bash script/preprocess/build_data.sh       # Tag facts, split cases, and create scoring prompts
bash script/eval/host_model.sh                         # Serve the base model in a separate terminal
MODEL_NAME=qwen3_4b_instruct bash script/eval/score_p.sh # Measure base-model accuracy for each evidence set
STEPS="4 5 6" bash script/preprocess/build_data.sh       # Create labels, episodes, and RL datasets in Parquet format
```

### Training

```bash
PARQUET_DIR=$PWD/processed_data/rl_parquet/qwen3_4b_instruct_filtered \
  bash script/train/run_grpo_medeviaware.sh
```

### Evaluation

```bash
# Convert the checkpoint for vLLM, then start the model server
bash script/eval/make_vllm_ckpt.sh results/checkpoints/medeviaware_rl/<experiment>/global_step_<N>
MODEL_PATH=results/checkpoints/medeviaware_rl/<experiment>/global_step_<N>/actor/huggingface_vllm \
SERVED_MODEL_NAME=medeviaware bash script/eval/host_model.sh

# Run the multi-turn test episodes in a separate terminal
API_MODEL=medeviaware MODEL_NAME=medeviaware bash script/eval/eval_episodes.sh
```

Results are written to `results/eval_episodes/<MODEL_NAME>/test/metrics.json`.

> This project is for research only and is not intended for clinical use.

## 🙏 Acknowledgements

This project was developed as part of the [NTI GStar Bootcamp 2026](https://gstar.newturing.ai/). I sincerely thank the GStar mentors for their guidance, thoughtful feedback, and support throughout the project.

MedEviAware builds on [MediQ](https://github.com/stellalisy/mediQ) for clinical cases, [verl](https://github.com/volcengine/verl) for RL training, [vLLM](https://github.com/vllm-project/vllm) for inference, and [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) as the base model. I am grateful to these teams for open-source their work and making this project possible.
