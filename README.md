# DeCert

**Repo for the paper *"Judging the Justification: Decision Certificate Supervision for Generative Reward Models"***


## 🧠 What is DeCert?

**DeCert** (**De**cision-**Cert**ificate Supervision) treats the rationale of a pairwise generative judge as a *decision certificate*: a comparative argument that should justify its verdict. Without any reference explanation, each certificate is scored along three dimensions:

- **Soundness**: are the observations grounded in the responses, and do the criterion-level conclusions follow from them?
- **Coverage**: does the certificate address the *material issues* that an independent analyzer extracts from the input without seeing the judgment?
- **Decision Support**: do the certificate's own criterion-level assessments support its final verdict?

Certificate quality is combined with verdict correctness to select SFT targets and build DPO preference pairs. When every sampled judgment for an input has the correct verdict, large differences in certificate quality still give **rationale preference pairs**. These pairs supervise *how* a correct decision is justified, using data that verdict-only training would discard.


## 🤗 Trained Models

The trained **DeCert-Llama-3.1-8B** judge is available on
[Hugging Face](https://huggingface.co/gqbond/DeCert-Llama-3.1-8B).


## 📂 Repository Structure

```text
DeCert-code/
├── decert/                     # core library
│   ├── data.py                 #   HelpSteer3 rows -> PairExample (gold label, deterministic A/B swap)
│   ├── rubrics.py              #   domain evaluation policies (criteria + weights)
│   ├── prompts.py              #   judge / analyzer / grounding / validity / matcher prompts
│   ├── parsing.py              #   certificate parser (never raises)
│   ├── llm.py                  #   OpenAI-compatible client + test double
│   ├── verify.py               #   LLM verifiers with pessimistic fallbacks
│   ├── scoring.py              #   S_snd, S_cov, S_dec, Q_proc, Q_total (pure math)
│   ├── pipeline.py             #   score one group of judgments end to end
│   ├── batch_score.py          #   Batch-API version of the scoring stage
│   └── pairs.py                #   SFT targets and Max-Min DPO pairs
├── data/README.md              # HelpSteer3 source and expected input fields
├── rollout/                    # Step 1: sample judgments
├── scoring/                    # Step 2: certificate scoring
├── construct/                  # Step 3: SFT / DPO data construction
├── tools/submit_openai_batch.py
├── tests/                      # offline unit tests (no API calls)
├── run_rollout.sh
├── run_score.sh
└── run_construct.sh
```


## 🛠️ Setup

Python ≥ 3.9.

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...       
```

The source data is the public
[NVIDIA HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3).

For local model rollout and verification models, install **vLLM** by following the official instructions:
🔗 [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)


## 📥 Training Data Source

The paper uses 12,000 samples selected from the
[`preference/train`](https://huggingface.co/datasets/nvidia/HelpSteer3/viewer/preference/train)
split of NVIDIA HelpSteer3.

Export the samples you want to process as JSONL with the fields listed in
[`data/README.md`](data/README.md), then pass that file through `DATA`. Use
`train` as `SPLIT_NAME` for the paper subset so every stage constructs the
same example IDs, such as `train:0`.


## 🔄 Step 1: Rollout

The judge writes a structured **decision certificate** under a domain-specific evaluation policy (`decert/rubrics.py`):

```text
<comparison>
<criterion name="correctness">
<obs side="A">A concrete, verifiable observation about Response A.</obs>
<obs side="B">Same for Response B.</obs>
<result winner="A">Why these observations decide this criterion.</result>
</criterion>
...
</comparison>
<decision>How the criterion results combine under the policy.</decision>
<answer>[[A]]</answer>
```

### ▶️ Run

Start the Qwen3-8B server, then run the rollout for one split:

```bash
vllm serve Qwen/Qwen3-8B --port 8000

DATA=/path/to/helpsteer3_12000.jsonl SPLIT_NAME=train \
OUT_DIR=rollouts TAG=train \
bash run_rollout.sh
```

The released pipeline samples GPT-4o three times and Qwen3-8B five times for
each input. Set `LIMIT=2` for a quick trial.

### ▶️ OpenAI Batch API (optional, about half the cost for GPT-4o)

```bash
DATA=/path/to/helpsteer3_12000.jsonl
python3 rollout/build_openai_batch.py --input "$DATA" --split-name train \
    --n 3 --out-dir batches/train_rollout
python3 tools/submit_openai_batch.py --requests batches/train_rollout/batch_*[0-9].jsonl
python3 rollout/merge_openai_batch.py \
    --batch-output batches/train_rollout/*_output.jsonl \
    --expected-n 3 --expected-examples 12000 \
    --out rollouts/gpt4o_train.jsonl
```

📄 Output (one line per example; abridged):

```json
{"example_id": "train:0", "model": "gpt-4o", "generation": ["<comparison>..."], "verdict": ["A", "B", "A"]}
```


## 🧮 Step 2: Certificate Scoring

### 🔹 Scores (`decert/scoring.py`)

A verifier LLM labels each observation's **grounding** *g* and each criterion conclusion's **inference validity** *ℓ<sub>k</sub>* as yes / partial / no (1 / 0.5 / 0). The structured certificate induces a claim graph: observations are leaves and each criterion conclusion is their child. Its support-chain reliability *q<sub>j</sub>* is the minimum local validity on the claim and its ancestors. A separate **analyzer** extracts up to six material issues *D\** with importance *ρ<sub>t</sub>* ∈ {1, 2, 3}. It sees only the input, never the judgment. A **matcher** scores every issue--claim pair (*m<sub>tj</sub>* ∈ {0, 0.5, 1}); a local solver then computes the maximum-weight partial one-to-one matching.

| Score | Definition |
| --- | --- |
| *S*<sub>snd</sub> | ∏<sub>k</sub> *s*<sub>k</sub><sup>α<sub>k</sub></sup>, where *s*<sub>k</sub> is the minimum support-chain reliability among the claims for criterion *k* |
| *S*<sub>cov</sub> | max<sub>*P*</sub> Σ<sub>*t,j*</sub> ρ<sub>t</sub> *P*<sub>tj</sub> *m*<sub>tj</sub> *q*<sub>j</sub> / Σ<sub>t</sub> ρ<sub>t</sub>, over partial one-to-one matchings |
| *S*<sub>dec</sub> | (1 + *B*<sub>r</sub>(*v*)) / 2, where *B*<sub>r</sub>(*v*) is the closed-form worst-case policy margin over verified criterion intervals |
| *Q*<sub>proc</sub> | (S<sub>snd</sub> · S<sub>cov</sub> · S<sub>dec</sub>)<sup>1/3</sup> |
| *Q*<sub>total</sub> | (O + λ Q<sub>proc</sub>) / (1 + λ) with O = 1[verdict correct] and λ = 0.5. Any correct verdict outranks any incorrect one. |

Unparseable judgments score zero and are excluded from data construction. Missing or malformed grounding, validity, and matcher outputs fall back to their most pessimistic values. If the analyzer returns an empty or malformed material-issue set, the entire example is marked as discarded and excluded from both SFT and DPO construction.

### ▶️ Run

Start Qwen3-14B for certificate verification (you can change it to any other model you want):

```bash
vllm serve Qwen/Qwen3-14B --port 8000

DATA=/path/to/helpsteer3_12000.jsonl SPLIT_NAME=train \
ROLLOUTS="rollouts/gpt4o_train.jsonl rollouts/qwen3_train.jsonl" \
OUT=scored/train.jsonl \
bash run_score.sh
```

We use GPT-4o as the analyzer, and Qwen3-14B as the matcher and the grounding and validity verifiers (the script defaults). Pass `--data` and `--split-name` exactly as in the rollout stage.

### ▶️ OpenAI Batch API (optional)

The matcher needs the analyzer's issues, so batch scoring runs in two phases followed by local assembly. Pass the same `--rollouts` files, in the same order, to all three sub-commands:

```bash
DATA=/path/to/helpsteer3_12000.jsonl
ROLL="rollouts/gpt4o_train.jsonl rollouts/qwen3_train.jsonl"
python3 scoring/score_batch.py phase1 --data "$DATA" --split-name train \
    --rollouts $ROLL --verifier-model gpt-4o --out-dir batches/score_train
python3 tools/submit_openai_batch.py --requests batches/score_train/phase1_*[0-9].jsonl
python3 scoring/score_batch.py phase2 --data "$DATA" --split-name train \
    --rollouts $ROLL --verifier-model gpt-4o-mini --out-dir batches/score_train \
    --phase1-output batches/score_train/phase1_*_output.jsonl
python3 tools/submit_openai_batch.py --requests batches/score_train/phase2_*[0-9].jsonl
python3 scoring/score_batch.py assemble --data "$DATA" --split-name train --rollouts $ROLL \
    --phase1-output batches/score_train/phase1_*_output.jsonl \
    --phase2-output batches/score_train/phase2_*_output.jsonl \
    --out scored/train.jsonl
```

### 🔹 Local verifiers

Any OpenAI-compatible server can stand in for each role. The analyzer uses `--analyzer-model` and `--analyzer-base-url`; grounding and validity use `--verifier-model` and `--verifier-base-url`; the matcher uses `--matcher-model` and `--matcher-base-url`. Thinking is disabled automatically when using reasoning model (`--[no-]disable-thinking`).

📄 Output (one line per example; abridged):

```json
{"example_id": "train:0", "domain": "multilingual", "gold": "A",
 "issues": [{"description": "...", "criterion": "instruction_following", "importance": 3.0, "favors": "B"}],
 "judgments": [{"model": "gpt-4o", "text": "<comparison>...", "parse_ok": true, "outcome": 1,
                "s_snd": 0.83, "s_cov": 0.67, "s_dec": 1.0, "q_proc": 0.82, "q_total": 0.94,
                "detail": {"grounding": {...}, "validity": {...}, "claim_q": {...},
                           "match_scores": [[...]], "matches": [...],
                           "criterion_intervals": {...}, ...}}],
 "verifier": {"analyzer_model": "gpt-4o", "verifier_model": "Qwen/Qwen3-14B",
              "matcher_model": "Qwen/Qwen3-14B", "lambda": 0.5,
              "local_validity_threshold": 1.0, ...}}
```


## 📚 Step 3: Construct SFT and DPO Data

```bash
bash run_construct.sh            # DELTA=0.25 by default
```

For each scored group (all judgments for one input):

* **SFT target**: the verdict-correct judgment with the highest *Q*<sub>proc</sub>. Groups with no correct judgment are skipped.
* **DPO pair (Max-Min)**: chosen = argmax *Q*<sub>total</sub>, rejected = argmin *Q*<sub>total</sub>.
  * mixed groups → **outcome pairs** (correct vs. incorrect verdict), always kept;
  * all-correct groups → **rationale preference pairs**, kept only if *Q*<sub>proc</sub>(chosen) − *Q*<sub>proc</sub>(rejected) > δ;
  * all-incorrect groups → discarded.

📂 Outputs in `training_data/`:

| File | Content |
| --- | --- |
| `sft_all.jsonl` | SFT targets |
| `dpo_mixed.jsonl` | outcome pairs only (outcome-only variant) |
| `dpo_same_verdict_d025.jsonl` | rationale preference pairs |
| `dpo_all_d025.jsonl` | outcome + rationale pairs (full DeCert) |

Records use conversational prompts:

```json
// SFT
{"example_id": "...", "prompt": [{"role": "system", ...}, {"role": "user", ...}],
 "target": "<comparison>...", "target_model": "gpt-4o", "target_scores": {...}}
// DPO
{"example_id": "...", "pair_type": "maxmin", "prompt": [...],
 "chosen": "<comparison>...", "rejected": "<comparison>...",
 "chosen_scores": {...}, "rejected_scores": {...}}
```

**Aggregation ablation.** `AGGREGATION=arithmetic bash run_construct.sh` ranks candidates by the arithmetic mean of the three scores instead of the geometric mean. Each record stores `process_aggregation` together with the selected `process_score`.


## 🏋️ Training

We train in two stages: SFT on `sft_all.jsonl`, then DPO from the SFT checkpoint on `dpo_all_d025.jsonl`. These files work with standard prompt-completion and preference trainers such as [TRL](https://github.com/huggingface/trl) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).


## 🧪 Tests

```bash
python3 -m pytest tests -q
```

The tests run offline. Every LLM call goes through a fake client.


## 📜 License

The code is released under the [MIT License](LICENSE). HelpSteer3 is obtained
from its upstream repository under the CC-BY-4.0 license.


## 🙏 Acknowledgements

This work builds on [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) (NVIDIA, CC-BY-4.0).


## 📖 Citation

If you use DeCert, the code, or the trained models in your research, please cite our paper:

```bibtex
@article{decert2026,
  title  = {What Makes a Judgment Justified? Reliable Generative Reward Modeling via Decision Certificate Supervision},
  author = {Sun, Guiquan and Bai, Bing and Min, Martin Renqiang and Song, Dongjin},
  year   = {2026}
}
```
