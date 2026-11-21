I've built the training-data and benchmark inventory. The most useful finding is that the raw text behind our Nemotron vectors is public: it is 62,451,070 Q&A pairs plus their source documents, keyed on the same `(doc_id, pair_idx)` our vector store uses. The biggest risks are licensing (several key sources can't be used commercially, or their redistribution is questionable) and contamination: our planned training data overlaps with source datasets used in the S1-Arena, Jev Decision Index and OpenDecisions test sets.

I made only read-only GET requests and did not write any files.

---

## Key findings

1. **Nemotron text is available.**
   - [jackyk02/Nemotron-CC-2.1-HQA-Doc-Negatives](https://huggingface.co/datasets/jackyk02/Nemotron-CC-2.1-HQA-Doc-Negatives) is raw text, 18.8 GB, ungated. It holds 7,806,965 documents, 62,451,070 Q&A pairs and 595,312,507 Gemini hard negatives.
   - Each row is one document: the prose (`context`) plus nested `qa_pairs` with `question`, `answer` and `hard_negatives`.
   - Our store [CLM-v0.1-Pretrain-Nemotron](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron) has meta parquet columns `doc_id` and `pair_idx` only. It has 62,451,070 rows and about 1.02 TB of `.npy` files, with **no README or license**. So it joins 1:1 with the text.
   - A tokenized twin exists: [nemotron-cc-v2.1-hq-dqa-qwen3-tokens](https://huggingface.co/datasets/jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens), 22 GB, with configs `docs`, `qa_pairs` and `neg_pairs`.
   - Its token counts: context 6.20B, questions 1.02B, answers 0.63B. Context length is p1 254, median 764, p99 1,935 tokens.
2. **Upstream source and license.**
   - The source is [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1), `High-Quality-DQA` config. It is gated (manual approval) under the [NVIDIA Data Agreement for Model Training](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md).
   - That agreement allows use "solely for the purpose of internal training of Company AI Solutions."
   - It forbids "distribute… or otherwise make available to others the Datasets."
   - The jackyk02 derivatives are ungated and tagged "nvidia-open-data-license". That looks inconsistent with §2.2.2 **(inference)**. Recommendation: request our own gated access to the NVIDIA dataset, and treat the mirrors as a convenience copy.
3. **Hard negatives are Gemini outputs.**
   - They were generated with `gemini-2.5-flash-lite`, up to 10 per pair, covering about 95.5% of pairs ([card](https://huggingface.co/datasets/jackyk02/Nemotron-CC-2.1-HQA-Doc-Negatives)).
   - The card warns that about 1 in 20 near-binary questions has a "negative" that paraphrases the true answer.
   - Google's API terms may restrict using outputs to build competing models. Check before commercial use **(inference; terms not fetched)**.
   - Our negative store [nemotron_dqa_qwen3_8b_neg_embeddings](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings) covers only shard 0, chunks 0000–0029.
   - That is 3M pairs, 26,696,585 negatives and about 229 GB, as float16 arrays of shape [100000, 10, 4096] per chunk.
   - The full set would be about 596M negatives and about 4.9 TB.
4. **jackyk02 has also embedded much of the CLM-style post-training pool with Qwen3.6-35B-A3B-FP8** ([listing](https://huggingface.co/api/datasets?author=jackyk02)). This covers ADP, endless-terminals, litecoder-terminal-sft, nemotron-sft-swe-v3, When2Call, helpsteer3, embed-nemotron and more. These are a different backbone, so they are not usable for Qwen3-8B heads. They do show that the next CLM is on that backbone **(inference)**.

---

## Training data by System One model

| Model | Stage → data (source) |
|---|---|
| **CLM-v0.1-8B** ([repo](https://github.com/Contrastive-LM/CLM)) | **Pre:** about 60M Nemotron DQA Q&A pairs (question = state, answer = action). **Mid:** about 30M synthetic hard negatives from Gemini 2.5 Flash-Lite. **Post:** about 1M ADP trajectories plus Endless-Terminals and LiteCoder-Terminal-SFT; the mix is 60% agentic and 40% Nemotron replay. **Task heads:** DeepSWE PRM, trained on [deepswe-clm-train-embeddings-8k](https://huggingface.co/datasets/Contrastive-LM/deepswe-clm-train-embeddings-8k) (405,919 steps, 4,701 trajectories, 113 tasks, MIT, embeddings only). Typed decisions use [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions). |
| **Kev-4B** ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)) | **decision-v7**, 12,576 records: 1,000 each from banking77, BoolQ, AG News, MultiNLI, SST-5, Yelp full, TREC, DBpedia-14, Amazon reviews (en) and IMDB, plus 896 policy minimal pairs and 1,680 rule-structure records with code labels. **Dates and missing evidence:** 1,425, generated. **documents-v1:** 5,219 CFPB complaints ([davidheineman/consumer-finance-complaints-large](https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large)); labels kept only where two open-weight teachers agreed with the filing. **hard-v1:** 6,000, programmatic. **devtools-v1:** 5,320 from CodeReviewer, CommitPackFT, FlakeFlagger and Aegis 2.0. **Kev-27B:** a 145,840-record corpus ([repo](https://github.com/jaredpalmer/kev)). **Public?** Partly. [jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) (211 MB, no card or license) contains train/dev/calibration/test JSONL for decision-v1 to v8, transfer-v1 to v9, hard-v1 (train only), documents-v1 (train only, which is the CFPB set), public-pool-v4 to v6, SemIf and scienthoon. I did not find devtools-v1 or breadth-v1 in the mirror. |
| **Jeff-1.1** ([arXiv](https://arxiv.org/abs/2610.17205)) | [DecisionBank-260k](https://huggingface.co/datasets/jgeuter/DecisionBank), LoRA, about 83% of one epoch. Details in Table 1. |
| **decider** (Mapika; the source of DecisionBank's largest part) ([repo](https://github.com/Mapika/decider)) | About 95 public datasets plus labels from a local Qwen3.5-27B teacher. One epoch is 1.47M examples and 455M tokens. The repo says "nothing was distilled from Jev." The built mixture is not published as a dataset; it is rebuilt by code. |
| **Laya** ([HF](https://huggingface.co/convaiinnovations/laya)) | **Not found.** No dataset is named. A Kaggle notebook builds the data; the teacher is unnamed. The fine-tuned checkpoint was trained on the typed-decisions train split. |
| **Decision 2.0** ([card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)) | **Not found.** The card says only "Training data audited at row level against all Index test items." The vllm-sr HF datasets contain Decision Index results and router datasets, not the decision training set. |
| **Bongard** ([arXiv HTML](https://arxiv.org/html/2609.39111v1)) | Stage 1 uses program-exact synthesis, retrieval judgments filtered by a teacher reranker, and structured lookups. Stage 2 uses JEPA pairs derived from stage 1. Stage 3 uses sandbox rollouts: games, grid worlds, business simulators and computer use, with outcome distributions from rollout frequencies or exact oracles (DP, endgame solvers). **No training data was released.** Weights are at [AgentBull/bongard-mini](https://huggingface.co/AgentBull/bongard-mini) under Gemma terms. |
| **SanSi** ([repo](https://github.com/minnesotanlp/Sansi)) | Main set: 12,800 train, 2,471 dev and 10,027 test, from 59 sources built by `data.build_main`. The sources are not named on the page. Depth tasks are synthetic (liar chains, object swaps). I found no built-data release. Code is Apache-2.0. |
| **Jeva** | **Not found** (no data or weights). |

---

## Table 1: Datasets

Abbreviations: **NC** = non-commercial; **CC**, **CC0**, **CC-BY**, **CC-BY-SA** = Creative Commons licenses; **S1A** = S1-Arena; **JDI** = Jev Decision Index; **LLM** = large language model.

| Name | Link | Rows / size | License (commercial?) | Format | Content | Label source | Used by | Contamination notes |
|---|---|---|---|---|---|---|---|---|
| Nemotron-CC-v2.1 HQ-DQA | [nvidia](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | 7.8M docs → 62.45M QA; about 7.86B tokens | NVIDIA Data Agreement: internal training only, no redistribution. Training a commercial model is not explicitly barred **(inference)** | text | Web doc plus synthetic Q&A | NVIDIA synthetic generation (LLM) | CLM pre | Web crawl, so it may contain benchmark text. Run n-gram decontamination. |
| HQA-Doc-Negatives (text) | [jackyk02](https://huggingface.co/datasets/jackyk02/Nemotron-CC-2.1-HQA-Doc-Negatives) | 62.45M QA, 595M negatives, 18.8 GB | Inherits NVIDIA terms; ungated mirror | text | QA plus 10 hard negatives | Gemini-2.5-flash-lite (negatives) | CLM mid | About 5% of binary-question negatives are paraphrases of the answer (false negatives) |
| CLM-v0.1-Pretrain-Nemotron | [Contrastive-LM](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron) | 62,451,070 pairs, 1.02 TB | None stated | Qwen3-8B last-token vectors, 4096-d fp16 | q and a vectors | as above | CLM, us | Same as above. Identical storage to [nemotron_dqa_qwen3_8b_embeddings](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_embeddings). |
| nemotron_dqa_qwen3_8b_neg_embeddings | [jackyk02](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings) | 3M pairs, 26.7M negatives, about 229 GB | "nvidia-open-data-license" | vectors | negatives | Gemini | us | Shard 0 only |
| embed-nemotron-dataset-v1 | [nvidia](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1) | 12.29M rows, 2.3 GB | CC-BY-SA-4.0 / CC-BY-4.0 / other, per config | text | Retrieval queries plus corpus (MS MARCO, NQ, HotpotQA, FEVER, SQuAD, SciFact, MIRACL, PAQ, synthetic classification) | Public labels plus synthetic | llama-embed-nemotron-8b | Overlaps the S1A sources hotpotqa, nq_open, ms_marco_qna, scifact and squad_v2, with different splits but the same passages |
| DecisionBank | [jgeuter](https://huggingface.co/datasets/jgeuter/DecisionBank) | 625k config (279M tokens) / 260k config (108M tokens) | **Mixed; the card says "Do not use commercially."** Own parts are CC-BY-SA-4.0; a `source_license` field allows filtering | text JSONL | Typed decisions: choice 319k, noul 171k, score 134k; 22 languages; soft targets | 50/50 source label + Qwen3.8-27B (thinking, 2 samples); code labels for programmatic rows | Jeff-1.1 | `decider` part includes CLINC, banking77, MNLI, BoolQ, ARC, MMLU aux, HellaSwag, PIQA, MS MARCO, MASSIVE, SST-5, COPA. `human_soft` includes Civil Comments and ChaosNLI, both of which are **S1A test sources**. `j11_generator` rows are **template siblings of OpenDecisions**. The card states no OpenDecisions or JevBench items and no Jev outputs. |
| kev-suites | [jaredpalmer](https://huggingface.co/datasets/jaredpalmer/kev-suites) | 211 MB | No license on the card; per-source manifests | text JSONL | Typed decisions; train/dev/calibration/test splits | Native dataset labels, code, two-teacher agreement (CFPB) | Kev | decision-v7 sources banking77, SST-5 and DBpedia also appear in S1A; CodeReviewer and conventional commits appear in S1A `code/commit_classification` |
| typed-decisions | [LocalLLaMA](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,600 cases (400 test, 2,000 decisions), 5 MB | Apache-2.0 | text | 4 workflows × 5 typed questions; distribution targets | Synthetic, from a teacher | CLM fine-tune, Laya | Train on the train split only |
| ADP v2 | [neulab](https://huggingface.co/datasets/neulab/adp-v2) | 52 configs; 17.68M std rows; 1.78 TB | "other", per source | text JSONL (ADP/ATIF, OpenHands SFT 24k/110k) | Agent trajectories (SWE, web, terminal, tools) | Teacher-agent rollouts | CLM post | `nebius_SWE-rebench-openhands` and `nebius_SWE-agent-trajectories` share tasks or runs with S1A `code/patch_correctness` (swe_rebench, swe_bench_extra). `openthoughts_tb_dev` relates to Terminal-Bench. `webarena`/`mind2web` relate to S1A web environments. `swe-gym`/`swe-smith` relate to SWE-bench repos. |
| Endless-Terminals | [obiwan96](https://huggingface.co/datasets/obiwan96/endless-terminals) | About 2,500 environments (HF shows 0 bytes stored) | MIT | env specs | Terminal tasks | Synthetic | CLM post | Terminal-Bench style |
| LiteCoder-Terminal-SFT | [Lite-Coder](https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT) | 11,255 trajectories, 2.3 GB | MIT | text | Terminal trajectories | Teacher agent | CLM post | Terminal-Bench |
| DeepSWE PRM embeddings | [train](https://huggingface.co/datasets/Contrastive-LM/deepswe-clm-train-embeddings-8k) / [eval](https://huggingface.co/datasets/tarsur385/deepswe-prm-embeddings-8k) | 405,919 / 44,409 steps; 113 tasks | MIT | vectors only | step (state, action, reward) | Rollout outcome | CLM DeepSWE head | **Train and eval use the same 113 tasks**; they are separated only by task folds. Text not included. |
| R2E-Gym-Subset (DeepSWE envs) | [R2E-Gym](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | not counted | Apache-2.0 | text + docker | SWE tasks | tests | DeepSWE | — |
| SWE-Gym / SWE-smith | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) / [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) | 2,438 / 59,136 | MIT / MIT | text | SWE tasks | tests | ADP sources | Overlaps SWE-bench repos |
| MMLU / MMLU-Pro | [cais](https://huggingface.co/datasets/cais/mmlu) / [TIGER-Lab](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 231,400 rows across all HF configs (includes the aux train set) / 12,102 | MIT / MIT | text | MCQ | human | DecisionBank (aux train) | JDI uses MMLU and MMLU-Pro; S1A uses MMLU-Pro. Train only on aux train. |
| Banking77 / CLINC150 | [PolyAI](https://huggingface.co/datasets/PolyAI/banking77) / [clinc](https://huggingface.co/datasets/clinc/clinc_oos) | 13,083 / 59,275 | CC-BY-4.0 / CC-BY-3.0 | text | intent | human | Kev, DecisionBank | Both are S1A and JDI test sources. Exclude their test items. |
| SNLI / MNLI | [snli](https://huggingface.co/datasets/stanfordnlp/snli) / [multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | 570k / 412k | CC-BY-SA-4.0 / CC-BY-3.0 + mixed | text | NLI | human | Kev, DecisionBank | ChaosNLI (an S1A test source) reuses SNLI/MNLI dev items |
| BoolQ, CommonsenseQA, ARC, HellaSwag | [boolq](https://huggingface.co/datasets/google/boolq), [csqa](https://huggingface.co/datasets/tau/commonsense_qa), [arc](https://huggingface.co/datasets/allenai/ai2_arc), [hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | 12.7k, 12.1k, 7.8k, 60k | CC-BY-SA-3.0, MIT, CC-BY-SA-4.0, none on card | text | QA/MCQ | human | Kev, DecisionBank | CSQA val is in S1A. ARC, HellaSwag and CSQA are in JDI. RouterBench in S1A uses MMLU, HellaSwag and ARC prompts. |
| MS MARCO / NQ hard negatives | [msmarco-hard-negatives](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives) (12.8 GB), [natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions) (100k) | — | No license on the cards. DecisionBank lists MS MARCO as **research use only** | text | Retrieval negatives | mined | — | S1A uses MS MARCO dev and TREC DL |
| BFCL | [gorilla](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | — | Apache-2.0 | text | tool calls | human/exec | CLM eval | BFCL v4 cases are in S1A `route/tool_selection` and in JDI |
| When2Call | [nvidia](https://huggingface.co/datasets/nvidia/When2Call) | 112 MB | CC-BY-4.0 | text | tool or no-tool decision | — | Kev evaluation only | In JDI |
| ToolBench | (GitHub OpenBMB/ToolBench) | not verified | not verified | — | — | — | — | JDI uses ToolRet |
| Bongard sandbox rollouts | — | not released | — | — | outcome distributions | oracle or rollout | Bongard | — |
| Laya / Decision 2.0 / Jeva training data | — | **not found** | — | — | — | — | — | — |

---

## Table 2: Benchmarks

| Name | Link | Size | License | Harness? | Fits a bi-encoder? | Notes |
|---|---|---|---|---|---|---|
| **S1-Arena** | [HF](https://huggingface.co/datasets/C-Tianyu/S1-Arena), [code](https://github.com/TianyuCodings/S1-Arena) | 49,592 requests, about 92,125 questions; 254 sources; 16 domains; 138 closed-loop environments | Code Apache-2.0; generated requests CC-BY-4.0; per-source otherwise. 14 sources must be rebuilt locally (`s1arena build --local`) | Yes. Adapter module plus `s1arena check-model`. Reports accuracy, Brier, ECE (expected calibration error), p50 latency. Refused or over-length inputs score 0 | Yes for the static set. Each question has `criteria` text per option, so we can encode the state (+ instructions) and each criterion separately. Boolean questions need a yes/no pair; score questions need level descriptions. Closed-loop environments need the task harness | **No train or dev split** (test only). The paper notes possible exposure of public sources. Many sources are test splits of datasets that appear in DecisionBank, Kev, ADP or embed-nemotron. |
| **OpenDecisions** | [HF](https://huggingface.co/datasets/jgeuter/OpenDecisions) | 2,352 items: 1,178 choice, 586 yes/no, 588 score; 599 code-labeled, 1,443 author+panel, 310 panel-only | CC-BY-SA-4.0; "do not use for training" | `score.py` and `predict_letters.py` | Yes. `criteria` gives a description per option | DecisionBank's `j11_generator` uses the same generators with a new seed. Fit temperatures elsewhere. |
| **JevBench** | [GitHub](https://github.com/fstandhartinger/jevbench), [overview](https://parallel.ai/articles/jevbench-benchmark.md) | v1.8: 1,500 items (300 public / 1,200 sealed) per Parallel. The repo has 72 original + 111 hard + 48 easy public items | MIT (harness and 72 original items) | Yes. `--adapter typesafe --endpoint <our /v1/systemone>` | Yes, through a `/v1/systemone` wrapper | Training on the public split "must be disclosed." Sealed scores require submission. |
| **Jev Decision Index** | [Space](https://huggingface.co/spaces/multimodalart/jev-decision-index), [methodology.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json) | v0.4.1: 110,201 requests, 37 static benchmarks; areas are knowledge, language, retrieval, tools, arts, games | Per source; the methodology carries a licence caveat | An "official kit" exists (the Decision 2.0 team ran the 0.2.1 kit). **I did not find a download link for the request corpus**; the Space ships results JSON | Mostly. Benchmarks include MMLU/-Pro, ARC, HellaSwag, WinoGrande, CSQA, BANKING77, CLINC150, ANLI, BFCL, When2Call, ToolRet, BRIGHT, ESCI, GSM8K, GPQA, HLE and games | Since v0.4 the headline score uses private tests. CLM-v0.1-8B is already listed. |
| **TypedBench** | [arXiv](https://arxiv.org/abs/2610.11392) | 7 generators + 9 suites | **not found** | **not found** | — | Data release not found |
| **DecisionBench** (Hanno Labs) | [HF](https://huggingface.co/datasets/Hanno-Labs/decision-bench) | 23,900 rows, 43 task IDs, 28 domains, up to 255 candidates; 56 MB | "other" | Issue tracker on GitHub (Hanno-Labs/decision-bench); a harness was not verified | Yes. Candidates are runtime-defined with descriptions | Eval split only. Bongard scored 78.05%. |
| **typed-decisions** | [HF](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 400 test cases, 2,000 decisions | Apache-2.0 | Leaderboard via HF discussions | Yes | Has a train split |
| **Kev suites** (transfer-v4/v9, decision-vN test) | [HF](https://huggingface.co/datasets/jaredpalmer/kev-suites) | transfer-v4 locked test: 656 questions | unstated | `kev.benchmark` in the [repo](https://github.com/jaredpalmer/kev) | Yes | breadth-v1 (3,089 questions) was not found in the mirror |
| **CLM: DeepSWE** | [eval embeddings](https://huggingface.co/datasets/tarsur385/deepswe-prm-embeddings-8k) | 113 tasks, 449 rollouts (Claude-Opus-5) | MIT | `evaluation/bon_deepswe_eval.py` | Native (step-state, action, reward) | Text not shipped. Re-embedding with another backbone requires re-running rollouts or finding the raw trajectories **(inference)** |
| **CLM: Terminal-Bench 2.1, T-Rex, WikiRacing, Mario, BFCL v4, held-out Nemotron DQA** | [CLM repo](https://github.com/Contrastive-LM/CLM) | TB2.1: 30 held-out tasks; DQA hold-out: about 100k questions × 10 negatives | Apache-2.0 (code) | T-Rex in `examples/t_rex`; others **not linked** | Native | The DQA hold-out must be excluded from our pretraining by `doc_id` |

---

## Recommended curriculum for our model

**Stage 0: Decontamination first (inference).**
- Build one "eval registry" before any training. It should hold the following, normalized (lowercased, whitespace-collapsed) and hashed:
  - S1-Arena (all 16 configs, plus `s1arena build --local` sources);
  - OpenDecisions;
  - the JevBench public items;
  - DecisionBench;
  - typed-decisions test;
  - Kev test/locked splits;
  - the held-out Nemotron DQA set.
- Drop training rows in two passes:
  - exact hash or 13-gram overlap on state, question or option text;
  - MinHash near-duplicates (Jaccard > 0.8).
- Also drop whole **source datasets** that serve as eval tests, unless their train and test splits are verifiably separate. Examples: ChaosNLI, Civil Comments and CLINC/banking77 *test* items. DecisionBank's `source_ref` field makes this possible.
- Drop DecisionBank `j11_generator` rows if we report OpenDecisions. At minimum, report OpenDecisions both with and without its 599 code-labeled items.
- For agent data, exclude ADP configs `nebius_SWE-rebench-openhands-trajectories`, `nebius_SWE-agent-trajectories` and `openthoughts_tb_dev` when reporting S1-Arena code or Terminal-Bench. Split DeepSWE by task, as CLM does.
- Hold out a validation slice of our own: DecisionBank by `group_id`, and Nemotron by `doc_id`. Fit temperatures on it, never on a benchmark.

**Stage 1: Pre (retrieval-style), about 60M pairs.**
- Use the existing Qwen3-8B vectors at zero backbone cost while we stay on Qwen3-8B last layer with last-token pooling.
- Add diversity from embed-nemotron-dataset-v1 (12.3M rows; per-config licenses; drop MS MARCO if we need commercial use).
- Loss: symmetric InfoNCE plus a Matryoshka loss.
- Free hard negatives: the other Q&A pairs from the same document (usually 8 per document) **(inference)**.

**Stage 2: Mid (hard negatives), 3M to 30M pairs.**
- Start from the 3M-pair shard we already have.
- Mine extra negatives **in embedding space**: approximate-nearest-neighbour search over the 62M answer vectors using our stage-1 head. This needs no backbone compute **(inference)**.
- Filter false negatives:
  - drop a candidate if its similarity is more than about 0.95 × the positive's similarity (positive-aware margin, NV-Retriever-style) **(inference)**;
  - drop Gemini negatives that paraphrase the answer, using a lexical-overlap or teacher check.
- To extend to the CLM-scale 30M pairs, either embed more Gemini negatives (cost below) or rely on mined negatives. Mining avoids the Gemini-output terms question.

**Stage 3: Typed decisions, about 250k to 650k rows.**
- Data:
  - DecisionBank-260k, decontaminated; filter by `source_license` for commercial use;
  - Kev suites train splits (decision-v7, hard-v1, documents-v1/CFPB, public-pool);
  - typed-decisions train.
- Bi-encoder format:
  - state side = `state + question.instructions`;
  - option side = option `description`/criteria;
  - target = KL to `target_distribution`, falling back to `label`.
- Use Kev-style augmentation: shuffled options, inserted "none of the above," and distractors.
- Then add the typed path in parallel on the same data: a pointer head or a Jeva-style late-interaction stack.
- Calibrate per type with temperatures fit on our own dev split.

**Stage 4: Agent/post, about 1M trajectories.**
- Data:
  - a decontaminated ADP v2 subset (std split; prioritize SWE, terminal and tool configs);
  - LiteCoder-Terminal-SFT;
  - Endless-Terminals;
  - SWE-smith/SWE-Gym trajectories.
- Mix: 40% Nemotron replay, as CLM does.
- Agent negatives: actions from other rollouts at a similar step, later or earlier actions in the same trajectory, and actions from failed (reward 0) rollouts **(inference)**.
- Then task heads on the DeepSWE and TB data with task-fold holdouts.
- License: ADP is "other", per source, so audit each config.

**Embedding compute if we re-embed with Qwen3-8B (all inference).**

Basis: about 16 GFLOPs per token, roughly 25k tokens/s per H100 at about 40% MFU (model FLOPs utilization). Real-world check: jackyk02 embedded 26.7M negatives (about 270M tokens at about 10 tokens each) in 19,800 s, about 13.5k tokens/s on unstated hardware ([metadata.json](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings/raw/main/metadata.json)).

| Corpus | Tokens | H100-hours | Storage at 4096-d fp16 |
|---|---|---|---|
| Nemotron Q + A | 1.66B | about 20–35 | 1.02 TB |
| All 595M Gemini negatives | about 6B (assumes 10.2 tokens/answer) | about 70–130 (about 123 h at jackyk02's rate) | about 4.9 TB |
| Document contexts, only if states are context-conditioned | 6.2B | about 70–140 | — |
| DecisionBank-625k | about 0.3B | under 5 | — |
| 1M ADP trajectories (assumes 10–20k tokens each) | 10–20B | about 110–220 | — |

- **One-pass trick for trajectories:** with last-token pooling on a causal model, every step's state embedding is the hidden state at that step's last token. So one forward pass over the whole trajectory gives all steps, with cost linear in trajectory length rather than steps × context. This requires dropping per-step left truncation and re-templating, or windowing at 8k.
- **Layer sweeps** (L/2, 3L/4, L) can be pulled from the same pass. That multiplies storage, not compute. Use int8 or PCA to keep it near 1 TB per layer.

---

## Not found

- Training data for Laya, Decision 2.0, Jeva and Bongard (Bongard's data is described in the paper but not released).
- SanSi's source list.
- TypedBench's data or harness.
- The Jev Decision Index request corpus download.
- Kev devtools-v1 and breadth-v1 files.
- A license for kev-suites and for CLM-v0.1-Pretrain-Nemotron.
- An S1-Arena train split (there is none).
- WikiRacing and Mario data links.

Sources: the links inline above, plus [jackyk02 dataset list](https://huggingface.co/api/datasets?author=jackyk02), [AI Weekly on JDI](https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models), [LLMReference Decision Index](https://www.llmreference.com/benchmark/decision-index), [S1-Arena paper](https://arxiv.org/html/2610.21244), [Bongard abstract](https://arxiv.org/abs/2609.39111), [SanSi abstract](https://arxiv.org/abs/2610.07730), [Jeff repo](https://github.com/j-geuter/Jeff).
