I found **about 40 open implementations the prior study missed**, plus **20 new arXiv papers**. That includes four implementation papers from teams that also ship code: StartLux, AnyJev, Intern-Decision and VideoJev. I also found the CLM team's earlier papers.

**How things were checked:**
- **GitHub:** star counts, license and last push came from `gh api repos/...`.
- **Hugging Face:** downloads, likes and license tags came from the model API.
- **arXiv:** every ID was checked against its abs page (title, date, authors and license read from the page metadata), and every `https://arxiv.org/pdf/<id>` returned HTTP 200.

The best single catalog of the field is [AnotiaWang/awesome-decision-models](https://github.com/AnotiaWang/awesome-decision-models) (724★, CC0, updated 2026-11-13). [andyrewlee/awesome-system-one](https://github.com/andyrewlee/awesome-system-one) is a smaller one (65★).

---

## Table A: Implementations, ranked by prominence

Stars are as of 2026-11-20. "HF dl" means Hugging Face downloads (the API's recent count) and "♥" means HF likes. "Training code?" was checked against each repo's top-level directory listing.

| # | Name | Team | Repo | Weights | License | Stars | Training code? | Prominence notes | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Laya** | Convai Innovations | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (16.0k dl, 5,989♥); laya-multilingual (8.4k dl) | Apache-2.0 | **35,727** | README describes the RLCD training, but I did not confirm a train directory (the listing was truncated) | Most-starred repo in the field. Ports exist for MLX, Core ML, laya.cpp, llama.cpp and Unsloth | Covered (the prior study missed the GitHub repo) |
| 2 | **Kev** | Jared Palmer | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | kev-0.8b, kev-4b, kev-9b, kev-27b (13.0k, 10.6k, 2.6k and 3.6k dl) | Apache-2.0 | 10,069 | Yes (experiments, scripts, runs, Modal) | Pushed 2026-11-19 | Covered |
| 3 | **vLLM-SR / Decision 2.0** | vLLM Semantic Router | [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) | Decision-2.0 Kai to Vega (Kai 1.1k dl, Vega 577 dl) | Apache-2.0 | 6,512 | Not for Decision 2.0 | Router framework; the decision weights live on HF | Covered |
| 4 | **SemIf** (formerly OpenJev) | TheoLeeCJ | [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | none (reads option logits from existing open models) | MIT | 4,970 | No, inference technique only | 68.66 on the BenchLM JevBench list | **NEW** |
| 5 | **CLM** | Stanford (Mirhoseini / Pavone / Ré groups) | [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) | CLM-v0.1-8B (986 dl, 795♥) | Apache-2.0 | 3,212 | Yes (`train/`, `preprocessing/`) | Last push 2026-10-05 | Covered |
| 6 | **NanoJev** | Tianyu Chen (also first author of S1-Arena) | [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) | 0.6B checkpoint | MIT | 2,602 | Yes, end-to-end training pipeline | Same team as S1-Arena (2610.21244) | **NEW** |
| 7 | **Clef / Clef-flash** | Cloudflare | none (HF only) | [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (78.2k dl, 2,380♥); clef-flash (33.4k dl, 933♥) | Apache-2.0 | n/a | No | Most-liked open decision weights; served on Workers AI | **NEW** |
| 8 | **Nimble** | Bespoke Labs | [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) | [Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) (256♥) | Repo: **no license file**; weights: Apache-2.0 | 2,319 | Data-curation code and eval suite | In the Ollama library as `nimble` | **NEW** |
| 9 | **StartLux-Decision** | StartLux Labs | [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) | 0.8B to 27B plus 35B-A3B | Code Apache-2.0; **weights CC BY-NC 4.0** | 1,776 | Partial (has an arXiv tech report) | Self-reported Decision Index 63.88 vs Jev 57.91 | **NEW** |
| 10 | **AnyJev / Tacit** | Nokia Applied Research | [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) | Tacit 1.7B to 9B (self-distilled) | Apache-2.0 | 1,629 | Training-free method plus self-distillation scripts | Order-invariant readout from any LLM | **NEW** |
| 11 | **Ollaya** (runtime) | ollaya-dev | [ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya) | none (serves others' weights) | Apache-2.0 | 1,473 | No | "Ollama for decision models" | **NEW** |
| 12 | **jevlike** | vinnylarouge | [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) | small trained scorer | MIT | 1,371 | Yes | Says it is not a Jev reproduction | **NEW** |
| 13 | **Strands Decider** | AWS Strands Labs | [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) | StrandsAgents HF org: 2B, plus Gemma-4 E2B, E4B, 12B and 26B-A4B | Apache-2.0 | 1,236 | **Yes: weights, data, configs and scripts** | Qwen3.5-2B with a pointer head of about 1M params and LoRA r16. **Closest public recipe to ours** | **NEW** |
| 14 | **decider** | Mapika | [Mapika/decider](https://github.com/Mapika/decider) | 0.8B, 2B, 4B, 35B-A3B, vision | Apache-2.0 | 1,036 | Yes (`teacher_data/`, scripts) | JevBench 71.28 (4B v2) | **NEW** |
| 15 | **jevos** | feder-cr | [feder-cr/jev](https://github.com/feder-cr/jev) | 1B GGUF | MIT | 970 | Export scripts | CPU only, Noul questions only | **NEW** |
| 16 | **Valen-Decision-1.5** | Valen Team | [Valen-Team/Valen](https://github.com/Valen-Team/Valen) | 0.8B to 27B, plus GGUF and ONNX | Apache-2.0 | 932 | Yes (configs, data, eval) | Multimodal; block-bidirectional attention; bbox head | **NEW** |
| 17 | **Von** | wfzyx | [wfzyx/von](https://github.com/wfzyx/von) | 395M option-marker model | Apache-2.0 | 922 (**archived 2026-10-19**) | Yes (`training/`) | Option-marker encoder in the style of Laya | **NEW** |
| 18 | **Intern-Decision** | InternLM (Shanghai AI Lab) | [InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision) | 0.8B, 2B, 4B | Apache-2.0 | 812 | **Yes** (XTuner configs, temperature calibration) | Has an arXiv tech report | **NEW** |
| 19 | **Phocinae-Largha-150M** | Phocinae Project | [repo](https://github.com/Phocinae/Phocinae-Largha-150M-v1) | 144M mmBERT-small | Apache-2.0 | 790 | Calibration and datasets | English/Chinese encoder | **NEW** |
| 20 | **OneJev / Flide** | OmniJev | [OneJev](https://github.com/OmniJev/OneJev) (282★), [Flide](https://github.com/OmniJev/Flide) (546★) | 0.8B to 27B | Apache-2.0 | 282 / 546 | OneJev has `train/` | Flide is an inference engine that forks one state into many questions | **NEW** |
| 21 | **Jeeves** | PostHog | [PostHog/jeeves](https://github.com/PostHog/jeeves) | 9B (Qwen3.5 with LoRA and a pointer head) | MIT | 461 | **Yes** (`train.py`, `calibrate.py`, CISPO, diffusion drafter) | Reports JevBench-231 0.935 vs Jev 0.866 | **NEW** |
| 22 | **VideoJev** | MCG-NJU (Nanjing University) | [MCG-NJU/VideoJev](https://github.com/MCG-NJU/VideoJev) | 4B and 35B-A3B | Apache-2.0 | 317 | Configs and scripts | Video; arXiv paper | **NEW** |
| 23 | **JEV-27B / JEV-9B** | AutoTrust | HF only | [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (356k dl); JEV-9B (**528k dl, the highest of any decision model I checked**) | Apache-2.0 | n/a | No | **Distilled from Jev 1.13 outputs, so check TypeSafe's terms before using it** | **NEW** |
| 24 | **GLiNER2.5-Decide** | Fastino | HF only | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (67.7k dl) | Apache-2.0 | n/a | No | DeBERTa-v3-large; HF tags arXiv 2507.18546 | **NEW** |
| 25 | **Liquid d1** | Liquid AI | HF only | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (20.3k dl); d1-2B (16.2k); d1-Omni-2B (6.1k) | **LFM Open License** (HF tag "other") | n/a | No | d1-3B card tags the LFM2 report (2511.23404) | **NEW** |
| 26 | **pplx-decider v1 / v1.1 27B** | Perplexity | HF only | [v1.1](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) (3.7k dl, 182♥) | Apache-2.0 | n/a | No | v1.1 removes the causal mask; card reports Decision Index 61.56 vs Jev 57.9 | **NEW** |
| 27 | Long tail of community models | various | — | Wald-4B (111k dl), Winnow-12B (14.6k dl; JevBench #2), openjev/openjev (6.9k dl, **CC BY-NC**), Jev-Omni (3.6k dl), Tev1 (Together, 2.1k dl), OpenThai-SystemOne, Standard One 8B, Lev (Interfaze, 30★), JevK5 (169★), reflex (175★), RSI-Jev (140★), jev-style (137★), Vev (33★), WebJev (7★) | mostly Apache-2.0 | <200 | some | Lower prominence | NEW |

**Runtimes and evaluation tools to know about:**
- Runtimes: [mode-io/vllm-jev](https://github.com/mode-io/vllm-jev) (954★), [razorback16/openjev](https://github.com/razorback16/openjev) (797★), [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) (660★), [LiteVar/system-one](https://github.com/LiteVar/system-one) (0★). SGLang, llama.cpp and Ollama also serve `/v1/systemone` natively.
- Evaluation: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (350★, MIT) and [Valen-Team/JevalKit](https://github.com/Valen-Team/JevalKit).

---

## Table B: Publications

Every arXiv row below was verified: the abs page resolved, the metadata was read, and the PDF returned HTTP 200. "Abs" means `https://arxiv.org/abs/<id>` and "PDF" means `https://arxiv.org/pdf/<id>`.

**What the license column means for us:**
- **CC-BY 4.0 / CC-BY-SA 4.0:** we may commit the PDF with attribution (SA: share-alike).
- **arXiv non-exclusive (NE):** do **not** redistribute; link to it instead.
- **CC BY-NC-ND:** redistribution only unmodified and non-commercial.

### B1. Papers from teams that ship code or weights

| System | Title | Authors / org | Date | ID | License |
|---|---|---|---|---|---|
| S1-Arena / NanoJev | System1-Arena: Which Decisions Can Be Left to System 1? | Tianyu Chen, Xinyang Liu, Ruyi An et al. | 2026-10-15 | [2610.21244](https://arxiv.org/abs/2610.21244) · [PDF](https://arxiv.org/pdf/2610.21244) | CC-BY 4.0 |
| Jeff-1.1 | Releasing the Inner Jev: A Simple Recipe for Open Decision Models | Geuter, Cheng, Alvarez-Melis (Harvard) | 2026-10-12 | [2610.17205](https://arxiv.org/abs/2610.17205) · [PDF](https://arxiv.org/pdf/2610.17205) | NE |
| Jeva | Jeva: System One Models from First Principles | Qi Li, Xinchao Wang | 2026-10-13 | [2610.18773](https://arxiv.org/abs/2610.18773) · [PDF](https://arxiv.org/pdf/2610.18773) | CC-BY 4.0 |
| Bongard | Bongard: Training Machine Intuition | Ding, Jin, Ji et al. | 2026-09-30 | [2609.39111](https://arxiv.org/abs/2609.39111) · [PDF](https://arxiv.org/pdf/2609.39111) | CC-BY 4.0 |
| SanSi | SanSi: A Looped Typed Decision Model for System 1.5 Thinking | Gan, Lee, Kang et al. | 2026-10-06 | [2610.07730](https://arxiv.org/abs/2610.07730) · [PDF](https://arxiv.org/pdf/2610.07730) | NE |
| **StartLux** | StartLux-Decision: AutoThink Decision Models with RL for Calibrated Deliberative Decisions | Zhuang, Su, Pan et al. | 2026-10-12 | [2610.17309](https://arxiv.org/abs/2610.17309) · [PDF](https://arxiv.org/pdf/2610.17309) | NE |
| **AnyJev** | AnyJev Technical Report | Jiamu Zhang, Tianze Yang, Yucheng Shi et al. (Nokia) | 2026-09-30 | [2610.00831](https://arxiv.org/abs/2610.00831) · [PDF](https://arxiv.org/pdf/2610.00831) | CC-BY 4.0 |
| **Intern-Decision** | Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling | Tong Zhu, Zhe Xu, Zefeng He et al. | 2026-10-15 | [2610.22131](https://arxiv.org/abs/2610.22131) · [PDF](https://arxiv.org/pdf/2610.22131) | **CC-BY-SA 4.0** |
| **VideoJev** | VideoJev: System One Model for Video Understanding and Interaction | Yuhan Zhu, Xiangyu Zeng, Qingyi Si et al. | 2026-10-13 | [2610.18615](https://arxiv.org/abs/2610.18615) · [PDF](https://arxiv.org/pdf/2610.18615) | CC-BY 4.0 |
| **this-that-model-1.0** (weights: [flock-io/this-that-model-1.0](https://huggingface.co/flock-io/this-that-model-1.0)) | this-that-model-1.0: A typed decision model that decides in 30 ms… | Zehua Cheng, Wei Dai, Jiahao Sun et al. | 2026-09-20 | [2609.23886](https://arxiv.org/abs/2609.23886) · [PDF](https://arxiv.org/pdf/2609.23886) | NE |
| **Beveridge** (repo bev-decisions-pypi, 0★) | Beveridge: Open Weight System One Decision Models with Calibrated Probabilities | Sathiskumar Jothi | 2026-10-17 | [2611.18655](https://arxiv.org/abs/2611.18655) · [PDF](https://arxiv.org/pdf/2611.18655) | CC-BY 4.0 |
| **Chinese-Jev** | Chinese-Jev: Bringing System One Model to Chinese-Language Tasks | Zexiao Wang, Zihao Zhang, Xudong Wang et al. | 2026-09-29 | [2609.36965](https://arxiv.org/abs/2609.36965) · [PDF](https://arxiv.org/pdf/2609.36965) | CC-BY 4.0 |
| **PixelJev** | From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model | Xunlan Zhou, Xianliang Yang, Li Zhao et al. | 2026-09-24 | [2609.29283](https://arxiv.org/abs/2609.29283) · [PDF](https://arxiv.org/pdf/2609.29283) | CC-BY 4.0 |
| **Narrow Decision Models** | Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions | Carlos André Antunes, Aldry Albuquerque, Gabriel de Melo Schmith et al. | 2026-10-18 | [2610.25553](https://arxiv.org/abs/2610.25553) · [PDF](https://arxiv.org/pdf/2610.25553) | CC-BY 4.0 |
| **AnyS1** (no code link on abs) | Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow | Rui Li, Liyang He, Zheng Zhang, Zhenya Huang, Qi Liu | 2026-11-18 | [2611.23751](https://arxiv.org/abs/2611.23751) · [PDF](https://arxiv.org/pdf/2611.23751) | CC-BY 4.0 |
| **Small Heads** | Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models | Ritvik Rastogi | 2026-11-11 | [2611.15592](https://arxiv.org/abs/2611.15592) · [PDF](https://arxiv.org/pdf/2611.15592) | CC-BY 4.0 |
| **Biomedical S1** | From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders | Pritam Deka | 2026-10-01 | [2610.02486](https://arxiv.org/abs/2610.02486) · [PDF](https://arxiv.org/pdf/2610.02486) | CC-BY 4.0 |

### B2. Background papers from the same teams

| System | Title | Authors / org | Date | ID | License |
|---|---|---|---|---|---|
| CLM team (Kwok) | RoboMonkey: Scaling Test-Time Sampling and Verification for VLA Models | Kwok, Agia, Sinha, …, Stoica, Mirhoseini, Pavone | 2025-06-21 | [2506.17811](https://arxiv.org/abs/2506.17811) · [PDF](https://arxiv.org/pdf/2506.17811) | CC-BY 4.0 |
| CLM team (Saad-Falcon) | Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver) | Saad-Falcon et al. | 2025-06-22 | [2506.18203](https://arxiv.org/abs/2506.18203) · [PDF](https://arxiv.org/pdf/2506.18203) | CC-BY 4.0 |
| CLM team (Saad-Falcon, Ré, Mirhoseini) | Archon: An Architecture Search Framework for Inference-Time Techniques | Saad-Falcon et al. | 2024-09-23 | [2409.15254](https://arxiv.org/abs/2409.15254) · [PDF](https://arxiv.org/pdf/2409.15254) | CC-BY 4.0 |
| CLM team (Suresh) | CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking | Tarun Suresh et al. | 2024-12-01 | [2412.01007](https://arxiv.org/abs/2412.01007) · [PDF](https://arxiv.org/pdf/2412.01007) | CC-BY 4.0 |
| CLM team (Mirhoseini, Ré) | Large Language Monkeys: Scaling Inference Compute with Repeated Sampling | Brown, Juravsky, …, Ré, Mirhoseini | 2024-07-31 | [2407.21787](https://arxiv.org/abs/2407.21787) · [PDF](https://arxiv.org/pdf/2407.21787) | CC-BY 4.0 |
| CLM team (Mirhoseini, Ré) | CodeMonkeys: Scaling Test-Time Compute for Software Engineering | Ehrlich et al. | 2025-01-24 | [2501.14723](https://arxiv.org/abs/2501.14723) · [PDF](https://arxiv.org/pdf/2501.14723) | CC-BY 4.0 |
| CLM team / cited by Kev | Hydragen: High-Throughput LLM Inference with Shared Prefixes | Juravsky, …, Ré, Mirhoseini | 2024-02-07 | [2402.05099](https://arxiv.org/abs/2402.05099) · [PDF](https://arxiv.org/pdf/2402.05099) | CC-BY 4.0 |
| Kev (cited) | DeFT: Flash Tree-attention | Yao et al. | 2024-03-30 | [2404.00242](https://arxiv.org/abs/2404.00242) · [PDF](https://arxiv.org/pdf/2404.00242) | NE |
| Kev (cited) | FIRST: Listwise Reranking with Single Token Decoding | Reddy et al. | 2024-06-21 | [2406.15657](https://arxiv.org/abs/2406.15657) · [PDF](https://arxiv.org/pdf/2406.15657) | CC-BY 4.0 |
| Laya (cited) | Parallel Context Windows for LLMs | Ratner et al. | 2022-12-21 | [2212.10947](https://arxiv.org/abs/2212.10947) · [PDF](https://arxiv.org/pdf/2212.10947) | CC-BY 4.0 |
| Jeeves (cited) | Orthrus: Parallel Token Generation via Dual-View Diffusion | Van Nguyen et al. | 2026-05-12 | [2605.12825](https://arxiv.org/abs/2605.12825) · [PDF](https://arxiv.org/pdf/2605.12825) | CC-BY 4.0 |
| Fastino | GLiNER2: An Efficient Multi-Task Information Extraction System | Zaratiana et al. | 2025-07-24 | [2507.18546](https://arxiv.org/abs/2507.18546) · [PDF](https://arxiv.org/pdf/2507.18546) | CC-BY 4.0 |
| Liquid d1 | LFM2 Technical Report | Amini et al. (Liquid AI) | 2025-11-28 | [2511.23404](https://arxiv.org/abs/2511.23404) · [PDF](https://arxiv.org/pdf/2511.23404) | CC-BY 4.0 |
| vLLM-SR | When to Reason: Semantic Router for vLLM | Chen Wang et al. | 2025-10-09 | [2510.08731](https://arxiv.org/abs/2510.08731) · [PDF](https://arxiv.org/pdf/2510.08731) | NE |
| vLLM-SR | Category-Aware Semantic Caching for Heterogeneous LLM Workloads | Chen Wang et al. | 2025-10-29 | [2510.26835](https://arxiv.org/abs/2510.26835) · [PDF](https://arxiv.org/pdf/2510.26835) | NE |

### B3. Benchmarks and evaluations (new; no model released)

| Title | Date | ID | License |
|---|---|---|---|
| Jev-ToM: Evaluating Theory of Mind in Typed Decision Models (HKUST; uses Valen) | 2026-10-13 | [2610.18610](https://arxiv.org/abs/2610.18610) · [PDF](https://arxiv.org/pdf/2610.18610) | CC-BY 4.0 |
| ProbabilityBench: Evaluation with Exact Ground-Truth Uncertainty | 2026-10-12 | [2610.17016](https://arxiv.org/abs/2610.17016) · [PDF](https://arxiv.org/pdf/2610.17016) | CC-BY 4.0 |
| S1CogBench: System One Decision Models on Cognitive Experiments | 2026-10-31 | [2611.00911](https://arxiv.org/abs/2611.00911) · [PDF](https://arxiv.org/pdf/2611.00911) | NE |
| A First Look at System One Decision Models for Malware Detection (uses CLM-8B embeddings) | 2026-11-16 | [2611.20699](https://arxiv.org/abs/2611.20699) · [PDF](https://arxiv.org/pdf/2611.20699) | CC-BY 4.0 |
| Calibration and Risk-Budgeted Decisions with System One Models | 2026-10-14 | [2610.19811](https://arxiv.org/abs/2610.19811) · [PDF](https://arxiv.org/pdf/2610.19811) | NE |
| Does System One Transfer? (Indonesian regional languages) | 2026-10-09 | [2610.14112](https://arxiv.org/abs/2610.14112) · [PDF](https://arxiv.org/pdf/2610.14112) | NE |
| Decision Models vs. LLMs as Rubric Judges (Rao, Callison-Burch) | 2026-09-24 | [2609.29769](https://arxiv.org/abs/2609.29769) · [PDF](https://arxiv.org/pdf/2609.29769) | NE |
| Typed Decision Models: An Early Evidence Audit and Evaluation Checklist | 2026-09-26 | [2609.32160](https://arxiv.org/abs/2609.32160) · [PDF](https://arxiv.org/pdf/2609.32160) | CC-BY 4.0 |
| Jev in the Wild | 2026-09-24 | [2609.30216](https://arxiv.org/abs/2609.30216) · [PDF](https://arxiv.org/pdf/2609.30216) | NE |
| Evaluating and Benchmarking the System One Model Jev | 2026-09-29 | [2609.37647](https://arxiv.org/abs/2609.37647) · [PDF](https://arxiv.org/pdf/2609.37647) | NE |
| Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms? | 2026-09-27 | [2609.33209](https://arxiv.org/abs/2609.33209) · [PDF](https://arxiv.org/pdf/2609.33209) | CC-BY 4.0 |
| Evaluating System One Models for Agent Security Decisions | 2026-09-27 | [2609.33401](https://arxiv.org/abs/2609.33401) · [PDF](https://arxiv.org/pdf/2609.33401) | **CC BY-NC-ND 4.0** |
| JevAdvBench | 2026-09-25 | [2609.31142](https://arxiv.org/abs/2609.31142) · [PDF](https://arxiv.org/pdf/2609.31142) | CC-BY 4.0 |
| Calibrated Decision Models for Pentest Harnesses (Jev and Laya) | 2026-09-24 | [2609.28940](https://arxiv.org/abs/2609.28940) · [PDF](https://arxiv.org/pdf/2609.28940) | CC-BY 4.0 |

**Papers already in the prior study, licenses now confirmed:**
- [2610.11392](https://arxiv.org/abs/2610.11392) TypedBench: NE.
- [2610.18233](https://arxiv.org/abs/2610.18233) Runoff: CC-BY.
- [2610.18579](https://arxiv.org/abs/2610.18579) Reading Right, Acting Wrong: CC-BY.
- [2610.00346](https://arxiv.org/abs/2610.00346): NE.
- All of their PDFs returned 200.

### B4. Non-arXiv publications (all returned HTTP 200 unless noted)

None of these state a reuse license, so treat them as link-only.

- **CLM:** "Contrastive Language Models", a Notion blog by Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré and Mirhoseini, 2026 ([contrastive-lm.notion.site](https://contrastive-lm.notion.site)). This is the repo's only citation; **there is no arXiv paper.** The page renders with JavaScript, so I could not read its contents. Launch threads: [@jackyk02](https://x.com/jackyk02/status/2102905335925424285), [@Azaliamirh](https://x.com/Azaliamirh/status/2102958024323428362).
- **Laya:** "From Laya to Vega: Open Non-Autoregressive Decision Models with Calibrated Probabilities", Preprints.org, DOI [10.20944/preprints202611.0781.v1](https://doi.org/10.20944/preprints202611.0781.v1). **The link returned 403 (bot-blocked), so the URL and license are unverified.** Preprints.org is usually CC-BY, but I could not confirm it.
- **TypeSafe / Jev:**
  - [launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  - [ML primer (RLCD)](https://docs.typesafe.ai/introduction/machine-learning-primer.md)
  - [Don't Call It a Decision Model](https://typesafe.ai/blog/dont-call-it-a-decision-model)
  - [Jev 1.14](https://typesafe.ai/blog/jev-1-14)
  - [Lies, Damned Lies, and Benchmarks](https://typesafe.ai/blog/antibenchmaxxing)
  - [The Bitterest Lesson](https://typesafe.ai/blog/bitterest-lesson)
- **Thinking Machines:**
  - [Introducing Inkling](https://thinkingmachines.ai/news/introducing-inkling/)
  - [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025-09-29)
  - [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025-10-27)
  - **Their blog index has no post on calibration or proper scoring rules.** That material appears only inside the Inkling post.
- **Vendor posts:**
  - Cloudflare: [Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
  - Liquid AI: [d1 blog](https://www.liquid.ai/blog/d1-decision-model) and [docs](https://docs.liquid.ai/lfm/models/decision-models)
  - AWS: [Introducing Strands Decider](https://strandsagents.com/blog/introducing-strands-decider)
  - Together: [How to train your own Jev for $17](https://www.together.ai/blog/how-to-train-your-own-jev)
  - StartLux: [blog](https://decision.startlux.com/)
  - Valen: [Valen-Decision-1.5 release post](https://valen-team.github.io/news/valen-decision-1.5/)
  - Superagent: [Security-One](https://www.superagent.sh/blog/introducing-security-one)
  - vLLM-SR: [decision-models blog](https://vllm-sr.ai/blog/decision-models/) and [Decision 2.0 news](https://modelsystem.one/news/vllm-sr-decision-2-family/)
- **Model cards:** the HF cards linked in Table A. Kev's cards are at [docs/model-cards](https://github.com/jaredpalmer/kev/tree/main/docs/model-cards).

---

## What I could not verify

- **Valen has no arXiv paper.** Its README's arXiv link (2610.18610) is HKUST's Jev-ToM evaluation, not a Valen paper.
- **No arXiv paper exists for CLM, Kev, Clef, pplx-decider, Strands Decider, Nimble, Jeeves, NanoJev or Liquid d1.** I found only blogs and model cards. Perplexity has published no technical report I could find.
- **Cygnet (blockbrain)** is #1 on BenchLM's JevBench list (73.70), but its HF repo returned 401 (gated or nonexistent), so it is unverified. The Jeff-1.1-31B HF path I guessed also returned 401. The prior study's collection link is the correct source.
- **Two leaderboards render with JavaScript:** the S1-Arena leaderboard and the Jev Decision Index Space (951♥). The fetch returned no table rows for either, so model presence there comes from the papers and BenchLM.
- **Closed systems with no papers:** Gemini Decision 1, HeyGen Decision 1, Celeris, Subquadratic Snap, Nace.AI Drex (an older Drex Max checkpoint is reportedly under Open RAIL-M), meraGPT, Trio-Spark (closed weights; its repo is 5★), Upstage Solar Decide, Inception Mercury Decide, and Tazo.
- **Two papers to verify before relying on dates:** the arXiv page gives Beveridge (2611.18655) a citation date of 2026-10-17, which looks odd for a 2611 ID. This-that-model (2609.23886) links its weights at `flock-io/this-that-model-1.0`, which I did not check.
- **Licence cautions:**
  - Nimble's GitHub repo has no license file, although its weights are Apache-2.0.
  - StartLux weights, openjev/openjev and Vev weights are CC BY-NC.
  - The AutoTrust JEV models are distilled from Jev outputs, so check TypeSafe's terms before using them.
  - Liquid d1 uses the LFM Open License, not an OSI-approved license.

**Most relevant to our CLM-8B rebuild (my judgment):**
- **Strands Decider:** a small pointer head (about 1M params) with LoRA on Qwen3.5-2B, with data and scripts public.
- **Jeeves:** pointer head, calibration code and full training code.
- **Intern-Decision:** training code and a temperature-calibration benchmark; the paper is CC-BY-SA.
- **NanoJev:** an end-to-end training pipeline, by the S1-Arena author.
- **"Small Heads, Strong Decisions" (2611.15592):** a controlled study of decision heads.
- **AnyS1 (2611.23751):** a 4B pointer model for decision points in agent workflows.
