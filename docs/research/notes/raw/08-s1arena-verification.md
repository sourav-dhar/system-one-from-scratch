Source: https://arxiv.org/html/2610.21244 (HTML v1, fetched directly, HTTP 200). I read the verbatim text through about character 40,000. The rest came back as a model-extracted summary, which I flag below. That summary wrongly said the authors and CLM's table figures were absent, so I checked them against the verbatim text.

(1) Title: "System1-Arena: Which Decisions Can Be Left to System 1?" Authors (title block): Tianyu Chen, Xinyang Liu, Ruyi An, Zhendong Wang, Mingyuan Zhou (University of Texas at Austin).

(2) Table 1 (non-agentic), CLM-v0.1-8B row: overall Acc 41.4, Brier 0.749. The paper calls it "CLM-v0.1-8B", not "CLM-8B".

(3) Table 1, latency column "p50 (ms)": 41. Section 4 defines latency as "the median time per request." Cost is 19 GPU-s per 1,000 requests.

(4) CLM has the lowest accuracy of the 9 System-1 rows in Table 1 (next lowest: Laya 44.5). The intro says: "We evaluate 30 models: 7 closed LLMs, 13 open LLMs evaluated through answer-token probabilities, and 10 System-1 models." Table 1 has only 9 System-1 rows. Decision-2.0-Sol-2B appears only in Table 2 (agentic), so I could not confirm CLM is lowest among all 10.

(5) Not found as a readout comparison. The summary says: "The Decisions API and CLM-v0.1-8B instead lose 2.6 and 17.8 percentage points, respectively." This is relative to base model Qwen3-8B (Section 5, "Paired comparisons"). It is unverified because I did not read that section. In Table 1, Qwen3-8B (answer-token read) is 59.3 and CLM is 41.4, a 17.9-point gap.

(6) Not found in the text I read. Appendix B "Inputs" is in the unread part (offset 139958 onward).

(7) Table 1: Decision-2.0-Vega-27B Acc 73.0 (Brier 0.343); Jev Acc 72.6 (Brier 0.349). Intro: "Jev and Decision-2.0-Vega-27B reach 72.6% and 73.0% accuracy, respectively."
