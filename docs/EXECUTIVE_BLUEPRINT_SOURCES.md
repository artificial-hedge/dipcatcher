# Executive blueprint: source verification and evaluation boundaries

Verified on **2026-10-01** against author papers, author repositories, published
patent text, and dataset providers. The supplied PDF's `【19†L…】`, `【20†L…】`,
`【40†L…】`, `【50†L…】`, and `【57†L…】` tokens are internal citation handles;
they contain no resolvable bibliography or source URL. The mappings below were
reconstructed from named works and descriptions. Matching descriptions does
not establish the identity of every original citation, especially handle 50,
which is reused for both multimodal methods and industry dark-pool claims.

## QuantCode model and the 400-task benchmark

[QuantCode Model, arXiv:2609.39420v1](https://arxiv.org/html/2609.39420v1)
supports continued pretraining on trading-framework code followed by
agent-validated supervised fine-tuning. It reports better executable-code and
semantic-alignment rates, while also exposing regressions in structured tool
calling and repository tasks. Its internal training checkpoints and historical
runs do not have a complete public reproduction package. The paper explicitly
excludes profitability, realistic transaction costs, robustness and production
risk from its benchmark claim. Paper license: CC BY 4.0.

[QuantCode-Bench paper](https://arxiv.org/abs/2604.15151) and the
[author repository](https://github.com/LimexAILab/QuantCode-Bench) provide 400
Backtrader strategy-description tasks. Their evaluation measures compilation,
runtime success, at least one trade, and LLM semantic judging; it does **not**
require positive returns or a target profit within tolerance. The task mix is
183 Reddit, 100 TradingView, 90 StackExchange, 19 GitHub and 8 synthetic tasks.
The repository declares MIT; rights to underlying forum content and vendor
market data still require their own provenance. Backtrader has its own
[GPLv3 license](https://github.com/mementum/backtrader/blob/master/LICENSE).

The following published files were fetched and hashed directly at revision
`f8bda951addb409a81aa316c00401dbde60774ae`:

| Artifact | Exact source | SHA-256 |
| --- | --- | --- |
| Task file, 400 unique ids 1–400 | [benchmark_tasks_multiframe.json](https://raw.githubusercontent.com/LimexAILab/QuantCode-Bench/f8bda951addb409a81aa316c00401dbde60774ae/data/benchmark_tasks_multiframe.json) | `b197e0271779f332c6808ea40167615e3b90061563544b8bdf3c48237a9f17d3` |
| Data requirements, 400 matching ids | [task_data_requirements.json](https://raw.githubusercontent.com/LimexAILab/QuantCode-Bench/f8bda951addb409a81aa316c00401dbde60774ae/data/task_data_requirements.json) | `7bc4039cfe971ec04de3618c652eca268c95ce07030c5f597c594209344f38b9` |

Inspection of that requirement file found 41 tasks with fixed prices, 106
with session timing, 17 with multiple assets, and 29 with external-data needs.
These are overlapping categories. A generic single-asset OHLCV test cannot
establish faithful execution of every task.

[Pinned upstream reward code](https://raw.githubusercontent.com/LimexAILab/QuantCode-Bench/f8bda951addb409a81aa316c00401dbde60774ae/quantcode_bench/reward.py)
executes generated Python in a subprocess with a time limit; that alone does
not establish OS isolation. It also accepts semantic alignment after judge
errors. Its yfinance fallback uses rolling windows for intraday inputs, so
rerunning without a frozen data bundle changes the evaluation data.

The local adapter is `src/fx1/eval/quantcode_bench.py`.
`load_quantcode_bench` verifies the externally downloaded bytes, schema, unique
ids, complete count, and matching data requirements. It records raw-file,
semantic-content, schema, source-identity and adapter hashes. A custom fixture
or partial file never becomes the official 400-task import.

`run_quantcode_eval` requires injected model, sandbox executor, semantic judge,
and authorized task-specific input-bundle receipts. It checks code/task/data
and judge/execution bindings, required data capabilities, timezone-aware data
windows, and training/development/evaluation split separation. Both single-turn
and bounded repair evaluation are supported. Missing components and judge
exceptions remain `not_evaluated`; subset results remain `partial` even at a
100% pass rate. The adapter internally executes no generated code and makes no
network request. An actual sandbox must enforce its own OS isolation,
credential exclusion, resource limits and constrained data access. A supplied
attestation hash is a provenance pointer, not independent proof of isolation.

For an import inspection without model execution:

```python
from fx1.eval.quantcode_bench import (
    EvaluationProtocol, load_quantcode_bench, run_quantcode_eval,
)

manifest = load_quantcode_bench(
    "data/fx1/quantcode/benchmark_tasks_multiframe.json",
    requirements_path="data/fx1/quantcode/task_data_requirements.json",
)
protocol = EvaluationProtocol(
    model_id="planned-checkpoint", model_revision="UNMEASURED",
    judge_id="planned-judge", judge_revision="UNMEASURED",
    evaluation_ids=tuple(task.task_id for task in manifest.tasks),
)
report = run_quantcode_eval(manifest, protocol)
assert report.status == "not_evaluated"
assert report.judge_pass_rate is None
```

The local protocol always marks `official_score=False`,
`upstream_protocol_equivalence=UNVERIFIED`, and `market_evidence=False`.
Public benchmark correctness, empirical predictive quality and official
leaderboard submission are separate requirements. There is no measured model
score in this source-verification document.

## Say, Echo, Do

[Author paper, arXiv:2609.38545v1](https://arxiv.org/abs/2609.38545) develops
the narrative/positioning covariance result, echo identification,
return-aligned embeddings, and timing statistics. Its AUC of 0.90 concerns
false-alarm detection in controlled simulations, not a real-market return
classifier. The 29 markets are simulated. Paper license: CC0.

[Author code](https://github.com/AliAtiah/say-echo-do) is MIT, with conference
style files under their separate terms. Its README explicitly leaves a
real-market study to future work. Practical next step: reproduce controlled
experiments, then use timestamped statements, media and publicly available
positioning on rolling, purged splits. Disclosure availability, rather than
the reporting-period end, governs feature timing. This repository supplies no
licensed proprietary institutional positioning panel or demonstrated market
edge.

The bounded local F02 implementation was checked against paper section 4.2 and
the [pinned official objective](https://raw.githubusercontent.com/AliAtiah/say-echo-do/4a201e77f8b50c423e9aae156125bea386984d50/src/say_echo_do/embeddings.py).
It minimizes soft return-neighbor cross entropy with self-neighbors excluded,
using a newly learned token/projection encoder. The author's
[text experiment](https://raw.githubusercontent.com/AliAtiah/say-echo-do/4a201e77f8b50c423e9aae156125bea386984d50/experiments/text_experiment.py)
uses TF-IDF inputs and a learned projection; this does not justify calling the
local encoder a pretrained transformer or claiming a full experimental reproduction.

## Graf and Mastrolia: auctions and RL

The current [arXiv:2601.17247v3](https://arxiv.org/html/2601.17247v3), revised
2026-09-29, is titled **Learning Optimal Liquidation with Closing Auctions**.
It compares DQN, DDPG, TD3 and SAC against stylized AS/TWAP policies. Historical
input contributes midprices; order flow, auction clearing and allocation remain
simulated. The agents improve **inventory-penalized** implementation shortfall;
their mean ordinary shortfalls remain higher than AS. The quadratic inventory
preference is not a cash expense. Thus the PDF's unrestricted execution
outperformance interpretation is too broad. Paper license: CC BY 4.0.

The [MIT software snapshot v0.1.0](https://github.com/juliusgraf/learning-optimal-liquidation/tree/v0.1.0)
resolves to commit `736a0c8ffbc29d831460fa915bc944cbb00fa708`.
[Reproduction instructions](https://github.com/juliusgraf/learning-optimal-liquidation/blob/v0.1.0/REPRODUCING.md)
provide synthetic smoke, saved-result checks and full retraining paths. The
reported campaign uses CPU; the PDF's multi-month TPU/GPU estimates are not a
requirement established by this implementation. Bundled summaries do not
include full run trees or model checkpoints. Historical Alpaca SIP inputs are
not redistributed; reconstruction needs appropriate entitlement. The release
also discloses development-visible reused test dates. Recomputed saved tables
therefore cannot prove fresh out-of-sample performance or live execution.

## FinGPT and multimodal claims

[Original FinGPT paper](https://arxiv.org/abs/2306.06031) and
[official MIT repository](https://github.com/AI4Finance-Foundation/FinGPT)
support financial-data curation and lightweight adaptation of language models.
[FinGPT-Forecaster documentation](https://raw.githubusercontent.com/AI4Finance-Foundation/FinGPT/master/fingpt/FinGPT_Forecaster/README.md)
describes news, basic financials and price information supplied as text, with
Llama-2 LoRA. It does not establish the PDF's particular chart/satellite,
cross-attention or contrastive fusion architecture. The released base model
has its own license and access conditions; the project MIT license does not
replace them.

A [public DOW30 dataset](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)
and [released LoRA](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)
exist. Dataset-specific redistribution rights and vendor-derived news rights
remain to be established before commercial redistribution. A chronological
text-plus-price ablation can use authorized inputs and proper forecasting
scores; adding pictures needs a separately specified dataset and evaluation.

[MFFM position paper, arXiv:2506.01973v2](https://arxiv.org/abs/2506.01973)
describes opportunities and challenges beyond language-centric FinGPT. A
position paper is not evidence that the proposed multimodal module wins a
forecasting benchmark.

## Router patent and market-data availability

[Published US11488243B2 text](https://patents.google.com/patent/US11488243)
describes ML estimates of fill probability and adverse selection together with
child-order optimization. It names external TMX data and proprietary internal
order data; logistic and linear regression are example estimators. It provides
an architectural prior-art reference, not a public training corpus,
reproducible execution-quality benchmark, permission to reproduce claimed
inventions, or empirical proof of superiority. Google displays Royal Bank of
Canada as assignee and an assumed active status; this is not a legal-status or
freedom-to-operate determination. Patent/IP review remains a commercialization
dependency.

[LOBSTER subscription information](https://data.lobsterdata.com/info/HowToJoin.php)
offers academic subscriptions and directs commercial users to a discussion.
Its [legacy FAQ](https://data.lobsterdata.com/info/help_faq_general.php) still
describes academic-only access. The [provider terms](https://data.lobsterdata.com/info/docs/legal/LOBSTER_TermsAndConditions.pdf)
and [documents page](https://data.lobsterdata.com/info/Documents.php) identify
NASDAQ agreements and academic waiver requirements. Consequently, commercial
use and redistribution must be established from the applicable current
contract rather than inferred from a sample download or old FAQ.

The PDF's “Kinetics-level Tape data” is not an identified financial dataset.
The actual [Kinetics paper](https://arxiv.org/abs/1705.06950) concerns human-action
videos. An LOB/order-flow evaluation still needs a named, authorized source
with timestamps, microstructure fields and a reproducible sampling contract.
The PDF's options institutional-intent labels, dark-pool accumulation labels,
exclusive alternative datasets, hardware partnerships and patent filings are
not supplied or proven by the verified references above.

## Additional algorithm references for the broader feature catalog

These primary publications were also checked on 2026-10-01. They identify
concrete algorithm candidates for the PDF's otherwise uncited features. Many
are foundational works from 2003–2021; they do **not** substantiate the PDF's
assertion that every feature is new or current SOTA. Candidate publication
identity is verified here; dataset/checkpoint licenses, implementation
equivalence and the proposed financial evaluations remain separate work.

| Feature family | Verified candidate primary reference | What can be carried into the design; remaining limit |
| --- | --- | --- |
| Synthetic time series | [Yoon et al., TimeGAN, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html); [Rasul et al., TimeGrad, 2021](https://arxiv.org/abs/2101.12072) | TimeGAN learns temporal representations; TimeGrad is autoregressive probabilistic forecasting with diffusion. Neither supplies a guaranteed realistic market-tail generator. |
| Asset graph learning | [Kipf and Welling, GCN, 2016](https://arxiv.org/abs/1609.02907); [Veličković et al., GAT, 2017](https://arxiv.org/abs/1710.10903) | Graph convolution and attention building blocks. Financial edge provenance and chronological asset-level prediction tests must be supplied. |
| Meta-learning | [Finn et al., MAML, ICML 2017](https://arxiv.org/abs/1703.03400) | Gradient-based fast adaptation across tasks; the original demonstrations are not portfolio allocation evidence. |
| Deep hedging / tail scenarios | [Buehler et al., Deep Hedging, 2018](https://arxiv.org/abs/1802.03042); [Kobyzev et al., normalizing-flow review, 2019](https://arxiv.org/abs/1908.09257) | Hedging under frictions and density-transform methods respectively. Deep Hedging itself is not a tail-scenario generator. |
| Credit graph prediction | [Wei et al., GDAN, 2024](https://arxiv.org/abs/2407.11615) | Enterprise-credit graph attention and an announced ECAD dataset. Counterparty exposures, applicable licenses and event-time default labels remain to be established. |
| Continual / online learning | [Chaudhry et al., Tiny Episodic Memories, 2019](https://arxiv.org/abs/1902.10486); [Zinkevich, Online Convex Programming, ICML 2003](https://www.martin.zinkevich.org/publications/ICML03.pdf); [Vitter, reservoir sampling, 1985](https://www.ittc.ku.edu/~jsv/Papers/Vit85.Reservoir.pdf); [Rolnick et al., CLEAR, NeurIPS 2019](https://arxiv.org/abs/1811.11682) | Limited-memory replay and online-gradient foundations. The implemented Gaussian learner uses Algorithm R and momentum gradients; it does not reproduce CLEAR's RL/behavioral-cloning method. Delayed financial labels, drift and rollback need their own protocol. |
| Adversarial robustness | [Madry et al., 2017](https://arxiv.org/abs/1706.06083) | Robust optimization against a specified adversary. Financially admissible perturbations and valid stress scenarios cannot be inferred from image-classification experiments. |
| Explanations | [Lundberg and Lee, SHAP, 2017](https://arxiv.org/abs/1705.07874) | Additive feature attribution framework. Explanations need stability/fidelity checks and do not by themselves establish regulatory adequacy. |
| Order-flow imbalance | [Cont, Kukanov and Stoikov, 2010/2014](https://arxiv.org/abs/1011.6402) | Empirical relation between order-book-event imbalance and price changes. Tick-level field completeness and causal sampling are required for a reproduction. |
| Policy-gradient market making | [Schulman et al., PPO, 2017](https://arxiv.org/abs/1707.06347) | A general policy-gradient algorithm; a quoting policy, fills, inventory constraints and simulated-market fidelity are additional design obligations. |
| Federated models | [McMahan et al., FedAvg, 2016/2017](https://arxiv.org/abs/1602.05629) | Communication-efficient decentralized training. Aggregation alone is not a privacy-leakage guarantee or evidence of an institutional collaboration. |
| Anomaly detection | [Liu, Ting and Zhou, Isolation Forest, ICDM 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf); [sklearn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html); [probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Isolation-based anomaly scoring and separate sigmoid calibration. Frozen traversal was independently matched to sklearn 1.9.1. An outlier or calibrated annotation score does not identify illegal or informed trading. |
| Economic/news event extraction | [Zheng et al., Doc2EDAG, EMNLP 2019](https://aclanthology.org/D19-1032/) | Document-level extraction from Chinese financial announcements; a timestamped economic calendar and surprise/volatility impact labels are different tasks. |
| Causal discovery | [Runge et al., PCMCI, 2017/2019](https://arxiv.org/abs/1702.07007) | Conditional-independence-based causal association discovery for nonlinear time series. Identification assumptions and interventions are required before claiming causation. |
| Alternative-data fusion | [Kwon et al., 2026](https://arxiv.org/abs/2609.11607) | Firm-revenue forecasting using context from four commercial alternative-data channels. A public reproducible input panel or commercial redistribution right is not supplied by that description. |
| Option pricing | [E, Han and Jentzen, deep BSDE methods, 2017](https://arxiv.org/abs/1706.04702); [Raissi et al., PINNs Part I, 2017](https://arxiv.org/abs/1711.10561) | Distinct neural PDE-solving approaches; pricing-boundary conditions, analytic/Monte Carlo comparisons and calibration runtime need direct verification. |
| Quantum / QUBO allocation | [Glover et al., QUBO tutorial, 2018](https://arxiv.org/abs/1811.11538); [Rosenberg et al., quantum trading trajectories, 2015](https://arxiv.org/abs/1508.06182) | QUBO formulations and small quantum-annealer portfolio examples. No general quantum speed advantage, hardware partnership or superior portfolio performance follows. |
| Dynamic correlation visualizer | [Mantegna, financial hierarchies, 1998/1999](https://arxiv.org/abs/cond-mat/9802256); [Jacomy et al., ForceAtlas2, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679) | Correlation-distance financial networks and a graph layout algorithm; edge uncertainty and layout coordinates must not be read as causal or spatial market relationships. |
| Options volume signals | [Pan and Poteshman, NBER 10925 / RFS 2006](https://www.nber.org/papers/w10925) | Research with buyer-initiated opening-volume information from a distinctive CBOE dataset. A generic public trade tape does not reproduce those nonpublic fields or create institutional-intent labels. |

## Acceptance implications

Implementations must preserve the repo's research-only honesty contract:
proper predictive scores and explicit synthetic labels; immutable provenance;
no broker connectivity or claimed live trading; no profitability inferred from
code correctness. Real-market narrative panels, entitlement-controlled LOB and
options data, actual checkpoint training, full model benchmarks, patent rights,
staffing, paid pilots and commercial deployments require their own evidence.
The source research establishes evaluation pathways and limitations, not that
these deliverables are already achieved.
