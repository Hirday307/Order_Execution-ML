# Adaptive Execution Engine

*Can short-term order book signals lower the cost of executing large equity orders? An event-driven replay study against TWAP, VWAP and Almgren–Chriss schedules on Nasdaq limit order book data.*

> **Status: in progress.** No results have been produced yet. Every number in the Results section will come from running the backtest (`results/<stage>/summary.md`), written into this README by `scripts/report.py`. Nothing is filled in by hand.

---

## The question

An execution desk that has to buy or sell a large block of shares cannot trade it all at once without moving the price against itself. Instead it splits the **parent order** into many small **child orders** spread over time. Standard schedules such as TWAP and VWAP decide in advance how much to trade in each interval, and they behave the same way whatever the order book is showing.

This project asks:

> If a model reads short-term order book signals, can it adjust **how fast** we trade and **whether each child order waits in the book or crosses the spread**, so that implementation shortfall falls after fees, on data the model has never seen?

"No" is a valid answer. The project is designed so that the answer can be trusted either way.

## How cost is measured

Every strategy is scored by **implementation shortfall (IS)** against the **arrival price**, which is the mid-price at the moment the parent order starts. IS is expressed in basis points (1 bp = 0.01%) and includes fees.

```
IS (bps) = side × (average fill price − arrival mid) / arrival mid × 10,000  +  fees (bps)

side = +1 for a buy, −1 for a sell.   Positive IS = cost.
```

Example: a buy with an arrival mid of $100.00 and an average fill of $100.05 costs 5 bps before fees.

Each order's IS is split into five parts that add up to the total: **spread** (fill price vs. the historical mid at the time of each fill; negative when a resting order earns the spread), **drift** (mid at each fill vs. arrival mid), **modeled impact**, **fees**, and **opportunity cost** for any shares left unfilled.

Slippage against interval VWAP is reported as a secondary benchmark only. A VWAP schedule tracks VWAP by design, and the VWAP benchmark hides the cost of trading in a trending market.

## What is compared

| Strategy | Schedule: how much per 30 s slice | Tactic: post or cross |
|---|---|---|
| TWAP | Equal shares in every slice | Baseline tactic |
| VWAP | Follows the historical intraday volume curve | Baseline tactic |
| Almgren–Chriss | Front-loaded curve; urgency κT tuned on validation | Baseline tactic |
| Heuristic adaptive | VWAP, sped up or slowed down within a ±10% band | Rule on level-1 queue imbalance |
| ML adaptive (tactic only) | VWAP, unchanged | Model's predicted 30 s price move |
| ML adaptive (tactic + speed) | VWAP, sped up or slowed down within a ±10% band | Model's predicted 30 s price move |

**Baseline tactic:** post each slice at the best bid (buys) or best ask (sells). At each decision, cancel whatever is unfilled and re-post at the current best price. Cross the spread only to catch up when the order falls more than 10% of its size behind schedule, and cross whatever remains at the end of the horizon.

The heuristic is there on purpose: the ML strategy has to beat a simple rule built from the same kind of signal, not just a fixed schedule. The "tactic only" variant separates how much of any gain comes from choosing post vs. cross and how much comes from changing speed.

## Data

| Stage | Source | What it is | Used for |
|---|---|---|---|
| 1 — build | LOBSTER free sample | One trading day (21 June 2012) of Nasdaq data for AAPL, AMZN, GOOG, INTC and MSFT: every order book event, plus the book up to 10 levels | Building and debugging the pipeline and simulator. One day is a demonstration, not evidence. |
| 2 — test | Databento `XNAS.ITCH`, schema `mbp-10` | Nasdaq TotalView-ITCH book: top 10 levels on every update, trades included | 10–20 consecutive recent trading days for 1–2 stocks, paid for with the $125 free credit for new accounts. Check the cost before every download. |
| 1 — build (alternative) | Nasdaq TotalView-ITCH 5.0 public sample | One full day (27 March 2019) of the raw Nasdaq feed that LOBSTER is built from, published by Nasdaq as a sample file | Stage 1 without the LOBSTER sample: `scripts/extract_itch.py` keeps the five stocks' order messages and `build_dataset.py` rebuilds their books order by order (`configs/stage1_itch.yaml`). |
| Helper | Yahoo Finance (`yfinance`) | 5-minute bars, last 60 days only | A proxy intraday volume curve for Stage 1. It has no quotes, depth or individual trades, so it cannot drive the simulator. |

Both order book sources are **Nasdaq only**; real desks route across many venues (see Limitations). Raw data is never committed to this repository. See `QUICKSTART.md` for download steps.

**LOBSTER sample access (checked September 2026):** the LOBSTER website now serves its samples through a request form (terms, a human check, then a review before download). If the request is slow or refused, Stage 1 runs unchanged on Nasdaq's public ITCH sample day (`configs/stage1_itch.yaml`) or on one recent Databento day (`configs/stage1_databento.yaml`).

## How the simulator works

The backtest replays the historical order book event by event and inserts our child orders into it.

- **Crossing the spread.** A marketable child order fills against the displayed opposite side, best price first ("walking the book"), up to 10 levels.
- **Posting in the book.** A resting child order joins the **back** of the queue at the best bid (buys) or best ask (sells). It fills only when historical trades at that price use up the whole queue ahead of it, or when the market trades through its price. Cancellations by other traders never move it forward. This is deliberately conservative. An optimistic variant, where cancellations are assumed to come from ahead of us, is reported alongside it to show how sensitive results are to this assumption.
- **Fees.** A configurable taker fee and maker rebate per share.
- **Market impact.** A replay cannot show how the market would have reacted to our orders. A transient impact overlay (Obizhaeva–Wang style) pushes our fill prices against us in proportion to the shares we have traded, and the push decays with a set half-life. Its strength is calibrated to the square-root impact law, and every backtest runs at three settings: none, medium and strong.
- **End of the horizon.** Anything left is crossed. Shares the displayed book cannot absorb are charged as opportunity cost at the final mid plus half the spread.

## Model and decision rule

**Features (18, all computed only from data at or before the decision time):** spread in ticks and in bps, level-1 and level-1-to-5 queue imbalance, weighted-mid gap, book depth, order flow imbalance over 10, 30 and 60 s, trade-sign imbalance over 30 and 60 s, trade count and relative volume over 60 s, mid-price returns over 10, 60 and 300 s, 5-minute realized volatility, and minutes since the open. Exact definitions are in `QUICKSTART.md`.

**Target:** the change in mid-price over the next 30 seconds, in bps. This exists in the data at every moment, so no label has to be invented. Because the model predicts the market's move rather than an action, one model serves both buys and sells; the decision rule flips the sign for sells.

**Models:** ridge regression (baseline) and XGBoost (main). The final choice is made on validation data by execution cost, not by prediction accuracy.

**Decision rule, every 30 seconds:**

```
signed     = side × predicted move              (positive = price expected to move against us)
cross_cost = spread + taker fee + maker rebate  (bps; extra cost of crossing vs. a filled resting order)

if signed >  m × cross_cost:   cross, and trade up to (1 + α) × the scheduled slice
if signed < −m × cross_cost:   post,  and trade (1 − α) × the scheduled slice
otherwise:                     baseline tactic, scheduled slice
```

Cumulative shares always stay within ±10% of the parent order around the VWAP schedule. The margin *m* and speed step *α* are tuned on validation data. The "tactic only" variant fixes *α* = 0.

## Evaluation protocol

- **Parent orders:** buys and sells with a 30-minute horizon, sized at 2%, 5% and 10% of the expected Nasdaq volume over that horizon, starting every 5 minutes in each evaluation window.
- **Stage 1 (one day):** split by time. Train 09:45–12:30, validate 12:40–14:00, test 14:10–16:00. The 10-minute gaps are longer than the 30 s label horizon, so no label crosses a boundary. Extra check: leave-one-stock-out (train and tune on four stocks, test on the fifth).
- **Stage 2 (multi-day):** walk-forward by day. For each test day *d*, train on the 10 trading days before *d − 1*, tune on *d − 1*, and test on *d*. Volume curves and feature normalizers come only from earlier days. With 20 days this gives 9 test days.
- **Tuning discipline:** every choice (model, *m*, *α*, the heuristic threshold, Almgren–Chriss urgency) is made on validation data. Each test period is scored once.
- **Paired comparisons:** every strategy executes the same parent orders, so differences are measured order by order.
- **Honest uncertainty:** orders that overlap in time are not independent, so 95% confidence intervals come from a block bootstrap (15-minute blocks in Stage 1, whole days in Stage 2).
- **Robustness:** results are broken down by side, order size, stock, impact setting and fill assumption.
- **Success criterion, fixed before testing:** the adaptive strategy counts as better than VWAP only if the 95% confidence interval of the paired difference excludes zero at medium impact with conservative fills, and the direction of the result holds for buys and sells separately and under the other two impact settings.

## Results

> Not yet run. These tables are generated by `scripts/report.py --readme README.md`, which replaces the text between each stage's markers. Nothing between the markers is edited by hand.

### Stage 1 (one day; a demonstration, not evidence)

<!-- BEGIN RESULTS: stage1 -->
_Not yet run._
<!-- END RESULTS: stage1 -->

### Stage 2, out-of-sample, medium impact, conservative fills

<!-- BEGIN RESULTS: stage2 -->
| Strategy | Mean IS (bps) | Median | Std. dev. | 95th pct | Δ vs VWAP (95% CI) | Share filled passively |
|---|---|---|---|---|---|---|
| TWAP | — | — | — | — | — | — |
| VWAP | — | — | — | — | baseline | — |
| Almgren–Chriss | — | — | — | — | — | — |
| Heuristic adaptive | — | — | — | — | — | — |
| ML adaptive (tactic only) | — | — | — | — | — | — |
| ML adaptive (tactic + speed) | — | — | — | — | — | — |
<!-- END RESULTS: stage2 -->

The cost decomposition, robustness, leave-one-stock-out, validation and assumption tables are written to `results/<stage>/summary.md`.

## Default settings

| Setting | Value |
|---|---|
| Feature grid | 1 second |
| Prediction horizon | 30 s (10 s and 60 s also tried on validation) |
| Decision interval | 30 s |
| Parent order horizon | 30 minutes (60 decisions) |
| Parent order size | 2%, 5%, 10% of expected Nasdaq volume over the horizon |
| Schedule band | ±10% of the parent order |
| Speed step α | tuned on validation from {0.25, 0.5} |
| Fees (Stage 1 defaults) | Taker $0.0030 per share (the Reg NMS access-fee cap in 2012), maker rebate $0.0020 per share (a rough typical value). For other periods, set them from the venue's published fee schedule. |
| Impact overlay | None; medium (Y = 0.5, half-life 5 min); strong (Y = 1.0, half-life 10 min) |

## Repository layout

```
adaptive-execution/
├── README.md
├── QUICKSTART.md                 # step-by-step build guide
├── PROJECT_SUMMARY.md            # overview, industry context, interview notes
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── configs/
│   ├── stage1_lobster.yaml
│   ├── stage1_itch.yaml          # Stage 1 on Nasdaq's public ITCH 5.0 sample day (27 March 2019)
│   ├── stage1_databento.yaml     # Stage 1 on one Databento day, if the LOBSTER sample is unavailable
│   ├── stage2_databento.yaml
│   └── synthetic.yaml            # dry run on synthetic LOBSTER-format files
├── data/                         # git-ignored; never committed
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_explore_order_book.ipynb   # Phase 1 plots + one parent order checked by hand
│   ├── 02_signals.ipynb              # feature signal, model accuracy by horizon, importance
│   └── 03_results.ipynb              # charts of the out-of-sample results
├── src/adaptive_exec/
│   ├── __init__.py
│   ├── loaders.py                # LOBSTER and Databento → one internal format
│   ├── itch.py                   # Nasdaq ITCH 5.0: stream the raw feed, rebuild each book order by order
│   ├── features.py               # 18 point-in-time features
│   ├── labels.py                 # 30 s forward mid-price move
│   ├── schedules.py              # TWAP, VWAP, Almgren–Chriss, volume curves
│   ├── fills.py                  # walking the book, resting-order queue model
│   ├── impact.py                 # transient impact overlay + square-root calibration
│   ├── strategies.py             # baseline tactic, heuristic, ML adaptive
│   ├── simulator.py              # parent-order replay loop
│   ├── model.py                  # ridge and XGBoost
│   ├── splits.py                 # time splits with gaps, walk-forward
│   ├── metrics.py                # IS, decomposition, block bootstrap
│   ├── pipeline.py               # config, processed-data IO, evaluation units (incl. leave-one-stock-out)
│   ├── experiment.py             # strategy candidates and per-period setup
│   ├── reporting.py              # summary tables and the pre-registered success test
│   ├── plotting.py               # notebook chart style
│   └── synthetic.py              # synthetic LOBSTER-format events for tests and dry runs
├── scripts/
│   ├── fetch_volume_curve.py     # Stage 1 Yahoo proxy curve
│   ├── download_databento.py     # Stage 2 download, cost estimate first
│   ├── extract_itch.py           # pass 1 over an ITCH sample file (keeps the chosen stocks)
│   ├── make_synthetic_lobster.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── backtest.py               # tunes on validation; scores each test period once
│   ├── report.py                 # summary.md, and the README results block
│   └── inspect_order.py          # one parent order, decision by decision and fill by fill
├── tests/                        # synthetic data only, so CI needs no market data
│   ├── conftest.py
│   ├── test_fills.py
│   ├── test_schedules.py
│   ├── test_impact.py
│   ├── test_metrics.py
│   ├── test_no_lookahead.py
│   ├── test_loaders.py
│   ├── test_databento_format.py  # a real DBN file, via Databento's own libraries
│   ├── test_itch.py              # ITCH parsing and book rebuilding on hand-built messages
│   ├── test_splits.py
│   ├── test_pipeline.py
│   ├── test_model.py
│   ├── test_simulator.py
│   ├── test_backtest_choice.py
│   └── test_end_to_end.py        # every script, in order, on a tiny synthetic project
├── .github/workflows/ci.yml
└── results/                      # generated; only summary.md, chosen_params.json and scoring_log.csv are committed
```

## Quick start

Download the data first (`QUICKSTART.md`, Phase 0). Then:

```bash
git clone https://github.com/<your-username>/adaptive-execution.git
cd adaptive-execution
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest                                               # simulator tests must pass first

C=configs/stage1_lobster.yaml
python scripts/fetch_volume_curve.py --config $C       # Stage 1 proxy volume curve
python scripts/build_dataset.py      --config $C       # read the [build] sanity checks
python scripts/train.py              --config $C
python scripts/backtest.py           --config $C --validate-only   # tune; test untouched
python scripts/inspect_order.py      --config $C       # check one validation order by hand
python scripts/backtest.py           --config $C       # score the test period, once
python scripts/report.py             --config $C --readme README.md
```

The notebooks need Jupyter (`pip install jupyter`) and read `configs/stage1_itch.yaml` unless the `ADAPTIVE_EXEC_CONFIG` environment variable names another config. `02_signals` needs `train.py`'s output and `03_results` needs the test run.

To run Stage 1 on Nasdaq's ITCH sample instead, download `03272019.NASDAQ_ITCH50.gz` (5.5 GB) from Nasdaq's ITCH sample directory (`emi.nasdaq.com/ITCH/Nasdaq ITCH/`) into `data/raw/itch/`, run `python scripts/extract_itch.py --config configs/stage1_itch.yaml`, then the steps above with `C=configs/stage1_itch.yaml`.

For Stage 2, set `DATABENTO_API_KEY`, run `python scripts/download_databento.py --config configs/stage2_databento.yaml` (it prints the cost estimate and downloads nothing without `--confirm`), then the same steps with that config.

To check the whole pipeline without market data, run it on synthetic LOBSTER-format files. That data has a planted signal, so its numbers only show that the plumbing works:

```bash
python scripts/make_synthetic_lobster.py
for s in build_dataset train backtest report; do python scripts/$s.py --config configs/synthetic.yaml; done
```

The backtest runs one worker per stock-day in parallel; set `n_jobs` in the config to limit it.

## Limitations

1. **The replay does not react to us.** In reality, other traders would respond to our orders. The impact overlay is a model of that response, not a measurement, so a conclusion is only trusted if it holds under all three impact settings.
2. **Resting-order fills are approximate.** We don't know our exact place in the queue. The conservative assumption understates passive fills, and the optimistic one overstates them; both are reported.
3. **One venue.** Both data sources are Nasdaq only. Real desks split orders across many venues with a smart order router, which this project does not model.
4. **Limited data.** Stage 1 is a single day from 2012, and Stage 2 covers a few weeks of one or two stocks. Results describe those stocks and periods, not markets in general.
5. **Market structure changes.** Fees, tick sizes and trading speed have changed since 2012, which is one reason Stage 2 uses recent data.
6. **Research code, not a trading system.** There is no live client flow, real-time risk control or order management integration.

## References

- Perold, A. (1988). The implementation shortfall: Paper versus reality. *Journal of Portfolio Management.*
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk.*
- Obizhaeva, A., & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets.*
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics.*
- Stoikov, S. (2018). The micro-price: A high-frequency estimator of future prices. *Quantitative Finance.*
- Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope.* Cambridge University Press.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Huang, R., & Polak, T. (2011). LOBSTER: Limit Order Book Reconstruction System. SSRN working paper.

---

**Author:** Your Name
**License:** MIT (code only; market data is subject to its providers' licenses)
