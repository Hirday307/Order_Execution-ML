# Adaptive Execution Engine — Project Summary

## In one paragraph

This project is a small-scale version of the research an electronic execution desk does every day. It replays real Nasdaq limit order book data event by event, simulates how child orders would have filled, and measures every strategy by **implementation shortfall against the arrival price**, the standard cost benchmark in the industry. It compares fixed schedules (TWAP, VWAP, Almgren–Chriss) with adaptive strategies that read short-term order book signals to decide how fast to trade and when to post versus cross the spread. The goal is not a big headline number. It is a trustworthy answer, positive or negative, produced with the same benchmark and validation discipline a desk would use.

**Status:** Stage 1 (one day of Nasdaq data, 27 March 2019) has been run once; Stage 2 (multi-day) has not.

---

## How execution desks work

A client, such as a pension fund, decides **what** to trade: for example, buy 500,000 shares. The execution desk decides **how**, and it is judged mainly on the cost of doing it.

A large order faces two opposing costs:

- **Market impact.** Trading fast uses up the available liquidity and pushes the price against you, and other traders notice.
- **Timing risk.** Trading slowly avoids impact, but the price can drift away while you wait.

Desks manage this with execution algorithms that split the parent order into child orders. TWAP spreads the order evenly over time. VWAP follows the market's usual intraday volume pattern. Implementation shortfall algorithms, based on the Almgren–Chriss model, trade faster early to limit drift. Inside each slice there is a second choice: **post** a limit order and wait (saving the spread, but risking no fill or a fill just before the price moves against you) or **cross** the spread (a certain fill, at the cost of the spread and a taker fee).

After the fact, transaction cost analysis (TCA) measures the result, most often as implementation shortfall in basis points against the arrival price.

## The question

> Can short-term order book signals adjust the speed and the post-versus-cross choice of each child order so that implementation shortfall falls after fees, on data the model has never seen?

## How this maps to a real execution stack

| Industry component | In this project | Simplification |
|---|---|---|
| Parent order from the client or order management system | Simulated buy and sell parent orders at 2%, 5% and 10% of expected volume | No real client flow |
| Pre-trade cost estimate | Impact overlay calibrated to the square-root law | A simple model, run at three strengths |
| Scheduler | TWAP, VWAP, Almgren–Chriss | No percentage-of-volume or auction algorithms |
| Micro-tactics (post or cross) | Baseline tactic, heuristic rule, ML-driven rule | Resting orders only at the best price |
| Smart order router across venues | Not included | Nasdaq only |
| Exchange matching | Historical replay with a conservative queue model | The market does not react to us beyond the overlay |
| Post-trade TCA | Implementation shortfall, five-part cost decomposition, bootstrap intervals | Same concept |
| Risk controls | Schedule band and forced catch-up | Minimal |

The project covers the scheduling, tactics and TCA layers faithfully. It deliberately leaves out routing, live risk and client flow, and says so.

## Design

### Data

- **Stage 1 (build):** the LOBSTER free sample. It covers one day (21 June 2012) of Nasdaq data for AAPL, AMZN, GOOG, INTC and MSFT, with every order book event and 10 price levels. The Stage 1 run so far used Nasdaq's public TotalView-ITCH 5.0 sample for 27 March 2019 instead, with the same five stocks, because LOBSTER's samples now need a reviewed request.
- **Stage 2 (test):** Databento Nasdaq TotalView-ITCH (`XNAS.ITCH`, `mbp-10`) for 10–20 recent trading days of one or two stocks, using the $125 free credit for new accounts.
- **Helper:** Yahoo Finance, only for a proxy intraday volume curve in Stage 1.

### Simulator

- Marketable child orders walk the displayed book, up to 10 levels.
- Resting child orders join the back of the queue and fill only when trades use up the queue ahead of them or the price trades through them. Cancellations never move them forward (conservative); an optimistic variant is reported for comparison.
- Taker fees and maker rebates are configurable.
- A transient impact overlay (Obizhaeva–Wang style) accounts for the reaction the replay cannot show. It is calibrated to the square-root law and run at three settings: none, medium and strong.
- Anything left at the end is crossed, and anything the book cannot absorb is charged as opportunity cost.

### Strategies

| Strategy | What it does |
|---|---|
| TWAP | Equal slices, baseline tactic |
| VWAP | Slices follow the historical volume curve, baseline tactic |
| Almgren–Chriss | Front-loaded slices, urgency tuned on validation |
| Heuristic adaptive | VWAP plus a simple rule on level-1 queue imbalance |
| ML adaptive (tactic only) | VWAP slices; the model decides post vs. cross |
| ML adaptive (tactic + speed) | The model also speeds up or slows down, within ±10% of the VWAP schedule |

The heuristic sets the bar that the ML strategy must clear. The tactic-only variant shows where any gain comes from.

### Model and decision rule

- **Features:** 18 point-in-time features covering the spread, queue imbalance, weighted-mid gap, depth, order flow imbalance, trade-sign imbalance, activity, recent returns, volatility and time of day.
- **Target:** the mid-price move over the next 30 seconds, which is directly observable in the data. One model serves both sides; the rule flips the sign for sells.
- **Models:** ridge regression (baseline) and XGBoost (main).
- **Rule:** cross and speed up only when the predicted move against us exceeds a tuned multiple of the extra cost of crossing (spread plus fees). Post and slow down when the prediction favours us. Otherwise follow the schedule.

## Evaluation protocol

- **Stage 1:** a single day split by time (train 09:45–12:30, validate 12:40–14:00, test 14:10–16:00), with gaps longer than the label horizon. Leave-one-stock-out is an extra check.
- **Stage 2:** walk-forward by day. For each test day, train on the 10 days before the previous day, tune on the previous day, and test on the day itself.
- All tuning happens on validation data, and each test period is scored once.
- Comparisons are paired: every strategy executes the same parent orders.
- 95% confidence intervals come from a block bootstrap (15-minute blocks in Stage 1, days in Stage 2).
- **Success criterion, fixed in advance:** the adaptive strategy beats VWAP only if the paired 95% interval excludes zero at medium impact with conservative fills, and the direction holds for buys and sells separately and under the other impact settings.

## Results

### Stage 1 (one day; a demonstration, not evidence)

<!-- BEGIN RESULTS: stage1 -->
> Stage 1: one day (27 March 2019) of Nasdaq TotalView-ITCH data from Nasdaq's public sample file. A demonstration, not evidence; intervals from a single day are wide.

Out-of-sample: 510 paired parent orders on AAPL, AMZN, GOOG, INTC, MSFT, test day 2019-03-27; medium impact, conservative fills. Generated by `scripts/report.py`; full tables in `results/stage1_itch/summary.md`.

| Strategy | Mean IS (bps) | Median | Std. dev. | 95th pct | Δ vs VWAP (95% CI) | Share filled passively |
|---|---|---|---|---|---|---|
| TWAP | 3.78 | 3.33 | 14.73 | 27.63 | -0.04 [-0.17, +0.11] | 54.1% |
| VWAP | 3.82 | 3.60 | 15.11 | 29.12 | baseline | 54.3% |
| Almgren-Chriss | 3.78 | 3.78 | 13.01 | 24.77 | -0.05 [-0.17, +0.10] | 53.8% |
| Heuristic adaptive | 3.97 | 3.78 | 14.85 | 28.46 | +0.15 [+0.04, +0.28] | 40.4% |
| ML adaptive (tactic only) | 3.78 | 3.65 | 14.92 | 28.48 | -0.04 [-0.14, +0.08] | 41.2% |
| ML adaptive (tactic + speed) | 3.85 | 3.65 | 14.69 | 28.21 | +0.02 [-0.14, +0.20] | 29.7% |

- **ML adaptive (tactic only):** not shown to be better than VWAP under the pre-registered criterion (Δ -0.04 [-0.14, +0.08] bps).
- **ML adaptive (tactic + speed):** not shown to be better than VWAP under the pre-registered criterion (Δ +0.02 [-0.14, +0.20] bps).
<!-- END RESULTS: stage1 -->

**Key findings** (one day, 27 March 2019; tables in `results/stage1_itch/summary.md`):

1. **Where the cost comes from.** VWAP cost about 3.8 bps per order: 2.35 bps of modeled impact, 0.88 bps of spread and 0.57 bps of drift. About 54% of its shares filled passively, waiting at the back of the queue.
2. **The ML strategies traded spread for drift, and it netted out to zero.** Crossing when the model predicted an adverse move cut drift (0.36 and 0.27 bps against VWAP's 0.57) but paid more spread and taker fees. Net: −0.04 bps [−0.14, +0.08] (tactic only) and +0.02 bps [−0.14, +0.20] (tactic + speed) vs VWAP, so neither meets the pre-registered criterion.
3. **The simple imbalance rule was reliably worse**, +0.15 bps [+0.04, +0.28] vs VWAP, and worse at every impact setting and under both fill assumptions.
4. **Validation flattered the tuned strategies.** On the validation window, ML tactic + speed looked 0.66 bps cheaper than VWAP (2.92 vs 3.58 bps); on the unseen test window the gap disappeared. This is why each test period is scored once and reported as it came out.
5. **The signal is weak.** Predictions correlated with the next 30 s move at about +0.08 to +0.10 on validation (R² near zero). Models chosen on the other four stocks did about as well as own-stock models (−0.04 bps pooled).

### Stage 2

Not yet run.

## What prior research suggests

These points come from published research, not from this project, and they set realistic expectations:

- Order flow imbalance and queue imbalance have well-documented relationships with short-term price changes (Cont, Kukanov and Stoikov, 2014; Stoikov, 2018).
- Their predictive power is weak and fades quickly, over seconds to minutes.
- So the most likely source of any gain is the post-versus-cross decision, not large changes to the schedule.
- Any gain may shrink or disappear after fees and conservative fill assumptions. Measuring that honestly is the point of the project.

## What the results can and cannot tell you

**They can show** whether, for these stocks and periods, order book signals improved the post-versus-cross and speed decisions relative to standard schedules and a simple heuristic, and how sensitive that conclusion is to the impact and fill assumptions.

**They cannot show** how the strategy would perform live. The replay does not react to our orders beyond a modeled overlay, it covers one venue, and it uses a few weeks of data for one or two stocks. The next step in industry would be **shadow mode**: running the model alongside a live algorithm and comparing its decisions without trading.

## What an improvement would be worth

An illustration of scale only, not a result:

| Improvement vs. baseline | Value per $10B traded per year |
|---|---|
| 0.1 bp | $100,000 |
| 0.5 bp | $500,000 |
| 1 bp | $1,000,000 |

Value = notional traded × improvement in bps ÷ 10,000. A result measured on one or two Nasdaq stocks in a replay would not transfer one-to-one to a desk's full flow.

---

## Interview talking points

Replace every bracket with your real results before using these.

**"Tell us about your project."**
> "I built an execution research framework on Nasdaq order book data. The question was whether short-term order book signals, such as order flow imbalance, queue imbalance and the spread, can lower implementation shortfall compared with TWAP, VWAP and Almgren–Chriss. I wrote an event-driven replay simulator with book-walking for marketable orders, a conservative queue model for resting orders, fees, and a transient impact overlay. A model predicts the next 30 seconds of mid-price movement, and a transparent rule turns that into speed and post-versus-cross decisions within a ±10% band around the VWAP schedule. On 510 paired out-of-sample orders across five stocks on one test day, the adaptive strategy was not measurably cheaper than VWAP after fees: −0.04 bps, with a 95% interval of −0.14 to +0.08. It cut drift by crossing ahead of predicted moves, but paid about the same back in spread and fees."

**"Why arrival price and not VWAP?"**
> "The VWAP benchmark measures how closely you tracked the market's average price. A VWAP schedule does that by design, and the benchmark hides the cost of trading in a trending market; it is also affected by your own trades. Implementation shortfall against arrival price captures the full cost from the moment the decision was made, which is why TCA is built around it."

**"What was the hardest part?"**
> "Making the simulator honest. The replay doesn't react to my orders, and I can't know my exact queue position for resting orders. I chose conservative fill assumptions, added an impact overlay at three strengths, and reported how the conclusion changed under each. The conclusion held under all three impact settings and both fill assumptions: no ML gain, and the simple imbalance rule was worse in every case. I also checked the fill logic by recomputing 1,080 resting-order slices independently from the raw events; all of them matched."

**"How did you avoid look-ahead bias?"**
> "Features use only data at or before each decision time, sampled with a backward as-of join, and a unit test checks that deleting future data doesn't change past features. Normalizers come from training data only, there are gaps between the training, validation and test periods, the VWAP volume curve comes from earlier days, and each test period was scored once."

**"Why XGBoost rather than deep learning or reinforcement learning?"**
> "The signal is weak and the data limited, so I started with a ridge baseline and shallow gradient-boosted trees, which are easy to inspect. For RL, the agent learns by acting, and a replay doesn't react to its actions, so it would learn to exploit the simulator. I'd want a reactive simulator first."

**"How would this work in production?"**
> "In a real execution stack, a parent order comes from the order management system into the algo engine, which runs the scheduler, the micro-tactics and a smart order router across venues, with risk checks throughout. My project covers the scheduler and tactics layer plus post-trade TCA. It doesn't cover routing, live risk or client flow. The next step would be shadow mode: run the model beside the live algorithm and compare decisions without trading."

**"What if the result was negative?"**
> "Then I'd say so and explain why using the cost breakdown, for example whether resting orders were adversely selected or whether the signal faded faster than the 30-second decision interval. Knowing when a strategy doesn't pay after costs is as valuable as finding one that does."

---

## Extensions

1. **Exact queue tracking.** LOBSTER includes order IDs, so queue position can be tracked exactly for visible orders instead of conservatively.
2. **Fill-probability model.** Predict the chance a resting order fills within the slice, and use it directly in the post-versus-cross rule.
3. **More algorithms.** Percentage-of-volume and closing-auction strategies.
4. **Multi-venue routing.** Add a second venue and a simple smart order router.
5. **A reactive simulator.** An agent-based or calibrated market model, which would make reinforcement learning meaningful.
6. **Deep order book models** such as DeepLOB, once enough data is available.
7. **Canadian market data.** Applying the same pipeline to TSX-listed stocks.

## What this project demonstrates

1. **Market structure:** how order books, queues, spreads and fees actually determine execution cost.
2. **The industry's measurement standard:** implementation shortfall, cost decomposition and TCA.
3. **Careful ML:** point-in-time features, walk-forward validation, fair baselines, a pre-set success criterion and honest uncertainty.
4. **Engineering:** a tested, reproducible simulator with CI.
5. **Judgment:** stating clearly what the results can and cannot show.

## References

- Perold, A. (1988). The implementation shortfall: Paper versus reality. *Journal of Portfolio Management.*
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk.*
- Obizhaeva, A., & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets.*
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics.*
- Stoikov, S. (2018). The micro-price: A high-frequency estimator of future prices. *Quantitative Finance.*
- Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope.* Cambridge University Press.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for optimized trade execution. *Proceedings of ICML.*
- Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing.*
- Huang, R., & Polak, T. (2011). LOBSTER: Limit Order Book Reconstruction System. SSRN working paper.
