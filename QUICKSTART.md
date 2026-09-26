# Adaptive Execution Engine — Build Guide

A step-by-step plan for a small-scale but industry-faithful execution research project. It uses the same question, benchmark and validation discipline as an electronic trading desk, on free or low-cost Nasdaq order book data.

**Time:** about 4–6 weeks at a few hours a day.
**Stack:** Python 3.10+, pandas, NumPy, scikit-learn, XGBoost, pytest.

---

## Ground rules

1. **Never write a number you did not produce.** This applies to the README, your resume, cover letters and interviews. The results tables stay empty until the backtest fills them.
2. **Each test period is scored once.** If you change anything after seeing test results, that period is spent. Say so, or evaluate on a fresh period.
3. **Point-in-time only.** Every feature at time *t* uses only data at or before *t*. A unit test enforces this.
4. **Label every proxy and assumption:** the 2012 data, the proxy volume curve, the fee values and the impact overlay.
5. **A negative result, well explained, is a good result.** Desks see many strategies that fail after costs. Showing that you can tell the difference is the skill they are hiring for.

## Timeline

| Phase | What you build | Time |
|---|---|---|
| 0 | Repository, environment, data download | 1 day |
| 1 | Load and explore the order book | 2–3 days |
| 2 | Execution simulator and its unit tests | 5–7 days |
| 3 | Baseline schedules and cost measurement | 2–3 days |
| 4 | Features, target and model | 3–4 days |
| 5 | Adaptive strategies and the Stage 1 backtest | 3–4 days |
| 6 | Stage 2: multi-day walk-forward on Databento data | 4–5 days |
| 7 | Report, README results, resume | 2 days |

Phase 2 is the largest on purpose. The simulator decides whether any result can be trusted.

---

## Phase 0 — Setup and data (1 day)

### Repository and environment

```bash
git clone https://github.com/<your-username>/adaptive-execution.git
cd adaptive-execution
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import adaptive_exec` work from anywhere
```

`requirements.txt`:

```
pandas>=2.0
numpy>=1.24
pyarrow>=14
scikit-learn>=1.3
xgboost>=2.0
pyyaml>=6.0
matplotlib>=3.7
yfinance
databento
pytest>=7.4
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "adaptive-exec"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

`.gitignore` must include `data/` and `.venv/`. Market data licenses generally forbid redistribution, so raw files never go into Git.

### LOBSTER sample (Stage 1)

1. Go to the LOBSTER website and open the free data samples page. As of September 2026 the samples are behind a request form: accept the terms, pass the human check, and wait for the request to be reviewed before the download is released.
2. Download the 10-level sample for AAPL, AMZN, GOOG, INTC and MSFT (21 June 2012).
3. Unzip into `data/raw/`. Each stock has a message file and an order book file, named like `MSFT_2012-06-21_34200000_57600000_message_10.csv` and `MSFT_2012-06-21_34200000_57600000_orderbook_10.csv`, plus a readme. Read the readme, since it defines every column.

If the request is slow or refused, run Stage 1 on one recent Databento day instead: `configs/stage1_databento.yaml` has the same design, with its day kept outside the Stage 2 range. Download it with `python scripts/download_databento.py --config configs/stage1_databento.yaml` (cost estimate first; nothing downloads without `--confirm`).

### Proxy volume curve (Stage 1 only)

The VWAP schedule needs to know how volume is usually spread across the day, and that must come from *other* days. Using the test day's own volume would be look-ahead. With only one LOBSTER day, use recent Yahoo data as a proxy:

```python
import yfinance as yf

bars = yf.Ticker("MSFT").history(period="60d", interval="5m")   # intraday bars: last 60 days only
bars = bars[bars["Volume"] > 0]
day_total = bars.groupby(bars.index.date)["Volume"].transform("sum")
share = bars["Volume"] / day_total
curve = share.groupby(bars.index.strftime("%H:%M")).mean()
curve = curve / curve.sum()
curve.to_csv("data/processed/volume_curve_proxy_MSFT.csv")
```

`python scripts/fetch_volume_curve.py --config configs/stage1_lobster.yaml` does this for all five stocks. This is recent consolidated volume, not 2012 Nasdaq volume. The part that matters is the U-shape (busy open and close, quiet midday). Label it as a proxy in the write-up. In Stage 2 the curve comes from earlier Databento days instead.

**Done when:** all ten LOBSTER files are in `data/raw/` and a proxy curve is saved for each stock.

---

## Phase 1 — Load and explore the order book (2–3 days)

### Loader

```python
import numpy as np
import pandas as pd

def load_lobster(msg_path, book_path, date="2012-06-21", levels=10):
    msg = pd.read_csv(msg_path, header=None, usecols=range(6),
                      names=["time", "type", "order_id", "size", "price", "direction"])
    cols = [f"{side}_{kind}_{i}" for i in range(1, levels + 1)
            for side, kind in (("ask", "px"), ("ask", "sz"), ("bid", "px"), ("bid", "sz"))]
    book = pd.read_csv(book_path, header=None, names=cols)

    # Prices are stored as dollars x 10,000
    px_cols = [c for c in cols if "_px_" in c]
    book[px_cols] = book[px_cols] / 10_000
    msg["price"] = msg["price"] / 10_000

    # Empty levels hold dummy prices (very large for asks, negative for bids): mask them
    for i in range(1, levels + 1):
        book.loc[book[f"ask_px_{i}"] > 100_000, [f"ask_px_{i}", f"ask_sz_{i}"]] = np.nan
        book.loc[book[f"bid_px_{i}"] < 0, [f"bid_px_{i}", f"bid_sz_{i}"]] = np.nan

    df = pd.concat([msg, book], axis=1)            # book row k = the state right after message k
    # One timestamp type everywhere (pandas 3 can mix ns and us, which breaks merge_asof)
    df["ts"] = (pd.Timestamp(date) + pd.to_timedelta(df["time"], unit="s")).astype("datetime64[ns]")
    df = df[df["type"] != 7]                       # drop trading-halt markers
    df["mid"] = (df["ask_px_1"] + df["bid_px_1"]) / 2
    df["spread"] = df["ask_px_1"] - df["bid_px_1"]
    return df.reset_index(drop=True)
```

LOBSTER's `time` is seconds after midnight, New York time, so 34200 is 09:30 and 57600 is 16:00.

### Trades and who started them

```python
trades = df[df["type"].isin([4, 5])].copy()      # 4 = visible execution, 5 = hidden execution
# `direction` is the side of the RESTING order that was hit:
# a sell limit order executed (-1) means a buyer-initiated trade.
trades["aggressor"] = -trades["direction"]       # +1 buyer-initiated, -1 seller-initiated
```

Type 6 (auction crosses) is not continuous trading, so leave it out of trade-flow features.

### Sanity checks

- Timestamps run from about 34200 to 57600 seconds and never go backwards.
- Prices are plausible. MSFT traded near $30 in June 2012.
- `spread > 0` on almost every row. Count the rows where it isn't (locked, crossed or one-sided books) and exclude them from features.
- Every execution's price matches the best bid or ask in the previous book row.

### Four plots for `notebooks/01_explore_order_book.ipynb`

1. Mid-price through the day.
2. Histogram of the spread in ticks (1 tick = $0.01), MSFT next to AAPL.
3. Level-1 imbalance, grouped into 10 bins, against the average mid-price move over the next 30 s. This is the first sign of whether the signal exists.
4. Trades per minute through the day, which should show the U-shape.

**What to notice:** MSFT and INTC (around $30) usually have a one-tick spread and long queues, so the spread barely changes and queue imbalance carries most of the information. These are called large-tick stocks. AAPL, AMZN and GOOG (hundreds of dollars) have spreads of several ticks that move around a lot. Expect the model and the value of posting to differ between the two groups.

**Done when:** you can explain each plot in two sentences.

---

## Phase 2 — The execution simulator (5–7 days)

This is the heart of the project. Build it with tests before any modeling.

### 2.1 The replay loop

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class ParentOrder:
    side: int                 # +1 buy, -1 sell
    qty: int                  # shares
    start: pd.Timestamp
    horizon_s: int = 1800     # 30 minutes
    slice_s: int = 30         # one decision every 30 seconds
```

For each parent order:

```
arrival_mid = mid at start
schedule    = strategy.schedule(qty, n_slices)           # shares per 30 s slice

for each decision time t_k (every 30 s):
    cancel any unfilled resting child order
    tactic, child_qty = strategy.decide(state at t_k, progress vs schedule)
    if tactic is "cross":  walk the book in the snapshot at t_k
    if tactic is "post":   rest at the best price, back of the queue, then feed it
                           every historical trade in (t_k, t_k+1]
    record each fill: time, price, historical mid, impact push, fee

at the horizon: cross what remains; charge opportunity cost for anything still unfilled
compute IS and its components
```

### 2.2 Crossing the spread: walking the book

```python
import numpy as np

def walk_book(qty, levels):
    """levels: [(price, size), ...] on the opposite side, best first.
    Returns the fills and the shares the displayed book could not absorb."""
    fills, left = [], qty
    for px, sz in levels:
        if left == 0 or np.isnan(px):
            break
        take = int(min(left, sz))
        if take > 0:
            fills.append((px, take))
            left -= take
    return fills, left
```

### 2.3 Posting in the book: a conservative queue model

```python
class PassiveOrder:
    """Limit order resting at `px`, at the back of the queue. side: +1 buy, -1 sell."""

    def __init__(self, side, px, qty, queue_ahead):
        self.side, self.px, self.qty = side, px, qty
        self.ahead, self.filled = queue_ahead, 0     # queue_ahead = displayed size at px when posted

    def on_trade(self, trade_px, trade_qty, aggressor):
        """Feed one historical trade. Returns the shares filled for us."""
        remaining = self.qty - self.filled
        if remaining == 0 or aggressor != -self.side:   # only opposite-side aggressors can hit us
            return 0
        through = trade_px < self.px if self.side == 1 else trade_px > self.px
        if through:                                     # the market traded through our price
            fill = remaining
        elif trade_px == self.px:                       # traded at our price: the queue ahead goes first
            used = min(self.ahead, trade_qty)
            self.ahead -= used
            fill = min(trade_qty - used, remaining)
        else:
            return 0
        self.filled += fill
        return fill
```

Cancellations never reduce `ahead`, which is the conservative assumption. For the optimistic variant, add an `on_cancel` method that reduces `ahead` by the cancelled size, as if every cancellation came from ahead of us. Report both.

**Tip:** compare prices as integers (LOBSTER's raw units of $0.0001) inside the simulator to avoid floating-point equality problems.

### 2.4 Fees

Each crossing fill pays the taker fee. Each resting fill earns the maker rebate, recorded as a negative fee. Keep both as config values. For a $30 stock, a $0.003 fee is 1 bp, which is about the size of half the spread, so fees can change which tactic wins.

### 2.5 Market impact overlay

```python
class TransientImpact:
    """Obizhaeva-Wang style: each fill pushes our later prices against us; the push decays."""

    def __init__(self, lam_bps_per_share, half_life_s):
        self.lam = lam_bps_per_share
        self.decay = np.log(2) / half_life_s
        self.level_bps, self.t_last = 0.0, None

    def current_bps(self, t):
        if self.t_last is None:
            return 0.0
        return self.level_bps * np.exp(-self.decay * (t - self.t_last))

    def add_fill(self, t, qty):
        self.level_bps = self.current_bps(t) + self.lam * qty
        self.t_last = t
```

Each fill price is adjusted by the push that exists just before it: `adjusted_px = px * (1 + side * impact.current_bps(t) / 1e4)`, and then `impact.add_fill(t, qty)` is called.

**Calibration:** choose `lam` so that a typical parent order executed on the VWAP schedule reaches a peak push of about `Y × σ_daily × sqrt(Q / V_daily)` bps, which is the square-root impact law. Here σ_daily is daily volatility in bps, Q is the order size and V_daily is daily volume. Using Nasdaq-only volume for V_daily makes the estimate higher, which is acceptable for a stress test. These are simulator settings, not strategy inputs.

| Setting | Y | Half-life |
|---|---|---|
| None | 0 | — |
| Medium | 0.5 | 5 minutes |
| Strong | 1.0 | 10 minutes |

One limitation to state: the overlay changes our prices but not which resting orders fill.

### 2.6 End of the horizon

Cross whatever remains in the last slice. If the displayed book cannot absorb it, charge the rest as opportunity cost at the final mid plus half the spread.

### 2.7 Traps to avoid

- Letting a resting order fill from trades that happened before it was posted.
- Crossing against the book snapshot *after* the event you are reacting to.
- Taking the same displayed liquidity twice. Cross at most once per decision, so the book has 30 s to refresh before the next crossing.

### 2.8 Unit tests

The tests use small hand-built books and a synthetic event stream from `tests/conftest.py`, so CI never needs market data.

```python
# tests/test_fills.py
from adaptive_exec.fills import walk_book, PassiveOrder

def test_walk_book_uses_best_prices_first():
    asks = [(10.00, 500), (10.01, 300), (10.02, 1000)]
    fills, left = walk_book(700, asks)
    assert fills == [(10.00, 500), (10.01, 200)]
    assert left == 0

def test_resting_buy_waits_for_the_queue_ahead():
    order = PassiveOrder(side=+1, px=9.99, qty=500, queue_ahead=1000)
    assert order.on_trade(9.99, 600, aggressor=-1) == 0      # queue ahead: 1000 -> 400
    assert order.on_trade(9.99, 600, aggressor=-1) == 200    # 400 clears the queue, 200 fill us
    assert order.on_trade(9.98, 100, aggressor=-1) == 300    # traded through: the rest fills
    assert order.filled == 500

def test_buyer_initiated_trades_do_not_fill_a_resting_buy():
    order = PassiveOrder(side=+1, px=9.99, qty=500, queue_ahead=0)
    assert order.on_trade(9.99, 1000, aggressor=+1) == 0
```

```python
# tests/test_impact.py
import pytest
from adaptive_exec.impact import TransientImpact

def test_impact_halves_after_one_half_life():
    imp = TransientImpact(lam_bps_per_share=0.001, half_life_s=300)
    imp.add_fill(t=0.0, qty=1000)                  # push = 1 bp
    assert imp.current_bps(300.0) == pytest.approx(0.5)
```

**Done when:** `pytest` passes, and you have checked one parent order's fills by hand in a notebook. `notebooks/01_explore_order_book.ipynb` (section 5) recomputes one slice's resting fills straight from the raw events and checks the cost breakdown; `scripts/inspect_order.py` prints any validation order decision by decision.

---

## Phase 3 — Baseline schedules and cost measurement (2–3 days)

### Schedules

```python
import numpy as np

def to_whole_shares(slices, total_qty):
    whole = np.floor(slices).astype(int)
    whole[-1] += total_qty - whole.sum()           # put the rounding remainder in the last slice
    return whole

def twap(total_qty, n_slices):
    return to_whole_shares(np.full(n_slices, total_qty / n_slices), total_qty)

def vwap(total_qty, curve):
    """curve: expected share of volume in each slice, from PAST days only."""
    w = np.asarray(curve, dtype=float)
    return to_whole_shares(total_qty * w / w.sum(), total_qty)

def almgren_chriss(total_qty, n_slices, kappa_T):
    """Front-loaded: shares remaining = X * sinh(kappa_T * (1 - t)) / sinh(kappa_T).
    kappa_T -> 0 gives TWAP; larger kappa_T trades more urgently."""
    if kappa_T < 1e-6:
        return twap(total_qty, n_slices)
    t = np.linspace(0.0, 1.0, n_slices + 1)
    remaining = total_qty * np.sinh(kappa_T * (1 - t)) / np.sinh(kappa_T)
    return to_whole_shares(-np.diff(remaining), total_qty)
```

```python
# tests/test_schedules.py
import numpy as np
from adaptive_exec.schedules import twap, vwap, almgren_chriss

def test_schedules_trade_the_whole_order():
    for s in (twap(10_000, 60), vwap(10_000, np.linspace(2, 1, 60)), almgren_chriss(10_000, 60, 2.0)):
        assert s.sum() == 10_000
        assert (s >= 0).all()

def test_almgren_chriss_is_front_loaded_and_reduces_to_twap():
    assert almgren_chriss(10_000, 60, 2.0)[0] > almgren_chriss(10_000, 60, 2.0)[-1]
    assert (almgren_chriss(10_000, 60, 1e-9) == twap(10_000, 60)).all()
```

### Implementation shortfall

```python
def implementation_shortfall_bps(side, arrival_mid, fills, fees):
    """fills: [(price, qty), ...] after the impact adjustment; fees: net dollars (rebates negative)."""
    qty = sum(q for _, q in fills)
    avg_px = sum(p * q for p, q in fills) / qty
    price_cost = side * (avg_px - arrival_mid) / arrival_mid * 1e4
    fee_cost = fees / (arrival_mid * qty) * 1e4
    return price_cost + fee_cost
```

```python
# tests/test_metrics.py
import pytest
from adaptive_exec.metrics import implementation_shortfall_bps

def test_paying_up_is_a_positive_cost_for_both_sides():
    assert implementation_shortfall_bps(+1, 100.0, [(100.05, 1000)], fees=0.0) == pytest.approx(5.0)
    assert implementation_shortfall_bps(-1, 100.0, [(99.95, 1000)], fees=0.0) == pytest.approx(5.0)
```

**Decomposition.** For each fill with raw price *p*, impact-adjusted price *p′* and historical mid *m*, with arrival mid *A*:

- spread = side × (p − m) / A
- impact = side × (p′ − p) / A
- drift = side × (m − A) / A

Weight each by the fill's shares, convert to bps and add fees and opportunity cost. The five parts add up exactly to the total IS, which makes a good unit test.

### Parent orders

```python
import itertools

def make_parent_orders(window_start, window_end, expected_volume_fn,
                       targets=(0.02, 0.05, 0.10), every_min=5, horizon_min=30):
    last_start = pd.Timestamp(window_end) - pd.Timedelta(minutes=horizon_min)
    starts = pd.date_range(window_start, last_start, freq=f"{every_min}min")
    orders = []
    for start, p, side in itertools.product(starts, targets, (+1, -1)):
        qty = int(round(p * expected_volume_fn(start, horizon_min)))
        orders.append(ParentOrder(side=side, qty=qty, start=start, horizon_s=horizon_min * 60))
    return orders
```

`expected_volume_fn` returns the expected Nasdaq volume over the order's horizon. In Stage 1, use the day's total Nasdaq volume times the proxy curve's share for that window. This uses the same day's total, which is acceptable only because order size is a test setting that no strategy sees; say so in the write-up. In Stage 2, use the average of earlier days.

The Stage 1 test window (14:10–16:00) gives 17 start times × 3 sizes × 2 sides = 102 parent orders per stock. Report the realized participation (our shares ÷ Nasdaq volume over the horizon) as a check.

### Run the baselines on validation

Run TWAP, VWAP and Almgren–Chriss on the validation window (12:40–14:00). Choose κT for Almgren–Chriss from {0.5, 1, 2, 4} by mean IS.

**Sanity checks:**

- On a day the price rose, buys show a positive drift cost and sells a negative one. Averaged over buys and sells, drift mostly cancels, which is why every test uses both sides.
- Posting-heavy tactics should show a negative spread component (earning part of the spread) but can show worse drift, because resting orders tend to fill just before the price moves against them. This is called adverse selection.
- Over 30 minutes at midday, the volume curve is nearly flat, so TWAP and VWAP should behave similarly there. Large differences deserve a look.

**Done when:** a validation table with all three baselines and their cost decomposition exists.

---

## Phase 4 — Features, target and model (3–4 days)

### The 18 features

All are computed from events at or before time *t*, then sampled every second.

| # | Feature | Definition |
|---|---|---|
| 1 | `spread_ticks` | (ask₁ − bid₁) / $0.01 |
| 2 | `spread_bps` | (ask₁ − bid₁) / mid × 10⁴ |
| 3 | `imb_l1` | (bid size₁ − ask size₁) / (bid size₁ + ask size₁) |
| 4 | `imb_l5` | Same, using total size in levels 1–5 |
| 5 | `micro_gap_bps` | (weighted mid − mid) / mid × 10⁴, where weighted mid = (ask₁ × bid size₁ + bid₁ × ask size₁) / (bid size₁ + ask size₁) |
| 6 | `depth_l5_log` | log of total shares on both sides in levels 1–5 |
| 7–9 | `ofi_10s`, `ofi_30s`, `ofi_60s` | Order flow imbalance summed over the window, divided by the training-period average level-1 depth |
| 10–11 | `trade_imb_30s`, `trade_imb_60s` | (buyer-initiated − seller-initiated volume) / total volume over the window; 0 if no trades |
| 12 | `trade_count_60s` | Number of trades in the last 60 s |
| 13 | `volume_60s_rel` | Volume in the last 60 s ÷ the training-period average 60 s volume |
| 14–16 | `ret_10s`, `ret_60s`, `ret_300s` | Mid-price change over the window, bps |
| 17 | `rv_300s` | Realized volatility of 1-second mid returns over the last 5 minutes, bps |
| 18 | `minutes_since_open` | Minutes since 09:30 |

Normalizers (features 7–9 and 13) come from the training period only.

### Splits

Stage 1 uses one day split by time: train 09:45–12:30, validate 12:40–14:00, test 14:10–16:00. The first 15 minutes are skipped because the open behaves differently. The 10-minute gaps are longer than the 30-second label horizon, so no training label uses prices from the validation or test windows.

### Order flow imbalance (Cont, Kukanov and Stoikov, 2014)

```python
def add_ofi(df):
    bp, bq = df["bid_px_1"], df["bid_sz_1"]
    ap, aq = df["ask_px_1"], df["ask_sz_1"]
    e = ((bp >= bp.shift()) * bq - (bp <= bp.shift()) * bq.shift()
         - (ap <= ap.shift()) * aq + (ap >= ap.shift()) * aq.shift())
    df["ofi_event"] = e.fillna(0.0)
    ev = df.set_index("ts")["ofi_event"]
    for w in ("10s", "30s", "60s"):
        df[f"ofi_{w}_raw"] = ev.rolling(w).sum().to_numpy()   # each window ends at, and includes, the event
    return df
```

Positive OFI means buying pressure: bids growing or moving up, or asks shrinking or moving up.

### Sampling on a 1-second grid, and the target

```python
grid = pd.DataFrame({"ts": pd.date_range(start, end, freq="1s", unit="ns")})

# Last known state at or before each grid time. LOBSTER rows are already in time order;
# don't re-sort with an unstable sort, or same-timestamp rows can change order.
snap = pd.merge_asof(grid, events[["ts"] + feature_inputs], on="ts", direction="backward")

# Target: mid-price move over the next 30 s, in bps
ahead = pd.merge_asof(grid.assign(ts=grid["ts"] + pd.Timedelta(seconds=30)),
                      events[["ts", "mid"]], on="ts", direction="backward")
snap["y_bps"] = (ahead["mid"].to_numpy() / snap["mid"].to_numpy() - 1) * 1e4
```

**Trap:** `df.resample("1s").last()` labels each bin by its left edge by default. The row labelled 10:00:00 then holds data up to 10:00:00.999, which is a hidden look-ahead. Use `merge_asof` as above, or `resample("1s", label="right", closed="right")`.

### Point-in-time test

```python
# tests/test_no_lookahead.py
import pandas as pd
from adaptive_exec.features import build_features, FEATURE_COLUMNS

def test_features_do_not_use_the_future(events):              # `events` is a synthetic fixture
    cut = events["ts"].iloc[len(events) // 2]
    full = build_features(events).set_index("ts")
    past = build_features(events[events["ts"] <= cut]).set_index("ts")
    rows = past.index[past.index <= cut]
    pd.testing.assert_frame_equal(full.loc[rows, FEATURE_COLUMNS], past.loc[rows, FEATURE_COLUMNS])
```

If deleting the future changes any past feature, something is leaking.

### Models

```python
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lo, hi = np.percentile(y_train, [0.5, 99.5])     # clip extreme targets using TRAINING data only
y_tr, y_va = np.clip(y_train, lo, hi), np.clip(y_val, lo, hi)

ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_train, y_tr)   # scaler fit on train only

booster = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
    early_stopping_rounds=100, eval_metric="rmse",
)
booster.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
```

Shallow trees and a large `min_child_weight` suit a noisy target.

**Prediction metrics (secondary):** out-of-sample R², correlation between prediction and outcome, and the sign hit rate on moves that are not zero. Large-tick stocks often show no mid change in 30 s, so count hits only on non-zero moves.

**What to expect:** predictive R² for 30-second returns is usually small, often a few percent or less. That is normal. A weak signal applied thousands of times can still matter, and the backtest, not R², decides.

**Done when:** both models are trained, you have validation metrics and a feature importance plot, and the no-look-ahead test passes.

---

## Phase 5 — Adaptive strategies and the Stage 1 backtest (3–4 days)

### Decision rules

```python
def ml_decision(side, pred_bps, spread_bps, taker_bps, rebate_bps, m, alpha):
    """Returns (tactic, speed multiplier) for this slice."""
    signed = side * pred_bps                       # > 0: price expected to move against us
    cross_cost = spread_bps + taker_bps + rebate_bps
    if signed > m * cross_cost:
        return "cross", 1 + alpha
    if signed < -m * cross_cost:
        return "post", 1 - alpha
    return "baseline", 1.0

def heuristic_decision(side, imb_l1, theta, alpha):
    signed = side * imb_l1                         # buying into a bid-heavy book: price likely to rise
    if signed > theta:
        return "cross", 1 + alpha
    if signed < -theta:
        return "post", 1 - alpha
    return "baseline", 1.0

def clamp_to_band(planned_cum, schedule_cum, total_qty, band=0.10):
    lo = max(0, schedule_cum - band * total_qty)
    hi = min(total_qty, schedule_cum + band * total_qty)
    return int(np.clip(planned_cum, lo, hi))
```

Each slice, the planned cumulative quantity (shares done + scheduled slice × speed multiplier) is clamped to the band. If the order has fallen below the band, the shortfall is crossed as a catch-up. Fee and spread values are converted to bps at the current mid.

### Tuning on validation

Pick the setting with the lowest mean IS on the validation window, at medium impact with conservative fills:

- ML adaptive: *m* ∈ {0.25, 0.5, 1, 2}, *α* ∈ {0.25, 0.5}, and ridge vs. XGBoost
- ML adaptive (tactic only): *m* ∈ {0.25, 0.5, 1, 2}, *α* = 0
- Heuristic: *θ* ∈ {0.3, 0.5, 0.7}, *α* ∈ {0.25, 0.5}

Every strategy gets the same validation data and a similar tuning budget, so the comparison is fair.

### The test run (once)

Tune first with `python scripts/backtest.py --config configs/stage1_lobster.yaml --validate-only`, which never touches the test window. Then run all six strategies on the test window (14:10–16:00) for every stock, under all three impact settings and both fill assumptions: `python scripts/backtest.py --config configs/stage1_lobster.yaml`. It refuses to score a test period twice unless given `--rescore --reason "..."`, and logs every scoring in `results/stage1/scoring_log.csv`. `scripts/report.py` writes `results/stage1/summary.md`, with bootstrap confidence intervals from 15-minute blocks. With a single day these intervals will be wide. Say so; that is what Stage 2 is for.

**Extra check, leave-one-stock-out:** train and tune on four stocks, test on the fifth. This shows whether the signal generalizes across stocks, especially between large-tick and small-tick names.

**Done when:** `results/stage1/summary.md` exists and you can explain where each strategy's cost came from.

---

## Phase 6 — Stage 2: multi-day walk-forward (4–5 days)

### Get the data

1. Create a Databento account; new accounts get $125 in free credits for historical data.
2. Put your API key in an environment variable, never in code or Git.
3. Estimate the cost before downloading anything:

```python
import databento as db

client = db.Historical()          # reads the DATABENTO_API_KEY environment variable

params = dict(
    dataset="XNAS.ITCH",          # Nasdaq TotalView-ITCH
    symbols=["MSFT"],
    schema="mbp-10",              # top 10 levels on every book update, trades included
    start="2026-03-02T14:30",     # UTC: 09:30 New York is 14:30 UTC in winter, 13:30 UTC in summer
    end="2026-03-02T21:00",
)
print(f"Estimated cost: ${client.metadata.get_cost(**params):.2f}")

# Only after checking the cost:
# df = client.timeseries.get_range(**params).to_df()
```

Aim for 10–20 consecutive trading days of one or two liquid stocks. If `mbp-10` is too expensive, pick a less active (but still liquid) stock, or fall back to `mbp-1` (top of book only). With `mbp-1`, drop features 4 and 6 and limit each crossing to the displayed level-1 size.

### Loader conventions

Map Databento's columns into the same internal format as LOBSTER, so everything downstream runs unchanged. Databento timestamps are in UTC; convert them to New York time and to the same `datetime64[ns]` type as the LOBSTER loader. Watch one trap: in LOBSTER, an execution's direction is the side of the **resting** order, while in Databento a trade's side is the **aggressor** (`B` = buyer-initiated, `A` = seller-initiated, `N` = not specified). Confirm this against Databento's schema documentation, and add a unit test for each loader's trade sign.

### Walk-forward

For each test day *d*: train on the 10 trading days before *d − 1*, tune on *d − 1*, test on *d*. The volume curve for the VWAP schedule, the parent order sizes and all feature normalizers come from days before *d*. With 20 days, that gives 9 test days.

Confidence intervals come from a bootstrap that resamples whole test days. Apply the success criterion from the README exactly as written.

**Done when:** `results/stage2/summary.md` holds the out-of-sample tables for Stage 2.

---

## Phase 7 — Write-up (2 days)

1. Run `python scripts/report.py --config configs/stage2_databento.yaml --readme README.md`, which writes the tables from `results/stage2/summary.md` into the README between the stage's markers, then remove the "Status: in progress" note.
2. Write three to five findings in plain language, each backed by a table: where the cost came from, whether the adaptive strategies beat VWAP and the heuristic, and whether the result held across sides, sizes, stocks, impact settings and fill assumptions.
3. List what surprised you and what you would test next.
4. Update the resume bullets below with your real numbers.

---

## Resume bullets (templates)

Fill the brackets only with numbers from `results/`. Delete any bullet that isn't true.

```
• Built an event-driven execution simulator on Nasdaq limit order book data (LOBSTER, Databento)
  with book-walking marketable orders, a conservative queue model for resting orders, fees, and a
  transient market-impact overlay calibrated to the square-root law

• Implemented TWAP, VWAP and Almgren–Chriss schedules and an adaptive strategy that uses an XGBoost
  model on 18 order-flow and book features to choose post vs. cross and adjust speed within a ±10% band

• Evaluated [N] paired parent orders over [D] out-of-sample days with walk-forward validation and
  block-bootstrap confidence intervals; the adaptive strategy changed implementation shortfall by
  [X] bps vs. VWAP after fees (95% CI [a, b])

• Wrote unit tests for fill logic, schedules, cost accounting and point-in-time features, including
  an automated look-ahead check, run in CI on every commit
```

If the result is negative, write it that way, for example: "…found no statistically significant improvement vs. VWAP after fees; the cost breakdown traced this to [your finding]."

## Cover letter paragraph (template)

> To understand how electronic execution works in practice, I built an execution research framework on Nasdaq limit order book data. It replays the order book event by event, simulates marketable and resting child orders with conservative fill assumptions and fees, and compares TWAP, VWAP and Almgren–Chriss schedules with an adaptive strategy that uses order-flow signals to decide when to post and when to cross. Every strategy is measured by implementation shortfall against the arrival price on out-of-sample days. [One sentence stating your actual finding.] I would like to bring the same care about measurement to CIBC's Electronic Trading team.

Interview preparation is in `PROJECT_SUMMARY.md`.

---

## CI workflow

`.github/workflows/ci.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt && pip install -e .
      - run: pytest
```

---

## FAQ

**Why not Yahoo Finance for everything?**
It only has bars (open, high, low, close, volume), with no quotes, depth or individual trades, and intraday history only goes back 60 days. The spread and imbalance signals, and the resting-order fill model, all need the order book.

**Is 2012 data still useful?**
For building and learning, yes. Price-time priority, queues and spreads work the same way. Fees, tick sizes and speed have changed, which is why Stage 2 uses recent data.

**Why XGBoost and not an LSTM or DeepLOB?**
The signal is noisy and the data is limited. A ridge baseline plus gradient boosting is the sensible first step, and it is easy to inspect. Deep order book models need far more data; they are a natural extension once Stage 2 works.

**Why not reinforcement learning?**
RL learns by acting, and a replay does not react to its actions. An RL agent would learn to exploit the simulator's blind spots, such as optimistic fills. A reactive simulator is the prerequisite, so RL is listed as an extension.

**Should I add a FastAPI server or Docker?**
Both are optional. In real execution systems, decision logic runs inside the algo engine rather than behind a web call, and latency is not the point with 30-second decisions. A clean package, tests, CI and one-command reproducibility show more. A Dockerfile is worth adding only for reproducibility.

**What if the ML strategy loses to the heuristic?**
Report it. It means the extra model complexity did not pay for itself here, which is exactly what a desk would want to know. Then use the cost decomposition to explain why.

**How many orders do I need?**
Enough that the confidence intervals can separate the strategies. In Stage 1 they will be wide. In Stage 2 the independent unit is the day, so more days matter more than more orders per day.

---

## Final checklist

- [ ] Unit tests pass for fills, schedules, impact, metrics and point-in-time features
- [ ] One parent order checked by hand
- [ ] Baselines run and κT chosen on validation
- [ ] Features and target built with `merge_asof`, normalizers from training only
- [ ] Ridge and XGBoost trained; choice made on validation by execution cost
- [ ] All tuning done on validation; each test period scored once
- [ ] Stage 1 results with wide intervals clearly labelled as a demonstration
- [ ] Stage 2 walk-forward results with day-level bootstrap intervals
- [ ] Success criterion applied exactly as written in the README
- [ ] Robustness tables: side, size, stock, impact setting, fill assumption
- [ ] README results filled from `results/`, with no numbers typed by hand
- [ ] Every proxy and assumption labelled
- [ ] Resume and cover letter use only real numbers
