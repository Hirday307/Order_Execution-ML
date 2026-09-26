from adaptive_exec.fills import PassiveOrder, walk_book


def test_walk_book_uses_best_prices_first():
    asks = [(10.00, 500), (10.01, 300), (10.02, 1000)]
    fills, left = walk_book(700, asks)
    assert fills == [(10.00, 500), (10.01, 200)]
    assert left == 0


def test_walk_book_reports_what_the_book_cannot_absorb():
    fills, left = walk_book(1000, [(10.00, 300), (float("nan"), 0)])
    assert fills == [(10.00, 300)] and left == 700


def test_resting_buy_waits_for_the_queue_ahead():
    order = PassiveOrder(side=+1, px=9.99, qty=500, queue_ahead=1000)
    assert order.on_trade(9.99, 600, aggressor=-1) == 0      # queue ahead: 1000 -> 400
    assert order.on_trade(9.99, 600, aggressor=-1) == 200    # 400 clears the queue, 200 fill us
    assert order.on_trade(9.98, 100, aggressor=-1) == 300    # traded through: the rest fills
    assert order.filled == 500


def test_buyer_initiated_trades_do_not_fill_a_resting_buy():
    order = PassiveOrder(side=+1, px=9.99, qty=500, queue_ahead=0)
    assert order.on_trade(9.99, 1000, aggressor=+1) == 0


def test_resting_sell_mirrors_the_buy():
    order = PassiveOrder(side=-1, px=10.01, qty=300, queue_ahead=100)
    assert order.on_trade(10.01, 250, aggressor=+1) == 150
    assert order.on_trade(10.00, 999, aggressor=+1) == 0     # below our ask: not a hit
    assert order.on_trade(10.02, 1, aggressor=+1) == 150     # traded through


def test_cancels_only_help_in_the_optimistic_variant():
    order = PassiveOrder(side=+1, px=9.99, qty=100, queue_ahead=500)
    order.on_cancel(9.99, 400, order_side=+1)
    order.on_cancel(9.98, 100, order_side=+1)                # different price: ignored
    order.on_cancel(9.99, 100, order_side=-1)                # other side: ignored
    assert order.ahead == 100


def test_float_prices_compare_exactly():
    order = PassiveOrder(side=+1, px=0.1 + 0.2, qty=100, queue_ahead=0)   # 0.30000000000000004
    assert order.on_trade(0.3, 100, aggressor=-1) == 100
