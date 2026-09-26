"""Fill models: walking the displayed book, and a queue model for resting orders.

Prices are compared as integers in LOBSTER's raw unit ($0.0001) to avoid
floating-point equality problems.
"""
import numpy as np


def px_int(px: float) -> int:
    return int(round(px * 10_000))


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


class PassiveOrder:
    """Limit order resting at `px`, at the back of the queue. side: +1 buy, -1 sell.

    Conservative by default: cancellations by others never move us forward.
    `on_cancel` implements the optimistic variant (every cancel is ahead of us).
    """

    def __init__(self, side, px, qty, queue_ahead):
        self.side, self.px, self.qty = side, px, qty
        self._px = px_int(px)
        self.ahead, self.filled = queue_ahead, 0     # queue_ahead = displayed size at px when posted

    @property
    def remaining(self):
        return self.qty - self.filled

    def on_trade(self, trade_px, trade_qty, aggressor):
        """Feed one historical trade. Returns the shares filled for us."""
        remaining = self.remaining
        if remaining == 0 or aggressor != -self.side:   # only opposite-side aggressors can hit us
            return 0
        tpx = px_int(trade_px)
        through = tpx < self._px if self.side == 1 else tpx > self._px
        if through:                                     # the market traded through our price
            fill = remaining
        elif tpx == self._px:                           # traded at our price: the queue ahead goes first
            used = min(self.ahead, trade_qty)
            self.ahead -= used
            fill = int(min(trade_qty - used, remaining))
        else:
            return 0
        self.filled += fill
        return fill

    def on_cancel(self, cancel_px, cancel_qty, order_side):
        """Optimistic variant: a cancel on our side at our price shrinks the queue ahead."""
        if order_side == self.side and px_int(cancel_px) == self._px:
            self.ahead = max(0, self.ahead - cancel_qty)
