from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import pandas as pd

from core.contracts import FillEvent, Order
from .broker_interface import IBrokerAdapter


class IGLiveAdapter(IBrokerAdapter):
    """
    Production adapter for IG using the 'trading-ig' library.

    Notes:
      - This module is not used by the backtest path.
      - For SELL fills, margin release is estimated from current_pos when IG does not
        return an explicit margin field for close confirmations.

    Expected config keys:
      username, password, api_key, acc_type (DEMO|LIVE), acc_number
    """

    def __init__(self, config: Dict, catalog):
        self.catalog = catalog
        self.logger = logging.getLogger(__name__)

        try:
            from trading_ig import IGService
        except Exception as e:
            raise ImportError("trading-ig is required for IGLiveAdapter. Install via requirements.txt") from e

        self.ig = IGService(
            config["username"],
            config["password"],
            config["api_key"],
            config.get("acc_type", "DEMO"),
        )
        self.session = self.ig.create_session()
        self.account_id = config.get("acc_number")

    def execute(
        self,
        order: Order,
        price: float,
        regime: str,
        atr_pts: float,
        current_pos: Optional[Dict] = None,
    ) -> FillEvent:
        inst = self.catalog.get(order.ticker)
        direction = "BUY" if order.side == "BUY" else "SELL"

        # size = stake (£/pt)
        if order.notional is not None:
            size = round(float(order.notional) / (float(price) * float(inst.value_per_point)), 2)
        else:
            size = float(order.qty)

        self.logger.info(f"Submitting {direction} {inst.epic} | size={size}")

        resp = self.ig.create_open_position(
            currency_code="GBP",
            direction=direction,
            epic=inst.epic,
            order_type="MARKET",
            expiry="DFB",
            force_open="true",
            guaranteed_stop="false",
            size=size,
        )

        deal_ref = resp.get("dealReference")
        if not deal_ref:
            raise RuntimeError(f"IG submission failed: {resp}")

        conf = self._poll_confirmation(deal_ref)
        if conf.get("dealStatus") != "ACCEPTED":
            raise RuntimeError(f"IG deal rejected: {conf.get('reason', 'Unknown')}")

        fill_price = float(conf.get("level", price))
        gross_notional = abs(float(size)) * fill_price * float(inst.value_per_point)

        if order.side == "BUY":
            margin_val = float(conf.get("marginDeposit", gross_notional * float(inst.margin_factor)))
            margin_change = margin_val
            net_cf = 0.0
            realized = 0.0
        else:
            if current_pos is None:
                raise ValueError("current_pos is required for SELL in live adapter")

            # Realized profit is sometimes returned, sometimes not.
            realized = float(conf.get("profit", 0.0))
            net_cf = realized

            # Release margin based on current held margin (best effort).
            current_qty = float(current_pos.get("qty", size))
            ratio = (float(size) / current_qty) if current_qty else 1.0
            margin_change = -(float(current_pos.get("margin", 0.0)) * ratio)

        return FillEvent(
            fill_id=str(conf.get("dealId", "")) or None,
            order_id=order.order_id,
            ticker=order.ticker,
            timestamp=pd.Timestamp.now(),
            event_type="TRADE",
            side=order.side,
            qty=float(size),
            price=float(fill_price),
            margin_change=float(margin_change),
            realized_pnl_cashflow=float(realized),
            net_cashflow=float(net_cf),
            gross_notional=float(gross_notional),
            regime=str(regime),
            reason=str(order.reason),
        )

    def _poll_confirmation(self, deal_ref: str, timeout: int = 15) -> Dict:
        start = time.time()
        while time.time() - start < timeout:
            conf = self.ig.fetch_deal_confirmation(deal_ref)
            if conf and conf.get("dealStatus"):
                return conf
            time.sleep(0.5)
        raise TimeoutError(f"IG confirmation timeout for dealRef: {deal_ref}")

    def calculate_funding(self, ticker: str, pos: Dict, price: float, date: pd.Timestamp) -> FillEvent:
        # In live, funding is applied by IG. This is a placeholder for reconciliation.
        return FillEvent(
            ticker=ticker,
            timestamp=pd.to_datetime(date),
            event_type="FUNDING",
            net_cashflow=0.0,
            gross_notional=0.0,
            reason="LIVE_FUNDING_RECONCILE_TODO",
        )
