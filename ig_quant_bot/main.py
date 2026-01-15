from __future__ import annotations

import uuid
import yaml
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# Internal Platform Imports
from core.contracts import Order, FillEvent
from core.state import PortfolioState
from core.feature_engine import FeatureEngine
from core.ledgers import TradeLedger
from execution.instrument_db import InstrumentCatalog
from execution.sim_broker import IGSyntheticBroker
from vault.run_manager import RunManager
from analytics.performance import PerformanceAnalyser


class QuantDesk:
    def __init__(self, config_path: str = "config.yaml"):
        # 1) Load Governance & Parameters
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f) or {}

        strat = self.cfg.get("strategy", {}) or {}
        risk = self.cfg.get("risk", {}) or {}
        costs = self.cfg.get("costs", {}) or {}
        vault_cfg = self.cfg.get("vault", {}) or {}

        # Strategy
        self.strategy_name = str(strat.get("name", "IG_Quant_Run"))
        self.universe = list(strat.get("universe", ["QQQ", "SPY"]))

        self.rsi_p = int(strat.get("rsi_period", 14))
        self.sma_p = int(strat.get("sma_period", 200))
        self.rsi_entry_threshold = float(strat.get("rsi_entry_threshold", 30))
        self.rsi_exit_threshold = float(strat.get("rsi_exit_threshold", 70))
        self.regime_filter = str(strat.get("regime_filter", "BULL_STABLE"))

        # Risk
        self.initial_capital = float(risk.get("initial_capital", 100000.0))
        self.max_slots = int(risk.get("max_slots", 3))
        self.max_dd_limit = float(risk.get("max_drawdown_limit", 0.15))

        self.per_slot_margin_fraction = float(risk.get("per_slot_margin_fraction", 0.15))
        self.min_free_margin_buffer = float(risk.get("min_free_margin_buffer", 0.10))
        self.allow_shorting = bool(risk.get("allow_shorting", False))

        # Costs (passed into broker where applicable)
        self.annual_funding_rate = float(costs.get("annual_funding_rate", 0.05))
        self.weekend_multiplier = int(costs.get("weekend_multiplier", 3))
        self.adaptive_spread_k = float(costs.get("adaptive_spread_k", 0.10))

        # Vault
        vault_path = str(vault_cfg.get("output_path", "vault/runs"))
        self.run_manager = RunManager(base_path=vault_path)

        # 2) Initialize Engine Room
        self.catalog = InstrumentCatalog()

        # Broker (Backtest)
        self.broker = IGSyntheticBroker(
            self.catalog,
            annual_funding_rate=self.annual_funding_rate,
            weekend_multiplier=self.weekend_multiplier,
            adaptive_spread_k=self.adaptive_spread_k,
        )

        self.state = PortfolioState(self.initial_capital, allow_shorting=self.allow_shorting)
        self.trade_ledger = TradeLedger()
        self.analyser = PerformanceAnalyser()

        # 3) Persistence Buffers
        self.fills: List[FillEvent] = []
        self.ledger_rows: List[dict] = []

    def _make_run_id(self) -> str:
        return f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}_{self.strategy_name}_{str(uuid.uuid4())[:6]}"

    def load_and_sync_data(self, start: str, end: Optional[str]) -> tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """
        Ensures all tickers share a master calendar to prevent date drift.
        Uses SPY as the calendar anchor.
        """
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")

        print(f"[*] Synchronizing {len(self.universe)} tickers against Master Calendar...")

        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if spy is None or spy.empty:
            raise RuntimeError("Failed to download SPY data for master calendar.")

        master_idx = spy.index

        factors: Dict[str, pd.DataFrame] = {}
        for t in self.universe:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                print(f"[!] Warning: No data for {t}. Skipping.")
                continue

            df["is_real_bar"] = True

            # Reindex to master calendar before computing features
            # Fill calendar holes forward for price continuity but mark them as not-real bars.
            df = df.reindex(master_idx).ffill()
            df["is_real_bar"] = df["is_real_bar"].fillna(False)

            factors[t] = FeatureEngine.compute(df, rsi_p=self.rsi_p, sma_p=self.sma_p)

        if not factors:
            raise RuntimeError("No ticker data loaded. Universe is empty after downloads.")

        return factors, master_idx

    def _record_event(self, fill: FillEvent) -> None:
        self.fills.append(fill)
        self.trade_ledger.add_fill(fill)

    def _snapshot_daily_ledger(self, date: pd.Timestamp) -> None:
        self.ledger_rows.append(
            {
                "date": date,
                "equity": float(self.state.equity),
                "cash_balance": float(self.state.cash_balance),
                "margin_used": float(self.state.margin_used),
                "free_margin": float(self.state.free_margin),
                "pos_count": int(len(self.state.positions)),
                "drawdown": float(self.state.get_drawdown()),
            }
        )

    def run(self, start: str = "2020-01-01", end: Optional[str] = None) -> None:
        run_id = self._make_run_id()
        factors, master_idx = self.load_and_sync_data(start, end)

        print(f"[*] Starting Backtest Run: {run_id}")

        for date in master_idx:
            # A) Mark-to-market using today's Open only for real bars
            prices_open = {
                t: float(f.at[date, "Open"])
                for t, f in factors.items()
                if bool(f.at[date, "is_real_bar"])
            }
            self.state.update_mtm(prices_open, self.catalog)

            # B) Killswitch Check (peak-to-trough drawdown)
            if self.state.get_drawdown() < -self.max_dd_limit:
                print(f"[!] DRAWDOWN KILLSWITCH ACTIVATED AT {date}")
                break

            # C) EXITS (Signal on t-1, execute on t-open)
            for t, pos in list(self.state.positions.items()):
                f = factors.get(t)
                if f is None:
                    continue

                loc = f.index.get_loc(date)
                if loc == 0:
                    continue
                prev = f.iloc[loc - 1]

                if not bool(prev.get("valid_signal", False)):
                    continue

                exit_signal = (float(prev["RSI"]) > self.rsi_exit_threshold) or (date == master_idx[-1])
                if not exit_signal:
                    continue

                px = prices_open.get(t, pos.get("last_p"))
                if px is None:
                    continue

                order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason="RSI_EXIT", timestamp=date)
                fill = self.broker.execute(
                    order,
                    float(px),
                    str(prev.get("Regime")),
                    float(prev.get("ATR_pts")),
                    current_pos=pos,
                )
                self.state.apply_fill(fill)
                self._record_event(fill)

            # D) FUNDING (daily charge on open positions)
            for t, pos in list(self.state.positions.items()):
                px = prices_open.get(t, pos.get("last_p"))
                if px is None:
                    continue
                f_event = self.broker.calculate_funding(t, pos, float(px), date)
                self.state.apply_fill(f_event)
                self._record_event(f_event)

            # E) ENTRIES (ranked candidate selection)
            self.state.update_mtm(prices_open, self.catalog)
            candidates = []

            for t, f in factors.items():
                if t in self.state.positions:
                    continue
                if not bool(f.at[date, "is_real_bar"]):
                    continue

                loc = f.index.get_loc(date)
                if loc == 0:
                    continue
                prev = f.iloc[loc - 1]

                if not bool(prev.get("valid_signal", False)):
                    continue
                if str(prev.get("Regime")) != self.regime_filter:
                    continue
                if float(prev["RSI"]) >= self.rsi_entry_threshold:
                    continue

                inst = self.catalog.get(t)
                atr_pts = float(prev.get("ATR_pts", 0.0))
                spread_pts = float(inst.spread_points)

                # Score = RSI edge - spread friction penalty
                spread_penalty = spread_pts / max(atr_pts, 1e-6)
                score = (self.rsi_entry_threshold - float(prev["RSI"])) - (1.5 * spread_penalty)

                candidates.append(
                    {
                        "ticker": t,
                        "score": float(score),
                        "regime": str(prev.get("Regime")),
                        "atr_pts": float(atr_pts),
                    }
                )

            candidates.sort(key=lambda x: x["score"], reverse=True)

            for cand in candidates:
                if len(self.state.positions) >= self.max_slots:
                    break

                # Margin-budget sizing
                m_budget = float(self.state.equity * self.per_slot_margin_fraction)
                inst = self.catalog.get(cand["ticker"])
                target_notional = float(m_budget / float(inst.margin_factor))

                # Safety buffer
                if (self.state.free_margin - m_budget) <= (self.state.equity * self.min_free_margin_buffer):
                    continue

                px = prices_open.get(cand["ticker"])
                if px is None:
                    continue

                order = Order(
                    ticker=cand["ticker"],
                    side="BUY",
                    notional=target_notional,
                    reason="RSI_ENTRY",
                    timestamp=date,
                )
                fill = self.broker.execute(order, float(px), cand["regime"], cand["atr_pts"])
                self.state.apply_fill(fill)
                self._record_event(fill)

            # F) Daily snapshot (authoritative ledger for analytics)
            self._snapshot_daily_ledger(date)

        # G) Finalize & Persist
        self._wrap_up(run_id)

    def _wrap_up(self, run_id: str) -> None:
        ledger_df = pd.DataFrame(self.ledger_rows).set_index("date")
        fills_df = pd.DataFrame([asdict(f) for f in self.fills])
        trades_df = self.trade_ledger.get_trades_df()

        metadata = {
            "strategy": self.strategy_name,
            "universe": self.universe,
            "version": "7.2",
        }

        self.run_manager.persist_run(
            run_id,
            self.cfg,
            ledger_df=ledger_df,
            fills_df=fills_df,
            trades_df=trades_df,
            metadata=metadata,
        )

        # Final audit gate: uses cash_balance column (now consistent)
        self.analyser.audit_replay_cash_recursion(ledger_df, fills_df)

        # Tear sheet + tables
        sheet, trades_attrib_df, regime_table = self.analyser.generate_tear_sheet(ledger_df, fills_df, trades_df)

        print("[*] Run complete.")
        print(f"    Final equity: {sheet.final_equity:.2f}")
        print(f"    CAGR: {sheet.cagr:.4f} | Sharpe: {sheet.sharpe:.3f} | MaxDD: {sheet.max_drawdown:.3f}")
        print(f"    Trades: {sheet.total_trades} | Win rate: {sheet.win_rate:.3f} | PF: {sheet.profit_factor:.3f}")

        # Optional: persist analytics outputs alongside the run
        try:
            run_dir = f"{self.run_manager.base_path}/{run_id}"
            if trades_attrib_df is not None and not trades_attrib_df.empty:
                trades_attrib_df.to_parquet(f"{run_dir}/trades_attrib.parquet")
            if regime_table is not None and not regime_table.empty:
                regime_table.to_parquet(f"{run_dir}/expectancy_by_regime.parquet")
        except Exception as e:
            print(f"[!] Warning: Could not persist analytics artifacts: {e}")


if __name__ == "__main__":
    QuantDesk().run(start="2020-01-01")
