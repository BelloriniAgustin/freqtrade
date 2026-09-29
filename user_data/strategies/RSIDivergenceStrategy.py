# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class RSIDivergenceStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {
        "120": 0.01,
        "60": 0.02,
        "30": 0.03,
        "0": 0.05,
    }

    stoploss = -0.05

    trailing_stop = False

    use_custom_stoploss = True

    timeframe = "5m"

    process_only_new_candles = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    rsi_period = IntParameter(low=7, high=21, default=14, space="buy", optimize=True, load=True)
    rsi_level = IntParameter(low=20, high=45, default=35, space="buy", optimize=True, load=True)
    rsi_short_level = IntParameter(low=55, high=80, default=70, space="buy", optimize=True, load=True)
    pivot_period = IntParameter(low=2, high=15, default=5, space="buy", optimize=True, load=True)
    pivot_lookback = IntParameter(low=20, high=50, default=30, space="buy", optimize=True, load=True)
    sma_window = IntParameter(low=20, high=300, default=200, space="buy", optimize=True, load=True)
    atr_multiplier = DecimalParameter(
        low=0.5, high=4.0, default=2.0, space="sell", optimize=True, load=True
    )
    use_sma_filter = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    startup_candle_count: int = 1500

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "sma200": {"color": "blue"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        if self.inf_tf and self.inf_tf != self.timeframe:
            return [(pair, self.inf_tf) for pair in self.config["exchange"]["pair_whitelist"]]
        return []

    @property
    def inf_tf(self) -> str:
        map = {
            "1m": "5m",
            "5m": "15m",
            "15m": "1h",
            "1h": "4h",
            "4h": "1d",
        }
        return map.get(self.timeframe, "")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["sma200"] = ta.SMA(dataframe, timeperiod=self.sma_window.value)

        if self.inf_tf and self.inf_tf != self.timeframe:
            inf = self.dp.get_pair_dataframe(metadata["pair"], self.inf_tf)
            if inf.empty:
                raise RuntimeError(f"No informative data for {self.inf_tf} - use download-data")
            inf["sma200_inf"] = ta.SMA(inf, timeperiod=self.sma_window.value)
            dataframe = merge_informative_pair(
                dataframe, inf, self.timeframe, self.inf_tf, ffill=True
            )
            dataframe["sma200_inf"] = dataframe[f"sma200_inf_{self.inf_tf}"]
        else:
            dataframe["sma200_inf"] = dataframe["sma200"]

        window = 2 * self.pivot_period.value + 1
        rolling_low = dataframe["low"].rolling(window=window, center=True).min()
        rolling_high = dataframe["high"].rolling(window=window, center=True).max()

        is_pivot_low = (dataframe["low"] == rolling_low)
        is_pivot_high = (dataframe["high"] == rolling_high)

        confirmed_pivot_low = is_pivot_low.shift(self.pivot_period.value).fillna(False).astype(bool)
        confirmed_pivot_high = is_pivot_high.shift(self.pivot_period.value).fillna(False).astype(bool)

        dataframe["confirmed_pivot_low"] = confirmed_pivot_low
        dataframe["confirmed_pivot_high"] = confirmed_pivot_high

        low_pivot_price = dataframe["low"].where(confirmed_pivot_low)
        low_pivot_rsi = dataframe["rsi"].where(confirmed_pivot_low)
        high_pivot_price = dataframe["high"].where(confirmed_pivot_high)
        high_pivot_rsi = dataframe["rsi"].where(confirmed_pivot_high)

        def _prev_pivot(low: pd.Series, confirmed: pd.Series, p_price: pd.Series, p_rsi: pd.Series) -> tuple[pd.Series, pd.Series]:
            """For each confirmed pivot candle, find the most recent confirmed pivot that
            happened at least `pivot_lookback` candles before it. No consecutive-pivot coupling."""
            lb = int(self.pivot_lookback.value)
            conf_idx = np.where(confirmed.to_numpy())[0]
            prev_p = pd.Series(np.nan, index=dataframe.index)
            prev_r = pd.Series(np.nan, index=dataframe.index)
            if len(conf_idx) == 0:
                return prev_p, prev_r
            pos = np.searchsorted(conf_idx, conf_idx - lb, side="right") - 1
            valid = pos >= 0
            if not valid.any():
                return prev_p, prev_r
            src_p = p_price.to_numpy()
            src_r = p_rsi.to_numpy()
            prev_p.iloc[conf_idx[valid]] = src_p[conf_idx[pos[valid]]]
            prev_r.iloc[conf_idx[valid]] = src_r[conf_idx[pos[valid]]]
            return prev_p, prev_r

        prev_piv_low_p, prev_piv_low_r = _prev_pivot(dataframe["low"], confirmed_pivot_low, low_pivot_price, low_pivot_rsi)
        prev_piv_high_p, prev_piv_high_r = _prev_pivot(dataframe["high"], confirmed_pivot_high, high_pivot_price, high_pivot_rsi)

        dataframe["last_pivot_low_price"] = low_pivot_price.ffill()
        dataframe["prev_pivot_low_price"] = prev_piv_low_p.ffill()
        dataframe["last_pivot_low_rsi"] = low_pivot_rsi.ffill()
        dataframe["prev_pivot_low_rsi"] = prev_piv_low_r.ffill()
        dataframe["last_pivot_high_price"] = high_pivot_price.ffill()
        dataframe["prev_pivot_high_price"] = prev_piv_high_p.ffill()
        dataframe["last_pivot_high_rsi"] = high_pivot_rsi.ffill()
        dataframe["prev_pivot_high_rsi"] = prev_piv_high_r.ffill()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.use_sma_filter.value:
            bull_trend = dataframe["close"] > dataframe["sma200_inf"]
            bear_trend = dataframe["close"] < dataframe["sma200_inf"]
        else:
            bull_trend = True
            bear_trend = True

        bull_divergence = (
            dataframe["confirmed_pivot_low"]
            & (dataframe["last_pivot_low_price"] < dataframe["prev_pivot_low_price"])
            & (dataframe["last_pivot_low_rsi"] > dataframe["prev_pivot_low_rsi"])
            & (dataframe["prev_pivot_low_rsi"] < self.rsi_level.value)
            & bull_trend
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[bull_divergence, "enter_long"] = 1

        bear_divergence = (
            dataframe["confirmed_pivot_high"]
            & (dataframe["last_pivot_high_price"] > dataframe["prev_pivot_high_price"])
            & (dataframe["last_pivot_high_rsi"] < dataframe["prev_pivot_high_rsi"])
            & (dataframe["prev_pivot_high_rsi"] > self.rsi_short_level.value)
            & bear_trend
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[bear_divergence, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        if not self.dp or self.atr_multiplier.value <= 0:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        atr = dataframe["atr"].iat[-1]
        # Anchor the stop to the ENTRY price, not the current price.
        # Gives the trade breathing room from the open instead of a tight trailing stop.
        if trade.is_short:
            sl_price = trade.open_rate + (self.atr_multiplier.value * atr)
        else:
            sl_price = trade.open_rate - (self.atr_multiplier.value * atr)
        return stoploss_from_absolute(
            sl_price, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )