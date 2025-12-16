from typing import cast

import pandas as pd
import pandera.pandas as pa
import yfinance as yf
from pandera.typing.pandas import DataFrame

from ..constants import REQUIRED_COLUMNS, col_CLOSE, col_DATE
from ..schema import MarketDataSchema


class YahooFinanceProvider:
    """Market data provider using Yahoo Finance API."""

    def clean_data(self, data: pd.DataFrame | None) -> pd.DataFrame:
        """Clean the data."""
        if data is None or data.empty:
            raise ValueError("No data found for {ticker}.")
        data.columns = [c.lower() for c in data.columns]
        data.index.name = col_DATE
        if "adj close" in data.columns:
            data[col_CLOSE] = data["adj close"]
            data = data.drop(columns=["adj close"])
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(arg=data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize(tz="UTC")
        else:
            data.index = data.index.tz_convert(tz="UTC")
        data = data[REQUIRED_COLUMNS]
        data.sort_index(inplace=True)
        return data

    @pa.check_types
    def get_data(self, ticker: str) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker."""
        data: pd.DataFrame | None = yf.download(
            tickers=ticker,
            period="max",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        data = self.clean_data(data=data)
        return cast(DataFrame[MarketDataSchema], data)

    @pa.check_types
    def get_data_between(
        self, ticker: str, start_date: str, end_date: str
    ) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker between two dates."""
        data: pd.DataFrame | None = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        data = self.clean_data(data=data)
        return cast(DataFrame[MarketDataSchema], data)
