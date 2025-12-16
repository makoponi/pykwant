from datetime import timedelta
from typing import Optional, cast

import pandas as pd
from pandera.typing.pandas import DataFrame

from .interfaces import IDataProvider
from .providers.yahoo import YahooFinanceProvider
from .schema import MarketDataSchema
from .storage import ParquetStorage


class MarketDataRepository:
    """Repository for storing and retrieving market data locally."""

    def __init__(
            self,
            base_path: str,
            provider: Optional[IDataProvider] = None
        ) -> None:
        self.storage: ParquetStorage = ParquetStorage(base_path=base_path)
        self.provider: IDataProvider = provider or YahooFinanceProvider()

    def get_data(self, ticker: str) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker."""
        existing_data: DataFrame[MarketDataSchema] | None = self.storage.load(
            ticker=ticker
        )

        today: pd.Timestamp = pd.Timestamp.now(tz="UTC").normalize()
        yesterday: pd.Timestamp = today - timedelta(days=1)

        if existing_data is None:
            print(f"[{ticker}] No local data found, fetching from provider...")
            new_data: DataFrame[MarketDataSchema] = self.provider.get_data(
                ticker=ticker,
            )
            if not new_data.empty:
                self.storage.save(ticker=ticker, data=new_data)
                return new_data
            else:
                raise ValueError(f"No data found for {ticker}.")

        else:
            last_date: pd.Timestamp = existing_data.index[-1]

            if last_date < today - timedelta(days=1):
                start_date: pd.Timestamp = last_date + timedelta(days=1)
                print(
                    f"[{ticker}] Updating data from {start_date.date()} to "
                    f"{yesterday.date()}..."
                )

                fresh_data: DataFrame[MarketDataSchema] = (
                    self.provider.get_data_between(
                        ticker=ticker,
                        start_date=start_date.strftime(format="%Y-%m-%d"),
                        end_date=yesterday.strftime(format="%Y-%m-%d"),
                    )
                )

                if not fresh_data.empty:
                    updated_data: DataFrame[MarketDataSchema] = cast(
                        DataFrame[MarketDataSchema], pd.concat(
                            objs=[existing_data, fresh_data]
                        )
                    )
                    updated_data.drop_duplicates(inplace=True)
                    updated_data.sort_index(inplace=True)
                    self.storage.save(ticker=ticker, data=updated_data)
                    return updated_data
                else:
                    print(f"[{ticker}] No new data to update.")
                    return existing_data
            
            else:
                print(f"[{ticker}] No new data to update.")
                return existing_data
  
