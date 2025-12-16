from typing import Protocol

from pandera.typing.pandas import DataFrame

from .schema import MarketDataSchema


class IDataProvider(Protocol):
    """Protocol for data providers."""

    def get_data(self, ticker: str) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker."""
        ...
    
    def get_data_between(
        self, ticker: str, start_date: str, end_date: str
    ) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker between two dates."""
        ...