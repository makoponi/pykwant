from pathlib import Path
from typing import cast

import pandas as pd
from pandera.typing.pandas import DataFrame

from .schema import MarketDataSchema


class ParquetStorage:
    """Manage storage of market data in Parquet format."""

    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)

    def save(self, ticker: str, data: DataFrame[MarketDataSchema]) -> None:
        """Save market data to a Parquet file."""
        filepath: Path = self.base_path / f"{ticker.lower()}.parquet"
        data.to_parquet(path=filepath, compression="snappy", index=True)

    def load(self, ticker: str) -> DataFrame[MarketDataSchema] | None:
        """Load market data from a Parquet file."""
        filepath: Path = self.base_path / f"{ticker.lower()}.parquet"
        if not filepath.exists():
            return None
        return cast(DataFrame[MarketDataSchema], pd.read_parquet(path=filepath))
