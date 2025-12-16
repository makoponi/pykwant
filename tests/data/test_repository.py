from pathlib import Path
from typing import cast

import pandas as pd
from pandera.typing.pandas import DataFrame

from pykwant.data.repository import MarketDataRepository
from pykwant.data.schema import MarketDataSchema


class FakeProvider:
    """A fake provider for testing purposes."""
    def __init__(self, df_to_return: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df_to_return
        self.call_count = 0

    def get_data(self, ticker: str) -> DataFrame[MarketDataSchema]:
        self.call_count += 1
        return cast(DataFrame[MarketDataSchema], self.df)
    
    def get_data_between(
        self, ticker: str, start_date: str, end_date: str
    ) -> DataFrame[MarketDataSchema]:
        return cast(DataFrame[MarketDataSchema], self.df)


def test_repository_flow(tmp_path: Path, valid_df: pd.DataFrame) -> None:
    """Test loading on repository if repository is empty."""
    fake_provider = FakeProvider(df_to_return=valid_df)
    repo = MarketDataRepository(base_path=str(object=tmp_path), provider=fake_provider)
    ticker = "FAKE"

    df1: DataFrame[MarketDataSchema] = repo.get_data(ticker=ticker)
    
    assert fake_provider.call_count == 1
    assert (tmp_path / "fake.parquet").exists()

    df2: DataFrame[MarketDataSchema] = repo.get_data(ticker=ticker)
    
    assert fake_provider.call_count == 1
    pd.testing.assert_frame_equal(left=df1, right=df2)
