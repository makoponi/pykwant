import pandas as pd
import pytest

from pykwant.data.constants import (
    col_CLOSE,
    col_DATE,
    col_HIGH,
    col_LOW,
    col_OPEN,
    col_VOLUME,
)


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Generates a valid dataframe."""
    dates: pd.DatetimeIndex = pd.date_range(
        start="2025-01-01", 
        periods=5,
        freq="D",
        tz="UTC"
    )
    data: dict[str, list[float]] = {
        col_OPEN: [100.0, 101.0, 102.0, 103.0, 104.0],
        col_HIGH: [105.0, 106.0, 107.0, 108.0, 109.0],
        col_LOW: [95.0, 96.0, 97.0, 98.0, 99.0],
        col_CLOSE: [102.0, 103.0, 104.0, 105.0, 106.0],
        col_VOLUME: [1000.0, 1500.0, 1200.0, 1800.0, 2000.0]
    }
    df = pd.DataFrame(data=data, index=dates)
    df.index.name = col_DATE
    return df

@pytest.fixture
def invalid_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Generates an invalid dataframe."""
    df: pd.DataFrame = valid_df.copy()
    df.loc[df.index[0],col_LOW] = 150.0
    return df
