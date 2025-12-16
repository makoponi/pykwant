import pandas as pd
import pandera.errors as pe
import pytest
from pandera.typing.pandas import DataFrame

from pykwant.data.schema import MarketDataSchema


def test_valid_schema(valid_df: pd.DataFrame) -> None:
    """Validates a valid dataframe."""
    validated: DataFrame[MarketDataSchema] = MarketDataSchema.validate(
        check_obj=valid_df
    )
    assert validated is not None

def test_invalid_high_low_logic(invalid_df: pd.DataFrame) -> None:
    """Error if high is lower than low."""
    with pytest.raises(expected_exception=pe.SchemaError):
        MarketDataSchema.validate(check_obj=invalid_df)

def test_missing_column(valid_df: pd.DataFrame) -> None:
    """Error if column is missing."""
    df: pd.DataFrame = valid_df.drop(columns=["close"])
    with pytest.raises(expected_exception=pe.SchemaError):
        MarketDataSchema.validate(check_obj=df)

def test_timezone_check(valid_df: pd.DataFrame) -> None:
    """Error if timezone is not UTC."""
    df: pd.DataFrame = valid_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(tz=None)
    with pytest.raises(expected_exception=pe.SchemaError) as excinfo:
        MarketDataSchema.validate(check_obj=df)
    assert "check_tz_utc" in str(object=excinfo.value)
