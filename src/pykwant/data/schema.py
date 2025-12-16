import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Index, Series

from .constants import col_CLOSE, col_HIGH, col_LOW, col_OPEN


class MarketDataSchema(pa.DataFrameModel):
    """Schema for market data DataFrame."""

    open: Series[float] = pa.Field(gt=0, nullable=True)
    high: Series[float] = pa.Field(gt=0, nullable=True)
    low: Series[float] = pa.Field(gt=0, nullable=True)
    close: Series[float] = pa.Field(gt=0)
    volume: Series[int] = pa.Field(ge=0, nullable=True)

    date: Index[pd.Timestamp] = pa.Field(unique=True, check_name=True)

    class Config:
        """Configuration for MarketDataSchema."""

        coerce = True
        strict = True

    @pa.dataframe_check
    def validate_timezone(cls, df: pd.DataFrame) -> bool:
        """Check that the index timezone is UTC."""
        return isinstance(
            df.index, pd.DatetimeIndex
        ) and df.index.tz is not None and str(object=df.index.tz) in [
            "UTC",
            "datetime.timezone.utc",
        ]

    @pa.dataframe_check
    def no_large_gaps(cls, df: pd.DataFrame) -> bool:
        """Check that there are no gaps larger than 5 days in the index."""
        if len(df.index) < 2:
            return True
        deltas: pd.Series[int] = pd.to_datetime(arg=df.index).to_series().diff().dt.days
        return deltas.max() <= 5

    @pa.dataframe_check
    def validate_high_low(cls, df: pd.DataFrame) -> bool:
        """Check that high prices are greater than low prices."""
        return all(df[col_HIGH] >= df[col_LOW])

    @pa.dataframe_check
    def validate_low_open_close(cls, df: pd.DataFrame) -> bool:
        """Check that low prices are smaller than or equal to open and close prices."""
        return all((df[col_LOW] <= df[col_OPEN]) & (df[col_LOW] <= df[col_CLOSE]))

    @pa.dataframe_check
    def validate_high_open_close(cls, df: pd.DataFrame) -> bool:
        """Check that high prices are greater than or equal to open and close prices."""
        return all((df[col_HIGH] >= df[col_OPEN]) & (df[col_HIGH] >= df[col_CLOSE]))
