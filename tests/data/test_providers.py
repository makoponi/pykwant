from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pandera.typing.pandas import DataFrame

from pykwant.data.providers.csv import CSVDataProvider
from pykwant.data.providers.yahoo import YahooFinanceProvider
from pykwant.data.schema import MarketDataSchema

# --- TEST CSV PROVIDER ---

def test_csv_provider_reads_correctly(tmp_path: Path, valid_df: pd.DataFrame) -> None:
    """Correctly reads a CSV file."""
    csv_file: Path = tmp_path / "TEST.csv"
    raw_data = """Date;Open;High;Low;Close;Vol
2025-01-01;100;105;95;102;1000
2025-01-02;101;106;96;103;1500"""
    csv_file.write_text(data=raw_data)

    provider = CSVDataProvider(
        base_path=str(object=tmp_path),
        column_map={
            "Date": "date",
            "Vol": "volume",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low"},
        csv_kwargs={"sep": ";"}
    )
    
    df: DataFrame[MarketDataSchema] = provider.get_data(ticker="TEST")
    
    assert len(df) == 2
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert "volume" in df.columns

# --- TEST YAHOO PROVIDER (MOCKED) ---

@patch("pykwant.data.providers.yahoo.yf.download")
def test_yahoo_provider_structure(
    mock_download: MagicMock, 
    valid_df: pd.DataFrame
) -> None:
    """
    Assuming yfinance returns a valid dataframe, test the dataframe.
    """
    raw_yahoo_df: pd.DataFrame = valid_df.copy()
    raw_yahoo_df.columns = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw_yahoo_df.index, pd.DatetimeIndex):
        raw_yahoo_df.index = raw_yahoo_df.index.tz_localize(tz=None) 
    
    mock_download.return_value = raw_yahoo_df

    provider: YahooFinanceProvider = YahooFinanceProvider()
    result: DataFrame[MarketDataSchema] = provider.get_data("AAPL")

    # Verifiche
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert "open" in result.columns
    mock_download.assert_called_once()
