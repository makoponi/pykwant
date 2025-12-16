from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

from ..constants import REQUIRED_COLUMNS, col_DATE, col_VOLUME
from ..schema import MarketDataSchema


class CSVDataProvider:
    """Market data provider using local CSV files."""

    def __init__(
            self, 
            base_path: str, 
            column_map: Optional[dict[str, str  ]] = None,
            csv_kwargs: Optional[dict[str, Any]] = None
        ) -> None:
        """
        :param base_path: Path to the base directory where CSV files will be stored.
        :param column_map: Dict to rename columns in the CSV files 
            (e.g. {'Adj Close': 'close'}).
        :param csv_kwargs: Extra args to pass to pd.read_csv (e.g. {'sep': ';'}).
        """
        self.base_path: Path = Path(base_path)
        self.column_map: dict[str, str] = column_map or {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'close',
            'Volume': 'volume'
        }
        self.csv_kwargs: dict[str, Any] = csv_kwargs or {}
    
    @pa.check_types
    def get_data(self, ticker: str) -> DataFrame[MarketDataSchema]:
        """Get market data for a given ticker from a CSV file."""
        filepath: Path = self.base_path / f"{ticker}.csv"
        if not filepath.exists():   
            filepath = self.base_path / f"{ticker.upper()}.csv"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"No CSV file found for {ticker} in {self.base_path}."
                )
        try:
            data: pd.DataFrame = pd.read_csv(
                filepath_or_buffer=filepath, **self.csv_kwargs
            )
        except Exception as e:
            raise ValueError(f"Error reading CSV file {filepath}: {e}")
        
        data.rename(columns=self.column_map, inplace=True)
        
        data.columns = [c.lower() for c in data.columns]
        
        if col_DATE in data.columns:
            data[col_DATE] = pd.to_datetime(arg=data[col_DATE], utc=True)
            data.set_index(col_DATE, inplace=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(arg=data.index, utc=True)
                data.index.name = col_DATE
            except Exception:
                raise ValueError("Impossible to find or convert date column.")
        
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is None:
                data.index = data.index.tz_localize(tz="UTC")
            else:
                data.index = data.index.tz_convert(tz="UTC")
        
        if col_VOLUME not in data.columns:
            data[col_VOLUME] = 0
        
        data = data[REQUIRED_COLUMNS].astype(float)
            
        data.sort_index(inplace=True)

        return cast(DataFrame[MarketDataSchema], data)
