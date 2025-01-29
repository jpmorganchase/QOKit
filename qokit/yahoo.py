""" Yahoo data provider. 

    Based on Qiskit YahooDataProvider: https://qiskit-community.github.io/qiskit-finance/_modules/qiskit_finance/data_providers/yahoo_data_provider.html#YahooDataProvider

"""

from typing import Optional, Union, List
import datetime
import logging
import tempfile
import yfinance as yf


logger = logging.getLogger(__name__)

# Sets Y!Finance cache path in a new temp folder.
# This is done to avoid race conditions in the same cache file
# from different processes.
# The path will be automatically deleted if this module unloads cleanly.
# This needs to be done during yfinance initialization before any call
_temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
yf.set_tz_cache_location(_temp_dir.name)


class YahooDataProvider:
    """Yahoo data provider.

    Please see:
    https://qiskit-community.github.io/qiskit-finance/tutorials/11_time_series.html
    for instructions on use.
    """

    def __init__(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        start: datetime.datetime = datetime.datetime(2020, 1, 1),
        end: datetime.datetime = datetime.datetime(2020, 1, 30),
    ) -> None:
        """
        Args:
            tickers: tickers
            start: start time
            end: end time
        """
        self._tickers = []
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace("\n", ";").split(";")
        self._n = len(self._tickers)

        self._start = start.strftime("%Y-%m-%d")
        self._end = end.strftime("%Y-%m-%d")
        self._data = []

    def run(self) -> None:
        """
        Loads data
        """
        if len(self._tickers) == 0:
            raise Exception("Missing tickers to download.")
        self._data = []
        stocks_notfound = []
        try:
            # download multiple tickers in single thread to avoid
            # race condition
            stock_data = yf.download(
                self._tickers,
                start=self._start,
                end=self._end,
                group_by="ticker",
                threads=False,
                progress=logger.isEnabledFor(logging.DEBUG),
            )
            if len(self._tickers) == 1:
                ticker_name = self._tickers[0]
                stock_value = stock_data["Adj Close"]
                if stock_value.dropna().empty:
                    stocks_notfound.append(ticker_name)
                self._data.append(stock_value)
            else:
                for ticker_name in self._tickers:
                    stock_value = stock_data[ticker_name]["Adj Close"]
                    if stock_value.dropna().empty:
                        stocks_notfound.append(ticker_name)
                    self._data.append(stock_value)
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug(ex, exc_info=True)
            raise Exception("Accessing Yahoo Data failed.") from ex

        if stocks_notfound:
            raise Exception(f"No data found for this date range, symbols may be delisted: {stocks_notfound}.")
