import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockAnalyzer:
    def __init__(self, ticker: str):
        """
        Initialize the StockAnalyzer with a stock ticker.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None

    @property
    def info(self) -> Dict:
        """Get basic information about the stock."""
        if self._info is None:
            self._info = self.stock.info
        return self._info

    def get_historical_data(
        self,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            start (str): Start date in YYYY-MM-DD format
            end (str): End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: Historical stock data
        """
        if start and end:
            data = self.stock.history(start=start, end=end, interval=interval)
        else:
            data = self.stock.history(period=period, interval=interval)
        return data

    def get_news_impact_analysis(
        self,
        news_date: str,
        days_before: int = 5,
        days_after: int = 5
    ) -> pd.DataFrame:
        """
        Analyze stock price movements around a news event.

        Args:
            news_date (str): Date of the news event (YYYY-MM-DD)
            days_before (int): Number of days to analyze before the event
            days_after (int): Number of days to analyze after the event

        Returns:
            pd.DataFrame: Price data around the news event
        """
        start_date = (datetime.strptime(news_date, '%Y-%m-%d') -
                     timedelta(days=days_before))
        end_date = (datetime.strptime(news_date, '%Y-%m-%d') +
                   timedelta(days=days_after))

        return self.get_historical_data(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators for the stock data.

        Args:
            data (pd.DataFrame): Historical stock data

        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        df = data.copy()

        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def plot_stock_data(
        self,
        data: pd.DataFrame,
        indicators: bool = True,
        volume: bool = True
    ) -> None:
        """
        Create an interactive plot of stock data using Plotly.

        Args:
            data (pd.DataFrame): Historical stock data
            indicators (bool): Whether to show technical indicators
            volume (bool): Whether to show volume data
        """
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2 if volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if volume else [1]
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        if indicators:
            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='red')
                ),
                row=1, col=1
            )

        if volume:
            # Add volume bar chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'{self.ticker} Stock Price',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white'
        )

        fig.show()

    def analyze_news_impact(
        self,
        news_data: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Analyze the impact of news on stock prices.

        Args:
            news_data (pd.DataFrame): DataFrame containing news data with columns:
                                    ['headline', 'date', 'stock']
            price_data (pd.DataFrame): Historical price data

        Returns:
            Dict: Analysis results including price changes and statistics
        """
        results = {}

        for _, news in news_data.iterrows():
            news_date = pd.to_datetime(news['date'])

            # Get price data around news event
            event_data = self.get_news_impact_analysis(
                news_date.strftime('%Y-%m-%d')
            )

            if not event_data.empty:
                # Calculate price changes
                price_before = event_data['Close'].iloc[0]
                price_after = event_data['Close'].iloc[-1]
                price_change = ((price_after - price_before) / price_before) * 100

                # Calculate volatility
                volatility = event_data['Close'].pct_change().std() * 100

                results[news_date.strftime('%Y-%m-%d')] = {
                    'headline': news['headline'],
                    'price_change': price_change,
                    'volatility': volatility,
                    'volume_change': (
                        (event_data['Volume'].iloc[-1] - event_data['Volume'].iloc[0]) /
                        event_data['Volume'].iloc[0] * 100
                    )
                }

        return results

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = StockAnalyzer('AAPL')

    # Get historical data
    data = analyzer.get_historical_data(period='1y')

    # Calculate technical indicators
    data_with_indicators = analyzer.calculate_technical_indicators(data)

    # Plot the data
    analyzer.plot_stock_data(data_with_indicators)

    # Example news impact analysis
    sample_news = pd.DataFrame({
        'headline': ['Apple announces new iPhone', 'Apple beats earnings'],
        'date': ['2023-09-12', '2023-10-26'],
        'stock': ['AAPL', 'AAPL']
    })

    impact_analysis = analyzer.analyze_news_impact(sample_news, data)
    print("\nNews Impact Analysis:")
    for date, impact in impact_analysis.items():
        # print(f"Indicators: {data_with_indicators}")
        print(f"\nDate: {date}")
        print(f"Headline: {impact['headline']}")
        print(f"Price Change: {impact['price_change']:.2f}%")
        print(f"Volatility: {impact['volatility']:.2f}%")
        print(f"Volume Change: {impact['volume_change']:.2f}%")