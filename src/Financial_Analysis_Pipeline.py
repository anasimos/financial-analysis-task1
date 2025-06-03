import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
import sys # For graceful exit

# --- NLTK Data Download (ensure these are downloaded before running) ---
# It's good practice to run these once manually or ensure they are part of your CI/CD setup
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon for NLTK...")
    nltk.download('vader_lexicon')
    print("VADER lexicon downloaded.")
except Exception as e:
    print(f"Warning: Could not download VADER lexicon: {e}. Sentiment analysis might fail.")

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading punkt tokenizer for NLTK...")
    nltk.download('punkt')
    print("Punkt tokenizer downloaded.")
except Exception as e:
    print(f"Warning: Could not download punkt tokenizer: {e}.")

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading stopwords for NLTK...")
    nltk.download('stopwords')
    print("Stopwords downloaded.")
except Exception as e:
    print(f"Warning: Could not download stopwords: {e}.")

# --- Attempt to import TA-Lib ---
try:
    import talib as ta
    TALIB_AVAILABLE = True
    print("TA-Lib imported successfully.")
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not installed. Technical indicators will not be calculated.")
except Exception as e:
    TALIB_AVAILABLE = False
    print(f"Warning: Error importing TA-Lib: {e}. Technical indicators will not be calculated.")

# --- Attempt to import pynance ---
try:
    # Note: pynance is generally not actively maintained and can be difficult to install.
    # Its core functionalities for financial metrics are often better handled by pandas/numpy.
    # This import is included as per user request, but its actual use will be minimal/illustrative.
    import pynance as pn
    PYNANCE_AVAILABLE = True
    print("PyNance imported successfully (Note: Functionality will be illustrative due to library status).")
except ImportError:
    PYNANCE_AVAILABLE = False
    print("Warning: PyNance not installed. PyNance-specific functionality will be skipped.")
except Exception as e:
    PYNANCE_AVAILABLE = False
    print(f"Warning: Error importing PyNance: {e}. PyNance-specific functionality will be skipped.")


class DataLoader:
    """
    Handles loading and initial cleaning of financial news and stock price data.
    Implements robust error handling for file operations and data integrity.
    """
    def __init__(self, news_file_path, stock_files_dict):
        """
        Initializes the DataLoader with file paths.

        Args:
            news_file_path (str): Path to the news headlines CSV.
            stock_files_dict (dict): Dictionary of stock symbols to their CSV file paths.
        """
        self.news_file_path = news_file_path
        self.stock_files_dict = stock_files_dict
        self.df_news = None
        self.df_stocks = None

    def load_news_data(self):
        """Loads and performs initial cleaning on news data."""
        print(f"\n--- Loading News Data from {self.news_file_path} ---")
        try:
            df = pd.read_csv(self.news_file_path)
            if df.empty:
                print(f"Warning: News data file '{self.news_file_path}' is empty. Returning empty DataFrame.")
                self.df_news = pd.DataFrame()
                return self.df_news

            # Validate essential columns
            required_cols = ['headline', 'date', 'stock']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                print(f"Error: Missing required columns in news data: {missing}. Exiting.")
                sys.exit(1)

            df['headline'] = df['headline'].fillna('').astype(str)
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True) # Assuming UTC-4 from problem, storing as UTC
            df.dropna(subset=['date', 'stock'], inplace=True) # Drop rows with invalid dates or missing stock
            df.rename(columns={'stock': 'STOCK_SYMBOL'}, inplace=True)

            if df.empty:
                print("Warning: News DataFrame is empty after initial cleaning. Returning empty DataFrame.")
            else:
                print("News data loaded and initially cleaned successfully.")
                print(f"Shape: {df.shape}")
                print("Head:\n", df.head())
            self.df_news = df
            return self.df_news

        except FileNotFoundError:
            print(f"Error: News data file '{self.news_file_path}' not found. Exiting.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: News data file '{self.news_file_path}' is empty or malformed. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading news data: {e}. Exiting.")
            sys.exit(1)

    def load_stock_data(self):
        """Loads and performs initial cleaning on stock price data from multiple files."""
        print("\n--- Loading Stock Data ---")
        all_stocks_df_list = []
        required_stock_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        for symbol, file_path in self.stock_files_dict.items():
            print(f"Attempting to load data for {symbol} from {file_path}...")
            try:
                df_single_stock = pd.read_csv(file_path)
                if df_single_stock.empty:
                    print(f"Warning: Stock data file for {symbol} is empty. Skipping this stock.")
                    continue

                # Validate original columns
                if not all(col in df_single_stock.columns for col in required_stock_cols):
                    missing_cols = [col for col in required_stock_cols if col not in df_single_stock.columns]
                    print(f"Error: Missing required columns for {symbol} in {file_path}: {missing_cols}. Skipping this stock.")
                    continue

                df_single_stock['Stock Symbol'] = symbol
                all_stocks_df_list.append(df_single_stock)
                print(f"Successfully loaded data for {symbol}.")

            except FileNotFoundError:
                print(f"Error: Stock data file for {symbol} not found at '{file_path}'. Skipping this stock.")
            except pd.errors.EmptyDataError:
                print(f"Error: Stock data file for {symbol} is empty or malformed. Skipping this stock.")
            except Exception as e:
                print(f"An unexpected error occurred while loading data for {symbol}: {e}. Skipping this stock.")

        if not all_stocks_df_list:
            print("No valid stock data was loaded from any file. Exiting.")
            sys.exit(1)

        df_raw_combined = pd.concat(all_stocks_df_list, ignore_index=True)
        print("\nAll valid stock data concatenated.")

        df_cleaned = df_raw_combined.copy()
        df_cleaned.columns = [col.upper().replace(' ', '_') for col in df_cleaned.columns]
        print("Columns standardized to uppercase.")

        if 'DATE' not in df_cleaned.columns:
            print("Critical Error: 'DATE' column not found after standardization. Exiting.")
            sys.exit(1)

        df_cleaned['DATE'] = pd.to_datetime(df_cleaned['DATE'], errors='coerce')
        initial_rows = len(df_cleaned)
        df_cleaned.dropna(subset=['DATE'], inplace=True)
        if len(df_cleaned) < initial_rows:
            print(f"Dropped {initial_rows - len(df_cleaned)} rows due to invalid dates.")

        if df_cleaned.empty:
            print("Warning: Stock DataFrame is empty after dropping invalid dates. Exiting.")
            sys.exit(1)

        df_cleaned.sort_values(by=['STOCK_SYMBOL', 'DATE'], inplace=True)
        df_cleaned.set_index('DATE', inplace=True)
        print("Data sorted and 'DATE' set as index.")

        # Handle missing values: forward fill within each stock group, then drop any remaining NaNs
        initial_rows_before_ffill = len(df_cleaned)
        df_cleaned = df_cleaned.groupby('STOCK_SYMBOL', group_keys=False).apply(lambda group: group.fillna(method='ffill'))
        df_cleaned.dropna(inplace=True)
        if len(df_cleaned) < initial_rows_before_ffill:
            print(f"Dropped {initial_rows_before_ffill - len(df_cleaned)} rows after ffill and final dropna.")

        final_required_numerical_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJ_CLOSE', 'VOLUME']
        for col in final_required_numerical_cols:
            if col not in df_cleaned.columns:
                print(f"Critical Error: Required numerical column '{col}' missing after cleaning. Exiting.")
                sys.exit(1)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        initial_rows_before_num_dropna = len(df_cleaned)
        df_cleaned.dropna(subset=final_required_numerical_cols, inplace=True)
        if len(df_cleaned) < initial_rows_before_num_dropna:
            print(f"Dropped {initial_rows_before_num_dropna - len(df_cleaned)} rows due to NaN in critical numerical columns after conversion.")

        if df_cleaned.empty:
            print("Warning: Stock DataFrame is empty after all cleaning steps. Exiting.")
            sys.exit(1)
        else:
            print("Stock data prepared and cleaned successfully.")
            print(f"Shape: {df_cleaned.shape}")
            print("Info:\n", df_cleaned.info())
            print("Head (first few rows of each stock):\n", df_cleaned.groupby('STOCK_SYMBOL').head(2))
        self.df_stocks = df_cleaned
        return self.df_stocks


class EDAProcessor:
    """
    Performs Exploratory Data Analysis on the news data.
    """
    def __init__(self, df_news, plots_output_dir):
        """
        Initializes EDAProcessor.

        Args:
            df_news (pd.DataFrame): Cleaned news DataFrame.
            plots_output_dir (str): Directory to save EDA plots.
        """
        self.df_news = df_news
        self.plots_output_dir = plots_output_dir
        os.makedirs(self.plots_output_dir, exist_ok=True)

    def analyze_headline_length(self):
        """Analyzes and plots the distribution of headline lengths."""
        print("\n--- EDA: Headline Length Analysis ---")
        if self.df_news.empty or 'headline' not in self.df_news.columns:
            print("Skipping headline length analysis: News DataFrame is empty or 'headline' column missing.")
            return

        self.df_news["headline_length"] = self.df_news["headline"].str.len()
        stats = self.df_news["headline_length"].describe()
        print("Summary Statistics for Headline Length:\n", stats)

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_news['headline_length'], bins=50, kde=True, color='steelblue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'headline_length_distribution.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(self.plots_output_dir, 'headline_length_distribution.png')}")

    def analyze_publishers(self):
        """Analyzes and plots top publishers by article count."""
        print("\n--- EDA: Articles per Publisher Analysis ---")
        if self.df_news.empty or 'publisher' not in self.df_news.columns:
            print("Skipping publisher analysis: News DataFrame is empty or 'publisher' column missing.")
            return

        top_publishers = self.df_news["publisher"].value_counts().head(10)
        print("Top 10 Publishers by Article Count:\n", top_publishers)

        plt.figure(figsize=(12, 7))
        sns.barplot(x=top_publishers.index, y=top_publishers.values, palette='viridis')
        plt.title("Top 10 Publishers by Article Count")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'top_publishers_bar_chart.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(self.plots_output_dir, 'top_publishers_bar_chart.png')}")

    def analyze_publication_date_trends(self):
        """Analyzes and plots article frequency over time."""
        print("\n--- EDA: Publication Date Trends Analysis ---")
        if self.df_news.empty or 'date' not in self.df_news.columns:
            print("Skipping publication date trends: News DataFrame is empty or 'date' column missing.")
            return

        articles_per_day = self.df_news["date"].dt.date.value_counts().sort_index()
        if articles_per_day.empty:
            print("No daily article counts found. Skipping plot.")
            return

        plt.figure(figsize=(14, 7))
        articles_per_day.plot(kind='line', color='darkorange', linewidth=1.5)
        plt.title("Article Publication Frequency Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'article_frequency_over_time.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(self.plots_output_dir, 'article_frequency_over_time.png')}")

    def analyze_text_keywords(self):
        """Analyzes and plots top keywords in headlines using CountVectorizer."""
        print("\n--- EDA: Text Analysis (Keyword Extraction) ---")
        if self.df_news.empty or 'headline' not in self.df_news.columns:
            print("Skipping keyword analysis: News DataFrame is empty or 'headline' column missing.")
            return

        from sklearn.feature_extraction.text import CountVectorizer # Import locally to avoid global dependency if not used
        
        # Ensure 'headline' column is string type and handle NaNs
        headlines_for_vectorization = self.df_news['headline'].fillna('').astype(str)

        # Check if there's any text to vectorize
        if headlines_for_vectorization.str.strip().empty:
            print("No valid headlines for keyword extraction. Skipping.")
            return

        try:
            cv = CountVectorizer(stop_words='english', max_features=20)
            keywords_matrix = cv.fit_transform(headlines_for_vectorization)
            
            word_frequencies = keywords_matrix.sum(axis=0)
            words = cv.get_feature_names_out()

            word_freq_df = pd.DataFrame({
                'word': words,
                'frequency': word_frequencies.tolist()[0]
            }).sort_values(by='frequency', ascending=False)

            print("Top 20 Keywords in Headlines:\n", word_freq_df)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='frequency', y='word', data=word_freq_df, palette='viridis')
            plt.title('Top 20 Keywords in Headlines')
            plt.xlabel('Frequency (Total Occurrences)')
            plt.ylabel('Keyword')
            plt.grid(axis='x', alpha=0.75)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, 'top_keywords_bar_chart.png'))
            plt.close()
            print(f"Plot saved to {os.path.join(self.plots_output_dir, 'top_keywords_bar_chart.png')}")
        except Exception as e:
            print(f"Error during keyword extraction: {e}. Skipping.")


    def analyze_hourly_publication(self):
        """Analyzes and plots article publication by hour of day."""
        print("\n--- EDA: Time Series Analysis (Hourly Publication) ---")
        if self.df_news.empty or 'date' not in self.df_news.columns:
            print("Skipping hourly publication analysis: News DataFrame is empty or 'date' column missing.")
            return

        # Ensure 'date' column is datetime and not NaT before extracting hour
        df_hourly = self.df_news.dropna(subset=['date']).copy()
        if df_hourly.empty:
            print("No valid dates for hourly analysis. Skipping plot.")
            return

        df_hourly['hour'] = df_hourly['date'].dt.hour
        hourly_counts = df_hourly['hour'].value_counts().sort_index()

        if hourly_counts.empty:
            print("No hourly article counts found. Skipping plot.")
            return

        print("Article Publication by Hour:\n", hourly_counts)

        plt.figure(figsize=(12, 7))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='Blues_d')
        plt.title('Article Publication by Hour of Day')
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Number of Articles')
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'hourly_publication_bar_chart.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(self.plots_output_dir, 'hourly_publication_bar_chart.png')}")

    def analyze_publisher_domains(self):
        """Analyzes and plots top publisher domains."""
        print("\n--- EDA: Publisher Analysis (Domain Extraction) ---")
        if self.df_news.empty or 'publisher' not in self.df_news.columns:
            print("Skipping publisher domain analysis: News DataFrame is empty or 'publisher' column missing.")
            return

        self.df_news['publisher_clean'] = self.df_news['publisher'].fillna('').astype(str)

        # Robust domain extraction
        # This regex tries to extract domains from email-like strings or simply uses the publisher name
        self.df_news['domain'] = self.df_news['publisher_clean'].str.extract(r'@([\w\.\-]+(?:\.com|\.org|\.net|\.co|\.gov|\.edu|\.io|\.info|\.biz|\.us|\.uk|\.ca|\.au|\.de|\.fr|\.jp|\.cn|\.ru|\.br|\.in|\.mx|\.es|\.it|\.nl|\.se|\.no|\.dk|\.fi|\.ch|\.at|\.gr|\.be|\.nz|\.pt|\.za|\.ae|\.sg|\.kr|\.hk|\.my|\.ph|\.vn|\.th|\.id|\.ar|\.cl|\.co|\.il|\.sa|\.eg|\.ng|\.pk|\.bd|\.lk|\.my|\.ke|\.et|\.ug)\b)', flags=0)
        self.df_news['domain'].fillna(self.df_news['publisher_clean'], inplace=True) # Fallback to original publisher name if no domain extracted

        domain_counts = self.df_news['domain'].value_counts().head(10)
        if domain_counts.empty:
            print("No publisher domains found. Skipping plot.")
            return

        print("Top 10 Publisher Domains:\n", domain_counts)

        plt.figure(figsize=(12, 7))
        sns.barplot(x=domain_counts.index, y=domain_counts.values, palette='cividis')
        plt.title('Top 10 Publisher Domains by Article Count')
        plt.xlabel('Publisher Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'top_publisher_domains_bar_chart.png'))
        plt.close()
        print(f"Plot saved to {os.path.join(self.plots_output_dir, 'top_publisher_domains_bar_chart.png')}")


class QuantitativeProcessor:
    """
    Calculates technical indicators (TA-Lib) and financial metrics (Pandas/NumPy).
    Includes illustrative use of PyNance if available.
    """
    def __init__(self, df_stocks, plots_output_dir):
        """
        Initializes QuantitativeProcessor.

        Args:
            df_stocks (pd.DataFrame): Cleaned stock DataFrame.
            plots_output_dir (str): Directory to save quantitative plots.
        """
        self.df_stocks = df_stocks
        self.plots_output_dir = plots_output_dir
        os.makedirs(self.plots_output_dir, exist_ok=True)

    def apply_talib_indicators(self):
        """Applies various technical indicators using TA-Lib."""
        print("\n--- Quantitative Analysis: Applying TA-Lib Indicators ---")
        if not TALIB_AVAILABLE:
            print("TA-Lib not available. Skipping technical indicator calculation.")
            return self.df_stocks
        if self.df_stocks.empty:
            print("Skipping TA-Lib indicators: Stock DataFrame is empty.")
            return self.df_stocks

        def _apply_indicators_to_group(group):
            if group.empty:
                return group

            # Ensure all required columns are available as numpy arrays for TA-Lib
            required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJ_CLOSE', 'VOLUME']
            if not all(col in group.columns for col in required_cols):
                print(f"Warning: Missing critical columns for TA-Lib in group {group.name}. Skipping indicators for this group.")
                return group

            open_price = np.array(group['OPEN'], dtype=float)
            high_price = np.array(group['HIGH'], dtype=float)
            low_price = np.array(group['LOW'], dtype=float)
            close_price = np.array(group['CLOSE'], dtype=float)
            volume = np.array(group['VOLUME'], dtype=float)
            adj_close_price = np.array(group['ADJ_CLOSE'], dtype=float)

            # Check for sufficient data points for TA-Lib calculations
            if len(adj_close_price) < max(10, 14, 20, 26): # Minimum period for common indicators
                print(f"Warning: Insufficient data for TA-Lib indicators in group {group.name}. Skipping indicators for this group.")
                return group

            group['SMA_10'] = ta.SMA(adj_close_price, timeperiod=10)
            group['EMA_20'] = ta.EMA(adj_close_price, timeperiod=20)
            group['SMA_50'] = ta.SMA(adj_close_price, timeperiod=50)

            group['RSI'] = ta.RSI(adj_close_price, timeperiod=14)

            macd, macdsignal, macdhist = ta.MACD(adj_close_price, fastperiod=12, slowperiod=26, signalperiod=9)
            group['MACD'] = macd
            group['MACD_Signal'] = macdsignal
            group['MACD_Hist'] = macdhist

            upper, middle, lower = ta.BBANDS(adj_close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            group['BB_Upper'] = upper
            group['BB_Middle'] = middle
            group['BB_Lower'] = lower

            fastk, fastd = ta.STOCH(high_price, low_price, adj_close_price,
                                    fastk_period=5, slowk_period=3, slowk_matype=0,
                                    slowd_period=3, slowd_matype=0)
            group['STOCH_K'] = fastk
            group['STOCH_D'] = fastd

            group['ATR'] = ta.ATR(high_price, low_price, adj_close_price, timeperiod=14)

            return group

        df_with_ta = self.df_stocks.groupby('STOCK_SYMBOL', group_keys=False).apply(_apply_indicators_to_group)
        print("TA-Lib indicators applied successfully to all stocks.")
        self.df_stocks = df_with_ta
        return self.df_stocks

    def calculate_financial_metrics(self):
        """Calculates common financial metrics using Pandas/NumPy."""
        print("\n--- Quantitative Analysis: Calculating Financial Metrics ---")
        if self.df_stocks.empty or 'ADJ_CLOSE' not in self.df_stocks.columns:
            print("Skipping financial metrics: Stock DataFrame is empty or 'ADJ_CLOSE' column missing.")
            return self.df_stocks

        def _calculate_metrics_to_group(group):
            if group.empty:
                return group
            
            # Check for sufficient data points for pct_change
            if len(group) < 2:
                print(f"Warning: Insufficient data for Daily_Return in group {group.name}. Skipping.")
                group['Daily_Return'] = np.nan
                group['Cumulative_Return'] = np.nan
                group['Rolling_Volatility_20D'] = np.nan
                return group

            group['Daily_Return'] = group['ADJ_CLOSE'].pct_change()
            group['Cumulative_Return'] = (1 + group['Daily_Return'].fillna(0)).cumprod() - 1
            
            # Check for sufficient data points for rolling std (20-day window)
            if len(group) < 20:
                print(f"Warning: Insufficient data for Rolling_Volatility_20D in group {group.name}. Skipping.")
                group['Rolling_Volatility_20D'] = np.nan
            else:
                group['Rolling_Volatility_20D'] = group['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

            return group

        df_with_metrics = self.df_stocks.groupby('STOCK_SYMBOL', group_keys=False).apply(_calculate_metrics_to_group)
        print("Financial metrics calculated successfully for all stocks.")
        self.df_stocks = df_with_metrics
        return self.df_stocks

    def illustrate_pynance_use(self):
        """Illustrates conceptual use of PyNance if available."""
        print("\n--- Quantitative Analysis: Illustrating PyNance Use (Conceptual) ---")
        if not PYNANCE_AVAILABLE:
            print("PyNance not available. Skipping PyNance illustration.")
            return

        print("Note: PyNance is generally not actively maintained and its installation/usage can be problematic.")
        print("Core financial metrics are robustly calculated using Pandas/NumPy in this pipeline.")
        print("This section provides a conceptual example if PyNance were fully functional and integrated.")

        # Example of what you *might* do with PyNance if it were reliable and installed
        # This code is illustrative and might not run without a specific pynance setup.
        # It's here to show the "intent" of using pynance as requested.
        try:
            sample_stock_symbol = self.df_stocks['STOCK_SYMBOL'].unique()[0]
            sample_df = self.df_stocks[self.df_stocks['STOCK_SYMBOL'] == sample_stock_symbol].copy()

            # Example: Calculating a simple moving average using pynance (conceptual)
            # This would likely involve converting to a pynance-specific object or using its functions
            # if pn.moving_average is available and works as expected.
            # For demonstration, we'll just print a message.
            print(f"Conceptually, one might use PyNance functions like pn.moving_average(sample_df['ADJ_CLOSE'], period=10)")
            print("or pn.sharpe_ratio(...) for advanced metrics.")
            print("However, due to PyNance's status, these are handled by TA-Lib and Pandas/NumPy for reliability.")

        except Exception as e:
            print(f"Error during conceptual PyNance illustration: {e}. This likely confirms PyNance's instability.")


class SentimentCorrelationProcessor:
    """
    Performs sentiment analysis using VADER and calculates correlations with stock returns.
    """
    def __init__(self, df_news, df_stocks, plots_output_dir):
        """
        Initializes SentimentCorrelationProcessor.

        Args:
            df_news (pd.DataFrame): Cleaned news DataFrame.
            df_stocks (pd.DataFrame): Cleaned stock DataFrame with financial metrics.
            plots_output_dir (str): Directory to save correlation plots.
        """
        self.df_news = df_news
        self.df_stocks = df_stocks
        self.plots_output_dir = plots_output_dir
        os.makedirs(self.plots_output_dir, exist_ok=True)
        self.analyzer = SentimentIntensityAnalyzer() # VADER analyzer

    def perform_sentiment_analysis(self):
        """Applies VADER sentiment analysis to news headlines and aggregates daily."""
        print("\n--- Sentiment Analysis: Applying VADER to News Headlines ---")
        if self.df_news.empty or 'headline' not in self.df_news.columns:
            print("Skipping sentiment analysis: News DataFrame is empty or 'headline' column missing.")
            return None

        # Ensure 'headline' column is string type and handle NaNs
        self.df_news['headline_clean'] = self.df_news['headline'].fillna('').astype(str)

        def get_vader_sentiment(text):
            try:
                scores = self.analyzer.polarity_scores(text)
                return scores['compound']
            except Exception as e:
                print(f"Warning: Failed to get VADER sentiment for text (first 50 chars): '{text[:50]}...'. Error: {e}")
                return np.nan # Return NaN for failed sentiment calculation

        self.df_news['sentiment_compound_score'] = self.df_news['headline_clean'].apply(get_vader_sentiment)
        
        initial_sentiment_rows = len(self.df_news)
        self.df_news.dropna(subset=['sentiment_compound_score'], inplace=True)
        if len(self.df_news) < initial_sentiment_rows:
            print(f"Dropped {initial_sentiment_rows - len(self.df_news)} rows from news data due to failed sentiment calculation.")

        if self.df_news.empty:
            print("Warning: News DataFrame is empty after sentiment analysis. Cannot proceed.")
            return None

        # Aggregate daily sentiment per stock
        daily_sentiment = self.df_news.groupby(['STOCK_SYMBOL', self.df_news['date'].dt.date])['sentiment_compound_score'].mean().reset_index()
        daily_sentiment.rename(columns={'date': 'DATE'}, inplace=True)
        daily_sentiment['DATE'] = pd.to_datetime(daily_sentiment['DATE'])

        if daily_sentiment.empty:
            print("Warning: Daily aggregated sentiment DataFrame is empty. Cannot proceed.")
            return None

        print("VADER sentiment analysis and daily aggregation complete.")
        print("Daily Aggregated Sentiment Head:\n", daily_sentiment.head())
        return daily_sentiment

    def merge_data_for_correlation(self, daily_sentiment_df):
        """Merges stock data with daily sentiment data."""
        print("\n--- Correlation Analysis: Merging Data ---")
        if self.df_stocks.empty or daily_sentiment_df.empty:
            print("Skipping merge: One or both DataFrames are empty.")
            return pd.DataFrame()

        # Ensure stock data is indexed by Date for merging
        if not isinstance(self.df_stocks.index, pd.DatetimeIndex):
            print("Error: Stock DataFrame not indexed by DATE. Cannot merge. Exiting.")
            sys.exit(1)
        
        # Ensure daily_sentiment_df is indexed by Date for merging
        if not isinstance(daily_sentiment_df.index, pd.DatetimeIndex):
            daily_sentiment_df.set_index('DATE', inplace=True)

        # Validate common columns before merge
        required_stock_merge_cols = ['STOCK_SYMBOL', 'Daily_Return']
        if not all(col in self.df_stocks.columns for col in required_stock_merge_cols):
            print(f"Error: Stock data missing required columns for merge: {required_stock_merge_cols}. Exiting.")
            sys.exit(1)
        
        required_sentiment_merge_cols = ['STOCK_SYMBOL', 'sentiment_compound_score']
        if not all(col in daily_sentiment_df.columns for col in required_sentiment_merge_cols):
            print(f"Error: Sentiment data missing required columns for merge: {required_sentiment_merge_cols}. Exiting.")
            sys.exit(1)

        df_merged = pd.merge(
            self.df_stocks[['STOCK_SYMBOL', 'Daily_Return']],
            daily_sentiment_df[['STOCK_SYMBOL', 'sentiment_compound_score']],
            on=['DATE', 'STOCK_SYMBOL'],
            how='inner'
        )

        if df_merged.empty:
            print("Warning: Merged DataFrame is empty. No overlapping dates/symbols. Cannot proceed with correlation.")
        else:
            print("Data merged successfully.")
            print("Merged DataFrame Head:\n", df_merged.head())
        return df_merged

    def calculate_correlation(self, df_merged):
        """Calculates Pearson correlation between daily returns and sentiment."""
        print("\n--- Correlation Analysis: Calculating Correlations ---")
        if df_merged.empty:
            print("Skipping correlation calculation: Merged DataFrame is empty.")
            return pd.DataFrame()

        # Check for sufficient data for correlation
        if len(df_merged) < 2:
            print(f"Warning: Merged DataFrame has only {len(df_merged)} rows. Not enough data for meaningful correlation. Returning empty results.")
            return pd.DataFrame()

        min_group_size = df_merged.groupby('STOCK_SYMBOL').size().min()
        if min_group_size < 2:
            print(f"Warning: Some stock groups in merged data have fewer than 2 data points ({min_group_size}). Correlation for these might be NaN.")

        correlation_results = df_merged.groupby('STOCK_SYMBOL').apply(
            lambda x: x['Daily_Return'].corr(x['sentiment_compound_score'])
        ).rename('Correlation_Sentiment_Return')

        correlation_results.dropna(inplace=True)

        if correlation_results.empty:
            print("Warning: No valid correlation results obtained for any stock. This might be due to insufficient data or constant values.")
        else:
            print("Correlation results:\n", correlation_results)
        
        correlation_df = correlation_results.reset_index()
        correlation_df.columns = ['STOCK_SYMBOL', 'Correlation_Sentiment_Return']
        return correlation_df


class Visualizer:
    """
    Handles all plotting functionalities for EDA, Quantitative, and Correlation analyses.
    Ensures plots are saved and closed to prevent memory issues.
    """
    def __init__(self, plots_output_dir):
        self.plots_output_dir = plots_output_dir
        os.makedirs(self.plots_output_dir, exist_ok=True)

    def plot_eda_headline_length(self, df_news):
        if df_news.empty or 'headline_length' not in df_news.columns: return
        plt.figure(figsize=(10, 6))
        sns.histplot(df_news['headline_length'], bins=50, kde=True, color='steelblue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'headline_length_distribution.png'))
        plt.close()

    def plot_eda_top_publishers(self, df_news):
        if df_news.empty or 'publisher' not in df_news.columns: return
        top_publishers = df_news["publisher"].value_counts().head(10)
        if top_publishers.empty: return
        plt.figure(figsize=(12, 7))
        sns.barplot(x=top_publishers.index, y=top_publishers.values, palette='viridis')
        plt.title("Top 10 Publishers by Article Count")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'top_publishers_bar_chart.png'))
        plt.close()

    def plot_eda_publication_date_trends(self, df_news):
        if df_news.empty or 'date' not in df_news.columns: return
        articles_per_day = df_news["date"].dt.date.value_counts().sort_index()
        if articles_per_day.empty: return
        plt.figure(figsize=(14, 7))
        articles_per_day.plot(kind='line', color='darkorange', linewidth=1.5)
        plt.title("Article Publication Frequency Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'article_frequency_over_time.png'))
        plt.close()

    def plot_eda_top_keywords(self, word_freq_df):
        if word_freq_df.empty: return
        plt.figure(figsize=(10, 8))
        sns.barplot(x='frequency', y='word', data=word_freq_df, palette='viridis')
        plt.title('Top 20 Keywords in Headlines')
        plt.xlabel('Frequency (Total Occurrences)')
        plt.ylabel('Keyword')
        plt.grid(axis='x', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'top_keywords_bar_chart.png'))
        plt.close()

    def plot_eda_hourly_publication(self, df_news):
        if df_news.empty or 'date' not in df_news.columns: return
        df_hourly = df_news.dropna(subset=['date']).copy()
        if df_hourly.empty: return
        df_hourly['hour'] = df_hourly['date'].dt.hour
        hourly_counts = df_hourly['hour'].value_counts().sort_index()
        if hourly_counts.empty: return
        plt.figure(figsize=(12, 7))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='Blues_d')
        plt.title('Article Publication by Hour of Day')
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Number of Articles')
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'hourly_publication_bar_chart.png'))
        plt.close()

    def plot_eda_publisher_domains(self, df_news):
        if df_news.empty or 'domain' not in df_news.columns: return
        domain_counts = df_news['domain'].value_counts().head(10)
        if domain_counts.empty: return
        plt.figure(figsize=(12, 7))
        sns.barplot(x=domain_counts.index, y=domain_counts.values, palette='cividis')
        plt.title('Top 10 Publisher Domains by Article Count')
        plt.xlabel('Publisher Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'top_publisher_domains_bar_chart.png'))
        plt.close()

    def plot_quantitative_indicators(self, df_stocks, stock_symbol):
        if df_stocks.empty or stock_symbol not in df_stocks['STOCK_SYMBOL'].unique(): return
        df_plot = df_stocks[df_stocks['STOCK_SYMBOL'] == stock_symbol].copy()
        if df_plot.empty: return

        # Plot 1: Close Price with Moving Averages
        plt.figure(figsize=(14, 7))
        plt.plot(df_plot.index, df_plot['ADJ_CLOSE'], label='Adjusted Close Price', color='blue', linewidth=1.5)
        if 'SMA_10' in df_plot.columns: plt.plot(df_plot.index, df_plot['SMA_10'], label='SMA 10', color='red', linestyle='--', linewidth=1)
        if 'EMA_20' in df_plot.columns: plt.plot(df_plot.index, df_plot['EMA_20'], label='EMA 20', color='green', linestyle=':', linewidth=1)
        if 'SMA_50' in df_plot.columns: plt.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50', color='purple', linestyle='-.', linewidth=1)
        plt.title(f'{stock_symbol} Stock Adjusted Close Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_close_price_with_mas.png'))
        plt.close()

        # Plot 2: RSI
        if 'RSI' in df_plot.columns:
            plt.figure(figsize=(14, 5))
            plt.plot(df_plot.index, df_plot['RSI'], label='RSI (14)', color='purple')
            plt.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
            plt.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
            plt.title(f'{stock_symbol} Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI Value')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_rsi_plot.png'))
            plt.close()

        # Plot 3: MACD
        if 'MACD' in df_plot.columns and 'MACD_Signal' in df_plot.columns and 'MACD_Hist' in df_plot.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(df_plot.index, df_plot['MACD'], label='MACD Line', color='blue', linewidth=1.5)
            ax1.plot(df_plot.index, df_plot['MACD_Signal'], label='Signal Line', color='red', linestyle='--', linewidth=1)
            ax1.set_title(f'{stock_symbol} Moving Average Convergence Divergence (MACD)')
            ax1.set_ylabel('MACD Value')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            colors = ['green' if val >= 0 else 'red' for val in df_plot['MACD_Hist']]
            ax2.bar(df_plot.index, df_plot['MACD_Hist'], color=colors, label='MACD Histogram')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Histogram')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_macd_plot.png'))
            plt.close()

        # Plot 4: Bollinger Bands
        if 'BB_Upper' in df_plot.columns and 'BB_Middle' in df_plot.columns and 'BB_Lower' in df_plot.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(df_plot.index, df_plot['ADJ_CLOSE'], label='Adjusted Close Price', color='blue', linewidth=1.5)
            plt.plot(df_plot.index, df_plot['BB_Upper'], label='Upper BB', color='orange', linestyle='--', linewidth=1)
            plt.plot(df_plot.index, df_plot['BB_Middle'], label='Middle BB (SMA 20)', color='green', linestyle=':', linewidth=1)
            plt.plot(df_plot.index, df_plot['BB_Lower'], label='Lower BB', color='red', linestyle='--', linewidth=1)
            plt.fill_between(df_plot.index, df_plot['BB_Lower'], df_plot['BB_Upper'], color='gray', alpha=0.1)
            plt.title(f'{stock_symbol} Stock Adjusted Close Price with Bollinger Bands')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_bollinger_bands_plot.png'))
            plt.close()

    def plot_financial_metrics(self, df_stocks, stock_symbol):
        if df_stocks.empty or stock_symbol not in df_stocks['STOCK_SYMBOL'].unique(): return
        df_plot = df_stocks[df_stocks['STOCK_SYMBOL'] == stock_symbol].copy()
        if df_plot.empty: return

        # Plot 1: Daily Returns
        if 'Daily_Return' in df_plot.columns:
            plt.figure(figsize=(14, 5))
            plt.plot(df_plot.index, df_plot['Daily_Return'], label='Daily Return', color='darkcyan', linewidth=0.8)
            plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
            plt.title(f'{stock_symbol} Daily Percentage Returns')
            plt.xlabel('Date')
            plt.ylabel('Percentage Change')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_daily_returns_plot.png'))
            plt.close()

        # Plot 2: Cumulative Returns
        if 'Cumulative_Return' in df_plot.columns:
            plt.figure(figsize=(14, 5))
            plt.plot(df_plot.index, df_plot['Cumulative_Return'], label='Cumulative Return', color='darkgreen', linewidth=1.5)
            plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
            plt.title(f'{stock_symbol} Cumulative Returns Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_cumulative_returns_plot.png'))
            plt.close()

        # Plot 3: Rolling Volatility
        if 'Rolling_Volatility_20D' in df_plot.columns:
            plt.figure(figsize=(14, 5))
            plt.plot(df_plot.index, df_plot['Rolling_Volatility_20D'], label='Rolling Volatility (20-Day Annualized)', color='darkred', linewidth=1.5)
            plt.title(f'{stock_symbol} Rolling Volatility of Daily Returns')
            plt.xlabel('Date')
            plt.ylabel('Annualized Volatility')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_rolling_volatility_plot.png'))
            plt.close()

    def plot_correlation_results(self, correlation_df):
        if correlation_df.empty:
            print("No correlation results to plot.")
            return
        plt.figure(figsize=(10, 6))
        sns.barplot(x='STOCK_SYMBOL', y='Correlation_Sentiment_Return', data=correlation_df, palette='coolwarm')
        plt.title('Correlation Between News Sentiment (VADER Compound) and Daily Stock Returns by Company')
        plt.xlabel('Stock Symbol')
        plt.ylabel('Correlation Coefficient')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.ylim(-1, 1)
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, 'sentiment_return_correlation_bar_chart_vader.png'))
        plt.close()

    def plot_sentiment_return_scatter(self, df_merged, stock_symbol):
        if df_merged.empty or stock_symbol not in df_merged['STOCK_SYMBOL'].unique(): return
        df_plot = df_merged[df_merged['STOCK_SYMBOL'] == stock_symbol].copy()
        if df_plot.empty or len(df_plot) < 2: return # Need at least 2 points for meaningful scatter
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='sentiment_compound_score', y='Daily_Return', data=df_plot, alpha=0.6, s=50)
        plt.title(f'VADER Sentiment Compound Score vs. Daily Return for {stock_symbol}')
        plt.xlabel('News Sentiment Compound Score (-1 Negative to 1 Positive)')
        plt.ylabel('Daily Stock Return')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_output_dir, f'{stock_symbol}_sentiment_return_scatter_plot_vader.png'))
        plt.close()


class FinancialAnalysisPipeline:
    """
    Orchestrates the entire financial analysis workflow,
    from data loading to correlation and visualization.
    """
    def __init__(self, news_file_path, stock_files_dict, output_base_dir='./analysis_output'):
        self.news_file_path = news_file_path
        self.stock_files_dict = stock_files_dict
        self.output_base_dir = output_base_dir

        self.eda_plots_dir = os.path.join(output_base_dir, 'eda_plots')
        self.quantitative_plots_dir = os.path.join(output_base_dir, 'quantitative_plots')
        self.correlation_plots_dir = os.path.join(output_base_dir, 'correlation_plots')

        os.makedirs(self.eda_plots_dir, exist_ok=True)
        os.makedirs(self.quantitative_plots_dir, exist_ok=True)
        os.makedirs(self.correlation_plots_dir, exist_ok=True)

        self.data_loader = DataLoader(self.news_file_path, self.stock_files_dict)
        self.df_news = None
        self.df_stocks = None
        self.df_merged_correlation = None
        self.correlation_results_df = None

    def run_pipeline(self):
        """Executes the full financial analysis pipeline."""
        print("--- Starting Financial Analysis Pipeline ---")

        # 1. Load Data
        self.df_news = self.data_loader.load_news_data()
        self.df_stocks = self.data_loader.load_stock_data()

        # 2. EDA
        print("\n--- Running Exploratory Data Analysis (EDA) ---")
        eda_processor = EDAProcessor(self.df_news.copy(), self.eda_plots_dir)
        eda_processor.analyze_headline_length()
        eda_processor.analyze_publishers()
        eda_processor.analyze_publication_date_trends()
        eda_processor.analyze_text_keywords()
        eda_processor.analyze_hourly_publication()
        eda_processor.analyze_publisher_domains()
        print("--- EDA Complete ---")

        # 3. Quantitative Analysis
        print("\n--- Running Quantitative Analysis ---")
        quantitative_processor = QuantitativeProcessor(self.df_stocks.copy(), self.quantitative_plots_dir)
        self.df_stocks = quantitative_processor.apply_talib_indicators()
        self.df_stocks = quantitative_processor.calculate_financial_metrics()
        quantitative_processor.illustrate_pynance_use() # Illustrative PyNance use
        print("--- Quantitative Analysis Complete ---")

        # Plot quantitative results for a sample stock
        if not self.df_stocks.empty:
            sample_stock_to_plot = self.df_stocks['STOCK_SYMBOL'].unique()[0]
            print(f"\n--- Plotting Quantitative Results for Sample Stock: {sample_stock_to_plot} ---")
            visualizer = Visualizer(self.quantitative_plots_dir)
            visualizer.plot_quantitative_indicators(self.df_stocks, sample_stock_to_plot)
            visualizer.plot_financial_metrics(self.df_stocks, sample_stock_to_plot)
            print("--- Quantitative Plotting Complete ---")


        # 4. Sentiment Analysis and Correlation
        print("\n--- Running Sentiment Analysis and Correlation ---")
        sentiment_correlation_processor = SentimentCorrelationProcessor(
            self.df_news.copy(), self.df_stocks.copy(), self.correlation_plots_dir
        )
        daily_sentiment_df = sentiment_correlation_processor.perform_sentiment_analysis()

        if daily_sentiment_df is not None and not daily_sentiment_df.empty:
            self.df_merged_correlation = sentiment_correlation_processor.merge_data_for_correlation(daily_sentiment_df)
            if not self.df_merged_correlation.empty:
                self.correlation_results_df = sentiment_correlation_processor.calculate_correlation(self.df_merged_correlation)
                print("--- Sentiment Analysis and Correlation Complete ---")

                # Plot correlation results
                print("\n--- Plotting Correlation Results ---")
                visualizer = Visualizer(self.correlation_plots_dir)
                visualizer.plot_correlation_results(self.correlation_results_df)
                if not self.df_merged_correlation.empty:
                    sample_stock_for_scatter = self.df_merged_correlation['STOCK_SYMBOL'].unique()[0]
                    visualizer.plot_sentiment_return_scatter(self.df_merged_correlation, sample_stock_for_scatter)
                print("--- Correlation Plotting Complete ---")
            else:
                print("Skipping correlation analysis due to empty merged DataFrame.")
        else:
            print("Skipping correlation analysis due to empty sentiment DataFrame.")


        print("\n--- Financial Analysis Pipeline Complete ---")


if __name__ == "__main__":
    # --- Configuration for running the pipeline ---
    NEWS_FILE_PATH = './data/raw_analyst_ratings.csv'
    STOCK_FILES = {
        'AAPL': './data/aapl.csv',
        'MSFT': './data/msft.csv',
        'GOOG': './data/goog.csv',
        'AMZN': './data/amzn.csv',
        'TSLA': './data/tsla.csv',
        'NVDA': './data/nvda.csv',
        'JPM': './data/jpm.csv'
        # Add paths for your other stock CSVs here
    }
    OUTPUT_BASE_DIR = './analysis_output' # Base directory for all plots

    # Create dummy data if files don't exist for testing purposes
    # In a real scenario, ensure your 'data/' directory contains the actual files.
    if not os.path.exists(NEWS_FILE_PATH) or not all(os.path.exists(f) for f in STOCK_FILES.values()):
        print("Creating dummy data files for demonstration. Please replace with your actual data.")
        os.makedirs('./data', exist_ok=True)
        # Dummy news data
        with open(NEWS_FILE_PATH, 'w') as f:
            f.write("headline,url,publisher,date,stock\n")
            f.write("Positive news for AAPL,http://example.com/aapl,Publisher A,2023-01-02 10:00:00,AAPL\n")
            f.write("Negative outlook for MSFT,http://example.com/msft,Publisher B,2023-01-02 11:00:00,MSFT\n")
            f.write("Neutral market report,http://example.com/market,Publisher C,2023-01-03 09:00:00,AAPL\n")
            f.write("Strong quarter for GOOG,http://example.com/goog,Publisher A,2023-01-04 12:00:00,GOOG\n")
            f.write("AMZN stock dips,http://example.com/amzn,Publisher D,2023-01-05 14:00:00,AMZN\n")

        # Dummy stock data for 7 companies
        for symbol in STOCK_FILES.keys():
            with open(STOCK_FILES[symbol], 'w') as f:
                f.write("Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits\n")
                f.write(f"2023-01-01,100,102,99,101,100,1000000,0,0\n")
                f.write(f"2023-01-02,101,103,100,102,101,1200000,0,0\n")
                f.write(f"2023-01-03,102,105,101,104,103,1500000,0,0\n")
                f.write(f"2023-01-04,103,106,102,105,104,1300000,0,0\n")
                f.write(f"2023-01-05,104,107,103,106,105,1100000,0,0\n")
                f.write(f"2023-01-06,105,108,104,107,106,1050000,0,0\n")
                f.write(f"2023-01-07,106,109,105,108,107,900000,0,0\n")
                f.write(f"2023-01-08,107,110,106,109,108,950000,0,0\n")
                f.write(f"2023-01-09,108,111,107,110,109,1100000,0,0\n")
                f.write(f"2023-01-10,109,112,108,111,110,1200000,0,0\n")
                f.write(f"2023-01-11,110,113,109,112,111,1300000,0,0\n")
                f.write(f"2023-01-12,111,114,110,113,112,1400000,0,0\n")
                f.write(f"2023-01-13,112,115,111,114,113,1500000,0,0\n")
                f.write(f"2023-01-14,113,116,112,115,114,1600000,0,0\n")
                f.write(f"2023-01-15,114,117,113,116,115,1700000,0,0\n")
                f.write(f"2023-01-16,115,118,114,117,116,1800000,0,0\n")
                f.write(f"2023-01-17,116,119,115,118,117,1900000,0,0\n")
                f.write(f"2023-01-18,117,120,116,119,118,2000000,0,0\n")
                f.write(f"2023-01-19,118,121,117,120,119,2100000,0,0\n")
                f.write(f"2023-01-20,119,122,118,121,120,2200000,0,0\n")
                f.write(f"2023-01-21,120,123,119,122,121,2300000,0,0\n")
                f.write(f"2023-01-22,121,124,120,123,122,2400000,0,0\n")
                f.write(f"2023-01-23,122,125,121,124,123,2500000,0,0\n")
                f.write(f"2023-01-24,123,126,122,125,124,2600000,0,0\n")
                f.write(f"2023-01-25,124,127,123,126,125,2700000,0,0\n")
                f.write(f"2023-01-26,125,128,124,127,126,2800000,0,0\n")
                f.write(f"2023-01-27,126,129,125,128,127,2900000,0,0\n")
                f.write(f"2023-01-28,127,130,126,129,128,3000000,0,0\n")
                f.write(f"2023-01-29,128,131,127,130,129,3100000,0,0\n")
                f.write(f"2023-01-30,129,132,128,131,130,3200000,0,0\n")


    # Instantiate and run the pipeline
    pipeline = FinancialAnalysisPipeline(NEWS_FILE_PATH, STOCK_FILES, OUTPUT_BASE_DIR)
    pipeline.run_pipeline()
