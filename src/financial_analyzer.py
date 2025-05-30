import pandas as pd
import talib as ta
import numpy as np

def load_data(file_path):
    """
    Loads financial data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
                      Returns None if the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

def clean_data(df):
    """
    Cleans the financial data:
    - Renames columns to uppercase for TA-Lib compatibility (if not already).
    - Converts 'Date' column to datetime and sets as index.
    - Handles missing values by forward filling or dropping.
    - Converts numerical columns to appropriate types.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if df is None:
        return None

    print("Cleaning data...")

    # Standardize column names to uppercase for TA-Lib
    df.columns = [col.upper() for col in df.columns]

    # Ensure required columns exist
    required_cols = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns for analysis: {missing}")
        return None

    # Convert 'DATE' to datetime and set as index
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df.set_index('DATE', inplace=True)
        df.sort_index(inplace=True)
    else:
        print("Error: 'DATE' column not found.")
        return None

    # Handle missing values (e.g., forward fill, then drop any remaining NaNs)
    # Consider using ffill for time-series data
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # Convert numerical columns to numeric types
    for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where critical financial data might be NaN after conversion
    df.dropna(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], inplace=True)

    if df.empty:
        print("Warning: DataFrame is empty after cleaning.")
    else:
        print("Data cleaned successfully.")
        print("Cleaned DataFrame Info:")
        df.info()
        print("\nCleaned DataFrame Head:")
        print(df.head())
    return df

def apply_talib_indicators(df):
    """
    Applies various technical indicators using TA-Lib.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with financial data.

    Returns:
        pd.DataFrame: The DataFrame with added technical indicators.
                      Returns None if the input DataFrame is None or empty.
    """
    if df is None or df.empty:
        print("Cannot apply TA-Lib indicators: DataFrame is None or empty.")
        return None

    print("\nApplying TA-Lib indicators...")

    # Ensure all required columns are available as numpy arrays for TA-Lib
    try:
        open_price = np.array(df['OPEN'], dtype=float)
        high_price = np.array(df['HIGH'], dtype=float)
        low_price = np.array(df['LOW'], dtype=float)
        close_price = np.array(df['CLOSE'], dtype=float)
        volume = np.array(df['VOLUME'], dtype=float)
    except KeyError as e:
        print(f"Error: Missing required column for TA-Lib: {e}")
        return df # Return original df if critical column is missing

    # Moving Averages
    df['SMA_10'] = ta.SMA(close_price, timeperiod=10)
    df['EMA_20'] = ta.EMA(close_price, timeperiod=20)

    # Relative Strength Index (RSI)
    df['RSI'] = ta.RSI(close_price, timeperiod=14)

    # Moving Average Convergence Divergence (MACD)
    macd, macdsignal, macdhist = ta.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # Stochastic Oscillator
    fastk, fastd = ta.STOCH(high_price, low_price, close_price,
                            fastk_period=5, slowk_period=3, slowk_matype=0,
                            slowd_period=3, slowd_matype=0)
    df['STOCH_K'] = fastk
    df['STOCH_D'] = fastd

    print("TA-Lib indicators applied successfully.")
    print("\nDataFrame with Indicators Head:")
    print(df.tail()) # Show tail to see recent indicator values
    return df

def main():
    """
    Main function to execute the data loading, cleaning, and TA-Lib application.
    """
    file_path = 'data/stock_data.csv' 
    

    # 1. Load Data
    df = load_data(file_path)
    if df is None:
        return

    # 2. Clean Data
    df_cleaned = clean_data(df.copy()) # Use a copy to avoid modifying original df
    if df_cleaned is None or df_cleaned.empty:
        print("Exiting: Data cleaning failed or resulted in empty DataFrame.")
        return

    # 3. Apply TA-Lib Indicators
    df_final = apply_talib_indicators(df_cleaned.copy()) # Use a copy
    if df_final is None:
        print("Exiting: Failed to apply TA-Lib indicators.")
        return

    print("\nAnalysis complete. Final DataFrame with indicators:")
    print(df_final.tail())
    print(f"DataFrame shape: {df_final.shape}")


if __name__ == "__main__":
    main()