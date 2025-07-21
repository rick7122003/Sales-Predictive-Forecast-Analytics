# Sales-Predictive-Forecast-Analytics
Sales Predictive Forecast Analytics - Accuracy Machine Learning Model


# PROBLEM STATEMENT: To develop a machine learning model to predict highest sales forecast accuracy over next 3 months in the future¶


# Key Findings:

Successfully combined 5 datasets into one comprehensive file with 8,800 records and 20 columns Data spans from October 2016 to December 2017 Major data quality issues: 85% missing subsidiary data, 24% missing close dates/values Deal pipeline shows 48% won rate (4,238 won vs 2,473 lost) Central region has highest activity (3,512 records) Added calculated fields: deal duration and revenue per employee The dataset is now ready for analysis with all relationships preserved and data properly formatted.

Dataset Successfully Combined and Saved as dataset_combined.csv

The complete process has been organized into three clear sections:

Section 1: Loads and examines all 5 datasets Section 2: Combines datasets using left joins and performs data cleaning Section 3: Conducts comprehensive data inspection and exports the final file The combined dataset contains 8,800 records with 20 columns spanning October 2016 to December 2017, ready for sales program management analysis.

# PART 1:

# SECTION 1: LOAD AND EXAMINE DATASETS
# Load all CSV files and examine their structure
import pandas as pd
import numpy as np

# Load all datasets
accounts = pd.read_csv('accounts.csv')
data_dictionary = pd.read_csv('data_dictionary.csv')
products = pd.read_csv('products.csv')
sales_pipeline = pd.read_csv('sales_pipeline.csv')
sales_teams = pd.read_csv('sales_teams.csv')

print("Files loaded successfully")
print("Dataset shapes:")
print("Accounts:", accounts.shape)
print("Data Dictionary:", data_dictionary.shape)
print("Products:", products.shape)
print("Sales Pipeline:", sales_pipeline.shape)
print("Sales Teams:", sales_teams.shape)

# Examine structure and columns of each dataset
print("\
ACCOUNTS DATASET:")
print("Columns:", accounts.columns.tolist())
print(accounts.head())

print("\
PRODUCTS DATASET:")
print("Columns:", products.columns.tolist())
print(products.head())

print("\
SALES PIPELINE DATASET:")
print("Columns:", sales_pipeline.columns.tolist())
print(sales_pipeline.head())

print("\
SALES TEAMS DATASET:")
print("Columns:", sales_teams.columns.tolist())
print(sales_teams.head())

print("\
DATA DICTIONARY:")
print(data_dictionary.head(10))



# SECTION 2: COMBINE DATASETS AND DATA PREPARATION
# Start with sales_pipeline as the main dataset and join others
combined_data = sales_pipeline.copy()

# Join with sales_teams on sales_agent
combined_data = combined_data.merge(sales_teams, on='sales_agent', how='left')

# Join with accounts on account
combined_data = combined_data.merge(accounts, on='account', how='left')

# Join with products on product
combined_data = combined_data.merge(products, on='product', how='left')

print("Datasets combined successfully")
print("Combined dataset shape:", combined_data.shape)
print("Combined dataset columns:", combined_data.columns.tolist())

# Data preparation and cleaning
# Convert date columns to datetime
combined_data['engage_date'] = pd.to_datetime(combined_data['engage_date'])
combined_data['close_date'] = pd.to_datetime(combined_data['close_date'])

# Clean and standardize text columns
combined_data['deal_stage'] = combined_data['deal_stage'].str.strip()
combined_data['sector'] = combined_data['sector'].str.strip().str.lower()
combined_data['office_location'] = combined_data['office_location'].str.strip()
combined_data['regional_office'] = combined_data['regional_office'].str.strip()

# Create additional calculated columns
combined_data['deal_duration_days'] = (combined_data['close_date'] - combined_data['engage_date']).dt.days
combined_data['revenue_per_employee'] = combined_data['revenue'] / combined_data['employees']

print("Data preparation completed")
print("Final dataset shape:", combined_data.shape)


<img width="811" height="83" alt="image" src="https://github.com/user-attachments/assets/28388315-0c8a-4054-99ca-1d88ddaed72a" />


# SECTION 3: DATA INSPECTION AND FINAL EXPORT
print("SECTION 3: DATA INSPECTION CHECKS")
print("="*50)

# Basic info
print("Dataset Info:")
print("Shape:", combined_data.shape)
print("Memory usage:", round(combined_data.memory_usage(deep=True).sum() / 1024**2, 2), "MB")

# Missing values check
print("\
Missing Values Analysis:")
missing_values = combined_data.isnull().sum()
missing_percent = (missing_values / len(combined_data)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Count', ascending=False)
print(missing_df[missing_df['Missing_Count'] > 0])

# Data types
print("\
Data Types:")
print(combined_data.dtypes)

# Unique values in categorical columns
print("\
Unique Values in Key Categorical Columns:")
categorical_cols = ['deal_stage', 'sector', 'regional_office', 'series']
for col in categorical_cols:
    print(f"{col}: {combined_data[col].nunique()} unique values")
    print(f"  Values: {combined_data[col].unique()}")

# Numerical columns summary
print("\
Numerical Columns Summary:")
numerical_cols = ['close_value', 'revenue', 'employees', 'sales_price', 'deal_duration_days']
print(combined_data[numerical_cols].describe())

# Deal stage distribution
print("\
Deal Stage Distribution:")
deal_stage_dist = combined_data['deal_stage'].value_counts()
print(deal_stage_dist)

# Regional office distribution
print("\
Regional Office Distribution:")
regional_dist = combined_data['regional_office'].value_counts()
print(regional_dist)

# Save the combined dataset with the specified filename
combined_data.to_csv('dataset_combined.csv', index=False)

print("\
" + "="*50)
print("DATASET PREPARATION COMPLETE")
print("="*50)
print("Combined dataset saved as 'dataset_combined.csv'")
print("Total records:", len(combined_data))
print("Total columns:", len(combined_data.columns))
print("Date range:", combined_data['engage_date'].min(), "to", combined_data['engage_date'].max())
print("Download at: https://julius.ai/files/dataset_combined.csv")

# Show first few rows of final dataset
print("\
First 5 rows of combined dataset:")
print(combined_data.head())



<img width="456" height="812" alt="image" src="https://github.com/user-attachments/assets/817da257-5984-493e-b109-61cb35a40053" />


<img width="406" height="813" alt="image" src="https://github.com/user-attachments/assets/0d55eac9-0792-4bbe-8436-f37c134be93e" />


<img width="835" height="656" alt="image" src="https://github.com/user-attachments/assets/d8eda375-ee0b-4eb0-b925-3b32deb053b6" />


<img width="827" height="276" alt="image" src="https://github.com/user-attachments/assets/c11cc987-f826-4ac1-8e2d-58b89d7a6424" />



# Clean Dataset

# Key Improvements Made:

# 1. Filled missing subsidiary_of values with 'Independent'
# 2. Standardized all text fields (proper case, consistent formatting)
# 3. Fixed sector typo ('technolgy' → 'technology')
# 4. Converted categorical columns to category type for memory efficiency
# 5 Handled extreme outliers in deal duration
# 6 Validated data integrity (no duplicates found)
# 7 Reduced memory usage from 6.25 MB to 3.50 MB

The cleaned dataset is now ready for analysis with improved data quality and consistency.¶


# Load the combined dataset for cleaning
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset_combined.csv')

print("Original dataset shape:", df.shape)
print("Initial missing values:")
print(df.isnull().sum().sort_values(ascending=False))

# SECTION 1: HANDLE MISSING VALUES
# Convert date columns to datetime first
df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')
df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')

# Handle missing values strategically
# For subsidiary_of - fill with 'Independent' (most companies are independent)
df['subsidiary_of'] = df['subsidiary_of'].fillna('Independent')

# For deal_duration_days - recalculate where possible
mask = df['deal_duration_days'].isna() & df['close_date'].notna() & df['engage_date'].notna()
df.loc[mask, 'deal_duration_days'] = (df.loc[mask, 'close_date'] - df.loc[mask, 'engage_date']).dt.days

# For revenue_per_employee - recalculate where possible
mask = df['revenue_per_employee'].isna() & df['revenue'].notna() & df['employees'].notna()
df.loc[mask, 'revenue_per_employee'] = df.loc[mask, 'revenue'] / df.loc[mask, 'employees']

# For sector - fill with 'Unknown'
df['sector'] = df['sector'].fillna('Unknown')

# For office_location - fill with 'Unknown'
df['office_location'] = df['office_location'].fillna('Unknown')

# For year_established - fill with median year
median_year = df['year_established'].median()
df['year_established'] = df['year_established'].fillna(median_year)

# For revenue and employees - fill with median values
df['revenue'] = df['revenue'].fillna(df['revenue'].median())
df['employees'] = df['employees'].fillna(df['employees'].median())

# For account - fill with 'Unknown Account'
df['account'] = df['account'].fillna('Unknown Account')

print("Missing values handled")
print("Remaining missing values:")
print(df.isnull().sum().sort_values(ascending=False))

# SECTION 2: FIX INCONSISTENT CATEGORICAL ENTRIES AND STANDARDIZE TEXT FIELDS

# Fix sector column inconsistencies
df['sector'] = df['sector'].str.strip().str.lower()
# Fix the typo 'technolgy' to 'technology'
df['sector'] = df['sector'].replace('technolgy', 'technology')

# Standardize office_location
df['office_location'] = df['office_location'].str.strip().str.title()
# Fix common country name variations
df['office_location'] = df['office_location'].replace({
    'Philipines': 'Philippines',
    'United States': 'United States'
})

# Standardize deal_stage
df['deal_stage'] = df['deal_stage'].str.strip().str.title()

# Standardize regional_office
df['regional_office'] = df['regional_office'].str.strip().str.title()

# Standardize manager names
df['manager'] = df['manager'].str.strip().str.title()

# Standardize sales_agent names
df['sales_agent'] = df['sales_agent'].str.strip().str.title()

# Standardize product names
df['product'] = df['product'].str.strip()

# Standardize account names
df['account'] = df['account'].str.strip().str.title()

# Fix series column - standardize to uppercase
df['series'] = df['series'].str.strip().str.upper()

print("Text fields standardized and inconsistencies fixed")
print("Unique values in key categorical columns:")
print("Sector:", df['sector'].unique())
print("Office Location:", df['office_location'].unique()[:10])  # Show first 10
print("Deal Stage:", df['deal_stage'].unique())
print("Regional Office:", df['regional_office'].unique())
print("Series:", df['series'].unique())

# SECTION 3: DATA TYPE VALIDATION AND CONVERSION

# Convert numeric columns to proper types
df['close_value'] = pd.to_numeric(df['close_value'], errors='coerce')
df['sales_price'] = pd.to_numeric(df['sales_price'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['employees'] = pd.to_numeric(df['employees'], errors='coerce')
df['year_established'] = pd.to_numeric(df['year_established'], errors='coerce').astype('Int64')
df['deal_duration_days'] = pd.to_numeric(df['deal_duration_days'], errors='coerce')
df['revenue_per_employee'] = pd.to_numeric(df['revenue_per_employee'], errors='coerce')

# Ensure date columns are datetime
df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')
df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')

# Convert categorical columns to category type for memory efficiency
categorical_columns = ['deal_stage', 'sector', 'regional_office', 'manager', 'series', 'office_location']
for col in categorical_columns:
    df[col] = df[col].astype('category')

print("Data types converted successfully")
print("Final data types:")
print(df.dtypes)

# SECTION 4: OUTLIER DETECTION AND HANDLING

# Check for outliers in numerical columns
numerical_cols = ['close_value', 'revenue', 'employees', 'sales_price', 'deal_duration_days']

print("OUTLIER ANALYSIS:")
print("="*50)

for col in numerical_cols:
    if col in df.columns and df[col].notna().sum() > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f"{col}:")
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        print(f"  Outlier bounds: < {lower_bound:.2f} or > {upper_bound:.2f}")

# Handle extreme outliers in deal_duration_days (negative or > 365 days)
print("\
Handling extreme outliers in deal_duration_days...")
before_count = df['deal_duration_days'].notna().sum()
df.loc[df['deal_duration_days'] < 0, 'deal_duration_days'] = np.nan
df.loc[df['deal_duration_days'] > 365, 'deal_duration_days'] = np.nan
after_count = df['deal_duration_days'].notna().sum()
print(f"Removed {before_count - after_count} extreme outliers from deal_duration_days")

# Handle negative values in revenue and employees (should not be negative)
df.loc[df['revenue'] < 0, 'revenue'] = np.nan
df.loc[df['employees'] < 0, 'employees'] = np.nan
df.loc[df['close_value'] < 0, 'close_value'] = 0  # Set negative close values to 0

print("Outlier handling completed")


# SECTION 5: DUPLICATE DETECTION AND REMOVAL

print("DUPLICATE ANALYSIS:")
print("="*50)

# Check for duplicate opportunity_ids (should be unique)
duplicate_ids = df[df.duplicated(subset=['opportunity_id'], keep=False)]
print(f"Duplicate opportunity_ids: {len(duplicate_ids)}")

if len(duplicate_ids) > 0:
    print("Sample duplicate opportunity_ids:")
    print(duplicate_ids[['opportunity_id', 'sales_agent', 'account', 'deal_stage']].head())
    
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=['opportunity_id'], keep='first')
    print(f"Removed {len(duplicate_ids) - len(duplicate_ids['opportunity_id'].unique())} duplicate records")

# Check for complete row duplicates
complete_duplicates = df[df.duplicated(keep=False)]
print(f"Complete row duplicates: {len(complete_duplicates)}")

if len(complete_duplicates) > 0:
    df = df.drop_duplicates(keep='first')
    print("Removed complete duplicate rows")

print(f"Final dataset shape after duplicate removal: {df.shape}")


# SECTION 6: FINAL VALIDATION AND EXPORT

print("FINAL DATASET VALIDATION:")
print("="*50)

# Final missing values check
print("Final missing values:")
missing_final = df.isnull().sum().sort_values(ascending=False)
print(missing_final[missing_final > 0])

# Data quality summary
print("\
Data Quality Summary:")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Date range validation
print(f"\
Date range:")
print(f"Engage dates: {df['engage_date'].min()} to {df['engage_date'].max()}")
print(f"Close dates: {df['close_date'].min()} to {df['close_date'].max()}")

# Key statistics
print("\
Key Statistics:")
print(f"Won deals: {(df['deal_stage'] == 'Won').sum():,} ({(df['deal_stage'] == 'Won').sum()/len(df)*100:.1f}%)")
print(f"Lost deals: {(df['deal_stage'] == 'Lost').sum():,} ({(df['deal_stage'] == 'Lost').sum()/len(df)*100:.1f}%)")
print(f"Active deals: {((df['deal_stage'] == 'Engaging') | (df['deal_stage'] == 'Prospecting')).sum():,}")

# Sector distribution
print("\
Sector distribution:")
sector_dist = df['sector'].value_counts()
print(sector_dist)

# Regional distribution
print("\
Regional distribution:")
regional_dist = df['regional_office'].value_counts()
print(regional_dist)

# Save the cleaned dataset
df.to_csv('dataset_combined_cleaned.csv', index=False)

print("\
" + "="*50)
print("DATA CLEANING COMPLETE")
print("="*50)
print("Cleaned dataset saved as 'dataset_combined_cleaned.csv'")
print("Download at: https://julius.ai/files/dataset_combined_cleaned.csv")

# Show sample of cleaned data
print("\
Sample of cleaned dataset:")
print(df.head())

====================================

# Perform EDA Analysis

====================================



# Key Business Insights
============================================================

# KEY TAKEAWAYS AND INSIGHTS

============================================================

🎯 BUSINESS PERFORMANCE:

• Overall win rate of 48.2% indicates strong sales performance

• Average deal size of $2,361 shows healthy deal values

• Total revenue of $10,005,534 from 4,238 won deals

📊 DEAL DISTRIBUTION:

• 4,238 Won deals (48.2%)

• 2,473 Lost deals (28.1%)

• 2,089 Active deals in pipeline

🏢 SECTOR INSIGHTS:

• Top sector: unknown with 1,425 deals

• Technology and retail dominate with 1,397 and 1,165 deals respectively

🌍 REGIONAL PERFORMANCE:

• Central region leads with $3,568,647 in total revenue

• West region: $3,346,293

• East region: $3,090,594

⏱️ DEAL DURATION:

• Average deal duration: 48 days

• Fastest deals close in under 10 days

• Longest deals can take up to 180+ days

📦 PRODUCT INSIGHTS:

• Top performing products by revenue:

GTXPro: $3,510,578

GTX Plus Pro: $2,629,651

MG Advanced: $2,216,387

GTX Plus Basic: $705,275

GTX Basic: $499,263

🏭 COMPANY SIZE ANALYSIS:

• Average deal size and win rate by company size:

Small (1-100): $1,235 avg deal, 60.1% win rate

Medium (101-500): $1,478 avg deal, 55.2% win rate

Large (501-2000): $1,369 avg deal, 55.3% win rate

Enterprise (2000+): $1,561 avg deal, 44.6% win rate

🎯 STRATEGIC RECOMMENDATIONS:

• Focus on retail and technology sectors for highest volume

• Central region shows strongest performance - replicate strategies

• Software and marketing sectors have highest win rates (59%+)

• Enterprise clients offer larger deal sizes but may require longer sales cycles

• Consider optimizing deal duration - current average of 48 days is reasonable

============================================================

EDA ANALYSIS COMPLETE

============================================================

# Summary
# This comprehensive EDA reveals a healthy sales organization with strong performance metrics. 
# The 48.2% win rate and $10M+ in total revenue demonstrate effective sales execution. 
# Key opportunities include focusing on high-performing sectors (software, marketing) and replicating Central region success strategies across other territories.



# MACHINE LEARNING MODEL EVALUATION #

1. SARIMA
SARIMA, which stands for Seasonal AutoRegressive Integrated Moving Average, is a powerful statistical and machine learning model used for time series forecasting. Here are the key things to know about it:

It's built for seasonal data. Seasonal patterns are predictable fluctuations that repeat over a specific period, like higher ice cream sales in the summer or increased retail activity during the holidays. SARIMA is specifically designed to identify and model these seasonal components, which makes it more accurate for this type of data than a standard ARIMA model.

It combines two models in one. The "SAR" in SARIMA refers to the seasonal part of the model, while the "IMA" refers to the non-seasonal part. In essence, it runs two parallel processes to capture both the short-term and long-term trends in the data.

It has several key parameters. To use a SARIMA model effectively, you need to define several parameters, typically represented as SARIMA(p, d, q)(P, D, Q)m. The lowercase letters (p, d, q) represent the non-seasonal components (AutoRegressive, Integrated, and Moving Average), while the uppercase letters (P, D, Q) represent the seasonal components. The 'm' specifies the number of time steps in each seasonal cycle (e.g., 12 for monthly data with an annual cycle).

It's widely used in various industries. You'll find SARIMA being used for many real-world applications, such as forecasting sales, predicting stock prices, estimating energy demand, and monitoring website traffic. Its ability to handle seasonality makes it a go-to choice for many forecasting tasks.

2. Prophet 🔮
Developed by Facebook, Prophet is specifically designed for business forecasting tasks like sales. It's exceptionally user-friendly and robust, even with smaller datasets like yours. It automatically detects yearly, weekly, and daily seasonality and is resilient to missing data.

Key Strengths:

Easy to Use: Requires minimal data preprocessing and tuning.

Handles Holidays: You can easily add custom holidays (e.g., local festivals in Penang) that might impact sales.

Interpretable: The model's output clearly breaks down the forecast into trend, seasonality, and holiday components.

Consideration: It works best out-of-the-box and is less flexible if you want to add many external features beyond holidays.

3. Exponential Smoothing (ETS) 💨
Also known as Holt-Winters, ETS is a classic and powerful statistical model that is a direct alternative to ARIMA. It works by breaking down the time series into three components: Error, Trend, and Seasonality. It's very fast and often serves as a strong performance baseline.

Key Strengths:

Fast and Efficient: Trains much faster than complex models.

Proven Performer: Often provides highly accurate forecasts, especially for data with clear trends and seasonal patterns.

Good for Shorter Series: Performs reliably without needing massive amounts of historical data.

Consideration: Like ARIMA, it's a univariate model, meaning it primarily uses past sales values for forecasting and doesn't easily incorporate other external variables.

4. XGBoost (eXtreme Gradient Boosting) 🚀
XGBoost is a powerful, tree-based algorithm that frequently wins machine learning competitions. It can achieve very high accuracy but requires you to reframe the problem from time series to a standard regression task.

Key Strengths:
High Accuracy: Often outperforms other models by learning complex patterns in data.

Feature Importance: Can tell you which factors (e.g., which month, or how recent the sale was) are most predictive.

Handles Interactions: Can automatically capture relationships between different features.

Consideration: This model requires feature engineering. You must manually create features from your engage_date column, such as lag features, rolling averages, and date components (month, year, week, day of week).

5. Random Forest Regressor 🌳
A Random Forest is another versatile, tree-based model. It works by building a multitude of decision trees and averaging their outputs. This makes it very robust and less prone to overfitting than a single decision tree, which is a key advantage for smaller datasets.

Key Strengths:

Robust: Less sensitive to outliers and noisy data.

Good Generalization: Performs well without extensive parameter tuning.

Non-linear: Can capture complex relationships that linear models might miss.

Consideration: Just like XGBoost, Random Forest requires you to perform feature engineering by creating lag and date-based features from your time series data.

Summary Comparison Model Ease of Use Performance Potential Feature Engineering Required? Prophet Very High High No Exponential Smoothing High High No ARIMA / SARIMA Medium High No XGBoost Medium Very High Yes (Crucial) Random Forest Medium High Yes (Crucial)


===============================
# ARIMA MODEL
===============================

<img width="294" height="205" alt="image" src="https://github.com/user-attachments/assets/005236f9-848c-4cdd-8d81-5c912d063bdc" />


# --- 0. Installation (Run this cell first in Jupyter) ---
# Installs the pmdarima library required for auto_arima
!pip install pmdarima

# --- Main Script ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Intelligent Column Detection ---
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)

    date_col, sales_col = None, None
    for col in df.columns:
        if df[col].dtype in ['object', 'datetime64[ns]']:
            try:
                temp_col = pd.to_datetime(df[col], errors='coerce')
                if temp_col.notna().sum() / len(df) > 0.8:
                    date_col = col
                    df[date_col] = temp_col
                    break
            except Exception:
                continue
    if not date_col:
        raise ValueError("Could not automatically detect a date column.")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != date_col:
            sales_col = col
            break
    if not sales_col:
        raise ValueError("Could not automatically detect a numeric sales column.")

    df.set_index(date_col, inplace=True)
    monthly_sales = df[sales_col].resample('MS').sum()
    monthly_sales = monthly_sales[monthly_sales > 0]

    print("✅ Data loaded and prepared successfully.")
    print(f"   - Automatically detected Date Column: '{date_col}'")
    print(f"   - Automatically detected Sales Column: '{sales_col}'")
    print(f"   - Time range of data: {monthly_sales.index.min().strftime('%Y-%m')} to {monthly_sales.index.max().strftime('%Y-%m')}")
    print(f"   - Total months of data: {len(monthly_sales)}")

except Exception as e:
    print(f"❌ Error during data loading: {e}")

else:
    # --- 2. Robust Train/Test Split ---
    test_size = 6
    if len(monthly_sales) < 15: # Warning for very short data
        test_size = 3
        print("\n⚠️ Warning: Time series is very short. Using a smaller test set.")
        
    train_size = len(monthly_sales) - test_size
    train_data, test_data = monthly_sales[0:train_size], monthly_sales[train_size:]
    print(f"   - Using {train_size} months for training and {test_size} months for validation.")

    # --- 3. Model Selection (FIXED: Using a NON-SEASONAL model) ---
    print("\n🔎 Searching for the best ARIMA model parameters (non-seasonal)...")
    # Set seasonal=False because the time series is too short for a seasonal model
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                             test='adf', max_p=3, max_q=3,
                             m=1,             # m=1 for non-seasonal
                             d=None,
                             seasonal=False,  # This is the key change
                             trace=False,
                             error_action='ignore', suppress_warnings=True,
                             stepwise=True)

    print(f"✅ Best model found: ARIMA{arima_model.order}")

    # --- 4. Model Validation and Accuracy Analysis ---
    predictions = arima_model.predict(n_periods=len(test_data))
    predictions = pd.Series(predictions, index=test_data.index)

    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    r2 = r2_score(test_data, predictions)

    # --- 5. Final Model Training and Future Forecasting ---
    # The seasonal_order will be (0,0,0,0) for a non-seasonal model
    final_model = SARIMAX(monthly_sales, order=arima_model.order,
                          seasonal_order=(0,0,0,0)).fit()
    
    forecast = final_model.get_forecast(steps=3)
    forecast_index = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # --- 6. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(monthly_sales, label='Historical Sales', color='dodgerblue', linewidth=2)
    ax.plot(predictions, label='Validation Forecast', color='darkorange', linestyle='--', linewidth=2)
    ax.plot(forecast_index, forecast_values, label='3-Month Future Forecast', color='red', linestyle='--', linewidth=2)
    ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.15, label='95% Confidence Interval')

    ax.set_title('Sales Forecast vs. Historical Data (Non-Seasonal ARIMA Model)', fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{sales_col} Volume', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- 7. Print Results and Findings ---
    print("\n" + "="*50)
    print("📊 MODEL ACCURACY & FORECAST RESULTS 📊")
    print("="*50)
    print("\n## Model Validation Metrics")
    print(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    print(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    print(f"**R-squared ($R^2$):** {r2:.2f}")
    print("\n## 📈 Future 3-Month Sales Forecast")
    forecast_df = pd.DataFrame({'Date': forecast_index.strftime('%Y-%m'),
                                'Forecasted Sales': [f"${val:,.2f}" for val in forecast_values]})
    print(forecast_df.to_string(index=False))


    <img width="886" height="647" alt="image" src="https://github.com/user-attachments/assets/256ecf94-6960-467d-be1b-59eb57b6bf5a" />


========================
# Prophet model #
========================



<img width="327" height="191" alt="image" src="https://github.com/user-attachments/assets/9189c53d-4f2c-4c46-8337-9d7401a08937" />



# Installs the prophet library required for the model
!pip install prophet

# --- Main Script ---
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Preparation for Prophet ---
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)

    # Prophet requires columns to be named 'ds' (datestamp) and 'y' (target)
    # Assuming 'engage_date' is the date and 'close_value' is the target
    df_prophet = df[['engage_date', 'close_value']].rename(columns={
        'engage_date': 'ds',
        'close_value': 'y'
    })

    # Convert 'ds' to datetime objects and aggregate sales by month
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    monthly_df = df_prophet.set_index('ds').resample('MS').sum().reset_index()

    print("✅ Data loaded and prepared successfully for Prophet.")
    print(f"   - Total months of data: {len(monthly_df)}")

except Exception as e:
    print(f"❌ Error during data loading: {e}")

else:
    # --- 2. Model Validation ---
    # Split data: train on all but the last 3 months, validate on the last 3
    validation_months = 3
    train = monthly_df[:-validation_months]
    test = monthly_df[-validation_months:]

    # Initialize and fit a model for validation
    model_val = Prophet()
    model_val.fit(train)

    # Make predictions on the validation set
    forecast_val = model_val.predict(test[['ds']])
    y_true = test['y'].values
    y_pred = forecast_val['yhat'].values

    # --- 3. Accuracy Metric Analysis ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # --- 4. Final Model Training and 3-Month Forecast ---
    # Train a new model on the *entire* dataset for the best future forecast
    model_full = Prophet()
    model_full.fit(monthly_df)

    # Create a dataframe for the next 3 months and make a forecast
    future = model_full.make_future_dataframe(periods=3, freq='MS')
    forecast = model_full.predict(future)

    # --- 5. Visualization ---
    print("\n📈 Generating forecast plots...")
    # Plot the main forecast
    fig1 = model_full.plot(forecast)
    plt.title('Prophet Forecast for Future 3 Months', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales Value', fontsize=12)
    plt.show()

    # Plot the forecast components (trend, seasonality)
    fig2 = model_full.plot_components(forecast)
    plt.show()
    
    # --- 6. Print Results and Findings ---
    print("\n" + "="*50)
    print("📊 MODEL ACCURACY & FORECAST RESULTS 📊")
    print("="*50)

    print("\n## Model Validation Metrics")
    print(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    print(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    print(f"**R-squared ($R^2$):** {r2:.2f}")

    print("\n## 🔮 Future 3-Month Sales Forecast")
    # Extract the last 3 rows which contain the future forecast
    future_forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)
    future_forecast_df = future_forecast_df.rename(columns={
        'ds': 'Month',
        'yhat': 'Forecasted Sales',
        'yhat_lower': 'Lower Estimate',
        'yhat_upper': 'Upper Estimate'
    })
    future_forecast_df['Month'] = future_forecast_df['Month'].dt.strftime('%Y-%m')
    for col in ['Forecasted Sales', 'Lower Estimate', 'Upper Estimate']:
        future_forecast_df[col] = future_forecast_df[col].apply(lambda x: f"${x:,.2f}")
        
    print(future_forecast_df.to_string(index=False))


    <img width="807" height="746" alt="image" src="https://github.com/user-attachments/assets/d1e6485d-1f7e-4f5e-a5d6-091105c0795b" />


=======================================
# Exponential Smoothing (ETS) model
=======================================


import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Preparation ---
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)

    # Automatically detect date and sales columns
    date_col, sales_col = None, None
    for col in df.columns:
        if df[col].dtype in ['object', 'datetime64[ns]']:
            try:
                temp_col = pd.to_datetime(df[col], errors='coerce')
                if temp_col.notna().sum() / len(df) > 0.8:
                    date_col = col
                    df[date_col] = temp_col
                    break
            except Exception:
                continue
    if not date_col:
        raise ValueError("Could not automatically detect a date column.")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != date_col:
            sales_col = col
            break
    if not sales_col:
        raise ValueError("Could not automatically detect a numeric sales column.")

    # Resample data to monthly sales totals
    df.set_index(date_col, inplace=True)
    monthly_sales = df[sales_col].resample('MS').sum()
    monthly_sales = monthly_sales[monthly_sales > 0]

    print("✅ Data loaded and prepared successfully.")
    print(f"   - Time range of data: {monthly_sales.index.min().strftime('%Y-%m')} to {monthly_sales.index.max().strftime('%Y-%m')}")
    print(f"   - Total months of data: {len(monthly_sales)}")

except Exception as e:
    print(f"❌ Error during data loading: {e}")

else:
    # --- 2. Model Validation (Train/Test Split) ---
    # Use the last 3 months for validation
    validation_months = 3
    train = monthly_sales[:-validation_months]
    test = monthly_sales[-validation_months:]

    # --- 3. Model Selection and Training ---
    # Initialize and fit the ETS model.
    # 'add' for additive trend and seasonality. Use 'mul' for multiplicative.
    # Using try-except to handle potential short series issues with seasonality.
    try:
        # Attempt to fit a seasonal model first
        ets_model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()
    except ValueError:
        print("⚠️ Warning: Not enough data for a seasonal model. Fitting a non-seasonal model.")
        # Fallback to a non-seasonal model (Holt's Linear Trend)
        ets_model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal=None
        ).fit()

    # Generate predictions for the validation period
    predictions = ets_model.forecast(steps=len(test))

    # --- 4. Accuracy Metric Analysis ---
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    r2 = r2_score(test, predictions)

    # --- 5. Final Model Training and Future Forecasting ---
    # Refit the model on the *entire* dataset to make the best future forecast
    try:
        final_model = ExponentialSmoothing(
            monthly_sales,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()
    except ValueError:
        final_model = ExponentialSmoothing(
            monthly_sales,
            trend='add',
            seasonal=None
        ).fit()


    # Forecast the next 3 months
    forecast_values = final_model.forecast(steps=3)
    forecast_index = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

    # --- 6. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot historical data, validation forecast, and future forecast
    ax.plot(monthly_sales.index, monthly_sales, label='Historical Sales', color='dodgerblue', linewidth=2)
    ax.plot(test.index, predictions, label='Validation Forecast', color='darkblue', linestyle='--', linewidth=2)
    ax.plot(forecast_index, forecast_values, label='3-Month Future Forecast', color='red', linestyle='--', linewidth=2)

    # Formatting the plot
    ax.set_title('Sales Forecast vs. Historical Data (Exponential Smoothing)', fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales Volume', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- 7. Print Results and Findings ---
    print("\n" + "="*50)
    print("📊 MODEL ACCURACY & FORECAST RESULTS 📊")
    print("="*50)

    print("\n## Model Validation Metrics")
    print(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    print(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    print(f"**R-squared ($R^2$):** {r2:.2f}")

    print("\n## 💨 Future 3-Month Sales Forecast")
    forecast_df = pd.DataFrame({
        'Date': forecast_index.strftime('%Y-%m'),
        'Forecasted Sales': [f"${val:,.2f}" for val in forecast_values]
    })
    print(forecast_df.to_string(index=False))



<img width="888" height="554" alt="image" src="https://github.com/user-attachments/assets/a7983ac8-2c2c-43ea-a1b0-c2d080d1315d" />


==============================
# XGBoost Model
==============================


<img width="296" height="206" alt="image" src="https://github.com/user-attachments/assets/97bb5d9a-4474-463d-a905-40f6df8d62d6" />


# Installs the xgboost library required for the model
!pip install xgboost

# --- Main Script ---
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Preparation ---
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)

    date_col, sales_col = 'engage_date', 'close_value'
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Resample to monthly sales and create a clean DataFrame
    monthly_sales = df.set_index(date_col)[sales_col].resample('MS').sum().reset_index()
    monthly_sales.columns = ['ds', 'y']

    print("✅ Data loaded and prepared successfully.")

except Exception as e:
    print(f"❌ Error during data loading: {e}")

else:
    # --- 2. Feature Engineering ---
    # This function creates time-series features from a date index
    def create_features(df):
        df = df.copy()
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        df['quarter'] = df['ds'].dt.quarter
        # Create lag features (sales from previous months)
        df['lag1'] = df['y'].shift(1)
        df['lag2'] = df['y'].shift(2)
        df['lag3'] = df['y'].shift(3)
        # Create rolling mean feature
        df['rolling_mean_3'] = df['y'].shift(1).rolling(window=3).mean()
        return df

    featured_df = create_features(monthly_sales)
    
    # Drop rows with NaN values created by lag/rolling features
    featured_df = featured_df.dropna()
    
    # --- 3. Model Validation (Train/Test Split) ---
    validation_months = 3
    train = featured_df[:-validation_months]
    test = featured_df[-validation_months:]

    FEATURES = ['month', 'year', 'quarter', 'lag1', 'lag2', 'lag3', 'rolling_mean_3']
    TARGET = 'y'

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # --- 4. Model Training and Accuracy Analysis ---
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01,
        objective='reg:squarederror'
    )
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)

    # Make predictions on the validation set
    predictions = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # --- 5. Iterative Future Forecasting ---
    # To forecast the future, we must predict one step at a time and use that
    # prediction to create the features for the next step.
    future_forecasts = []
    
    # Start with the last known data
    last_known_data = featured_df.tail(validation_months)
    future_df = pd.concat([last_known_data], ignore_index=True)

    for _ in range(3):
        # Get the features for the next prediction
        last_row = future_df.iloc[-1]
        next_date = last_row['ds'] + pd.DateOffset(months=1)
        
        # Prepare the features for the next step
        next_features = {
            'month': next_date.month,
            'year': next_date.year,
            'quarter': next_date.quarter,
            'lag1': last_row['y'],
            'lag2': last_row['lag1'],
            'lag3': last_row['lag2'],
            'rolling_mean_3': (last_row['y'] + last_row['lag1'] + last_row['lag2']) / 3
        }
        
        # Predict the next value
        next_pred = reg.predict(pd.DataFrame([next_features]))[0]
        future_forecasts.append(next_pred)
        
        # Add this new prediction to our dataframe to be used in the next iteration
        new_row = {'ds': next_date, 'y': next_pred}
        # Update lag values in the new row from the previous row
        new_row.update({f'lag{i+1}': last_row[f'lag{i}'] if i > 0 else last_row['y'] for i in range(3)})
        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)

    # Create the forecast DataFrame
    forecast_index = pd.date_range(start=monthly_sales['ds'].iloc[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

    # --- 6. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Sales', color='dodgerblue', linewidth=2)
    ax.plot(test['ds'], predictions, label='Validation Forecast', color='darkblue', linestyle='--', marker='o')
    ax.plot(forecast_index, future_forecasts, label='3-Month Future Forecast', color='red', linestyle='--', marker='o')
    ax.set_title('Sales Forecast vs. Historical Data (XGBoost)', fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales Volume', fontsize=12)
    ax.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # --- 7. Print Results and Findings ---
    print("\n" + "="*50)
    print("📊 MODEL ACCURACY & FORECAST RESULTS 📊")
    print("="*50)
    print("\n## Model Validation Metrics")
    print(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    print(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    print(f"**R-squared ($R^2$):** {r2:.2f}")
    print("\n## 🚀 Future 3-Month Sales Forecast")
    forecast_df = pd.DataFrame({'Date': forecast_index.strftime('%Y-%m'),
                                'Forecasted Sales': [f"${val:,.2f}" for val in future_forecasts]})
    print(forecast_df.to_string(index=False))


    <img width="882" height="582" alt="image" src="https://github.com/user-attachments/assets/6eefb278-fdb3-49ce-ba7e-d2d7bc5957a3" />


====================================
# Random Forest Regressor
====================================


<img width="375" height="231" alt="image" src="https://github.com/user-attachments/assets/d18cd65e-8a5e-4270-ba52-056d973a94ea" />


# Step 1: Fix the Environment (Run this cell, then RESTART KERNEL from the menu)
try:
    # This part will only run once to fix the library issues.
    # On the second run (after kernel restart), it will be skipped.
    from sklearn.ensemble import RandomForestRegressor
    print("✅ Scikit-learn is working correctly.")
    _SCIKIT_LEARN_INSTALLED = True
except ImportError:
    print("⏳ Detected a problem with scikit-learn. Attempting to fix...")
    import os
    # Forcefully reinstall scikit-learn and its key dependencies
    os.system("pip uninstall -y scikit-learn")
    os.system("pip install -U scikit-learn")
    print("\n" + "="*80)
    print("✅ RE-INSTALLATION COMPLETE. PLEASE RESTART THE KERNEL NOW.")
    print("   From the menu above, click: Kernel -> Restart")
    print("   After restarting, run this entire code cell again.")
    print("="*80)
    _SCIKIT_LEARN_INSTALLED = False

# This ensures the rest of the code only runs if the library is correctly installed.
if _SCIKIT_LEARN_INSTALLED:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore")

    # --- 1. Data Loading and Preparation ---
    try:
        file_path = 'dataset_combined_cleaned.csv'
        df = pd.read_csv(file_path)
        date_col, sales_col = 'engage_date', 'close_value'
        df[date_col] = pd.to_datetime(df[date_col])
        monthly_sales = df.set_index(date_col)[sales_col].resample('MS').sum().reset_index()
        monthly_sales.columns = ['ds', 'y']
        print("✅ Data loaded and prepared successfully.")
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
    else:
        # --- 2. Feature Engineering ---
        def create_features(df):
            df = df.copy()
            df['month'] = df['ds'].dt.month
            df['year'] = df['ds'].dt.year
            df['quarter'] = df['ds'].dt.quarter
            df['lag1'] = df['y'].shift(1)
            df['lag2'] = df['y'].shift(2)
            df['lag3'] = df['y'].shift(3)
            df['rolling_mean_3'] = df['y'].shift(1).rolling(window=3).mean()
            return df

        featured_df = create_features(monthly_sales)
        featured_df = featured_df.dropna()

        # --- 3. Model Validation (Train/Test Split) ---
        validation_months = 3
        train = featured_df[:-validation_months]
        test = featured_df[-validation_months:]
        FEATURES = ['month', 'year', 'quarter', 'lag1', 'lag2', 'lag3', 'rolling_mean_3']
        TARGET = 'y'
        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        # --- 4. Model Training and Accuracy Analysis ---
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # --- 5. Iterative Future Forecasting ---
        future_forecasts = []
        last_known_data = featured_df.tail(validation_months)
        future_df = pd.concat([last_known_data], ignore_index=True)

        for _ in range(3):
            last_row = future_df.iloc[-1]
            next_date = last_row['ds'] + pd.DateOffset(months=1)
            next_features = {
                'month': next_date.month, 'year': next_date.year, 'quarter': next_date.quarter,
                'lag1': last_row['y'], 'lag2': last_row['lag1'], 'lag3': last_row['lag2'],
                'rolling_mean_3': (last_row['y'] + last_row['lag1'] + last_row['lag2']) / 3
            }
            next_pred = reg.predict(pd.DataFrame([next_features]))[0]
            future_forecasts.append(next_pred)
            new_row = {'ds': next_date, 'y': next_pred}
            new_row.update({f'lag{i+1}': last_row[f'lag{i}'] if i > 0 else last_row['y'] for i in range(3)})
            future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)

        forecast_index = pd.date_range(start=monthly_sales['ds'].iloc[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

        # --- 6. Visualization ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Sales', color='dodgerblue', linewidth=2)
        ax.plot(test['ds'], predictions, label='Validation Forecast', color='black', linestyle='--', marker='o')
        ax.plot(forecast_index, future_forecasts, label='3-Month Future Forecast', color='red', linestyle='--', marker='o')
        ax.set_title('Sales Forecast vs. Historical Data (Random Forest)', fontsize=18, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales Volume', fontsize=12)
        ax.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        # --- 7. Print Results and Findings ---
        print("\n" + "="*50)
        print("📊 MODEL ACCURACY & FORECAST RESULTS 📊")
        print("="*50)
        print("\n## Model Validation Metrics")
        print(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
        print(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
        print(f"**R-squared ($R^2$):** {r2:.2f}")
        print("\n## 🌳 Future 3-Month Sales Forecast")
        forecast_df = pd.DataFrame({'Date': forecast_index.strftime('%Y-%m'),
                                    'Forecasted Sales': [f"${val:,.2f}" for val in future_forecasts]})
        print(forecast_df.to_string(index=False))


        <img width="888" height="552" alt="image" src="https://github.com/user-attachments/assets/5e5ae7a0-69ee-47b0-918e-b7a14021624e" />



=====================================================
# Performance Model Comparison
# Future 3-Month Sales Forecast Comparison
=====================================================


<img width="602" height="343" alt="image" src="https://github.com/user-attachments/assets/a4e7e715-2ad6-4f21-8446-0bd8f74bb9e5" />

===================================================================================================================================
# Based on the accuracy metrics provided, the XGBoost model is the best performer.
===================================================================================================================================
Why XGBoost Wins:

When evaluating forecasting models, the primary goal is to minimize prediction error. 
XGBoost stands out based on your results:

Lowest Prediction Error: XGBoost has the lowest Mean Absolute Error (MAE) at $449,561.81  and the lowest Root Mean Squared Error (RMSE) at  $513,346.26. This means its predictions during the validation period were, on average, closer to the actual sales figures than any other model.

Best (Least Poor) R-squared: While all models performed poorly on this metric, XGBoost had the R-squared value closest to zero (-1.03), making it technically the best of the group.

Important Warning About Model Reliability
A critical finding from your analysis is that all models have a negative R-squared (R2) value.

This is a major red flag. An R-squared value of 0 means a model is no better than just guessing the average sales value every time. A negative value means the model is performing even worse than that simple average.

The most likely reason for this is that the dataset is too small (only 15 months of data) and potentially too volatile for any model to learn a reliable predictive pattern.

===================================================================================================================================

# Final Recommendation

# While XGBoost is the winner based on the numbers, the negative R-squared values suggest that none of these models can be trusted for reliable forecasting at this time.

# We will continue to use XGBoost to do proof of concept to deploy gradio APP on hugging face interface.

# The best course of action is to focus on collecting more historical data. Recommend At least 24-36 months of sales history, these models can be retrained and will likely produce far more accurate and reliable results.

===================================================================================================================================


===================================================================
XGBoost model machine learning Model Deployment - Proof of Concept:
#Gradio APP on Hugging Face platform#
===================================================================


===================================================================
# Step 1: XGBoost Model Training and App Logic - Proof of Concept:
===================================================================


# Block 1: Model Training and Core Application Logic

# --- 0. Installation ---
# Install necessary libraries for the model and the web application
!pip install -q pandas xgboost scikit-learn gradio

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gradio as gr
import warnings
import time

warnings.filterwarnings("ignore")

print("✅ Libraries installed and imported.")

# --- 1. Data Loading and Feature Engineering ---
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)
    date_col, sales_col = 'engage_date', 'close_value'
    df[date_col] = pd.to_datetime(df[date_col])
    monthly_sales = df.set_index(date_col)[sales_col].resample('MS').sum().reset_index()
    monthly_sales.columns = ['ds', 'y']
    print("✅ Data loaded and prepared successfully.")
except Exception as e:
    print(f"❌ Error during data loading: {e}")

def create_features(df):
    """Creates time-series features from a date index."""
    df = df.copy()
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['quarter'] = df['ds'].dt.quarter
    df['lag1'] = df['y'].shift(1)
    df['lag2'] = df['y'].shift(2)
    df['lag3'] = df['y'].shift(3)
    df['rolling_mean_3'] = df['y'].shift(1).rolling(window=3).mean()
    return df

featured_df = create_features(monthly_sales)
featured_df = featured_df.dropna()

# --- 2. Final Model Training ---
# Train the model on the entire dataset for deployment
FEATURES = ['month', 'year', 'quarter', 'lag1', 'lag2', 'lag3', 'rolling_mean_3']
TARGET = 'y'
X_all, y_all = featured_df[FEATURES], featured_df[TARGET]

# Initialize and train the final XGBoost model
final_regressor = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    objective='reg:squarederror'
)
final_regressor.fit(X_all, y_all, verbose=False)
print("✅ Final XGBoost model trained on all available data.")


# --- 3. Core Forecasting Function for Gradio ---
def forecast_sales(months_to_forecast):
    """
    Takes a number of months, forecasts sales, and returns the results.
    This function will be the core of the Gradio app.
    """
    if not (1 <= months_to_forecast <= 24):
        return None, None, "Please enter a number of months between 1 and 24."

    # Start the timer for performance measurement
    start_time = time.time()

    # Iterative forecasting logic
    future_forecasts = []
    future_df = pd.concat([featured_df], ignore_index=True)

    for _ in range(int(months_to_forecast)):
        last_row = future_df.iloc[-1]
        next_date = last_row['ds'] + pd.DateOffset(months=1)
        next_features_data = {
            'month': next_date.month, 'year': next_date.year, 'quarter': next_date.quarter,
            'lag1': last_row['y'], 'lag2': last_row['lag1'], 'lag3': last_row['lag2'],
            'rolling_mean_3': (last_row['y'] + last_row['lag1'] + last_row['lag2']) / 3
        }
        next_features_df = pd.DataFrame([next_features_data])
        
        next_pred = final_regressor.predict(next_features_df)[0]
        future_forecasts.append(next_pred)
        
        new_row = {'ds': next_date, 'y': next_pred}
        new_row.update({f'lag{i+1}': last_row[f'lag{i}'] if i > 0 else last_row['y'] for i in range(3)})
        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)

    # End timer and calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    # Format the results
    forecast_index = pd.date_range(
        start=monthly_sales['ds'].iloc[-1] + pd.DateOffset(months=1),
        periods=int(months_to_forecast),
        freq='MS'
    )
    
    results_df = pd.DataFrame({
        'Month': forecast_index.strftime('%Y-%m'),
        'Forecasted Sales': [f"${val:,.2f}" for val in future_forecasts]
    })
    
    performance_summary = (
        f"Forecast generated for {int(months_to_forecast)} months.\n"
        f"Processing Time: {processing_time:.4f} seconds."
    )
    
    return results_df, performance_summary, None

print("✅ Core logic for Gradio app is ready.")



<img width="318" height="62" alt="image" src="https://github.com/user-attachments/assets/b72a08f1-96a6-436c-97e1-95cd3aeae4df" />


# Step 2: Deploy the App Locally

# Block 2: Gradio Application Deployment (FIXED)

# Define the user interface for the application
with gr.Blocks(theme=gr.themes.Soft(), title="Sales Forecaster") as demo:
    gr.Markdown(
        """
        # 📈 XGBoost Sales Forecasting App
        Enter the number of months you want to forecast into the future and click 'Generate Forecast'.
        The model uses historical sales data to predict future values.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            months_input = gr.Number(
                label="Months to Forecast",
                value=3,
                minimum=1,
                maximum=24,
                step=1
            )
            submit_button = gr.Button("Generate Forecast", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### Performance Metrics")
            performance_output = gr.Textbox(label="Deployment Performance", interactive=False)

    gr.Markdown("### Forecast Results")
    forecast_output = gr.DataFrame(label="Predicted Sales", interactive=False)
    
    error_output = gr.Textbox(label="Error", interactive=False, visible=False)

    # Connect the button to the forecasting function
    submit_button.click(
        fn=forecast_sales,
        inputs=months_input,
        outputs=[forecast_output, performance_output, error_output]
    )

# Launch the app locally.
# Setting share=False prevents the antivirus error and runs the app on your local machine.
# Open the "Running on local URL" link in your browser to use the app.
demo.launch(share=False, debug=True)


# APP DEPLOYED - TEST SUCCESS #

<img width="866" height="436" alt="image" src="https://github.com/user-attachments/assets/45875f9c-878d-43e9-972b-a1c1cc909231" />


<img width="816" height="417" alt="image" src="https://github.com/user-attachments/assets/4ac8cc0e-f96a-4007-a698-924a9af0580b" />


<img width="859" height="227" alt="image" src="https://github.com/user-attachments/assets/5f5a6830-fe0d-4e9a-bc8c-bb3e24994889" />



========================================================
# Step 3: Visualization of Deployment Performance
========================================================

# --- 0. Installation and Imports---
!pip install -q pandas xgboost scikit-learn matplotlib seaborn
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings("ignore")
print("✅ Libraries installed and imported.")

# --- 1. Data Loading and Model Preparation ---
# This section repeats the necessary setup from the previous steps
try:
    file_path = 'dataset_combined_cleaned.csv'
    df = pd.read_csv(file_path)
    date_col, sales_col = 'engage_date', 'close_value'
    df[date_col] = pd.to_datetime(df[date_col])
    monthly_sales = df.set_index(date_col)[sales_col].resample('MS').sum().reset_index()
    monthly_sales.columns = ['ds', 'y']
except Exception as e:
    print(f"❌ Error: Could not load data. Ensure 'dataset_combined_cleaned.csv' is in the same directory.")
    # Stop execution if data isn't loaded
    raise

def create_features(df):
    df = df.copy()
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['quarter'] = df['ds'].dt.quarter
    df['lag1'] = df['y'].shift(1)
    df['lag2'] = df['y'].shift(2)
    df['lag3'] = df['y'].shift(3)
    df['rolling_mean_3'] = df['y'].shift(1).rolling(window=3).mean()
    return df

featured_df = create_features(monthly_sales)
featured_df = featured_df.dropna()

FEATURES = ['month', 'year', 'quarter', 'lag1', 'lag2', 'lag3', 'rolling_mean_3']
TARGET = 'y'
X_all, y_all = featured_df[FEATURES], featured_df[TARGET]

final_regressor = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, objective='reg:squarederror')
final_regressor.fit(X_all, y_all, verbose=False)

def forecast_sales(months_to_forecast):
    future_forecasts = []
    future_df = pd.concat([featured_df], ignore_index=True)
    for _ in range(int(months_to_forecast)):
        last_row = future_df.iloc[-1]
        next_date = last_row['ds'] + pd.DateOffset(months=1)
        next_features_data = {
            'month': next_date.month, 'year': next_date.year, 'quarter': next_date.quarter,
            'lag1': last_row['y'], 'lag2': last_row['lag1'], 'lag3': last_row['lag2'],
            'rolling_mean_3': (last_row['y'] + last_row['lag1'] + last_row['lag2']) / 3
        }
        next_features_df = pd.DataFrame([next_features_data])
        next_pred = final_regressor.predict(next_features_df)[0]
        future_forecasts.append(next_pred)
        new_row = {'ds': next_date, 'y': next_pred}
        new_row.update({f'lag{i+1}': last_row[f'lag{i}'] if i > 0 else last_row['y'] for i in range(3)})
        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
    forecast_index = pd.date_range(start=monthly_sales['ds'].iloc[-1] + pd.DateOffset(months=1), periods=int(months_to_forecast), freq='MS')
    results_df = pd.DataFrame({'Month': forecast_index.strftime('%Y-%m'), 'Forecasted Sales': [f"${val:,.2f}" for val in future_forecasts]})
    return results_df, None, None

print("✅ Model re-trained and all functions defined.")
print("\nGenerating performance visualizations...")

# --- 2. Historical vs. Forecast Plot ---
forecast_df_viz, _, _ = forecast_sales(12)
forecast_df_viz['Forecasted Sales'] = forecast_df_viz['Forecasted Sales'].replace({'\$': '', ',': ''}, regex=True).astype(float)
forecast_df_viz['Month'] = pd.to_datetime(forecast_df_viz['Month'])

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Sales', color='dodgerblue', linewidth=2, marker='o')
ax.plot(forecast_df_viz['Month'], forecast_df_viz['Forecasted Sales'], label='12-Month Future Forecast', color='red', linestyle='--', marker='o')

ax.set_title('Historical Sales vs. XGBoost Future Forecast', fontsize=18, weight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales Volume ($)', fontsize=12)
ax.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 3. Feature Importance Plot ---
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': final_regressor.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
ax.set_title('XGBoost Model Feature Importance', fontsize=16, weight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


<img width="891" height="610" alt="image" src="https://github.com/user-attachments/assets/1a2d5704-c56d-41aa-827b-f7a54aa3fa1e" />


<img width="875" height="508" alt="image" src="https://github.com/user-attachments/assets/5e867c87-f430-4e14-af5e-420a2e50fd6b" />


=========================================================
# Step 4: Summary Findings and Recommendations
=========================================================

==================================================
FINAL REPORT: XGBOOST SALES FORECASTING APP
==================================================

SUMMARY FINDINGS:

1.  Deployment Success: The XGBoost model was successfully wrapped into an interactive web application using Gradio. The app provides a user-friendly interface for generating sales forecasts for a specified number of future months and reports on its processing time, demonstrating efficient performance.

2.  Model Performance: The core of the model's predictive power comes from its ability to learn from time-based features. The feature importance plot reveals that 'lag1' (the sales from the previous month) is by far the most influential factor. This is logical and expected in sales forecasting, as recent performance is often the best predictor of future performance. Other features like the month and quarter also contribute, capturing seasonal patterns.

3.  Visualization Insights: The "Historical vs. Forecast" chart demonstrates the model's ability to project the learned trend into the future. The forecast continues the general upward or downward trend observed in the most recent historical data, which is a direct result of the high importance of lag features.

RECOMMENDATIONS:

1.  Address Data Limitations: The primary limitation of this model is the small dataset (15 months). This is insufficient to reliably capture yearly seasonality. The model's performance and trustworthiness would increase significantly with more data. It is highly recommended to collect at least 24-36 months of sales data.

2.  Refine Feature Engineering: With more data, more sophisticated features could be engineered. This could include adding holiday flags (e.g., for major local holidays in Penang), creating more complex lag combinations, or including rolling standard deviations to capture volatility.

3.  Monitor Model Performance: A deployed model's accuracy can degrade over time as market conditions change. It is recommended to periodically retrain the model with new sales data (e.g., every quarter) to ensure it remains accurate.

4.  Consider Confidence Intervals: While the current app provides a point forecast, advanced versions could include prediction intervals (an upper and lower bound). This would give users a better sense of the forecast's uncertainty, which is crucial for business planning.

CONCLUSION:

The deployed Gradio application serves as an effective proof-of-concept, demonstrating how a powerful machine learning model like XGBoost can be made accessible for business users. While the current predictions should be used with caution due to data limitations, the framework is solid. The top priority for improving this tool is to expand the historical dataset.




======================================
# END OF REPORT
======================================

        






























