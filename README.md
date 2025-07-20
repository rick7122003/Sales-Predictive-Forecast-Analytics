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













