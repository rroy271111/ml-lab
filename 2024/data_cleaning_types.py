import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
from scipy.stats import zscore
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLDataCleaner:
    """Comprehensive data cleaning toolkit for Machine Learning"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    # ============================================================
    # 1. MISSING DATA HANDLING
    # ============================================================
    
    def handle_missing_deletion(self, threshold=0.5):
        """Remove columns/rows with too many missing values"""
        # Remove columns with more than threshold missing
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_keep = missing_pct[missing_pct < threshold].index
        self.df = self.df[cols_to_keep]
        
        # Remove rows with any missing values
        self.df = self.df.dropna()
        print(f"Removed columns and rows with >{threshold*100}% missing")
        return self.df
    
    def impute_simple(self, strategy='mean', columns=None):
        """Simple imputation: mean, median, mode, constant"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
        else:
            numeric_cols = columns
            categorical_cols = []
        
        # Numeric imputation
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        
        # Categorical imputation (mode)
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
        
        print(f"Applied {strategy} imputation")
        return self.df
    
    def impute_knn(self, n_neighbors=5, columns=None):
        """KNN imputation for missing values"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        print(f"Applied KNN imputation with {n_neighbors} neighbors")
        return self.df
    
    def impute_iterative(self, columns=None):
        """MICE (Multivariate Imputation by Chained Equations)"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        imputer = IterativeImputer(random_state=42, max_iter=10)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        print("Applied iterative (MICE) imputation")
        return self.df
    
    def forward_backward_fill(self, columns=None, method='ffill'):
        """Forward or backward fill for time series"""
        if columns is None:
            columns = self.df.columns
        
        self.df[columns] = self.df[columns].fillna(method=method)
        print(f"Applied {method} imputation")
        return self.df
    
    # ============================================================
    # 2. DUPLICATE DATA REMOVAL
    # ============================================================
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove exact duplicates"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        print(f"Removed {before - after} duplicate rows")
        return self.df
    
    def remove_fuzzy_duplicates(self, columns, threshold=0.8):
        """Remove fuzzy/similar duplicates using string similarity"""
        from difflib import SequenceMatcher
        
        def similar(a, b):
            return SequenceMatcher(None, str(a), str(b)).ratio()
        
        to_drop = []
        for i in range(len(self.df)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(self.df)):
                if j in to_drop:
                    continue
                similarity = np.mean([similar(self.df.iloc[i][col], self.df.iloc[j][col]) 
                                     for col in columns])
                if similarity >= threshold:
                    to_drop.append(j)
        
        self.df = self.df.drop(self.df.index[to_drop])
        print(f"Removed {len(to_drop)} fuzzy duplicates")
        return self.df
    
    # ============================================================
    # 3. OUTLIER DETECTION AND TREATMENT
    # ============================================================
    
    def detect_outliers_zscore(self, columns=None, threshold=3):
        """Detect outliers using Z-score method"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame()
        for col in columns:
            z_scores = np.abs(zscore(self.df[col].dropna()))
            outliers[col] = z_scores > threshold
        
        return outliers
    
    def detect_outliers_iqr(self, columns=None):
        """Detect outliers using IQR method"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame(index=self.df.index)
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        
        return outliers
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        """Remove outliers using Z-score"""
        outliers = self.detect_outliers_zscore(columns, threshold)
        before = len(self.df)
        self.df = self.df[~outliers.any(axis=1)]
        after = len(self.df)
        print(f"Removed {before - after} outlier rows using Z-score")
        return self.df
    
    def cap_outliers_iqr(self, columns=None):
        """Cap outliers using IQR (Winsorization)"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Capped outliers for {len(columns)} columns using IQR")
        return self.df
    
    def detect_outliers_isolation_forest(self, columns=None, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(self.df[columns])
        return outliers == -1
    
    # ============================================================
    # 4. DATA TYPE CORRECTION
    # ============================================================
    
    def convert_to_numeric(self, columns, errors='coerce'):
        """Convert columns to numeric type"""
        for col in columns:
            self.df[col] = pd.to_numeric(self.df[col], errors=errors)
        print(f"Converted {len(columns)} columns to numeric")
        return self.df
    
    def convert_to_datetime(self, columns, format=None, errors='coerce'):
        """Convert columns to datetime"""
        for col in columns:
            self.df[col] = pd.to_datetime(self.df[col], format=format, errors=errors)
        print(f"Converted {len(columns)} columns to datetime")
        return self.df
    
    def convert_to_categorical(self, columns):
        """Convert columns to categorical type"""
        for col in columns:
            self.df[col] = self.df[col].astype('category')
        print(f"Converted {len(columns)} columns to categorical")
        return self.df
    
    # ============================================================
    # 5. INCONSISTENCY RESOLUTION
    # ============================================================
    
    def standardize_case(self, columns, case='lower'):
        """Standardize text case"""
        for col in columns:
            if case == 'lower':
                self.df[col] = self.df[col].str.lower()
            elif case == 'upper':
                self.df[col] = self.df[col].str.upper()
            elif case == 'title':
                self.df[col] = self.df[col].str.title()
        print(f"Standardized case to {case} for {len(columns)} columns")
        return self.df
    
    def replace_values(self, column, mapping_dict):
        """Replace values using mapping dictionary"""
        self.df[column] = self.df[column].replace(mapping_dict)
        print(f"Replaced values in {column}")
        return self.df
    
    def standardize_categories(self, column, mapping_dict):
        """Merge similar categories"""
        self.df[column] = self.df[column].map(lambda x: mapping_dict.get(x, x))
        print(f"Standardized categories in {column}")
        return self.df
    
    # ============================================================
    # 6. STRUCTURAL ERRORS
    # ============================================================
    
    def strip_whitespace(self, columns=None):
        """Remove leading/trailing whitespace"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        for col in columns:
            self.df[col] = self.df[col].str.strip()
        print(f"Stripped whitespace from {len(columns)} columns")
        return self.df
    
    def remove_special_characters(self, columns, pattern=r'[^a-zA-Z0-9\s]'):
        """Remove special characters"""
        for col in columns:
            self.df[col] = self.df[col].str.replace(pattern, '', regex=True)
        print(f"Removed special characters from {len(columns)} columns")
        return self.df
    
    def standardize_column_names(self):
        """Standardize column names (lowercase, underscores)"""
        self.df.columns = (self.df.columns
                          .str.lower()
                          .str.replace(' ', '_')
                          .str.replace('[^a-zA-Z0-9_]', '', regex=True))
        print("Standardized column names")
        return self.df
    
    # ============================================================
    # 7. RANGE AND CONSTRAINT VIOLATIONS
    # ============================================================
    
    def clip_values(self, column, lower=None, upper=None):
        """Clip values to specified range"""
        self.df[column] = self.df[column].clip(lower=lower, upper=upper)
        print(f"Clipped values in {column} to range [{lower}, {upper}]")
        return self.df
    
    def remove_invalid_ranges(self, column, min_val=None, max_val=None):
        """Remove rows with values outside valid range"""
        before = len(self.df)
        if min_val is not None:
            self.df = self.df[self.df[column] >= min_val]
        if max_val is not None:
            self.df = self.df[self.df[column] <= max_val]
        after = len(self.df)
        print(f"Removed {before - after} rows with invalid range in {column}")
        return self.df
    
    def validate_date_logic(self, start_col, end_col):
        """Ensure end date is after start date"""
        before = len(self.df)
        self.df = self.df[self.df[end_col] >= self.df[start_col]]
        after = len(self.df)
        print(f"Removed {before - after} rows with invalid date logic")
        return self.df
    
    # ============================================================
    # 8. CATEGORICAL DATA CLEANING
    # ============================================================
    
    def merge_rare_categories(self, column, threshold=0.01, new_label='Other'):
        """Merge rare categories below threshold"""
        value_counts = self.df[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        self.df[column] = self.df[column].replace(rare_categories, new_label)
        print(f"Merged {len(rare_categories)} rare categories in {column}")
        return self.df
    
    def handle_unknown_categories(self, column, unknown_labels=['?', 'Unknown', 'N/A', 'NA']):
        """Standardize unknown category labels"""
        self.df[column] = self.df[column].replace(unknown_labels, 'Unknown')
        print(f"Standardized unknown categories in {column}")
        return self.df
    
    # ============================================================
    # 9. TEXT DATA CLEANING
    # ============================================================
    
    def remove_html_tags(self, columns):
        """Remove HTML/XML tags"""
        for col in columns:
            self.df[col] = self.df[col].str.replace(r'<[^>]+>', '', regex=True)
        print(f"Removed HTML tags from {len(columns)} columns")
        return self.df
    
    def remove_urls(self, columns):
        """Remove URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for col in columns:
            self.df[col] = self.df[col].str.replace(url_pattern, '', regex=True)
        print(f"Removed URLs from {len(columns)} columns")
        return self.df
    
    def remove_emails(self, columns):
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for col in columns:
            self.df[col] = self.df[col].str.replace(email_pattern, '', regex=True)
        print(f"Removed emails from {len(columns)} columns")
        return self.df
    
    def remove_punctuation(self, columns):
        """Remove punctuation"""
        for col in columns:
            self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
        print(f"Removed punctuation from {len(columns)} columns")
        return self.df
    
    def remove_numbers(self, columns):
        """Remove numbers from text"""
        for col in columns:
            self.df[col] = self.df[col].str.replace(r'\d+', '', regex=True)
        print(f"Removed numbers from {len(columns)} columns")
        return self.df
    
    # ============================================================
    # 10. FEATURE SCALING
    # ============================================================
    
    def standard_scale(self, columns=None):
        """Standardize features (mean=0, std=1)"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        print(f"Applied standard scaling to {len(columns)} columns")
        return self.df
    
    def minmax_scale(self, columns=None, feature_range=(0, 1)):
        """Scale features to a range"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler(feature_range=feature_range)
        self.df[columns] = scaler.fit_transform(self.df[columns])
        print(f"Applied min-max scaling to {len(columns)} columns")
        return self.df
    
    def robust_scale(self, columns=None):
        """Robust scaling using median and IQR"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        scaler = RobustScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        print(f"Applied robust scaling to {len(columns)} columns")
        return self.df
    
    def log_transform(self, columns):
        """Log transformation for skewed data"""
        for col in columns:
            self.df[col] = np.log1p(self.df[col])
        print(f"Applied log transformation to {len(columns)} columns")
        return self.df
    
    # ============================================================
    # 11. TEMPORAL DATA CLEANING
    # ============================================================
    
    def sort_by_time(self, time_column):
        """Sort dataframe by time"""
        self.df = self.df.sort_values(by=time_column)
        print(f"Sorted by {time_column}")
        return self.df
    
    def extract_time_features(self, date_column):
        """Extract time-based features"""
        self.df[f'{date_column}_year'] = self.df[date_column].dt.year
        self.df[f'{date_column}_month'] = self.df[date_column].dt.month
        self.df[f'{date_column}_day'] = self.df[date_column].dt.day
        self.df[f'{date_column}_dayofweek'] = self.df[date_column].dt.dayofweek
        self.df[f'{date_column}_quarter'] = self.df[date_column].dt.quarter
        print(f"Extracted time features from {date_column}")
        return self.df
    
    # ============================================================
    # 12. DATA CARDINALITY ISSUES
    # ============================================================
    
    def remove_constant_columns(self):
        """Remove columns with zero variance"""
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        self.df = self.df.drop(columns=constant_cols)
        print(f"Removed {len(constant_cols)} constant columns")
        return self.df
    
    def remove_high_correlation(self, threshold=0.95):
        """Remove highly correlated features"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.df = self.df.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} highly correlated columns")
        return self.df
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def get_summary(self):
        """Get summary of data cleaning"""
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Original shape: {self.original_shape}")
        print(f"Current shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Columns removed: {self.original_shape[1] - self.df.shape[1]}")
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        print("\nData types:")
        print(self.df.dtypes)
        print("="*60)
        
    def get_data(self):
        """Return cleaned dataframe"""
        return self.df


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Create sample messy data
    np.random.seed(42)
    sample_data = {
        'id': range(1, 101),
        'name': ['John Doe', 'JANE SMITH', 'jane smith', 'Bob Johnson', 'Alice Davis'] * 20,
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.randint(30000, 150000, 100),
        'email': ['john@email.com', 'JANE@EMAIL.COM', 'bob@test.com', None, 'alice@email.com'] * 20,
        'category': ['A', 'B', 'C', 'D', 'E'] * 20,
        'date_joined': pd.date_range('2020-01-01', periods=100, freq='D')
    }
    
    # Add some missing values
    df = pd.DataFrame(sample_data)
    df.loc[10:15, 'age'] = np.nan
    df.loc[20:25, 'salary'] = np.nan
    
    # Add outliers
    df.loc[5, 'salary'] = 1000000
    df.loc[15, 'age'] = 150
    
    print("Original Data:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    
    # Initialize cleaner
    cleaner = MLDataCleaner(df)
    
    # Apply various cleaning operations
    print("\n" + "="*60)
    print("APPLYING DATA CLEANING OPERATIONS")
    print("="*60 + "\n")
    
    # 1. Handle missing data
    cleaner.impute_simple(strategy='median', columns=['age', 'salary'])
    
    # 2. Remove duplicates
    cleaner.remove_duplicates(subset=['name'])
    
    # 3. Standardize text
    cleaner.standardize_case(['name', 'email'], case='lower')
    cleaner.strip_whitespace()
    
    # 4. Handle outliers
    cleaner.cap_outliers_iqr(columns=['age', 'salary'])
    
    # 5. Remove constant columns (if any)
    cleaner.remove_constant_columns()
    
    # Get cleaned data
    cleaned_df = cleaner.get_data()
    
    print("\nCleaned Data:")
    print(cleaned_df.head(10))
    
    # Get summary
    cleaner.get_summary()