import os

# FILE PATHS
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'vehicles.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# DATA CLEANING CONSTANTS
LOWER_PRICE_BOUND = 500
UPPER_PRICE_BOUND = 150000

COLS_TO_DROP_INITIAL = ['county', 'id', 'url', 'region_url', 'VIN', 'image_url', 'description']
COLS_TO_DROP_FINAL = ['lat', 'long', 'posting_date']

# IMPUTATION CONSTANTS
CATEGORICAL_MISSING_COLS = ['size', 'condition', 'cylinders', 'drive', 'paint_color', 'type']
MINOR_CATS = ['fuel', 'title_status', 'transmission']

# ENCODING CONSTANTS & MAPPINGS
COND_MAP = {'unknown': 0, 'salvage': 1, 'fair': 2, 'good': 3, 'excellent': 4, 'like new': 5, 'new': 6}
CYL_MAP = {
    'unknown': 0, 'other': 0, '3 cylinders': 3, '4 cylinders': 4, 
    '5 cylinders': 5, '6 cylinders': 6, '8 cylinders': 8, 
    '10 cylinders': 10, '12 cylinders': 12
}
SIZE_MAP = {'unknown': 0, 'sub-compact': 1, 'compact': 2, 'mid-size': 3, 'full-size': 4}

HIGH_CARD_COLS = ['manufacturer', 'state', 'region', 'model']
OHE_COLS = ['fuel', 'title_status', 'transmission', 'drive', 'paint_color', 'type']

# Feature order saved from training (helps in avoiding feature mismatch errors in production)
TRAINING_COLUMNS_PATH = os.path.join(MODEL_DIR, 'training_columns.pkl')