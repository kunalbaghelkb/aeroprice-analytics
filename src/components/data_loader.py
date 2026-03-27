import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype
from src.logger import logging

class VehicleDataLoader:
    """
    Handles memory-efficient loading and downcasting of massive datasets.
    """
    def __init__(self, file_path, chunk_size=100000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def _reduce_memory_usage(self, df):
        """Internal method to compress memory safely."""
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                
                if is_integer_dtype(df[col]):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                elif is_float_dtype(df[col]):
                    df[col] = df[col].astype(np.float32)
            else:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
                    
        end_mem = df.memory_usage().sum() / 1024**2
        logging.info(f"Memory compressed: {start_mem:.2f}MB -> {end_mem:.2f}MB")
        return df

    def load_data(self):
        """Loads data in chunks and concatenates them."""
        logging.info(f"Starting to load data from {self.file_path} in chunks of {self.chunk_size}")
        chunks = []
        try:
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, on_bad_lines='skip', low_memory=False):
                optimized_chunk = self._reduce_memory_usage(chunk)
                chunks.append(optimized_chunk)
            
            df = pd.concat(chunks, axis=0, ignore_index=True)
            logging.info(f"Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e