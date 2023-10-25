"""
filter_and_save_data.py

This script contains a function for transforming the raw flight satisfaction dataset.
It filters out unnecessary columns and saves the resulting DataFrame to a new CSV file.

Functions:
- filter_and_save_csv(input_path: str, output_path: str) -> dict: Takes in the path to the raw CSV file and the path where the filtered CSV will be saved. Returns a dictionary containing details about the saved file.

Usage:
from filter_and_save_data import filter_and_save_csv

result = filter_and_save_csv('path/to/raw_data.csv', 'path/to/save/filtered_data.csv')
"""

import os
import pandas as pd


def filter_and_save_csv(input_path: str, output_path: str) -> dict:
    try:
        df = pd.read_csv(input_path)

        columns_to_keep = [
            'satisfaction',
            'Customer Type',
            'Age',
            'Type of Travel',
            'Class',
            'Flight Distance',
            'Ease of Online booking',
            'Online boarding'
        ]

        df = df[columns_to_keep]
        df.to_csv(output_path, index=False)

        full_output_path = os.path.abspath(output_path)

        file_info = {
            'status': 'File saved successfully',
            'path': full_output_path,
            'file_size': os.path.getsize(full_output_path),
        }

        return file_info

    except Exception as e:
        return {
            'status': 'Failed to save file',
            'error': str(e)
        }
