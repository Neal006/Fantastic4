"""
Run the preprocessing pipeline on all raw CSVs and save to data_cleaned/.
"""
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing.preprocess import preprocess_pipeline
from config.settings import DATA_RAW_DIR, DATA_CLEANED_DIR, RAW_FILES, PLANT_FILES

def main():
    os.makedirs(DATA_CLEANED_DIR, exist_ok=True)
    
    print(f"Starting bulk preprocessing...")
    print(f"Reading from: {DATA_RAW_DIR}")
    print(f"Writing to:   {DATA_CLEANED_DIR}\n")
    
    for plant_id, raw_rel_path in RAW_FILES.items():
        raw_path = os.path.join(DATA_RAW_DIR, raw_rel_path)
        clean_name = PLANT_FILES[plant_id]
        clean_path = os.path.join(DATA_CLEANED_DIR, clean_name)
        
        if not os.path.exists(raw_path):
            print(f"⚠️ SKIPPING: Raw file not found: {raw_path}")
            continue
            
        print(f"Processing {plant_id}...")
        try:
            # The preprocessing pipeline sorts by timestamp and sets it as the index
            df = preprocess_pipeline(raw_path)
            
            # Save to CSV
            df.to_csv(clean_path)
            print(f"✓ Saved cleaned data to {clean_path}\n")
        except Exception as e:
            print(f"❌ ERROR processing {plant_id}: {e}\n")

if __name__ == "__main__":
    main()
