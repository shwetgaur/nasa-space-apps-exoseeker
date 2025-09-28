# process_data.py
import pandas as pd

def clean_data(input_path='data/koi_data.csv', output_path='data/cleaned_koi_data.csv'):
    """Loads, cleans, and saves the Kepler exoplanet data."""

    # Load the dataset
    df = pd.read_csv(input_path, comment='#') # Automatically skip all comment lines
    print("Dataset loaded successfully.")

    # --- Data Cleaning and Feature Selection ---
    # 1. Select relevant columns. We're choosing a mix of transit and stellar properties.
    #    This is a crucial step; these features will determine your model's success.
    features = [
        'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
        'koi_fpflag_ec', 'koi_period', 'koi_time0bk', 'koi_impact', 
        'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
        'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad'
    ]
    df_clean = df[features].copy()

    # 2. Handle missing values. We'll fill them with the median of each column.
    #    Median is often better than mean for skewed data, which is common in astronomy.
    for col in df_clean.columns:
        if df_clean[col].dtype == 'float64':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    print("Missing values handled.")

    # 3. Create the target variable. We'll simplify this to a binary problem:
    #    1 = Exoplanet (CONFIRMED or CANDIDATE), 0 = Not Exoplanet (FALSE POSITIVE)
    df_clean['is_exoplanet'] = df_clean['koi_disposition'].apply(
        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
    )

    # We can now drop the original disposition column
    df_clean.drop('koi_disposition', axis=1, inplace=True)

    # Save the cleaned data to a new file
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print("\nData processing complete. Final data shape:", df_clean.shape)

if __name__ == '__main__':
    clean_data()