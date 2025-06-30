import pandas as pd


def calculate_heat_index(temp_F, humidity):
    """Calculate NOAA/NWS standard heat index"""
    HI = (-42.379 + 2.04901523*temp_F + 10.14333127*humidity 
          - 0.22475541*temp_F*humidity - 0.00683783*temp_F**2 
          - 0.05481717*humidity**2 + 0.00122874*temp_F**2*humidity 
          + 0.00085282*temp_F*humidity**2 - 0.00000199*temp_F**2*humidity**2)
    return HI

def fertilizer_feature_engineering(df):
    #NPK Ratio Features

    # NPK Ratios
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)


    # Soil Fertility Index

    df['fertility_index'] = (
    0.45 * (df['Nitrogen'] / df['Nitrogen'].max()) +
    0.35 * (df['Phosphorous'] / df['Phosphorous'].max()) +
    0.2 * (df['Potassium'] / df['Potassium'].max())
    )

    # Heat Index

    df['temp_F'] = df['Temparature'] * 9/5 + 32
    df['heat_index'] = df.apply(lambda x: calculate_heat_index(x['temp_F'], x['Humidity']), axis=1)

    return df