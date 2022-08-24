import pandas as pd

def sqi_smoothen(
    sqi: pd.Series,
    fs: int,
    window_s: int=5,
    min_ok_ratio=0.5,
    operation='and',
    output_name="SQI_smoothened",
) -> pd.Series:
    w_size = int(window_s * fs)
    ok_sum = sqi.rolling(w_size + (w_size % 2 - 1), center=True).sum()

    if operation == 'and':
        out = (sqi & ((ok_sum / w_size) >= min_ok_ratio)).rename(output_name)
    elif operation == 'or':
        out = (sqi | ((ok_sum / w_size) >= min_ok_ratio)).rename(output_name)

    return out
