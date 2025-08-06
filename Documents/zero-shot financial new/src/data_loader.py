import pandas as pd
import os

def load_headlines_from_csv(filepath: str, text_column: str = "Headlines", n: int = None):
    
    df = pd.read_csv(filepath)

    headlines = df[text_column].dropna().astype(str).tolist()

    if n:
        headlines = headlines[:n]

    return headlines

