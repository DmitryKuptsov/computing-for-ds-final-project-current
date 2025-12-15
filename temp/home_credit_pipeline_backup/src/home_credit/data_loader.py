from typing import Optional
import pandas as pd


class DataLoader:
    """Loads CSV dataset and optionally samples rows."""

    def __init__(self, path: str, sample_size: Optional[int] = None):
        self.path = path
        self.sample_size = sample_size

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        if self.sample_size:
            df = df.sample(self.sample_size, random_state=42)
        return df
