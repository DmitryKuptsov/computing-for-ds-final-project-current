import pandas as pd

from data_loader import DataLoader


def test_data_loader_sampling_is_deterministic(tmp_path):
    # Create a small CSV on the fly
    df = pd.DataFrame(
        {
            "SK_ID_CURR": list(range(1, 101)),
            "X": list(range(100)),
        }
    )
    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)

    loader1 = DataLoader(path=str(csv_path), sample_size=10)
    loader2 = DataLoader(path=str(csv_path), sample_size=10)

    s1 = loader1.load()
    s2 = loader2.load()

    # Should pick the same rows because random_state is fixed in DataLoader
    pd.testing.assert_frame_equal(
        s1.sort_values("SK_ID_CURR").reset_index(drop=True),
        s2.sort_values("SK_ID_CURR").reset_index(drop=True),
    )
