from pathlib import Path
from spotify_recommender.scripts import train


def test_train(example_db_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "train"
    train.__main__(
        [
            "--num-layers=2",
            "--num-features=16",
            "--num-hidden=32",
            str(example_db_path),
            str(output_path),
        ]
    )
