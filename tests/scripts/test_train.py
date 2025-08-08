import numpy
import pandas as pd
from pathlib import Path
from spotify_recommender.scripts import train
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


def test_train(example_db_path: Path, tmp_path: Path) -> None:
    # Simple end-to-end run.
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


def test_train_resume(example_db_path: Path, tmp_path: Path) -> None:
    # Run once end to end.
    output_path1 = tmp_path / "train1"
    train.__main__(
        [
            "--num-layers=2",
            "--num-features=16",
            "--num-hidden=32",
            "--valid-every=2",
            "--unk-proba=0.05",
            "--num-steps=7",
            str(example_db_path),
            str(output_path1),
        ]
    )

    # Run in two parts.
    output_path2 = tmp_path / "train2"
    train.__main__(
        [
            "--num-layers=2",
            "--num-features=16",
            "--num-hidden=32",
            "--valid-every=2",
            "--unk-proba=0.05",
            "--num-steps=3",
            str(example_db_path),
            str(output_path2),
        ]
    )
    train.__main__(
        [
            "--num-layers=2",
            "--num-features=16",
            "--num-hidden=32",
            "--valid-every=2",
            "--unk-proba=0.05",
            "--num-steps=7",
            "--resume",
            str(example_db_path),
            str(output_path2),
        ]
    )

    # We could do an exhaustive comparison here, but let's just load the training and
    # validation losses and compare them.
    paths = [output_path1, output_path2]
    events_by_run = []
    for path in paths:
        events = []
        for filename in (path / "logdir").glob("*.tfevents*"):
            stream = EventFileLoader(str(filename)).Load()
            for event in stream:
                for (
                    summary
                ) in event.summary.value:  # pyright: ignore[reportAttributeAccessIssue]
                    events.append(
                        {
                            "tag": summary.tag,
                            "value": summary.tensor.float_val[0],
                            "step": event.step,  # pyright: ignore[reportAttributeAccessIssue]
                        }
                    )
        events_by_run.append(pd.DataFrame(events))
    merged = pd.merge(*events_by_run, on=["step", "tag"], how="outer")
    numpy.testing.assert_allclose(merged.value_x, merged.value_y)
