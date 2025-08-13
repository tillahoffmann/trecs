import collectiontools
import numpy
from flax import nnx
import optax
import jax
from jax import numpy as jnp
import numpy
import pandas as pd
from pathlib import Path
from trecs.scripts import train
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from unittest.mock import patch
import pytest


decoder_only_mini_experiment_path = str(
    Path(__file__).parent.parent / "assets/decoder_only_mini_experiment.py"
)


@pytest.mark.parametrize("dry_run", [False, True])
def test_train(example_db_path: Path, tmp_path: Path, dry_run: bool) -> None:
    with patch.dict("os.environ", MPD=str(example_db_path)):
        # Simple end-to-end run.
        output_path = tmp_path / "train"
        argv = [
            str(output_path),
            decoder_only_mini_experiment_path,
        ]
        if dry_run:
            argv.append("--dry-run")
        train.__main__(argv)


def test_train_resume(example_db_path: Path, tmp_path: Path) -> None:
    with patch.dict("os.environ", MPD=str(example_db_path)):
        # Run once end to end.
        output_path1 = tmp_path / "train1"
        train.__main__(
            [
                "--num-steps=7",
                "--eval-every=2",
                str(output_path1),
                decoder_only_mini_experiment_path,
            ]
        )

        # Run in two parts.
        output_path2 = tmp_path / "train2"
        train.__main__(
            [
                "--num-steps=4",
                "--eval-every=2",
                str(output_path2),
                decoder_only_mini_experiment_path,
            ]
        )
        train.__main__(
            [
                "--num-steps=7",
                "--eval-every=2",
                "--resume",
                str(output_path2),
                decoder_only_mini_experiment_path,
            ]
        )

    # We could do an exhaustive comparison here, but let's just load the training and
    # validation losses and compare them.
    paths = [output_path1, output_path2]
    events_by_run = []
    for path in paths:
        events = []
        for filename in (path / "logdir").glob("**/*.tfevents*"):
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
                            "split": filename.parent.stem,
                        }
                    )
        events_by_run.append(pd.DataFrame(events))
    merged = pd.merge(*events_by_run, on=["step", "tag", "split"], how="outer")
    numpy.testing.assert_allclose(merged.value_x, merged.value_y)


@pytest.mark.parametrize("jit", [False, True])
def test_as_train_step(jit: bool) -> None:
    ok_batch = jnp.arange(10).reshape((5, 2)).astype(float)
    bad_batch = ok_batch.at[0, 0].set(jnp.nan)
    labels = jnp.arange(15).reshape((5, 3))

    def loss_fn(model, inputs, labels, key):
        pred = model(inputs)
        return jnp.mean(jnp.square(pred - labels))

    unsafe_train_step = train.as_train_step(loss_fn)
    safe_train_step = train.as_train_step(loss_fn, safe_update=True)
    if jit:
        unsafe_train_step = nnx.jit(unsafe_train_step)
        safe_train_step = nnx.jit(safe_train_step)

    ref_model = model = nnx.Linear(2, 3, rngs=nnx.Rngs(7))
    ref_state = nnx.state(ref_model)

    configs = {
        "ok-safe": (ok_batch, safe_train_step),
        "bad-safe": (bad_batch, safe_train_step),
        "ok-unsafe": (ok_batch, unsafe_train_step),
        "bad-unsafe": (bad_batch, unsafe_train_step),
    }
    models = {}
    losses = {}

    for key, (batch, train_step) in configs.items():
        rngs = nnx.Rngs(7)
        model = nnx.Linear(2, 3, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adamw(0.001), wrt=nnx.Param)
        loss = train_step(model, optimizer, batch, labels, rngs())
        models[key] = model
        losses[key] = loss

    assert jnp.isfinite(losses["ok-safe"])
    assert jnp.allclose(losses["ok-safe"], losses["ok-unsafe"])
    assert jnp.isnan(losses["bad-unsafe"])
    assert jnp.isnan(losses["bad-safe"])

    states = collectiontools.map_values(nnx.state, models)
    all_finite = {
        key: jax.tree.map(lambda x: jnp.isfinite(x).all(), value)
        for key, value in states.items()
    }
    all_finite = {
        key: jax.tree.reduce(lambda x, y: x and y, value, True)
        for key, value in all_finite.items()
    }
    assert all_finite == {
        "ok-safe": True,
        "ok-unsafe": True,
        "bad-safe": True,
        "bad-unsafe": False,
    }
    assert (
        optax.tree_utils.tree_norm(
            optax.tree_utils.tree_sub(states["ok-safe"], states["ok-unsafe"])
        )
        < 1e-6
    )
    assert (
        optax.tree_utils.tree_norm(
            optax.tree_utils.tree_sub(states["ok-safe"], ref_state)
        )
        > 1e-3
    )
    assert (
        optax.tree_utils.tree_norm(
            optax.tree_utils.tree_sub(states["bad-safe"], ref_state)
        )
        < 1e-6
    )


def test_data_loader_regression(example_db_path: Path, tmp_path: Path) -> None:
    # This test checks that the output of the data iterator conforms to a specific
    # output as a regression test.
    experiment = train.load_experiment_from_file(
        Path(decoder_only_mini_experiment_path)
    )
    with patch.dict("os.environ", MPD=str(example_db_path)):
        experiment.setup_output(tmp_path)
        data_source = experiment.create_data_source("train")
        data_loader = experiment.create_data_loader("train", data_source, 17)

    inputs, labels = next(iter(data_loader))

    expected_pos = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0],
    ]
    numpy.testing.assert_array_equal(inputs["pos"], expected_pos)

    expected_track_id = [
        [0, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 2, 3845, 3846],
        [0, 2869, 56, 1398, 2870, 2871, 2872, 2873, 2874, 2875, 1124, 2876, 2877, 2878],
        [0, 2228, 2229, 2230, 2231, 2232, 2233, 2, 2235, 2236, 2237, 2238, 1, 1],
        [0, 3652, 1182, 2761, 1130, 2384, 3653, 3654, 211, 615, 3655, 1687, 3656, 1715],
        [0, 283, 284, 285, 286, 287, 288, 289, 290, 291, 2, 293, 294, 295],
        [
            0,
            342,
            4135,
            4136,
            4137,
            4138,
            4139,
            3658,
            4140,
            4141,
            4142,
            3734,
            4143,
            4144,
        ],
        [0, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311],
        [0, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 2, 802, 1110, 1],
    ]
    numpy.testing.assert_array_equal(inputs["track_id"], expected_track_id)
