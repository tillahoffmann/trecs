from trecs.experiments.decoder_only import DecoderOnlyExperiment


def setup() -> DecoderOnlyExperiment:
    train_size = 800_000
    batch_size = 16
    num_steps = train_size // batch_size
    return DecoderOnlyExperiment(
        seed=42,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=0.0005,
        weight_decay=0.01,
        context_length=50,
        num_layers=6,
        num_features=128,
        num_hidden=256,
        num_heads=8,
        dropout=0.1,
        eval_every=100,
        checkpoint_every=1000,
        loss_function="uniform",
        unk_proba=0.01,
        # This will be determined by the encoder.
        num_tracks=None,
    )
