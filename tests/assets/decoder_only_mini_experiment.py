from trecs.experiments.decoder_only import DecoderOnlyExperiment


def setup():
    return DecoderOnlyExperiment(
        num_steps=7,
        batch_size=8,
        learning_rate=0.0005,
        seed=17,
        eval_every=1000,
        checkpoint_every=1000,
        context_length=14,
        num_layers=3,
        num_heads=8,
        num_features=16,
        num_hidden=32,
        dropout=0.1,
        unk_proba=0.05,
        weight_decay=0.01,
        num_tracks=None,
        loss_function="uniform",
    )
