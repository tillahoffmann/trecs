from trecs.experiments.decoder_only import DecoderOnlyExperiment


def setup():
    experiment = DecoderOnlyExperiment()
    experiment.num_steps = 7
    experiment.batch_size = 8
    experiment.learning_rate = 0.0005
    experiment.seed = 17
    experiment.num_layers = 3
    experiment.num_features = 16
    experiment.num_hidden = 32
    experiment.context_length = 13
    experiment.unk_proba = 0.05
    experiment.num_tracks = None
    return experiment
