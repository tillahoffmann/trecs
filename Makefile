.PHONY : experiments tests

tests :
	pytest tests --cov=trecs --cov-report=term-missing -v

# Data processing.

data/spotify_million_playlist_dataset.zip :
	$(info Download the Million Playlist Dataset from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge and place it at 'data/spotify_million_playlist_dataset.zip'.)
	false

data/spotify_million_playlist_dataset/md5sums : data/spotify_million_playlist_dataset.zip
	unzip -d data/spotify_million_playlist_dataset $<
	touch $@

data/spotify_million_playlist_dataset/md5sums.check : data/spotify_million_playlist_dataset/md5sums
	# Write to temp file and then move so we don't end up with a file despite the check failing.
	cd data/spotify_million_playlist_dataset && md5sum --check md5sums > $(notdir $@).tmp
	mv $@.tmp $@

data/mpd.db : data/spotify_million_playlist_dataset/md5sums.check
	# Build the database.
	python -m trecs.scripts.build_db data/mpd.db data/spotify_million_playlist_dataset/data/mpd.slice.*.json

# Training.

EXPERIMENT_SETUPS = $(filter-out $(wildcard src/trecs/experiments/*/_*.py),$(wildcard src/trecs/experiments/*/*.py))
EXPERIMENT_OUTPUTS = $(addprefix workspace/,${EXPERIMENT_SETUPS:src/trecs/experiments/%.py=%})
$(info ${EXPERIMENT_OUTPUTS})

experiments : ${EXPERIMENT_OUTPUTS}

${EXPERIMENT_OUTPUTS} : workspace/% : src/trecs/experiments/%.py data/mpd.db
	MPD=data/mpd.db python -m trecs.scripts.train $@ $<
