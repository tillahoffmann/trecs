PRAGMA foreign_keys = ON;

CREATE TABLE albums (
    id INTEGER PRIMARY KEY,
    uri TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL
);

CREATE TABLE artists (
    id INTEGER PRIMARY KEY,
    uri TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL
);

/* Tracks which could include audio features from
https://developer.spotify.com/documentation/web-api/reference/get-audio-features. The
API has been deprecated, but there are a number of scraped sources online, including

1. https://github.com/rezaakb/spotify-recommender
2. https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks
3. https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
4. https://www.kaggle.com/datasets/tomigelo/spotify-audio-features
5. https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21
6. https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
*/
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    uri TEXT NOT NULL UNIQUE,
    album_id NOT NULL REFERENCES albums(id),
    artist_id NOT NULL REFERENCES artists(id),
    name TEXT NOT NULL,
    duration_ms INTEGER NOT NULL
    /*
    acousticness REAL,
    danceability REAL,
    energy REAL,
    explicit BOOLEAN,
    instrumentalness REAL,
    key INTEGER,
    liveness REAL,
    loudness REAL,
    mode INTEGER,
    speechiness REAL,
    tempo REAL,
    time_signature REAL,
    valence REAL
    */
);

CREATE TABLE playlists (
    -- We use the explicit `pid` from the MPD dataset.
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    collaborative BOOLEAN NOT NULL,
    modified_at INTEGER NOT NULL,
    num_followers INTEGER NOT NULL,
    num_edits INTEGER NOT NULL,
    -- Placeholder field for synthetic descriptions used to train the cross-attention layers.
    synthetic_description TEXT
);

-- Which tracks belong to which playlist.
CREATE TABLE playlist_track_memberships (
    id INTEGER PRIMARY KEY,
    playlist_id INTEGER NOT NULL REFERENCES playlists(id),
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    pos INTEGER NOT NULL
);

-- Train, validation, and test splits.
CREATE TABLE splits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Which playlists belong to which split.
CREATE TABLE split_playlist_memberships(
    id INTEGER PRIMARY KEY,
    split_id INTEGER NOT NULL REFERENCES splits(id),
    -- This is UNIQUE because playlists cannot be in two different splits.
    playlist_id INTEGER UNIQUE NOT NULL REFERENCES playlists(id)
);

/*
Adding these indices is important for querying. But if we create them BEFORE data
insertion, that can really slow down the insertion because the index needs to be
rebuilt. So we wrap them in delimiters here and will create them AFTER having inserted
the data. This is dirty. But it does the trick.

<INDICES>
-- Index for JOINs between splits and split_playlist_memberships.
CREATE INDEX idx_spm_split_id ON split_playlist_memberships(split_id);
-- Index for JOINs between split_playlist_memberships and playlist_track_memberships.
CREATE INDEX idx_ptm_playlist_id ON playlist_track_memberships(playlist_id);
</INDICES>

*/
