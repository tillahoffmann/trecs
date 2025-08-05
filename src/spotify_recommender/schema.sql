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

-- Tracks with audio features from https://developer.spotify.com/documentation/web-api/reference/get-audio-features.
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    uri TEXT NOT NULL UNIQUE,
    album_id NOT NULL REFERENCES albums(id),
    artist_id NOT NULL REFERENCES artists(id),
    name TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
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
);
    valence REAL

CREATE TABLE playlists (
    -- We use the explicit `pid` from the MPD dataset.
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    collaborative BOOLEAN NOT NULL,
    modified_at INTEGER NOT NULL,
    num_followers INTEGER NOT NULL,
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
    playlist_id INTEGER NOT NULL REFERENCES playlists(id)
);
