from pydantic import BaseModel
from typing import Literal


class Track(BaseModel):
    """Track as presented in the Spotify Million Playlist dataset.
    """
    pos: int
    artist_name: str
    track_uri: str
    artist_uri: str
    track_name: str
    album_uri: str
    duration_ms: int
    album_name: str

    @property
    def track_id(self) -> str:
        return self.track_uri.split(":")[-1]


class Playlist(BaseModel):
    name: str
    collaborative: bool
    pid: int
    modified_at: int
    num_tracks: int
    num_albums: int
    num_followers: int
    tracks: list[Track]


class PlaylistSliceInfo(BaseModel):
    generated_on: str
    slice: str
    version: Literal["v1"]


class PlaylistSlice(BaseModel):
    info: PlaylistSliceInfo
    playlists: list[Playlist]
