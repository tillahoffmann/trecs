from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import httpx
import os
from urllib.parse import urlencode
import uvicorn


REDIRECT_URI = "http://127.0.0.1:8000/callback"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/token")
async def token() -> str:
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": os.environ["SPOTIFY_CLIENT_ID"],
        "client_secret": os.environ["SPOTIFY_CLIENT_SECRET"],
    }

    response = httpx.post(url, data=payload)
    response.raise_for_status()
    return response.text


@app.get("/callback")
def callback(code: str) -> str:
    url = "https://accounts.spotify.com/api/token"
    client_id =os.environ["SPOTIFY_CLIENT_ID"]
    assert client_id
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    assert client_secret
    payload = {
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    response = httpx.post(url, data=payload)
    response.raise_for_status()
    return response.text


@app.get("/login")
async def login() -> RedirectResponse:

    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    assert client_id
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": "user-read-private user-read-email",
        "redirect_uri": REDIRECT_URI,
    }

    url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return RedirectResponse(url)


if __name__ == "__main__":
    uvicorn.run(app)
