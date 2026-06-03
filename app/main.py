from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from app.decision_engine import analyze_website


app = FastAPI(
    title="PhishGuard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


class URLRequest(BaseModel):

    url: str


@app.get("/health")
def health_check():

    return {

        "status": "ok",

        "message": "PhishGuard is running"
    }


@app.post("/analyze")
def analyze_url(data: URLRequest):

    return analyze_website(data.url)