from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import init_models
from app.routers import style
from app.routers import reels


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    yield


app = FastAPI(
    title="Styleek AI Server",
    description="السيرفر الذي يستضيف قلب الذكاء الاصطناعي لتطبيق ستايلك (نُقل من main.dart)",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(style.router)
app.include_router(reels.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
