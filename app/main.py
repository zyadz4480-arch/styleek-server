from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import init_models
from app.routers import style

# ---- تشخيص استيراد reels ----
try:
    from app.routers import reels
    print("✅ reels imported successfully")
except Exception as e:
    print("❌ reels import failed:")
    print(repr(e))
    raise
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    yield


app = FastAPI(
    title="Styleek AI Server",
    description="خادم الذكاء الاصطناعي لتطبيق ستايليك",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(style.router)
app.include_router(reels.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
