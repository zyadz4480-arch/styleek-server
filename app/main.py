import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import init_models
from app.ml.automata.config import Config as AutomataConfig
from app.routers import style, automata_internal

logging.basicConfig(level=logging.INFO, force=True)

# ---- تشخيص استيراد reels ----
try:
    from app.routers import reels
    print("✅ reels imported successfully")
except Exception as e:
    print("❌ reels import failed:")
    print(repr(e))
    raise
# -----------------------------

# ---- [مؤقت] راوتر تشخيصي لعرض interactions_v2 من المتصفح --

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [جديد] فحص إلزامي عند الإقلاع — propagation_mode غير laplacian هو
    # النمط الوحيد المثبَت أنه لا ينفجر عدديًا (integration_guide.md §6).
    assert AutomataConfig().propagation_mode == "laplacian", (
        "FATAL: app/ml/automata propagation_mode ليس laplacian عند الإقلاع"
    )
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
app.include_router(automata_internal.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
