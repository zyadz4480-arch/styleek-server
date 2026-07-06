"""
إعداد SQLAlchemy (Async) للاتصال بـ PostgreSQL
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_models():
    """ينشئ الجداول إن لم تكن موجودة (بديل بسيط لـ Alembic أثناء التطوير)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # [جديد] migration يدوي بديل — يضيف عمود embedding_source لجدول
        # reel_embeddings إن لم يكن موجودًا بعد. آمن للتشغيل في كل مرة
        # يُقلع فيها السيرفر (IF NOT EXISTS)، لأن create_all وحده لا يضيف
        # أعمدة جديدة لجدول موجود مسبقًا. يمكن حذف هذا السطر لاحقًا بعد
        # التأكد أن العمود أُضيف فعليًا على قاعدة الإنتاج.
        await conn.execute(text(
            "ALTER TABLE reel_embeddings "
            "ADD COLUMN IF NOT EXISTS embedding_source VARCHAR(16) "
            "NOT NULL DEFAULT 'cold_start'"
        ))
