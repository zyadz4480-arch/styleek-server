"""
إعدادات التطبيق — تُقرأ من متغيرات البيئة (.env)
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # رابط قاعدة بيانات PostgreSQL الجديدة (بيانات التدريب فقط)
    # مثال: postgresql+asyncpg://styleek:password@localhost:5432/styleek_ml
    database_url: str = "postgresql+asyncpg://styleek:styleek@db:5432/styleek_ml"

    # مسار حفظ ملفات النماذج المدرَّبة (joblib)
    models_dir: str = "/app/model_store"

    # مفتاح API بسيط لحماية السيرفر (استبدله بشيء حقيقي في الإنتاج)
    api_key: str = "CHANGE_ME_SECRET_KEY"

    # عدد التفاعلات الجديدة قبل إعادة تدريب كل نماذج المستخدم تلقائيًا
    retrain_every: int = 25

    class Config:
        env_file = ".env"


settings = Settings()
