from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_api_key(x_api_key: str = Header(...)):
    """حماية بسيطة بمفتاح ثابت في الهيدر. استبدلها بـ OAuth2/JWT حقيقي قبل الإنتاج
    (يمكن ربطه بنفس Firebase Auth الذي يستخدمه التطبيق حاليًا عبر التحقق من الـ ID Token)."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="مفتاح API غير صالح")
