"""GET/PUT /v1/settings — runtime toggles per project (whitelisted keys).
The dashboard's on/off switches call this; changes apply within the cache TTL
(default 30s), no restart."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from memgram import settings_store

router = APIRouter()


@router.get("")
async def get_settings(request: Request, project_id: str):
    out = {}
    for key in settings_store.SETTING_KEYS:
        out[key] = await settings_store.get_bool(request.app.state.pool, project_id, key)
    return {"project_id": project_id, "settings": out}


class SettingBody(BaseModel):
    project_id: str
    key: str
    value: bool


@router.put("")
async def put_setting(request: Request, body: SettingBody):
    if body.key not in settings_store.SETTING_KEYS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown setting; allowed: {sorted(settings_store.SETTING_KEYS)}")
    await settings_store.set_setting(request.app.state.pool, body.project_id,
                                     body.key, "1" if body.value else "0")
    return {"project_id": body.project_id, "key": body.key, "value": body.value}
