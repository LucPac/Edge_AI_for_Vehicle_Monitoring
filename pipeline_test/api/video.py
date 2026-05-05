# api/video.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.camera import gen_frames

router = APIRouter()

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')