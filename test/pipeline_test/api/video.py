# api/video.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

# Import 2 hàm sinh luồng video mới từ camera.py
from services.camera import gen_frames_in, gen_frames_out

router = APIRouter()

# API cho Camera Lối Vào (Hikvision)
@router.get("/video_feed_in")
def video_feed_in():
    return StreamingResponse(gen_frames_in(), media_type='multipart/x-mixed-replace; boundary=frame')

# API cho Camera Lối Ra (Laptop)
@router.get("/video_feed_out")
def video_feed_out():
    return StreamingResponse(gen_frames_out(), media_type='multipart/x-mixed-replace; boundary=frame')