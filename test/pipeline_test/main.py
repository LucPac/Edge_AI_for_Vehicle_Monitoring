# main.py
import socket
import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from api.routes import router as api_router
from api.video import router as video_router
from services.camera import camera_loop

app = FastAPI(title="Hệ thống Quản lý Bãi đỗ xe thông minh")

# --- 1. CẤU HÌNH BẢO MẬT CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CẤU HÌNH THƯ MỤC TĨNH VÀ TRANG CHỦ ---
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/crops", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# --- 3. KẾT NỐI CÁC LUỒNG API ---
app.include_router(api_router)
app.include_router(video_router)

# --- 4. KHỞI ĐỘNG LUỒNG CAMERA ---
def get_local_ip():
    """Hàm tự động dò tìm địa chỉ IP mạng Wi-Fi của laptop"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.on_event("startup")
def startup_event():
    # In ra đường link trực tiếp
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("🚗 HỆ THỐNG BÃI ĐỖ XE ĐÃ SẴN SÀNG! 🚗")
    print(f"💻 Xem trên Laptop:  http://localhost:8000/")
    print("[*] Khởi động luồng Camera ngầm...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()