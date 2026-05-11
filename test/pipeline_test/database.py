# database.py
import sqlite3

DATABASE_URL = "./parking.db"

def get_db_connection():
    """Tạo và trả về kết nối đến cơ sở dữ liệu SQLite"""
    # check_same_thread=False bắt buộc phải có cho FastAPI
    return sqlite3.connect(DATABASE_URL, check_same_thread=False)