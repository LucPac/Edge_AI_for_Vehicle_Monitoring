# database.py
import psycopg2
from config import DB_CONFIG

def get_db_connection():
    """Tạo và trả về kết nối đến cơ sở dữ liệu PostgreSQL"""
    return psycopg2.connect(**DB_CONFIG)