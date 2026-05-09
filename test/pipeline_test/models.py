# models.py
from pydantic import BaseModel

class RFIDData(BaseModel):
    rfid_code: str 