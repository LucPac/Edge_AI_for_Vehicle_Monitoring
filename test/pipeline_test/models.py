# models.py
from pydantic import BaseModel

class RFIDData(BaseModel):
    rfid_code: str 

class RegisterData(BaseModel):
    rfid_code: str
    owner_name: str
    plate_number: str
    phone: str