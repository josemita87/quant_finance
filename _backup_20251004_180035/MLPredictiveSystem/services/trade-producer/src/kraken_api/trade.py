from pydantic import BaseModel

class Trade(BaseModel):
    product_id: str
    price: float
    volume: float
    timestamp_ms: int