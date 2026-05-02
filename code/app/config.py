# app/config.py
CONFIG = {
    "top_k": 5,
    "confidence_threshold": 0.65,
    "risk_keywords": [
        "fraud", "unauthorized", "payment failed",
        "money deducted", "account hacked"
    ]
}