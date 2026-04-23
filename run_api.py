"""Entry point to run the FastAPI backend.

Usage:
    python run_api.py
    uvicorn src.api.routes:app --reload
"""

import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
