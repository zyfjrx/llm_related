from fastapi import FastAPI
from web.predict_router import predict_router
import uvicorn
app = FastAPI()
app.include_router(predict_router)


def run_app():
    uvicorn.run("web.app:app", port=8000)