from fastapi import FastAPI
from web.predict_router import predict_router
app = FastAPI()
app.include_router(predict_router)



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
