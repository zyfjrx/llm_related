from fastapi import APIRouter,HTTPException
from web.schemas import PredictRequest, PredictResponse
from web.service import predict_title

predict_router = APIRouter(tags=["预测接口"])
@predict_router.post("/predict")
def predict(req: PredictRequest)->PredictResponse:
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=401,detail="请输入标题")
    return PredictResponse(title=title,label=predict_title(title))