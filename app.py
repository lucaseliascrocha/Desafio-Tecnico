import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model, predict_model

class Data(BaseModel):
    feature0: float
    feature1: int
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: int
    feature7: float
    feature8: float
    feature9: int
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float
    feature15: int

app = FastAPI()

@app.get('/')

def index():
    return {'message': "This is the home page of this API. Go to /predict"}

@app.post('/predict')
def get_predict(data: Data):

    entry_data = data.dict()

    model = load_model('modelo')

    predict = predict_model(
                    model,
                    data=pd.DataFrame([entry_data])
            )#['Label'][0]

    return {'predict': predict}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)