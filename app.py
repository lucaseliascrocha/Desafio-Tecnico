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
    return {'predict': 'carregou o modelo'}

    predict = predict_model(
                    model,
                    data=pd.DataFrame([{
                        'feature0' : entry_data['feature0'],
                        'feature1' : entry_data['feature1'],
                        'feature2' : entry_data['feature2'],
                        'feature3' : entry_data['feature3'],
                        'feature4' : entry_data['feature4'],
                        'feature5' : entry_data['feature5'],
                        'feature6' : entry_data['feature6'],
                        'feature7' : entry_data['feature7'],
                        'feature8' : entry_data['feature8'],
                        'feature9' : entry_data['feature9'],
                        'feature10' : entry_data['feature10'],
                        'feature11' : entry_data['feature11'],
                        'feature12' : entry_data['feature12'],
                        'feature13' : entry_data['feature13'],
                        'feature14' : entry_data['feature14'],
                        'feature15' : entry_data['feature15']
                    }])
            )['Label'][0]

    return {'predict': predict}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)