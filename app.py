from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import glob
import sys
from keras_models.run import preload_model, preprocess

class Item(BaseModel):
    text: str

attributes = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']
A, C, E, O, N = preload_model(model_no=2)
app = FastAPI()

@app.get("/")
def get():
    return "POST some text to /"
    
@app.post("/")
async def post(item: Item):
    pObj = preprocess(item.text)
    res = {}
    for i,model in enumerate([A, C, E, O, N]):
        res[attributes[i]] = round(model.predict(pObj.X, verbose=2)[0][0])
    return res


def run():
    uvicorn.run(app)


if __name__ == '__main__':
    run()
