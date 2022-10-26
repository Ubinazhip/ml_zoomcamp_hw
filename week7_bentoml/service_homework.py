import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from pydantic import BaseModel


#model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()


svc = bentoml.Service("hw_task", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(vector):
    print(f'input is {vector}')
    prediction = model_runner.predict.run(vector)
    print(f'output is {prediction[0]}')
    return prediction
