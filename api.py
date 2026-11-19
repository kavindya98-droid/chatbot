from fastapi import FastAPI,Form
from health_assistant_rag import chat,create_rag
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    #print("API is starting up")
    create_rag("BMI Category Guidelines & Regional Dietary Adaptation.pdf")
    yield
    #print("API is shutting down")

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
def chat_bot(user_name: str = Form(...), country: str = Form(...), preferred_metric: str = Form(...), food_preference: str = Form(...), preferred_workout_type: str = Form(...), height: str = Form(...), weight: str = Form(...)):
    response = chat(user_name,country, preferred_metric, food_preference, preferred_workout_type, height, weight)
    return response





# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}