from pydantic import Field
from pydantic import BaseModel as PydanticBaseModel, Field
from bot import therapy_bot
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class Chatting(PydanticBaseModel):
    user: str = Field(description="The message of user")


@app.post("/aibot")
async def health_data(request: Chatting):
    return {"response": therapy_bot(request.user)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
