import asyncio
import datetime
import json
from typing import Iterable
import uuid
from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from SVCFusion.api.utils.task import Progress, Task
from SVCFusion.api.utils import task


router = APIRouter(
    prefix="/api/test",
    tags=["test"],
)


class FooModel(BaseModel):
    hello: str = "bb"


@Task(router, "/sse/")
async def test(model: FooModel):
    """
    测试函数。
    """
    print("我特么收到了", model.dict())

    print("我特么开始了")
    task.info(f"Test {id} function called.")
    await asyncio.sleep(1)
    task.info("Task info after sleep.")

    for i in Progress(range(10)):
        await asyncio.sleep(0.1)

    task.shutdown()


def get_new_messages():
    return {
        "event": "new_message",
        "retry": 1,
        "data": json.dumps(
            {
                "message": "test message",
                "datetime": datetime.datetime.now().isoformat(sep="T", timespec="auto"),
            }
        ),
        "id": uuid.uuid4(),
    }


async def event_generator(request: Request):
    while True:
        if await request.is_disconnected():
            break
        for _ in range(20):
            yield get_new_messages()
            await asyncio.sleep(0.1)
        break
    yield {
        "event": "completed",
        "data": json.dumps({"status": "done"}),
        "id": str(uuid.uuid4()),
    }


@router.get("/sse2")
async def message_stream(request: Request):
    return EventSourceResponse(event_generator(request))


@router.post("/foo")
async def foo(model: FooModel):
    return {"message": "foo", "data": model.dict()}
