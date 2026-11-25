"""Bounded, in-memory handoff from one API process to the private GPU worker."""
import hmac
import os
import time
from concurrent.futures import Future, TimeoutError
from queue import Queue, Empty, Full
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Response

token = os.environ.get("SATQUERY_WORKER_TOKEN", "")
jobs = Queue(maxsize=3)
waiting = {}
lock = Lock()
last_poll = 0.0


def authorize(authorization: str = Header(default="")):
    if not token or not hmac.compare_digest(authorization.encode(), ("Bearer " + token).encode()):
        raise HTTPException(401)


router = APIRouter(prefix="/worker", dependencies=[Depends(authorize)])


def available():
    return last_poll > 0 and time.monotonic() - last_poll < 120


def infer(specialist, payload):
    if not available():
        raise RuntimeError("The judging PC is offline. Please try again shortly.")
    job_id = uuid4().hex
    future = Future()
    with lock:
        waiting[job_id] = future
    try:
        jobs.put_nowait((job_id, specialist, payload))
        response = future.result(timeout=120)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]
    except Full:
        raise RuntimeError("The GPU queue is full. Please retry shortly.") from None
    except TimeoutError:
        raise RuntimeError("The GPU job timed out. Please retry.") from None
    finally:
        with lock:
            waiting.pop(job_id, None)


@router.post("/next")
def next_job():
    global last_poll
    last_poll = time.monotonic()
    try:
        while True:
            job_id, specialist, payload = jobs.get(timeout=20)
            with lock:
                if job_id in waiting:
                    return {"id": job_id, "specialist": specialist, "payload": payload}
    except Empty:
        return Response(status_code=204)


@router.post("/{job_id}/result")
def finish(job_id: str, response: dict):
    with lock:
        future = waiting.get(job_id)
        if not future or future.done():
            raise HTTPException(410)
        future.set_result(response)
    return Response(status_code=204)
