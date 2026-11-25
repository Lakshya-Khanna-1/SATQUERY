"""Run on the judging PC with SATQUERY_API_URL and SATQUERY_WORKER_TOKEN set."""
import logging
import os
import time

import requests

url = os.environ["SATQUERY_API_URL"].rstrip("/")
cloud = requests.Session()
cloud.headers["Authorization"] = "Bearer " + os.environ["SATQUERY_WORKER_TOKEN"]
ports = {"specialist_a": 8001, "specialist_b": 8002, "specialist_c": 8003}


def run():
    while True:
        try:
            reply = cloud.post(url + "/worker/next", timeout=40, allow_redirects=False)
            reply.raise_for_status()
            if reply.status_code == 204:
                continue
            job = reply.json()
            try:
                port = ports[job["specialist"]]
                result = requests.post(f"http://127.0.0.1:{port}/infer", json=job["payload"], timeout=95)
                result.raise_for_status()
                output = {"result": result.json()}
            except Exception:
                logging.exception("Local specialist failed")
                output = {"error": "The local specialist could not complete this job."}
            saved = cloud.post(url + "/worker/" + job["id"] + "/result", json=output, timeout=20, allow_redirects=False)
            if saved.status_code != 410:
                saved.raise_for_status()
        except (requests.RequestException, ValueError, KeyError):
            logging.exception("GPU relay connection failed")
            time.sleep(5)


if __name__ == "__main__":
    run()
