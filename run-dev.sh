uvicorn main:app --reload --host 0.0.0.0 --port 6006


gunicorn main:app -b 0.0.0.0:6006 -k uvicorn.workers.UvicornWorker