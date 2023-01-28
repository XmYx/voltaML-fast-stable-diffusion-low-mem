#!/bin/sh
LD_PRELOAD=${PLUGIN_LIBS} uvicorn fastapi_server:app --port 5003 --host 0.0.0.0 --reload