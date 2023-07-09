#!/bin/bash

set -x

# Install python dependencies
pip install -e ./api || {
	echo 'Failed to install python dependencies'
	exit 1
}

# Install python bindings
pip install llama-cpp-python==0.1.69 || {
	echo 'Failed to install llama-cpp-python'
	exit 1
}

# Start Redis instance
redis-server /etc/redis/redis.conf &

# Start the web server
cd /usr/src/app/web
npm run dev -- --host 0.0.0.0 --port 8008 &

# Start the API
cd /usr/src/app/api
uvicorn src.serge.main:api_app --reload --host 0.0.0.0 --port 9124 --root-path /api/ || {
	echo 'Failed to start main app'
	exit 1
}
