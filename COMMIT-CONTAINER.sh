#!/bin/bash

################################################################################

# Source the environment file if it exists
if [ -f "docker.env" ]; then
    source docker.env
fi

# Set the Docker container name from a project name (first argument).
# If no argument is given, use DOCKER_PROJECT_NAME from env file, or fall back to current user name
PROJECT=$1
if [ -z "${PROJECT}" ]; then
    if [ -n "${DOCKER_PROJECT_NAME}" ]; then
        PROJECT=${DOCKER_PROJECT_NAME}
    else
        PROJECT=${USER}
    fi
fi
CONTAINER="${PROJECT}-compact_env-1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"

# Get current date in day-month-year format
DATE_TAG=$(date +%d-%m-%Y)
IMAGE_NAME="compact_env:${DATE_TAG}"

# Update the image name in docker-compose.yml
sed -i "s|image: compact_env:[^[:space:]]*|image: ${IMAGE_NAME}|" ./docker/docker-compose.yml

# Stop the current container
echo "Stopping container ${CONTAINER}..."
docker stop ${CONTAINER}

# Commit the current container state to a new image
echo "Committing container ${CONTAINER} to new image ${IMAGE_NAME}..."
docker commit ${CONTAINER} ${IMAGE_NAME}

# Remove the old container
echo "Removing old container ${CONTAINER}..."
docker rm ${CONTAINER}
