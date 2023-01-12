#!/bin/bash

# Declare needed variables
points_source_path=$1
footprints_source_path=$2
output_path=$3

# Start the Docker container with PostgreSQL
docker run --name skynet \
-e=POSTGRES_PASSWORD=mysecretpassword \
-d postgis/postgis 

# Wait for the container to start up
sleep 10

# Copy the data files to the container
docker cp "$points_source_path" skynet:/tmp/points.csv
docker cp "$footprints_source_path" skynet:/tmp/footprints.csv

# Copy the SQL file to the container
docker cp preprocessing.sql skynet:/tmp

# Execute the SQL file in the database
docker exec -i skynet psql -U postgres -f /tmp/preprocessing.sql \
-v points_source_path="'/tmp/points.csv'" \
-v footprints_source_path="'/tmp/footprints.csv'" \
-v output_path="'/tmp/output_data.csv'"

# Copy back the output file to the host 
docker cp skynet:/tmp/output_data.csv "$output_path"

# Stop and remove the Docker container
docker stop skynet
docker rm skynet
