#!/bin/bash
# run this script in the root of the project to recreate the index.yaml file and package the chart to docs/

helm lint k8s/serge/ &&
	helm package -d docs k8s/serge/ &&
	helm repo index docs --url https://serge-chat.github.io/serge
