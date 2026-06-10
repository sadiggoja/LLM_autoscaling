#!/bin/bash
set -e
microk8s helm install ingress-nginx ingress-nginx/ingress-nginx \
  --set controller.nodeSelector.cluster=vm \
  --set controller.admissionWebhooks.patch.nodeSelector.cluster=vm \
  --set defaultBackend.nodeSelector.cluster=vm
