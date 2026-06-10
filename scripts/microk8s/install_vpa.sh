#!/bin/bash
# Install the Kubernetes Vertical Pod Autoscaler (vpa-recommender, vpa-updater,
# vpa-admission-controller) into the cluster. Unlike HPA, VPA is NOT bundled
# with kube-controller-manager — it's a separate add-on from the
# kubernetes/autoscaler repo and there is no `microk8s enable vpa` shortcut.
#
# Usage:
#   ./scripts/microk8s/install_vpa.sh            # latest release branch
#   VPA_VERSION=vpa-release-1.3 ./scripts/microk8s/install_vpa.sh
#
# After running this script, apply the VPA objects with:
#   microk8s kubectl apply -f configs/localization/vpa/config.yaml
set -e

VPA_VERSION="${VPA_VERSION:-vpa-release-1.3}"
WORKDIR="${WORKDIR:-/tmp/k8s-autoscaler}"

log() { echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"; }

# vpa-up.sh + gencerts.sh hard-code `kubectl` in several places (the $KUBECTL
# override only covers some call sites). On a microk8s-only box we need a real
# `kubectl` on PATH — the standard fix is the snap alias.
if ! command -v kubectl >/dev/null 2>&1; then
    log "kubectl not on PATH — creating snap alias 'microk8s.kubectl' -> 'kubectl'"
    sudo snap alias microk8s.kubectl kubectl
fi

if [ ! -d "$WORKDIR" ]; then
    log "Cloning kubernetes/autoscaler ($VPA_VERSION) into $WORKDIR"
    # NOTE: not using --depth 1 — vpa-up.sh internally `git checkout`s a tag
    # (e.g. vertical-pod-autoscaler-1.3.1) that only exists in full history.
    git clone --branch "$VPA_VERSION" --no-single-branch \
        https://github.com/kubernetes/autoscaler.git "$WORKDIR"
    git -C "$WORKDIR" fetch --tags
else
    log "Reusing existing checkout at $WORKDIR (delete it if you want a clean install)"
    git -C "$WORKDIR" fetch --tags
fi

cd "$WORKDIR/vertical-pod-autoscaler"

log "Installing VPA components via hack/vpa-up.sh"
# vpa-up.sh shells out to `kubectl`; point it at microk8s.
export KUBECTL="microk8s kubectl"
./hack/vpa-up.sh

log "Waiting for VPA pods to become ready"
microk8s kubectl -n kube-system wait --for=condition=Ready pod \
    -l "app in (vpa-recommender,vpa-updater,vpa-admission-controller)" \
    --timeout=180s || {
    log "VPA pods did not become Ready in time — check 'microk8s kubectl -n kube-system get pods | grep vpa'"
    exit 1
}

log "VPA installed. Verify with: microk8s kubectl -n kube-system get pods | grep vpa"
log "Apply the localization VPA objects with:"
log "  microk8s kubectl apply -f configs/localization/vpa/config.yaml"
