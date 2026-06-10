from kubernetes import client, config

from utils import load_config


def _get_apps_api(debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    return client.AppsV1Api()


def _get_core_api(debug=False):
    if debug:
        config.load_kube_config()
    else:
        config.load_incluster_config()
    return client.CoreV1Api()


def get_deployment_replicas(deployment_name, namespace='default', debug=False):
    apps_api = _get_apps_api(debug)
    try:
        deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
        return deployment.spec.replicas
    except Exception as e:
        # Silently ignore 404 — cluster may use standalone Pods instead of Deployments
        if getattr(e, 'status', None) != 404:
            print(f"Error getting replicas for {deployment_name}: {e}")
        return None


def scale_deployment(deployment_name, replicas, namespace='default', debug=False):
    apps_api = _get_apps_api(debug)
    try:
        body = {"spec": {"replicas": replicas}}
        apps_api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        print(f"Scaled {deployment_name} to {replicas} replicas")
    except Exception as e:
        print(f"Error scaling {deployment_name}: {e}")


def resolve_pod_name(name, namespace='default', debug=False):
    """If `name` is a Deployment, return the first running pod's name; else return name as-is."""
    pods = get_deployment_pod_names(name, namespace=namespace, debug=debug)
    return pods[0] if pods else name


def get_deployment_pod_names(deployment_name, namespace='default', debug=False):
    apps_api = _get_apps_api(debug)
    v1 = _get_core_api(debug)
    try:
        dep = apps_api.read_namespaced_deployment(deployment_name, namespace)
        match = dep.spec.selector.match_labels or {}
        label_selector = ",".join(f"{k}={v}" for k, v in match.items())
    except Exception as e:
        if getattr(e, 'status', None) != 404:
            print(f"Error reading deployment {deployment_name}: {e}")
        cfg = load_config()
        label_selector = cfg['target_app_label']
    try:
        pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        return [pod.metadata.name for pod in pods.items if pod.status.phase == "Running"]
    except Exception as e:
        print(f"Error listing pods for {deployment_name}: {e}")
        return []


def get_deployment_avg_cpu(deployment_name, nodes, debug=False):
    pod_names = get_deployment_pod_names(deployment_name, debug=debug)
    if not pod_names:
        return 0.0

    cpu_percentages = []
    for node in nodes:
        for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
            if pod_name in pod_names:
                try:
                    (_, _, cpu_percentage), _, _, _ = node.get_container_usage(container_id)
                    cpu_percentages.append(cpu_percentage)
                except Exception:
                    pass

    if cpu_percentages:
        return sum(cpu_percentages) / len(cpu_percentages)
    return 0.0


if __name__ == '__main__':
    # Example usage:
    # print(get_deployment_replicas('localization-api', debug=True))
    # scale_deployment('localization-api', 3, debug=True)
    # print(get_deployment_pod_names('localization-api', debug=True))
    pass
