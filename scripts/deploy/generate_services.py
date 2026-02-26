"""
Service Generator for scalability testing.

Generates Kubernetes YAML manifests for N microservices distributed across nodes.
Usage: python scripts/deploy/generate_services.py --count 10 --output configs/generated/
"""

import argparse
import os

import yaml


def generate_deployment(service_id: int, node_selector: str = None) -> dict:
    """Generate a Deployment manifest for one microservice."""
    name = f"localization-api{service_id}"
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": 2,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "nodeSelector": {"kubernetes.io/arch": "arm64"},
                    "containers": [{
                        "name": "localization-api",
                        "image": "wrathchild14/localization-reg:latest",
                        "resizePolicy": [
                            {"resourceName": "cpu", "restartPolicy": "NotRequired"},
                            {"resourceName": "memory", "restartPolicy": "NotRequired"},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "150m", "memory": "512Mi"},
                        },
                        "ports": [{"containerPort": 8000}],
                    }],
                },
            },
        },
    }

    if node_selector:
        deployment["spec"]["template"]["spec"]["nodeSelector"]["cluster"] = node_selector

    return deployment


def generate_service(service_id: int) -> dict:
    """Generate a Service manifest."""
    name = f"localization-api{service_id}"
    port = 8080 + service_id
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"localization-service{service_id}"},
        "spec": {
            "type": "ClusterIP",
            "ports": [{"port": port, "targetPort": 8000}],
            "selector": {"app": name},
        },
    }


def generate_ingress(count: int) -> dict:
    """Generate a combined Ingress for all services."""
    paths = []
    for i in range(1, count + 1):
        paths.append({
            "pathType": "ImplementationSpecific",
            "path": f"/api{i}/(.*)",
            "backend": {
                "service": {
                    "name": f"localization-service{i}",
                    "port": {"number": 8080 + i},
                },
            },
        })

    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": "localization-ingress",
            "annotations": {
                "nginx.ingress.kubernetes.io/rewrite-target": "/$1",
            },
        },
        "spec": {
            "ingressClassName": "nginx",
            "rules": [{
                "host": "localhost",
                "http": {"paths": paths},
            }],
        },
    }


def distribute_nodes(count: int, nodes: list[str]) -> list[str]:
    """Round-robin distribute services across available nodes."""
    if not nodes:
        return [None] * count
    return [nodes[i % len(nodes)] for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description="Generate K8s manifests for N microservices")
    parser.add_argument('--count', type=int, required=True, help="Number of microservices")
    parser.add_argument('--output', type=str, default='configs/generated',
                        help="Output directory for YAML files")
    parser.add_argument('--nodes', type=str, nargs='*', default=['rasp1', 'rasp2'],
                        help="Node names for distribution")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    node_assignments = distribute_nodes(args.count, args.nodes)

    # Generate all resources
    all_resources = []
    for i in range(1, args.count + 1):
        all_resources.append(generate_deployment(i, node_assignments[i - 1]))
        all_resources.append(generate_service(i))

    all_resources.append(generate_ingress(args.count))

    # Write to single YAML file with document separators
    output_file = os.path.join(args.output, f"services_{args.count}.yaml")
    with open(output_file, 'w') as f:
        for i, resource in enumerate(all_resources):
            if i > 0:
                f.write("---\n")
            yaml.dump(resource, f, default_flow_style=False)

    # Also generate matching elasticity config
    config = {
        "debug_deployment": False,
        "target_app_label": "app=localization-api",
        "target_container_name": "localization-api",
        "max_cpu": 1000,
        "min_cpu": 50,
        "upper_cpu": 60,
        "lower_cpu": 30,
        "action_interval": 1,
        "max_steps": 60,
        "scale_action": 50,
        "discrete_increment": 25,
        "state_history": 6,
        "max_replicas": 5,
        "min_replicas": 1,
        "hpa_cooldown_steps": 30,
        "target_deployment": "localization-api",
        "replica_change_penalty": 0.3,
    }

    config_file = os.path.join(args.output, f"elasticity_config_{args.count}.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Generated {args.count} services:")
    print(f"  Manifests: {output_file}")
    print(f"  Config:    {config_file}")
    print(f"  Node distribution: {dict(zip(range(1, args.count + 1), node_assignments))}")


if __name__ == '__main__':
    main()
