import json
import re

import numpy as np
import pandas as pd
import requests
import yaml
from kubernetes import client, config

from node import Node


# returns list(nodes) from the connected cluster
def init_nodes(debug=False, custom_label='type=ray'):
    nodes = []
    config.load_kube_config() if debug else config.load_incluster_config()
    v1 = client.CoreV1Api()

    ret = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor')
    # multi-arch cadvisor image
    # ret_arm64 = v1.list_pod_for_all_namespaces(label_selector='name=cadvisor-arm64')
    for pod in ret.items:  # + ret_arm64.items:
        if pod.metadata.namespace == "kube-system":
            for container in pod.status.container_statuses:
                if container.name == "cadvisor":  # or status.name == "cadvisor-arm64":
                    node_ip = None
                    if pod.status.host_ip:
                        node_ip = pod.status.host_ip
                    nodes.append(Node(pod.spec.node_name, node_ip, node_ip))

    for node in nodes:
        node.update_containers(debug=debug, custom_label=custom_label)

    return nodes


_PROBE_LOG_PATH = "results/probe_errors.log"
_probe_log_fh = None


def _probe_log(line):
    global _probe_log_fh
    import os
    if _probe_log_fh is None:
        os.makedirs(os.path.dirname(_PROBE_LOG_PATH), exist_ok=True)
        _probe_log_fh = open(_PROBE_LOG_PATH, "a", buffering=1)
    _probe_log_fh.write(line + "\n")


def make_request(url, data, probe=False):
    headers = {'Content-Type': 'application/json'}
    response = None
    exc = None
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=30)
    except Exception as e:
        exc = e

    if response is None:
        if probe:
            from datetime import datetime
            etype = type(exc).__name__ if exc else "NoResponse"
            emsg = str(exc)[:160] if exc else ""
            _probe_log(f"{datetime.now():%H:%M:%S.%f}  url={url}  EXC  {etype}: {emsg}")
        return None
    if response.status_code != 200:
        if probe:
            from datetime import datetime
            body = (response.text or "")[:160].replace("\n", " ")
            _probe_log(f"{datetime.now():%H:%M:%S.%f}  url={url}  HTTP_{response.status_code}  body={body!r}")
        return None
    return response.elapsed.total_seconds()


def increment_last_number(input_string):
    match = re.search(r'(\d+)$', input_string)

    if match:
        last_number = int(match.group(1))
        new_number = last_number + 1
        result_string = re.sub(r'\d+$', str(new_number), input_string)
        return result_string
    else:
        return input_string + '1'


def load_config():
    with open('configs/elasticity_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_deployment_name(agent_id, config=None):
    """Look up the deployment name for a given 1-based agent_id.

    Reads `service_deployments` from elasticity_config.yaml. Falls back to the
    legacy `localization-api{id}` naming if the list is unset or too short, so
    existing single-service setups keep working.
    """
    cfg = config if config is not None else load_config()
    roster = cfg.get('service_deployments') or []
    if 1 <= agent_id <= len(roster):
        return roster[agent_id - 1]
    base = cfg.get('target_deployment', 'localization-api')
    return f'{base}{agent_id}'


def calculate_dynamic_rps(episode, reqs_per_second, min_rps, max_limit_rps=100, scale_factor=0.005,
                          randomize_reqs=True):
    dynamic_max_rps = int(reqs_per_second + episode * scale_factor * reqs_per_second)
    dynamic_max_rps = min(dynamic_max_rps, max_limit_rps)
    random_rps = np.random.randint(min_rps, dynamic_max_rps) if randomize_reqs else reqs_per_second
    return dynamic_max_rps, random_rps


def save_training_data(path, rewards, mean_rts, agents_summed_rewards, resource_dev=None, agent_mean_rts=None):
    ep_summed_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
    ep_summed_rewards_df.to_csv(f'{path}/ep_summed_rewards.csv', index=False)

    ep_latencies_df = pd.DataFrame({'Episode': range(len(mean_rts)), 'Mean Latency': mean_rts})
    ep_latencies_df.to_csv(f'{path}/ep_latencies.csv', index=False)

    for agent_idx, rewards in enumerate(agents_summed_rewards):
        filename = f'{path}/agent_{agent_idx}_ep_summed_rewards.csv'
        agent_rewards_df = pd.DataFrame({'Episode': range(len(rewards)), 'Reward': rewards})
        agent_rewards_df.to_csv(filename, index=False)

    if resource_dev:
        resource_dev_df = pd.DataFrame({'Episode': range(len(resource_dev)), 'Resource Deviation': resource_dev})
        resource_dev_df.to_csv(f'{path}/resource_dev.csv', index=False)

    if agent_mean_rts:
        for agent_idx, latencies in enumerate(agent_mean_rts):
            filename = f'{path}/agent_{agent_idx}_ep_mean_latencies.csv'
            agent_latenices_df = pd.DataFrame({'Episode': range(len(latencies)), 'Latency': latencies})
            agent_latenices_df.to_csv(filename, index=False)
