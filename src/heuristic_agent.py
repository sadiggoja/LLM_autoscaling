import time

import numpy as np

from pod_controller import patch_pod
from deployment_controller import scale_deployment, get_deployment_replicas, get_deployment_avg_cpu
from utils import load_config, init_nodes


class ThresholdJointAgent:
    """Threshold-based joint HPA+VPA agent.

    VPA: increase CPU if utilization > upper, decrease if < lower.
    HPA: scale replicas up if sustained high CPU across pods, down if low.
    """

    def __init__(self, pod_name: str = None, deployment_name: str = "localization-api"):
        config = load_config()
        self.pod_name = pod_name
        self.deployment_name = deployment_name
        self.upper_cpu = config['upper_cpu']
        self.lower_cpu = config['lower_cpu']
        self.increment = config['discrete_increment']
        self.max_cpu = config['max_cpu']
        self.min_cpu = config['min_cpu']
        self.max_replicas = config.get('max_replicas', 5)
        self.min_replicas = config.get('min_replicas', 1)
        self.hpa_cooldown_steps = config.get('hpa_cooldown_steps', 30)
        self.debug = config['debug_deployment']

        # HPA state
        self.hpa_cooldown_counter = 0
        self.sustained_high_count = 0
        self.sustained_low_count = 0
        self.sustained_threshold = 10  # Steps of sustained condition before HPA action

    def get_action(self, state):
        """Returns joint continuous action: [vpa_action, hpa_action] in [-1, 1]."""
        state = np.array(state).flatten()

        # Extract latest state dimensions
        state_len = len(state)
        if state_len % 10 == 0:
            dims = 10
        elif state_len % 7 == 0:
            dims = 7
        else:
            dims = 5
        latest = state[-dims:]

        cpu_percentage = latest[3] * 100

        # VPA decision
        if cpu_percentage > self.upper_cpu:
            vpa_action = 0.5  # Increase
        elif cpu_percentage < self.lower_cpu:
            vpa_action = -0.5  # Decrease
        else:
            vpa_action = 0.0  # Maintain

        # HPA decision (only for joint envs with replica info)
        hpa_action = 0.0
        if dims >= 10:
            avg_cpu_replicas = latest[8] * 100
            self.hpa_cooldown_counter += 1

            if avg_cpu_replicas > self.upper_cpu:
                self.sustained_high_count += 1
                self.sustained_low_count = 0
            elif avg_cpu_replicas < self.lower_cpu * 0.5:
                self.sustained_low_count += 1
                self.sustained_high_count = 0
            else:
                self.sustained_high_count = 0
                self.sustained_low_count = 0

            if (self.sustained_high_count >= self.sustained_threshold and
                    self.hpa_cooldown_counter >= self.hpa_cooldown_steps):
                hpa_action = 0.5  # Scale up
                self.hpa_cooldown_counter = 0
                self.sustained_high_count = 0
            elif (self.sustained_low_count >= self.sustained_threshold and
                    self.hpa_cooldown_counter >= self.hpa_cooldown_steps):
                hpa_action = -0.5  # Scale down
                self.hpa_cooldown_counter = 0
                self.sustained_low_count = 0

        return np.array([vpa_action, hpa_action], dtype=np.float32)

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class KubernetesHPABaseline:
    """Wrapper around native K8s HPA behavior for benchmarking.

    Does NOT take actions itself — relies on K8s HPA controller.
    Records metrics passively for comparison with other agents.
    """

    def __init__(self, deployment_name: str = "localization-api"):
        config = load_config()
        self.deployment_name = deployment_name
        self.debug = config['debug_deployment']
        self.nodes = init_nodes(debug=self.debug, custom_label=config['target_app_label'])

        self.metrics_history: list[dict] = []

    def get_action(self, state):
        """No-op action. K8s HPA handles scaling. We just record metrics."""
        replicas = get_deployment_replicas(self.deployment_name, debug=self.debug)
        avg_cpu = get_deployment_avg_cpu(self.deployment_name, self.nodes, debug=self.debug)

        self.metrics_history.append({
            "timestamp": time.time(),
            "replicas": replicas,
            "avg_cpu": avg_cpu,
        })

        # Return no-op (joint continuous format)
        return np.array([0.0, 0.0], dtype=np.float32)

    def get_recorded_metrics(self) -> list[dict]:
        return self.metrics_history

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
