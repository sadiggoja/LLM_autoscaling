import time
from collections import deque

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

        # Response-time-based back-pressure override (see configs/elasticity_config.yaml).
        rt_cfg = config.get('threshold_rt', {})
        self.rt_high_s = rt_cfg.get('rt_high_s', 0.5)
        self.rt_high_window = rt_cfg.get('rt_high_window', 3)
        self.rt_low_s = rt_cfg.get('rt_low_s', 0.2)
        self.rt_low_window = rt_cfg.get('rt_low_window', 5)
        self._last_rt = 0.0
        self._rt_high_count = 0
        self._rt_low_count = 0

    def observe_response_time(self, rt):
        """Called by runner before get_action. Updates rolling rt counters."""
        self._last_rt = float(rt) if rt is not None else 2.0
        if self._last_rt >= self.rt_high_s:
            self._rt_high_count += 1
            self._rt_low_count = 0
        elif self._last_rt <= self.rt_low_s:
            self._rt_low_count += 1
            self._rt_high_count = 0

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
        rt_pressure = self._rt_high_count >= self.rt_high_window
        n_cpu_limit_norm = latest[0]
        cpu_at_max = (n_cpu_limit_norm * self.max_cpu) >= 0.95 * self.max_cpu

        # VPA decision — rt-pressure overrides cpu observation when there is headroom.
        # Under 503 storms, traffic is shed at nginx so pod cpu_percentage stays low
        # even while the system is failing — rt is the truer back-pressure signal.
        if rt_pressure and not cpu_at_max:
            vpa_action = 0.5
        elif cpu_percentage > self.upper_cpu:
            vpa_action = 0.5
        elif cpu_percentage < self.lower_cpu and not rt_pressure:
            vpa_action = -0.5
        else:
            vpa_action = 0.0

        # HPA decision (only for joint envs with replica info).
        # Scale-down is intentionally suppressed: it can kill the pod the env
        # is observing (container_id then goes stale and patch_pod 404s).
        # Only scale-up is emitted; reset_cluster handles replica reset
        # between iterations.
        hpa_action = 0.0
        if dims >= 10:
            avg_cpu_replicas = latest[8] * 100
            self.hpa_cooldown_counter += 1

            # rt-pressure HPA promotion: when sustained back-pressure persists
            # and VPA is already capped at max_cpu, the only remaining lever is
            # to add a replica. Reuses the same sustained_high_count gate so it
            # respects hpa_cooldown_steps and sustained_threshold.
            if rt_pressure and cpu_at_max:
                self.sustained_high_count += 1
                self.sustained_low_count = 0
            elif avg_cpu_replicas > self.upper_cpu:
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
                hpa_action = 0.5  # Scale up only
                self.hpa_cooldown_counter = 0
                self.sustained_high_count = 0
            # Scale-down intentionally suppressed — see comment above.

        return np.array([vpa_action, hpa_action], dtype=np.float32)

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class InPlaceVPAAgent:
    """In-place VPA heuristic — an RL-comparable, drop-in autoscaling baseline.

    Reproduces the Kubernetes VPA *recommender* logic (a high percentile of
    recent CPU usage plus a safety margin) but actuates the recommendation
    IN-PLACE through the same env action path the RL agents use
    (``env.step`` -> ``_apply_delta`` -> ``patch_pod``, no pod restart), at the
    same ``action_interval``. This makes it an apples-to-apples replacement for
    the RL policy: only the decision rule differs, the actuator/cadence/budget
    are identical.

    Vertical only: the HPA component of the action is always 0, matching real
    VPA which never changes replica count.

    Contrast with ``KubernetesVPABaseline``, which is passive and relies on the
    real (restart-based, multi-minute) VPA operator running in the cluster.
    """

    def __init__(self, pod_name: str = None, deployment_name: str = "localization-api"):
        config = load_config()
        self.pod_name = pod_name
        self.deployment_name = deployment_name
        self.max_cpu = config['max_cpu']
        self.min_cpu = config['min_cpu']
        self.scale_action = config['scale_action']
        self.debug = config['debug_deployment']

        # VPA recommender parameters. Defaults mirror upstream VPA:
        # 90th-percentile target with a 15% safety margin over a rolling window.
        vpa_cfg = config.get('vpa', {})
        self.target_percentile = vpa_cfg.get('target_percentile', 90)
        self.safety_margin = vpa_cfg.get('safety_margin', 0.15)
        self.history_window = vpa_cfg.get('history_window', 60)

        self.usage_history = deque(maxlen=self.history_window)

    def _latest_dims(self, state):
        """Slice the most recent state vector out of the flattened FIFO.
        State width is 10 (joint), 7 (joint independent / flat), or 5."""
        state = np.array(state).flatten()
        n = len(state)
        dims = 10 if n % 10 == 0 else (7 if n % 7 == 0 else 5)
        return state[-dims:]

    def get_action(self, state):
        """Return joint continuous action [vpa_action, hpa_action] in [-1, 1].

        hpa_action is always 0 (VPA is vertical only). vpa_action is the
        per-step delta that moves the current CPU limit toward the VPA target;
        the env scales it by ``scale_action`` and clamps it to the shared CPU
        budget and [min_cpu, max_cpu]."""
        latest = self._latest_dims(state)
        # State indices (see envs.BaseJointElasticityEnv.get_current_usage):
        #   latest[0] = cpu_limit / max_cpu,  latest[1] = cpu_usage / max_cpu
        cpu_limit_mc = float(latest[0]) * self.max_cpu
        usage_mc = float(latest[1]) * self.max_cpu

        self.usage_history.append(usage_mc)

        # VPA recommendation: percentile of recent usage + safety margin.
        target = np.percentile(list(self.usage_history), self.target_percentile)
        target *= (1.0 + self.safety_margin)
        target = float(np.clip(target, self.min_cpu, self.max_cpu))

        # Convert the absolute target into the env's per-step delta action.
        # env applies: delta_mc = vpa_action * scale_action (then clamps).
        delta_mc = target - cpu_limit_mc
        vpa_action = float(np.clip(delta_mc / self.scale_action, -1.0, 1.0))

        return np.array([vpa_action, 0.0], dtype=np.float32)

    def observe_response_time(self, rt):
        """VPA is usage-driven only; response time is ignored. Kept so the
        benchmark runner's optional observe_response_time() call is a no-op."""
        pass

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class KubernetesHPABaseline:
    """Wrapper around native K8s HPA behavior for benchmarking.

    Does NOT take actions itself — relies on K8s HPA controller.
    Records metrics passively, and mirrors the controller-driven replica count
    onto the env each step so the benchmark CSV's `replica_count` column
    reflects what the K8s HPA controller is actually doing (otherwise it stays
    frozen at the value read at env __init__).
    """

    def __init__(self, env=None, deployment_name: str = "localization-api"):
        config = load_config()
        self.env = env
        self.deployment_name = deployment_name
        self.debug = config['debug_deployment']
        self.nodes = init_nodes(debug=self.debug, custom_label=config['target_app_label'])

        self.metrics_history: list[dict] = []

    def get_action(self, state):
        """No-op action. K8s HPA handles scaling. We just record metrics."""
        target = self.env.target_deployment if self.env is not None else self.deployment_name
        replicas = get_deployment_replicas(target, debug=self.debug)
        if self.env is not None and replicas:
            self.env.current_replicas = replicas
        avg_cpu = get_deployment_avg_cpu(target, self.nodes, debug=self.debug)

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


class KubernetesVPABaseline:
    """Wrapper around native K8s VPA behavior for benchmarking.

    Does NOT take actions itself — relies on the VPA operator
    (vpa-recommender + vpa-updater + vpa-admission-controller) to resize pods.
    Records metrics passively, and mirrors the VPA-driven CPU limit back onto
    the env each step so the benchmark CSV's `cpu_limit` column reflects what
    VPA is actually doing (otherwise it stays frozen at the env's `ALLOCATED`
    value, which this baseline never changes itself).
    """

    def __init__(self, env=None, deployment_name: str = "localization-api"):
        config = load_config()
        self.env = env
        self.deployment_name = deployment_name
        self.debug = config['debug_deployment']
        self.nodes = init_nodes(debug=self.debug, custom_label=config['target_app_label'])

        self.metrics_history: list[dict] = []

    def get_action(self, state):
        """No-op action. K8s VPA handles resizing. We just record metrics."""
        cpu_limit = None
        if self.env is not None:
            try:
                (live_cpu_limit, _, _), *_ = self.env.node.get_container_usage(
                    self.env.container_id)
                if live_cpu_limit and live_cpu_limit > 0:
                    cpu_limit = float(live_cpu_limit)
                    self.env.ALLOCATED = cpu_limit
            except Exception:
                pass

        self.metrics_history.append({
            "timestamp": time.time(),
            "cpu_limit": cpu_limit,
        })

        # Return no-op (joint continuous format)
        return np.array([0.0, 0.0], dtype=np.float32)

    def get_recorded_metrics(self) -> list[dict]:
        return self.metrics_history

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
