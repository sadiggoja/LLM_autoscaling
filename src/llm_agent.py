import time
from collections import deque
from typing import Optional

import numpy as np
import yaml

from llm_providers import LLMProvider, LLMResponse


def load_llm_config():
    with open('configs/llm_config.yaml', 'r') as f:
        return yaml.safe_load(f)


# Tool definitions in Anthropic format (also converted for Ollama)
SCALING_TOOLS = [
    {
        "name": "scale_cpu",
        "description": "Adjust the CPU limit of a pod by a delta in millicores. Positive values increase, negative decrease.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pod": {"type": "string", "description": "Pod name to scale"},
                "delta_millicores": {"type": "integer", "description": "CPU change in millicores (-500 to 500)"},
            },
            "required": ["pod", "delta_millicores"],
        },
    },
    {
        "name": "scale_replicas",
        "description": "Set the target replica count for a deployment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deployment": {"type": "string", "description": "Deployment name"},
                "target": {"type": "integer", "description": "Desired replica count (1-5)"},
            },
            "required": ["deployment", "target"],
        },
    },
    {
        "name": "no_action",
        "description": "Take no scaling action this step. Use when current resource allocation is appropriate.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

SYSTEM_PROMPT = """You are an autoscaling agent for a Kubernetes cluster running microservices on edge devices (Raspberry Pi).
Your job is to manage CPU resources efficiently by calling the provided tools.

Goals:
- Keep CPU utilization between 30-60% per pod (target range)
- Minimize response latency (keep under 250ms)
- Avoid over-provisioning (wasted resources) and under-provisioning (high latency)
- Scale replicas only when vertical scaling alone is insufficient

Rules:
- CPU limits range from 50m to 1000m per pod
- Available shared resources are limited; increasing one pod reduces what's available for others
- Replica changes have a cooldown period; avoid frequent scaling
- Call exactly ONE tool per decision step"""


class LLMAgent:
    """LLM-based autoscaling agent implementing the same get_action(state) interface as RL agents."""

    def __init__(self, provider: LLMProvider, pod_name: str, deployment_name: str = "localization-api",
                 history_window: int = 5):
        self.provider = provider
        self.pod_name = pod_name
        self.deployment_name = deployment_name
        self.history_window = history_window

        # Decision history for context
        self.decision_history: deque = deque(maxlen=history_window)

        # Tracking metrics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.decision_latencies: list[float] = []
        self.total_decisions = 0

        # Timeout handling
        self.last_action = None
        self.action_timeout_ms = 3000  # 3 second timeout

    def _state_to_prompt(self, state: np.ndarray) -> str:
        """Convert flat state vector to structured text prompt with metrics and trends."""
        # State layout depends on whether it's joint (10 dims per step) or standard (7 dims per step)
        state = np.array(state).flatten()
        state_len = len(state)

        # Try to detect if joint (10-dim) or standard (7-dim) based on total length
        if state_len % 10 == 0:
            dims_per_step = 10
            history_len = state_len // 10
        elif state_len % 8 == 0 and state_len % 7 != 0:
            dims_per_step = 8
            history_len = state_len // 8
        elif state_len % 7 == 0:
            dims_per_step = 7
            history_len = state_len // 7
        elif state_len % 5 == 0:
            dims_per_step = 5
            history_len = state_len // 5
        else:
            dims_per_step = 7
            history_len = state_len // 7

        # Extract the latest step's state
        latest = state[-dims_per_step:]

        cpu_limit_norm = latest[0]
        cpu_usage_norm = latest[1]
        available_norm = latest[2]
        cpu_percentage = latest[3] * 100

        lines = [
            f"Pod: {self.pod_name}",
            f"CPU limit: {cpu_limit_norm * 1000:.0f}m (normalized: {cpu_limit_norm:.3f})",
            f"CPU usage: {cpu_usage_norm * 1000:.0f}m (normalized: {cpu_usage_norm:.3f})",
            f"CPU utilization: {cpu_percentage:.1f}%",
            f"Available shared resources: {available_norm * 1000:.0f}m (normalized: {available_norm:.3f})",
        ]

        if dims_per_step >= 7:
            other_util = latest[4] * 100
            priority = latest[5]
            other_priorities = latest[6]
            lines.extend([
                f"Other pods avg CPU utilization: {other_util:.1f}%",
                f"Pod priority: {priority:.2f}",
                f"Other pods avg priority: {other_priorities:.2f}",
            ])

        if dims_per_step >= 10:
            replica_norm = latest[7]
            avg_cpu_replicas = latest[8] * 100
            request_rate = latest[9]
            max_replicas = 5  # From config
            lines.extend([
                f"Replica count: {int(replica_norm * max_replicas)}/{max_replicas}",
                f"Avg CPU across all replicas: {avg_cpu_replicas:.1f}%",
                f"Request rate estimate: {request_rate:.3f}",
            ])

        # Add trend from history
        if history_len >= 2:
            prev = state[-(2 * dims_per_step):-(dims_per_step)]
            cpu_trend = (latest[3] - prev[3]) * 100
            trend_dir = "increasing" if cpu_trend > 1 else "decreasing" if cpu_trend < -1 else "stable"
            lines.append(f"CPU utilization trend: {trend_dir} ({cpu_trend:+.1f}%)")

        # Add recent decision history
        if self.decision_history:
            lines.append("\nRecent decisions:")
            for decision in self.decision_history:
                lines.append(f"  - {decision}")

        return "\n".join(lines)

    def get_action(self, state) -> any:
        """Get action from LLM. Returns numeric action compatible with environment.step().

        For joint discrete envs: returns int 0-8
        For joint continuous envs: returns np.array([vpa, hpa]) in [-1, 1]
        For standard discrete envs: returns int 0-2
        For standard continuous envs: returns float in [-1, 1]
        """
        prompt = self._state_to_prompt(state)

        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nCurrent cluster state:\n{prompt}\n\nDecide the best scaling action."},
        ]

        start = time.time()
        response = self.provider.query(messages, SCALING_TOOLS)
        latency_ms = (time.time() - start) * 1000

        # Track metrics
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.decision_latencies.append(latency_ms)
        self.total_decisions += 1

        # Timeout: if too slow, reuse previous action
        if latency_ms > self.action_timeout_ms and self.last_action is not None:
            self.decision_history.append(f"TIMEOUT ({latency_ms:.0f}ms) - reused previous action")
            return self.last_action

        action = self._parse_action(response)
        self.last_action = action
        return action

    def _parse_action(self, response: LLMResponse):
        """Convert LLM tool call response to numeric environment action."""
        if not response.tool_calls:
            self.decision_history.append("no_action (no tool call returned)")
            return self._default_action()

        tool_call = response.tool_calls[0]
        name = tool_call.name
        args = tool_call.arguments

        if name == "no_action":
            self.decision_history.append("no_action")
            return self._default_action()

        elif name == "scale_cpu":
            delta = args.get("delta_millicores", 0)
            self.decision_history.append(f"scale_cpu({delta}m)")
            # Convert to continuous action in [-1, 1] range (scale_action default is 50)
            vpa_action = np.clip(delta / 50.0, -1.0, 1.0)
            return np.array([vpa_action, 0.0], dtype=np.float32)

        elif name == "scale_replicas":
            target = args.get("target", 0)
            self.decision_history.append(f"scale_replicas({target})")
            # HPA action: >0.33 means scale up, <-0.33 means scale down
            # We output a strong signal toward the desired direction
            hpa_action = np.clip((target - 2) / 2.0, -1.0, 1.0)  # Assume current ~2 replicas
            return np.array([0.0, hpa_action], dtype=np.float32)

        else:
            self.decision_history.append(f"unknown tool: {name}")
            return self._default_action()

    def _default_action(self):
        """Return a no-op action (joint continuous format)."""
        return np.array([0.0, 0.0], dtype=np.float32)

    def get_metrics(self) -> dict:
        """Return tracking metrics."""
        return {
            "total_decisions": self.total_decisions,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "mean_decision_latency_ms": np.mean(self.decision_latencies) if self.decision_latencies else 0,
            "p95_decision_latency_ms": np.percentile(self.decision_latencies, 95) if self.decision_latencies else 0,
        }

    # Methods for compatibility with RL agent interface
    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
