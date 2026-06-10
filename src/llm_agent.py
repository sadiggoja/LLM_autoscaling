import os
import re
import time
from collections import deque
from datetime import datetime
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

ACTION_MODES = ("vpa", "hpa", "both")


def _build_system_prompt(action_mode: str, inference_mode: str, max_cpu: int) -> str:
    """Produce a system prompt tailored to which scaling levers the agent is
    allowed to pull. `action_mode` is one of: 'vpa' (CPU only), 'hpa' (replicas
    only), 'both'."""
    intro = ("You are an autoscaling agent for a Kubernetes cluster running "
             "microservices on edge devices (Raspberry Pi).")

    if action_mode == "vpa":
        job = "Your job is to manage CPU resources by adjusting per-pod CPU limits."
        goals = [
            "Keep CPU utilization between 30-60% per pod (target range)",
            "Minimize response latency (keep under 250ms)",
            "Avoid over-provisioning (wasted resources) and under-provisioning (high latency)",
        ]
        rules = [
            f"CPU limits range from 50m to {max_cpu}m per pod",
            "Available shared resources are limited; increasing one pod reduces what's available for others",
            "You CANNOT change replica counts in this run — vertical scaling only",
        ]
        parser_actions = [
            "scale_cpu(<delta_millicores>)   integer delta, e.g. scale_cpu(50) or scale_cpu(-100)",
            "no_action()",
        ]
    elif action_mode == "hpa":
        job = "Your job is to manage capacity by adjusting the deployment's replica count."
        goals = [
            "Keep CPU utilization between 30-60% per pod (target range)",
            "Minimize response latency (keep under 250ms)",
            "Avoid over-provisioning (wasted replicas) and under-provisioning (high latency)",
        ]
        rules = [
            "Replica count ranges from 1 to 5",
            "Replica changes have a cooldown period; avoid frequent scaling",
            "You CANNOT change per-pod CPU limits in this run — horizontal scaling only",
        ]
        parser_actions = [
            "scale_replicas(<target>)        integer 1-5, e.g. scale_replicas(3)",
            "no_action()",
        ]
    else:  # "both"
        job = "Your job is to manage CPU resources efficiently by calling the provided tools."
        goals = [
            "Keep CPU utilization between 30-60% per pod (target range)",
            "Minimize response latency (keep under 250ms)",
            "Avoid over-provisioning (wasted resources) and under-provisioning (high latency)",
            "Scale replicas only when vertical scaling alone is insufficient",
        ]
        rules = [
            f"CPU limits range from 50m to {max_cpu}m per pod",
            "Available shared resources are limited; increasing one pod reduces what's available for others",
            "Replica changes have a cooldown period; avoid frequent scaling",
        ]
        parser_actions = [
            "scale_cpu(<delta_millicores>)   integer delta, e.g. scale_cpu(50) or scale_cpu(-100)",
            "scale_replicas(<target>)        integer 1-5,  e.g. scale_replicas(3)",
            "no_action()",
        ]

    goals_str = "\n".join(f"- {g}" for g in goals)
    rules_str = "\n".join(f"- {r}" for r in rules)

    if inference_mode == "function_calling":
        return (
            f"{intro}\n{job}\n\n"
            f"Goals:\n{goals_str}\n\n"
            f"Rules:\n{rules_str}\n"
            f"- Call exactly ONE tool per decision step"
        )
    # parser mode: no tool schema sent — model is told to end with an explicit call
    actions_block = "\n  ".join(parser_actions)
    return (
        f"{intro}\n{job}\n\n"
        f"Goals:\n{goals_str}\n\n"
        f"Rules:\n{rules_str}\n\n"
        f"After your reasoning, end your response with EXACTLY ONE of the following calls on its own line:\n"
        f"  {actions_block}"
    )


def _allowed_action_names(action_mode: str) -> set:
    if action_mode == "vpa":
        return {"scale_cpu", "no_action"}
    if action_mode == "hpa":
        return {"scale_replicas", "no_action"}
    return {"scale_cpu", "scale_replicas", "no_action"}


class LLMAgent:
    """LLM-based autoscaling agent implementing the same get_action(state) interface as RL agents."""

    # Class-level shared log file — one file for the entire benchmark run
    _log_fh = None
    _log_path: str = ""

    @classmethod
    def _init_log(cls):
        """Open the log file on first use. Safe to call multiple times."""
        if cls._log_fh is not None:
            return
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cls._log_path = f"results/LLM_model_logs_{ts}.txt"
        cls._log_fh = open(cls._log_path, "w", buffering=1)  # line-buffered
        cls._log_fh.write(f"LLM Benchmark Log — {datetime.now().isoformat()}\n")
        cls._log_fh.write("=" * 80 + "\n")
        print(f"  [LLMAgent] Logging to {cls._log_path}")

    @classmethod
    def log_section(cls, title: str):
        """Write a section header (algorithm name, phase, etc.) into the shared log."""
        if cls._log_fh:
            cls._log_fh.write(f"\n{'─' * 80}\n{title}\n{'─' * 80}\n")

    def __init__(self, provider: LLMProvider, pod_name: str, deployment_name: str = "localization-api",
                 history_window: int = 5, inference_mode: str = "function_calling",
                 max_cpu: int = 1000, action_mode: str = "both"):
        self.provider = provider
        self.pod_name = pod_name
        self.deployment_name = deployment_name
        self.history_window = history_window
        self.max_cpu = max_cpu
        if inference_mode not in ("function_calling", "parser"):
            raise ValueError(f"inference_mode must be 'function_calling' or 'parser', got {inference_mode!r}")
        self.inference_mode = inference_mode
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        self.action_mode = action_mode
        self._allowed_actions = _allowed_action_names(action_mode)

        # Decision history for context
        self.decision_history: deque = deque(maxlen=history_window)

        # Response-time tracking (set by runner via observe_response_time)
        self._rt_history: deque = deque(maxlen=5)

        # Current replica count from the latest state vector — used by the
        # scale_replicas parser to convert the LLM's absolute target into the
        # differential action the env expects.
        self._current_replicas: int = 1

        # Tracking metrics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.decision_latencies: list[float] = []
        self.total_decisions = 0

        # Timeout handling
        self.last_action = None
        self.action_timeout_ms = 30000  # 30 second timeout (Cloudflare tunnel adds latency)

        LLMAgent._init_log()

    def observe_response_time(self, rt):
        """Called by the benchmark runner before get_action; gives the LLM the
        latency signal it needs to honor the 'minimize response latency' goal."""
        self._rt_history.append(float(rt) if rt is not None else 2.0)

    # ------------------------------------------------------------------
    # Internal logging helpers
    # ------------------------------------------------------------------

    def _log(self, text: str):
        """Write a line to the shared log file."""
        if LLMAgent._log_fh:
            LLMAgent._log_fh.write(text + "\n")

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
            f"CPU limit: {cpu_limit_norm * self.max_cpu:.0f}m",
            f"CPU usage: {cpu_usage_norm * self.max_cpu:.0f}m",
            f"CPU utilization: {cpu_percentage:.1f}%",
            f"Available shared resources: {available_norm * self.max_cpu:.0f}m",
        ]
        if self._rt_history:
            latest_rt = self._rt_history[-1]
            lines.append(f"Latest response time: {latest_rt * 1000:.0f}ms (SLA target: 250ms)")
            if len(self._rt_history) >= 2:
                rt_trend = (latest_rt - self._rt_history[-2]) * 1000
                trend_dir = "rising" if rt_trend > 10 else "falling" if rt_trend < -10 else "stable"
                lines.append(f"Response time trend: {trend_dir} ({rt_trend:+.0f}ms)")

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
            self._current_replicas = max(1, int(round(replica_norm * max_replicas)))
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

        Behaviour depends on self.inference_mode:
          "function_calling" — tools are sent to the LLM; structured tool-call
                               responses are decoded directly; text fallback used
                               only when the model ignores the schema.
          "parser"           — no tools are sent; the LLM is asked to end its
                               response with a bare call (scale_cpu(N) etc.) which
                               is then extracted by _parse_text_action.
        """
        prompt = self._state_to_prompt(state)
        step_num = self.total_decisions + 1

        system = _build_system_prompt(self.action_mode, self.inference_mode, self.max_cpu)
        if self.inference_mode == "function_calling":
            tools = [t for t in SCALING_TOOLS if t["name"] in self._allowed_actions]
        else:  # "parser"
            tools = []

        messages = [
            {"role": "user", "content": f"{system}\n\nCurrent cluster state:\n{prompt}\n\nDecide the best scaling action."},
        ]

        start = time.time()
        response = self.provider.query(messages, tools)
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
            self._log(f"  step={step_num:3d}  pod={self.pod_name}  {latency_ms:.0f}ms  TIMEOUT → reused {self.last_action}")
            print(f"  [{self.pod_name}] ⏱ TIMEOUT ({latency_ms:.0f}ms) — reusing previous action: {self.last_action}")
            return self.last_action

        if self.inference_mode == "parser":
            # Always go through text parsing; tool calls won't be present
            action = self._parse_text_action(response.text) if response.text else self._default_action()
        else:
            action = self._parse_action(response)
        self.last_action = action

        # One compact log line per step
        if self.inference_mode == "function_calling" and response.tool_calls:
            tc = response.tool_calls[0]
            self._log(f"  step={step_num:3d}  pod={self.pod_name}  {latency_ms:.0f}ms  "
                      f"[fc] tool={tc.name}  args={tc.arguments}  → action={action}")
        else:
            mode_tag = "parser" if self.inference_mode == "parser" else "fc/no-tool"
            raw = response.text.replace('\n', ' ').strip() if response.text else '(empty)'
            self._log(f"  step={step_num:3d}  pod={self.pod_name}  {latency_ms:.0f}ms  "
                      f"[{mode_tag}]  raw_text={raw!r}  → action={action}")

        # --- console logging ---
        self._log_decision(response, action, latency_ms)

        return action

    def _log_decision(self, response, action, latency_ms: float):
        """Print a readable log of the LLM decision."""
        # Check if action was parsed from text (decision_history was already updated)
        last_decision = self.decision_history[-1] if self.decision_history else ""
        from_text = "(parsed from text)" in last_decision or "(inferred" in last_decision or "(could not parse text)" in last_decision
        source_tag = " 📝" if from_text else ""

        # Decision summary
        if response.tool_calls and not from_text:
            tc = response.tool_calls[0]
            if tc.name == "scale_cpu":
                delta = tc.arguments.get("delta_millicores", 0)
                direction = "⬆ SCALE UP" if delta > 0 else "⬇ SCALE DOWN" if delta < 0 else "— NO CHANGE"
                decision_str = f"{direction} CPU by {abs(delta)}m"
            elif tc.name == "scale_replicas":
                target = tc.arguments.get("target", "?")
                decision_str = f"🔄 SET REPLICAS → {target}"
            elif tc.name == "no_action":
                decision_str = "✓ NO ACTION (resources OK)"
            else:
                decision_str = f"❓ UNKNOWN TOOL: {tc.name}"
        elif from_text:
            # Reconstruct from decision_history
            if "scale_cpu" in last_decision:
                decision_str = f"📝 {last_decision}"
            elif "scale_replicas" in last_decision:
                decision_str = f"📝 {last_decision}"
            elif "no_action" in last_decision:
                decision_str = f"✓ {last_decision}"
            else:
                decision_str = f"📝 {last_decision}"
        else:
            if response.text:
                decision_str = "✓ NO ACTION (no tool call, text unparseable)"
            else:
                decision_str = "✓ NO ACTION (no tool call, no text)"

        # Reasoning (if the LLM provided text alongside the tool call)
        reasoning = response.text.strip() if response.text else ""

        print(f"  [{self.pod_name}] {decision_str}  |  action={action}  |  {latency_ms:.0f}ms  |  "
              f"tokens: {response.input_tokens}→{response.output_tokens}")
        if reasoning:
            # Show first 200 chars of reasoning to avoid flooding
            short = reasoning[:200] + ("..." if len(reasoning) > 200 else "")
            print(f"    💬 LLM reasoning: {short}")

    def _parse_action(self, response: LLMResponse):
        """Convert LLM tool call response to numeric environment action."""
        if not response.tool_calls:
            if response.text:
                return self._parse_text_action(response.text)
            self.decision_history.append("no_action (no tool call, no text)")
            return self._default_action()

        tool_call = response.tool_calls[0]
        name = tool_call.name
        args = tool_call.arguments

        if name not in self._allowed_actions:
            self._log(f"    WARN: tool '{name}' is disallowed in action_mode={self.action_mode!r}, treating as no_action")
            self.decision_history.append(f"no_action (disallowed tool: {name})")
            return self._default_action()

        if name == "no_action":
            self.decision_history.append("no_action")
            return self._default_action()

        elif name == "scale_cpu":
            delta = args.get("delta_millicores", 0)
            self.decision_history.append(f"scale_cpu({delta}m)")
            vpa_action = np.clip(delta / 500.0, -1.0, 1.0)
            return np.array([vpa_action, 0.0], dtype=np.float32)

        elif name == "scale_replicas":
            target = args.get("target", 0)
            self.decision_history.append(f"scale_replicas({target})")
            hpa_action = np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)
            return np.array([0.0, hpa_action], dtype=np.float32)

        else:
            self._log(f"    WARN: unknown tool name='{name}'")
            self.decision_history.append(f"unknown tool: {name}")
            return self._default_action()

    def _default_action(self):
        """Return a no-op action (joint continuous format)."""
        return np.array([0.0, 0.0], dtype=np.float32)

    def _parse_text_action(self, text: str):
        """Fallback: extract scaling decision from LLM's natural language text.

        Collects ALL candidate actions with their text position, then picks the
        LAST one — LLMs typically state their final/concluded decision at the end
        (e.g. "let's try scale_cpu(50m) ... actually no_action()" → no_action wins).

        Explicit-call pass (positions collected, last wins):
          1. Embedded JSON  {"name":"scale_cpu","arguments":{...}}
          2. Direct function-call notation  scale_cpu(Nm) / scale_cpu(pod, N) /
             scale_cpu("Nm") / no_action() / scale_replicas(N)
          3. "Function call: scale_cpu(...)" format
        Fallback pass (only reached when no explicit call found):
          4. Verbose CPU delta phrases  — all matches, last wins
          5. Verbose replica phrases   — all matches, last wins
          6. No-action keyword signals
          7. Vague directional hints
        """
        # Collapse newlines so multi-line LLM output doesn't break regex `.` patterns
        text_normalized = " ".join(text.split())
        text_lower = text_normalized.lower()

        # (pos, action_name, action_array, history_description)
        candidates: list[tuple[int, str, np.ndarray, str]] = []

        # --- 1. Embedded JSON tool call ---
        for json_match in re.finditer(
            r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*(?:"arguments"|"parameters")\s*:\s*(\{[^{}]*\})',
            text_normalized,
        ):
            try:
                import json
                tool_name = json_match.group(1)
                args = json.loads(json_match.group(2))
                if tool_name == "scale_cpu":
                    delta = int(args.get("delta_millicores", 0))
                    if delta != 0:
                        candidates.append((
                            json_match.start(),
                            "scale_cpu",
                            np.array([np.clip(delta / 500.0, -1.0, 1.0), 0.0], dtype=np.float32),
                            f"scale_cpu({delta}m) (parsed from text)",
                        ))
                elif tool_name == "scale_replicas":
                    target = int(args.get("target", 2))
                    candidates.append((
                        json_match.start(),
                        "scale_replicas",
                        np.array([0.0, np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)], dtype=np.float32),
                        f"scale_replicas({target}) (parsed from text)",
                    ))
                elif tool_name == "no_action":
                    candidates.append((json_match.start(), "no_action", self._default_action(), "no_action (parsed from text)"))
            except (ValueError, KeyError) as e:
                self._log(f"    WARN: embedded JSON matched but failed to parse: {e}  raw={json_match.group(0)!r}")

        # --- 2. Direct function-call notation ---
        # Handles quoted and unquoted args:
        #   scale_cpu(50m)  scale_cpu(-50m)  scale_cpu("300m")  scale_cpu('50')
        #   scale_cpu(localization-api2, 150)  scale_cpu("pod", "150m")
        # Pattern: optional non-greedy pod-name prefix ending in comma, then optional
        # quote, then the signed integer, then optional "m" and closing quote/paren.
        for m in re.finditer(
            r'scale_cpu\s*\(\s*(?:[^,)]*?,\s*)?["\']?([+-]?\d+)\s*m?["\']?\s*\)',
            text_lower,
        ):
            delta = int(m.group(1))
            if delta != 0:
                candidates.append((
                    m.start(),
                    "scale_cpu",
                    np.array([np.clip(delta / 500.0, -1.0, 1.0), 0.0], dtype=np.float32),
                    f"scale_cpu({delta}m) (parsed from text)",
                ))

        for m in re.finditer(
            r'scale_replicas\s*\(\s*(?:[^,)]*?,\s*)?(?:target\s*=\s*)?(\d+)\s*\)',
            text_lower,
        ):
            target = int(m.group(1))
            candidates.append((
                m.start(),
                "scale_replicas",
                np.array([0.0, np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)], dtype=np.float32),
                f"scale_replicas({target}) (parsed from text)",
            ))

        for m in re.finditer(r'no_action\s*\(\s*\)', text_lower):
            candidates.append((m.start(), "no_action", self._default_action(), "no_action (parsed from text)"))

        # --- 3. "Function call: scale_cpu(...)" format ---
        fn_match = re.search(r'function call:\s*scale_cpu\s*\(([^)]*)\)', text_lower)
        if fn_match:
            delta_match = re.search(r'delta_millicores\s*=\s*([+-]?\d+)', fn_match.group(1))
            if delta_match:
                delta = int(delta_match.group(1))
                if delta != 0:
                    candidates.append((
                        fn_match.start(),
                        "scale_cpu",
                        np.array([np.clip(delta / 500.0, -1.0, 1.0), 0.0], dtype=np.float32),
                        f"scale_cpu({delta}m) (parsed from text)",
                    ))

        # Use the LAST allowed explicit call (the LLM's concluded decision)
        allowed = self._allowed_actions
        filtered = [c for c in candidates if c[1] in allowed]
        if filtered:
            filtered.sort(key=lambda x: x[0])
            _, _, action, desc = filtered[-1]
            self.decision_history.append(desc)
            return action

        # --- 4. Verbose CPU delta patterns (no explicit call found) ---
        # Use finditer so all matches are collected; last position wins.
        verbose_candidates: list[tuple[int, str, np.ndarray, str]] = []
        if "scale_cpu" in allowed:
            cpu_patterns = [
                (r'(?:scale|adjust|change|set).{0,50}cpu.{0,50}(?:by|to|delta)?\s*([+-]?\d+)\s*m', 0),
                (r'(?:increase|add|raise|bump).{0,50}(?:cpu|limit).{0,80}(?:by)?\s*(\d+)\s*m', +1),
                (r'(?:decrease|reduce|lower|cut).{0,50}(?:cpu|limit).{0,80}(?:by)?\s*(\d+)\s*m', -1),
                (r'(?:increase|add|raise|bump).{0,50}(?:cpu|limit).{0,80}(?:by)?\s*(\d+)', +1),
                (r'(?:decrease|reduce|lower|cut).{0,50}(?:cpu|limit).{0,80}(?:by)?\s*(\d+)', -1),
                (r'(?:adding|increasing)\s*(\d+)\s*m', +1),
                (r'(?:reducing|decreasing)\s*(\d+)\s*m', -1),
                (r'delta_millicores["\']?\s*[:=]\s*([+-]?\d+)', 0),
            ]
            for pattern, sign in cpu_patterns:
                for m in re.finditer(pattern, text_lower):
                    delta = int(m.group(1))
                    if sign != 0:
                        delta = sign * abs(delta)
                    if delta != 0:
                        verbose_candidates.append((
                            m.start(),
                            "scale_cpu",
                            np.array([np.clip(delta / 500.0, -1.0, 1.0), 0.0], dtype=np.float32),
                            f"scale_cpu({delta}m) (parsed from text)",
                        ))

        # --- 5. Verbose replica pattern ---
        if "scale_replicas" in allowed:
            for m in re.finditer(r'(?:scale|set|change).{0,20}replica.{0,20}(?:to|=)\s*(\d+)', text_lower):
                target = int(m.group(1))
                verbose_candidates.append((
                    m.start(),
                    "scale_replicas",
                    np.array([0.0, np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)], dtype=np.float32),
                    f"scale_replicas({target}) (parsed from text)",
                ))

        if verbose_candidates:
            verbose_candidates.sort(key=lambda x: x[0])
            _, _, action, desc = verbose_candidates[-1]
            self.decision_history.append(desc)
            return action

        # --- 6. No-action signals ---
        no_action_patterns = [
            r'\bno.?action\b', r'\bno scaling\b', r'\bno change\b', r'\bmaintain current\b',
            r'resources? (?:are |is )?(?:ok|fine|appropriate|sufficient|adequate)',
        ]
        for pattern in no_action_patterns:
            if re.search(pattern, text_lower):
                self.decision_history.append("no_action (parsed from text)")
                return self._default_action()

        # --- 7. Vague directional intent ---
        up_match = re.search(r'(?:should|will|need to|recommend)\s+(?:increase|scale up|raise)', text_lower)
        down_match = re.search(r'(?:should|will|need to|recommend)\s+(?:decrease|scale down|reduce|lower)', text_lower)
        if up_match and "scale_cpu" in allowed:
            self.decision_history.append("scale_cpu(+50m) (inferred increase from text)")
            return np.array([1.0, 0.0], dtype=np.float32)
        if down_match and "scale_cpu" in allowed:
            self.decision_history.append("scale_cpu(-50m) (inferred decrease from text)")
            return np.array([-1.0, 0.0], dtype=np.float32)
        # HPA-only fallback: bump or shrink the replica count by one
        if up_match and "scale_replicas" in allowed:
            target = min(5, self._current_replicas + 1)
            self.decision_history.append(f"scale_replicas({target}) (inferred increase from text)")
            return np.array([0.0, np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)], dtype=np.float32)
        if down_match and "scale_replicas" in allowed:
            target = max(1, self._current_replicas - 1)
            self.decision_history.append(f"scale_replicas({target}) (inferred decrease from text)")
            return np.array([0.0, np.clip((target - self._current_replicas) / 2.0, -1.0, 1.0)], dtype=np.float32)

        # Nothing matched — log the full raw text so we can improve the parser later
        self._log(f"    PARSE FAILED: could not extract action from text below:")
        self._log(f"    {text_normalized!r}")
        self.decision_history.append("no_action (could not parse text)")
        return self._default_action()

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
