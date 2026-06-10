"""
Unified Benchmark Runner for MARLISE.

Runs all configured algorithms through identical load scenarios and collects metrics.
Usage: python scripts/eval/benchmark_runner.py [--config configs/benchmark_config.yaml] [--scenario small]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from envs import (JointContinuousElasticityEnv, JointDiscreteElasticityEnv,
                  set_available_resource, set_other_utilization, set_other_priorities)
from infer import initialize_agent
from heuristic_agent import (ThresholdJointAgent, KubernetesHPABaseline, KubernetesVPABaseline,
                             InPlaceVPAAgent)
from llm_agent import LLMAgent, load_llm_config
from llm_providers import AnthropicProvider, OllamaProvider
from node import Node
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times


def _sanitize_model_name(model: str) -> str:
    """Turn a model string like 'qwen2.5:7b' into a filename-safe tag like 'qwen2_5_7b'."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')


def expand_algorithms(algorithms: list) -> list:
    """Expand LLM entries that carry a `models:` list into one entry per model.

    An entry like:
        {name: llm, type: llm, provider: ollama, models: [qwen2.5:7b, gemma2:9b]}
    becomes two entries with distinct names (llm_ollama_qwen2_5_7b, llm_ollama_gemma2_9b)
    so each model gets its own metrics CSV in the output directory.

    When `action_mode` is set on the entry to anything other than 'both', it is
    included in the generated name so e.g. running both vpa and hpa side-by-side
    produces distinct output files.
    """
    expanded = []
    for alg in algorithms:
        if alg.get('type') == 'llm' and 'models' in alg:
            base_name = alg.get('name', 'llm')
            provider = alg.get('provider', 'ollama')
            mode = alg.get('action_mode', 'both')
            mode_tag = f"_{mode}" if mode in ("vpa", "hpa") else ""
            for model in alg['models']:
                entry = {k: v for k, v in alg.items() if k != 'models'}
                entry['model'] = model
                entry['name'] = f"{base_name}{mode_tag}_{provider}_{_sanitize_model_name(model)}"
                expanded.append(entry)
        else:
            expanded.append(alg)
    return expanded


def load_benchmark_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_yaml(path: str):
    """Apply a YAML and wait for what it declared to actually be ready.

    Deployments: wait for rollout to complete.
    HorizontalPodAutoscalers: wait until metrics-server data has populated
    `status.currentMetrics` (otherwise the HPA reports <unknown> and won't act
    during the first benchmark steps, biasing measurements).
    """
    print(f"Applying {path}")
    subprocess.run(['microk8s', 'kubectl', 'apply', '-f', path], check=True)

    with open(path) as f:
        docs = [d for d in yaml.safe_load_all(f) if d]
    deployments = [d['metadata']['name'] for d in docs if d.get('kind') == 'Deployment']
    hpas = [d['metadata']['name'] for d in docs if d.get('kind') == 'HorizontalPodAutoscaler']

    for dep in deployments:
        print(f"  Waiting for deployment/{dep} rollout")
        subprocess.run(
            ['microk8s', 'kubectl', 'rollout', 'status',
             f'deployment/{dep}', '--timeout=180s'],
            check=True,
        )

    for hpa in hpas:
        print(f"  Waiting for hpa/{hpa} to populate metrics")
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            r = subprocess.run(
                ['microk8s', 'kubectl', 'get', 'hpa', hpa,
                 '-o', 'jsonpath={.status.currentMetrics[0].resource.current.averageUtilization}'],
                capture_output=True, text=True,
            )
            if r.stdout.strip().isdigit():
                print(f"    hpa/{hpa} ready (utilization={r.stdout.strip()}%)")
                ready = True
                break
            time.sleep(3)
        if not ready:
            print(f"    WARNING: hpa/{hpa} still reports <unknown> after 120s — "
                  f"check `microk8s enable metrics-server` and `kubectl describe hpa {hpa}`")


def delete_yaml(path: str):
    print(f"Deleting {path}")
    subprocess.run(
        ['microk8s', 'kubectl', 'delete', '-f', path, '--ignore-not-found'],
        check=False,
    )


def reset_cluster(n_agents: int = 3, cpu: int = 100, replicas: int = 1, settle_timeout: float = 60.0):
    """Reset all pods to a deterministic baseline CPU and replica count.

    Earlier algorithms (especially RL ones) can leave pods at low cpu_limit and
    elevated replica counts. Without a per-iteration reset, downstream
    algorithms inherit that state and the threshold agent in particular gets
    stuck (cpu observation stays sub-threshold under nginx-shed traffic).
    """
    from deployment_controller import (
        scale_deployment, get_deployment_replicas, get_deployment_pod_names)

    # Skip the scale call when the deployment is already at the target replica
    # count — avoids needless K8s API churn and graceful-shutdown 503 windows
    # when the previous iter ended at 1 replica anyway.
    from utils import get_deployment_name

    needs_settle = []
    for i in range(1, n_agents + 1):
        dep = get_deployment_name(i)
        current = get_deployment_replicas(dep, debug=True)
        pods = get_deployment_pod_names(dep, debug=True)
        if current != replicas or len(pods) != replicas:
            scale_deployment(dep, replicas, debug=True)
            needs_settle.append(i)

    if needs_settle:
        deadline = time.time() + settle_timeout
        pending = list(needs_settle)
        while pending and time.time() < deadline:
            still = []
            for i in pending:
                dep = get_deployment_name(i)
                r = get_deployment_replicas(dep, debug=True)
                pods = get_deployment_pod_names(dep, debug=True)
                if r != replicas or len(pods) != replicas:
                    still.append(i)
            pending = still
            if pending:
                time.sleep(2)
        if pending:
            print(f"WARNING: deployments {pending} did not settle to {replicas} replicas in {settle_timeout}s")

    set_container_cpu_values(cpus=cpu, n=n_agents)
    time.sleep(2)


def create_agent(algorithm_config: dict, agent_id: int, resources: int, pod_name: str = None):
    """Create an agent based on algorithm configuration."""
    name = algorithm_config['name']
    alg_type = algorithm_config['type']

    if alg_type == 'heuristic':
        if name == 'threshold':
            agent = ThresholdJointAgent(pod_name=pod_name)
            env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
        elif name == 'k8s_hpa':
            env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
            agent = KubernetesHPABaseline(env=env)
        elif name == 'k8s_vpa':
            # In-place VPA heuristic: same actuator/cadence as the RL agents,
            # VPA recommender decision logic. Does NOT need the real VPA operator.
            env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
            agent = InPlaceVPAAgent(pod_name=pod_name)
        elif name == 'k8s_vpa_real':
            # Passive baseline driven by the real (restart-based) VPA operator.
            # Requires the VPA CRD applied via apply_yaml in benchmark_config.yaml.
            env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
            agent = KubernetesVPABaseline(env=env)
        return env, agent

    if alg_type == 'llm':
        llm_cfg = load_llm_config()
        agent_cfg = llm_cfg.get('agent', {})
        deployment_name = agent_cfg.get('deployment_name', 'localization-api')
        history_window = agent_cfg.get('history_window', 5)
        inference_mode = agent_cfg.get('inference_mode', 'function_calling')
        # action_mode: 'vpa' (CPU only), 'hpa' (replicas only), or 'both'.
        # Per-algorithm override wins over the llm_config.yaml default.
        action_mode = algorithm_config.get('action_mode', agent_cfg.get('action_mode', 'both'))

        provider_name = algorithm_config.get('provider', 'ollama')
        model = algorithm_config.get('model')
        if provider_name == 'anthropic':
            anth_cfg = llm_cfg.get('anthropic', {})
            provider = AnthropicProvider(
                model=model or anth_cfg.get('model', 'claude-sonnet-4-20250514'),
                max_tokens=anth_cfg.get('max_tokens', 512),
            )
        elif provider_name == 'ollama':
            oll_cfg = llm_cfg.get('ollama', {})
            provider = OllamaProvider(
                model=model or oll_cfg.get('model', 'mistral:latest'),
                base_url=algorithm_config.get('base_url', oll_cfg.get('base_url', 'http://localhost:11434')),
                max_tokens=oll_cfg.get('max_tokens', 512),
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name!r} (expected 'ollama' or 'anthropic')")

        from utils import get_deployment_name
        env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
        env.MAX_CPU_LIMIT = resources
        agent = LLMAgent(
            provider,
            pod_name=pod_name or get_deployment_name(agent_id),
            deployment_name=deployment_name,
            history_window=history_window,
            inference_mode=inference_mode,
            max_cpu=resources,
            action_mode=action_mode,
        )
        return env, agent

    # RL agents
    model_path = algorithm_config.get('model_path', None)
    env, agent = initialize_agent(
        id=agent_id,
        resources=resources,
        tl_agent=0,
        model=model_path,
        algorithm=name,
        independent=False,
        priority=1.0,
        pod_name=pod_name,
    )
    return env, agent


def run_single_benchmark(algorithm_config: dict, load_patterns: list, config: dict,
                         iteration: int, output_dir: str, n_agents: int = 3):
    """Run a single benchmark for one algorithm through all load phases."""
    alg_name = algorithm_config['name']
    resources = config['resources']
    action_interval = config['action_interval']

    print(f"\n--- Running {alg_name} (iteration {iteration}) ---")
    LLMAgent.log_section(f"Algorithm: {alg_name}  (iteration {iteration})")

    # Create agents and environments
    envs, agents = [], []
    for i in range(1, n_agents + 1):
        env, agent = create_agent(algorithm_config, i, resources)
        envs.append(env)
        agents.append(agent)

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))]
    set_available_resource(envs, resources)

    # Prepare metrics file
    metrics_file = os.path.join(output_dir, f"{alg_name}_iter{iteration}_metrics.csv")
    fieldnames = ['step', 'phase', 'rps', 'agent_id', 'cpu_limit', 'cpu_usage', 'cpu_percentage',
                  'response_time', 'action', 'decision_latency_ms', 'replica_count',
                  'available_resources', 'reward']
    rows = []

    diag_path = os.path.join(output_dir, f"{alg_name}_iter{iteration}_diag.log")
    diag_log = open(diag_path, "w", buffering=1)
    diag_log.write(f"# {alg_name} iter={iteration}  started={datetime.now().isoformat()}\n")
    diag_log.write(f"# columns: ts step phase phase_step rt_per_agent step_dur_ms cb_per_host exception\n")

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"

    global_step = 0
    states = [np.array(env.reset()).flatten() for env in envs]

    for phase in load_patterns:
        phase_name = phase['name']
        phase_rps = phase['rps']
        phase_steps = phase['duration_steps']

        states = [np.array(env.reset()).flatten() for env in envs]

        print(f"  Phase: {phase_name} ({phase_rps} RPS, {phase_steps} steps)")
        LLMAgent.log_section(f"  Phase: {phase_name}  {phase_rps} RPS  {phase_steps} steps")

        # Start load generator
        command = ['python', 'src/spam_cluster.py', '--users', str(phase_rps),
                   '--interval', '1000', '--variable', '--all',
                   '--n_services', str(n_agents)]
        spam_process = subprocess.Popen(command)

        for step in range(phase_steps):
            step_start = time.time()
            rts = []
            exc_summary = "-"

            try:
                # Get response times
                for env in envs:
                    rt_list = get_response_times(1, f'{url}/api{env.id}/predict')
                    rt = np.mean([r if r is not None else 2.0 for r in rt_list])
                    rts.append(rt)

                # Agent actions
                for i, agent in enumerate(agents):
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                    if hasattr(agent, 'observe_response_time'):
                        agent.observe_response_time(rts[i])

                    decision_start = time.time()
                    action = agent.get_action(states[i])
                    decision_latency = (time.time() - decision_start) * 1000

                    state, reward, done, _ = envs[i].step(action, 2)
                    set_available_resource(envs, resources)

                    states[i] = np.array(state).flatten()

                    # Record metrics
                    rows.append({
                        'step': global_step,
                        'phase': phase_name,
                        'rps': phase_rps,
                        'agent_id': i,
                        'cpu_limit': envs[i].ALLOCATED,
                        'cpu_usage': envs[i].last_cpu_percentage,
                        'cpu_percentage': envs[i].last_cpu_percentage,
                        'response_time': rts[i],
                        'action': str(action),
                        'decision_latency_ms': decision_latency,
                        'replica_count': getattr(envs[i], 'current_replicas', 1),
                        'available_resources': envs[i].AVAILABLE,
                        'reward': reward,
                    })
            except Exception as e:
                exc_summary = f"{type(e).__name__}:{str(e)[:120]}"
                print(f"    Step {step} error (skipping): {exc_summary}")
                diag_log.write(f"# TRACEBACK step={global_step}\n{traceback.format_exc()}\n")

            step_dur_ms = (time.time() - step_start) * 1000
            cb_parts = []
            for ca_ip, cb in Node._circuit.items():
                cb_parts.append(f"{ca_ip}:{'OK' if cb['available'] else 'OPEN'}/{cb['fails']}")
            rt_str = ",".join(f"{x:.3f}" for x in rts) if rts else ""
            diag_log.write(
                f"{datetime.now():%H:%M:%S.%f}  step={global_step:>4}  phase={phase_name:<6}  "
                f"phase_step={step:>3}  rt=[{rt_str}]  step_dur={step_dur_ms:>6.0f}ms  "
                f"cb=[{' '.join(cb_parts)}]  exc={exc_summary}\n"
            )

            global_step += 1

            elapsed = time.time() - step_start
            if elapsed < action_interval:
                time.sleep(action_interval - elapsed)

        spam_process.terminate()
        time.sleep(2)

    # Write metrics
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved metrics to {metrics_file}")
    diag_log.write(f"# finished={datetime.now().isoformat()}\n")
    diag_log.close()
    print(f"  Saved diagnostics to {diag_path}")

    # Collect LLM-specific metrics
    for agent in agents:
        if hasattr(agent, 'get_metrics'):
            llm_metrics = agent.get_metrics()
            llm_file = os.path.join(output_dir, f"{alg_name}_iter{iteration}_llm_metrics.yaml")
            with open(llm_file, 'w') as f:
                yaml.dump(llm_metrics, f)


def main():
    parser = argparse.ArgumentParser(description="MARLISE Benchmark Runner")
    parser.add_argument('--config', type=str, default='configs/benchmark_config.yaml')
    parser.add_argument('--scenario', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                        help="Specific algorithms to run (default: all)")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    scenario = config['scenarios'][args.scenario]
    n_agents = scenario['service_count']
    config['resources'] = scenario['total_resources']

    output_dir = os.path.join(config['output_dir'], args.scenario)
    os.makedirs(output_dir, exist_ok=True)

    algorithms = config['algorithms']
    if args.algorithms:
        algorithms = [a for a in algorithms if a['name'] in args.algorithms]
    # Expand any LLM entry with `models:` into one entry per model
    algorithms = expand_algorithms(algorithms)
    # Algorithms that need an extra K8s resource applied (e.g. k8s_hpa needs an
    # HPA object) are run last, and their resource is deleted after, so they
    # don't contaminate measurements for the other algorithms.
    algorithms.sort(key=lambda a: 1 if a.get('apply_yaml') else 0)

    iterations = config['iterations']
    load_patterns = config['load_patterns']

    print(f"Benchmark: {args.scenario} scenario, {n_agents} services, {iterations} iterations")
    print(f"Algorithms: {[a['name'] for a in algorithms]}")
    print(f"Load phases: {[p['name'] for p in load_patterns]}")

    print("Resetting cluster to baseline (once, before all runs)...")
    reset_cluster(n_agents)

    for alg_config in algorithms:
        extra_yaml = alg_config.get('apply_yaml')
        if extra_yaml:
            apply_yaml(extra_yaml)
        try:
            for iteration in range(iterations):
                print(f"Resetting cluster before {alg_config['name']} iter {iteration}...")
                reset_cluster(n_agents)
                run_single_benchmark(
                    alg_config, load_patterns, config, iteration, output_dir, n_agents
                )
        finally:
            if extra_yaml:
                delete_yaml(extra_yaml)

    print(f"\nBenchmark complete. Results in {output_dir}/")


if __name__ == '__main__':
    main()
