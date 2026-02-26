"""
Unified Benchmark Runner for MARLISE.

Runs all configured algorithms through identical load scenarios and collects metrics.
Usage: python scripts/eval/benchmark_runner.py [--config configs/benchmark_config.yaml] [--scenario small]
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from envs import (JointContinuousElasticityEnv, JointDiscreteElasticityEnv,
                  set_available_resource, set_other_utilization, set_other_priorities)
from infer import initialize_agent
from heuristic_agent import ThresholdJointAgent, KubernetesHPABaseline
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times


def load_benchmark_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def reset_cluster(n_agents: int = 3, cpu: int = 100):
    """Reset all pods to baseline CPU."""
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
            agent = KubernetesHPABaseline()
            env = JointContinuousElasticityEnv(agent_id, pod_name=pod_name)
        return env, agent

    # RL and LLM agents
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

    # Reset cluster
    reset_cluster(n_agents)

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

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"

    global_step = 0
    states = [np.array(env.reset()).flatten() for env in envs]

    for phase in load_patterns:
        phase_name = phase['name']
        phase_rps = phase['rps']
        phase_steps = phase['duration_steps']

        print(f"  Phase: {phase_name} ({phase_rps} RPS, {phase_steps} steps)")

        # Start load generator
        command = ['python', 'src/spam_cluster.py', '--users', str(phase_rps),
                   '--interval', '1000', '--variable', '--all']
        spam_process = subprocess.Popen(command)

        for step in range(phase_steps):
            step_start = time.time()

            # Get response times
            rts = []
            for env in envs:
                rt_list = get_response_times(1, f'{url}/api{env.id}/predict')
                rt = np.mean([r if r is not None else 2.0 for r in rt_list])
                rts.append(rt)

            # Agent actions
            for i, agent in enumerate(agents):
                set_other_utilization(envs[i], other_envs[i])
                set_other_priorities(envs[i], other_envs[i])

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

    # Collect LLM-specific metrics
    for agent in agents:
        if hasattr(agent, 'get_metrics'):
            llm_metrics = agent.get_metrics()
            llm_file = os.path.join(output_dir, f"{alg_name}_iter{iteration}_llm_metrics.yaml")
            with open(llm_file, 'w') as f:
                yaml.dump(llm_metrics, f)

    # Reset after benchmark
    reset_cluster(n_agents)


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

    iterations = config['iterations']
    load_patterns = config['load_patterns']

    print(f"Benchmark: {args.scenario} scenario, {n_agents} services, {iterations} iterations")
    print(f"Algorithms: {[a['name'] for a in algorithms]}")
    print(f"Load phases: {[p['name'] for p in load_patterns]}")

    for alg_config in algorithms:
        for iteration in range(iterations):
            run_single_benchmark(
                alg_config, load_patterns, config, iteration, output_dir, n_agents
            )

    print(f"\nBenchmark complete. Results in {output_dir}/")


if __name__ == '__main__':
    main()
