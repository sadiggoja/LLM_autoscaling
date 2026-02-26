"""
Scalability Evaluation for MARLISE.

Progressive test: 3 -> 10 -> 20 services.
Measures decision quality degradation, latency scaling, and LLM context window impact.
Usage: python scripts/eval/scalability_eval.py [--config configs/benchmark_config.yaml]
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from envs import JointContinuousElasticityEnv, set_available_resource, set_other_utilization, set_other_priorities
from infer import initialize_agent
from heuristic_agent import ThresholdJointAgent
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times


def run_scalability_test(algorithm: str, n_agents: int, resources: int, steps: int = 60,
                          action_interval: float = 1.0, model_path: str = None):
    """Run a single scalability test and return metrics."""
    print(f"  Testing {algorithm} with {n_agents} agents, {resources} resources...")

    # Create agents
    envs, agents = [], []
    for i in range(1, n_agents + 1):
        if algorithm == 'threshold':
            env = JointContinuousElasticityEnv(i)
            agent = ThresholdJointAgent()
        else:
            env, agent = initialize_agent(
                id=i, resources=resources, tl_agent=0,
                model=model_path, algorithm=algorithm,
                independent=False, priority=1.0
            )
        envs.append(env)
        agents.append(agent)

    other_envs = [[env for env in envs if env != envs[i]] for i in range(len(envs))]
    set_available_resource(envs, resources)

    states = [np.array(env.reset()).flatten() for env in envs]

    url = f"http://localhost:{get_loadbalancer_external_port(service_name='ingress-nginx-controller')}"

    # Start load
    command = ['python', 'src/spam_cluster.py', '--users', '30', '--interval', '1000', '--variable', '--all']
    spam_process = subprocess.Popen(command)

    step_latencies = []
    decision_latencies = []
    rewards = []

    for step in range(steps):
        step_start = time.time()

        # Get response times
        rts = []
        for env in envs:
            rt_list = get_response_times(1, f'{url}/api{env.id}/predict')
            rt = np.mean([r if r is not None else 2.0 for r in rt_list])
            rts.append(rt)

        # Agent decisions
        step_decision_latencies = []
        step_rewards = []
        for i, agent in enumerate(agents):
            if i < len(other_envs):
                set_other_utilization(envs[i], other_envs[i])
                set_other_priorities(envs[i], other_envs[i])

            decision_start = time.time()
            action = agent.get_action(states[i])
            step_decision_latencies.append((time.time() - decision_start) * 1000)

            state, reward, done, _ = envs[i].step(action, 2)
            set_available_resource(envs, resources)
            states[i] = np.array(state).flatten()
            step_rewards.append(reward)

        step_latencies.append(np.mean(rts))
        decision_latencies.extend(step_decision_latencies)
        rewards.extend(step_rewards)

        elapsed = time.time() - step_start
        if elapsed < action_interval:
            time.sleep(action_interval - elapsed)

    spam_process.terminate()

    # Collect LLM-specific metrics
    llm_metrics = {}
    for agent in agents:
        if hasattr(agent, 'get_metrics'):
            llm_metrics = agent.get_metrics()
            break

    return {
        'algorithm': algorithm,
        'n_agents': n_agents,
        'mean_response_time': np.mean(step_latencies),
        'p95_response_time': np.percentile(step_latencies, 95) if step_latencies else 0,
        'mean_decision_latency_ms': np.mean(decision_latencies),
        'p95_decision_latency_ms': np.percentile(decision_latencies, 95) if decision_latencies else 0,
        'max_decision_latency_ms': np.max(decision_latencies) if decision_latencies else 0,
        'mean_reward': np.mean(rewards),
        'total_reward': np.sum(rewards),
        'steps_within_interval': sum(1 for d in decision_latencies if d < action_interval * 1000) / max(len(decision_latencies), 1) * 100,
        **llm_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="MARLISE Scalability Evaluation")
    parser.add_argument('--config', type=str, default='configs/benchmark_config.yaml')
    parser.add_argument('--scales', type=int, nargs='+', default=[3, 10, 20])
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['joint_ppo', 'joint_ddpg', 'threshold', 'llm_claude'])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--output', type=str, default='results/scalability')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    results = []
    for scale in args.scales:
        resources = scale * 333  # Roughly 1000 per 3 services
        print(f"\n=== Scale: {scale} services, {resources} total resources ===")

        for algorithm in args.algorithms:
            # Find model path from config
            model_path = None
            for alg_config in config.get('algorithms', []):
                if alg_config['name'] == algorithm:
                    model_path = alg_config.get('model_path')
                    break

            try:
                result = run_scalability_test(
                    algorithm=algorithm,
                    n_agents=scale,
                    resources=resources,
                    steps=args.steps,
                    model_path=model_path,
                )
                results.append(result)
                print(f"    {algorithm}: decision_lat={result['mean_decision_latency_ms']:.1f}ms, "
                      f"rt={result['mean_response_time']:.4f}s, reward={result['mean_reward']:.3f}")
            except Exception as e:
                print(f"    {algorithm}: FAILED - {e}")

            # Reset between tests
            set_container_cpu_values(cpus=100, n=scale)
            time.sleep(3)

    # Save results
    if results:
        output_file = os.path.join(args.output, 'scalability_results.csv')
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")

        # Print summary
        print("\n=== Scalability Summary ===")
        print(f"{'Algorithm':<15} {'Scale':>6} {'Lat(ms)':>10} {'RT(s)':>10} {'In-time%':>10}")
        print("-" * 55)
        for r in results:
            print(f"{r['algorithm']:<15} {r['n_agents']:>6} "
                  f"{r['mean_decision_latency_ms']:>10.1f} "
                  f"{r['mean_response_time']:>10.4f} "
                  f"{r['steps_within_interval']:>10.1f}")


if __name__ == '__main__':
    main()
