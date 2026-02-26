"""
Chaos/Reliability Testing for MARLISE.

Simulates failures to measure graceful degradation of each agent type:
- cAdvisor metric blackout
- K8s API timeout
- LLM provider unavailability

Usage: python scripts/eval/chaos_test.py [--algorithm joint_ppo] [--chaos cadvisor]
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from unittest.mock import patch, MagicMock

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from envs import JointContinuousElasticityEnv, set_available_resource, set_other_utilization, set_other_priorities
from infer import initialize_agent
from heuristic_agent import ThresholdJointAgent
from pod_controller import set_container_cpu_values, get_loadbalancer_external_port
from spam_cluster import get_response_times


class ChaosSimulator:
    """Simulates various failure modes during autoscaling inference."""

    def __init__(self, chaos_type: str, start_step: int = 20, duration_steps: int = 20):
        self.chaos_type = chaos_type
        self.start_step = start_step
        self.duration_steps = duration_steps
        self.end_step = start_step + duration_steps
        self._original_funcs = {}
        self._active = False

    def should_activate(self, step: int) -> bool:
        return self.start_step <= step < self.end_step

    def activate(self):
        """Inject the failure."""
        if self._active:
            return
        self._active = True
        print(f"  [CHAOS] Activating {self.chaos_type} failure")

        if self.chaos_type == 'cadvisor':
            self._inject_cadvisor_failure()
        elif self.chaos_type == 'k8s_api':
            self._inject_k8s_timeout()
        elif self.chaos_type == 'llm_provider':
            self._inject_llm_failure()

    def deactivate(self):
        """Remove the failure."""
        if not self._active:
            return
        self._active = False
        print(f"  [CHAOS] Deactivating {self.chaos_type} failure")

        for name, func in self._original_funcs.items():
            # Restore is handled by context manager or manual reassignment
            pass
        self._original_funcs = {}

    def _inject_cadvisor_failure(self):
        """Make cAdvisor return stale/zero metrics."""
        import node as node_module
        self._original_funcs['get_container_usage'] = node_module.Node.get_container_usage

        def failing_get_usage(self_node, container_id):
            # Return zeros - simulating metric blackout
            return (0, 0, 0), (0, 0, 0), (0, 0), False

        node_module.Node.get_container_usage = failing_get_usage

    def _inject_k8s_timeout(self):
        """Make K8s API calls timeout."""
        import pod_controller
        self._original_funcs['patch_pod'] = pod_controller.patch_pod

        def slow_patch_pod(*args, **kwargs):
            time.sleep(5)  # Simulate timeout
            raise TimeoutError("K8s API timeout (simulated)")

        pod_controller.patch_pod = slow_patch_pod

    def _inject_llm_failure(self):
        """Make LLM provider return errors."""
        import llm_providers
        self._original_funcs['anthropic_query'] = getattr(
            llm_providers.AnthropicProvider, 'query', None)

        original_query = llm_providers.AnthropicProvider.query

        def failing_query(self_provider, messages, tools):
            raise ConnectionError("LLM provider unavailable (simulated)")

        llm_providers.AnthropicProvider.query = failing_query

    def restore(self):
        """Restore all original functions."""
        if self.chaos_type == 'cadvisor' and 'get_container_usage' in self._original_funcs:
            import node as node_module
            node_module.Node.get_container_usage = self._original_funcs['get_container_usage']
        elif self.chaos_type == 'k8s_api' and 'patch_pod' in self._original_funcs:
            import pod_controller
            pod_controller.patch_pod = self._original_funcs['patch_pod']
        elif self.chaos_type == 'llm_provider' and 'anthropic_query' in self._original_funcs:
            import llm_providers
            if self._original_funcs['anthropic_query']:
                llm_providers.AnthropicProvider.query = self._original_funcs['anthropic_query']
        self._original_funcs = {}
        self._active = False


def run_chaos_test(algorithm: str, chaos_type: str, n_agents: int = 3,
                    resources: int = 1000, steps: int = 60, model_path: str = None):
    """Run a chaos test for a single algorithm and failure type."""
    print(f"\n  Testing {algorithm} with {chaos_type} failure...")

    chaos = ChaosSimulator(chaos_type, start_step=20, duration_steps=20)

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

    command = ['python', 'src/spam_cluster.py', '--users', '30', '--interval', '1000', '--variable', '--all']
    spam_process = subprocess.Popen(command)

    metrics = {
        'pre_failure': {'rewards': [], 'rts': [], 'errors': 0},
        'during_failure': {'rewards': [], 'rts': [], 'errors': 0},
        'post_failure': {'rewards': [], 'rts': [], 'errors': 0},
    }

    crashed = False

    for step in range(steps):
        # Determine phase
        if step < chaos.start_step:
            phase = 'pre_failure'
        elif step < chaos.end_step:
            phase = 'during_failure'
            if not chaos._active:
                chaos.activate()
        else:
            phase = 'post_failure'
            if chaos._active:
                chaos.restore()

        step_start = time.time()

        # Get response times
        rts = []
        for env in envs:
            try:
                rt_list = get_response_times(1, f'{url}/api{env.id}/predict')
                rt = np.mean([r if r is not None else 2.0 for r in rt_list])
            except Exception:
                rt = 2.0
            rts.append(rt)

        # Agent decisions
        step_rewards = []
        for i, agent in enumerate(agents):
            try:
                if i < len(other_envs):
                    set_other_utilization(envs[i], other_envs[i])
                    set_other_priorities(envs[i], other_envs[i])

                action = agent.get_action(states[i])
                state, reward, done, _ = envs[i].step(action, 2)
                set_available_resource(envs, resources)
                states[i] = np.array(state).flatten()
                step_rewards.append(reward)
            except Exception as e:
                metrics[phase]['errors'] += 1
                step_rewards.append(0)

        metrics[phase]['rewards'].extend(step_rewards)
        metrics[phase]['rts'].extend(rts)

        elapsed = time.time() - step_start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    spam_process.terminate()
    chaos.restore()

    # Compile results
    result = {
        'algorithm': algorithm,
        'chaos_type': chaos_type,
        'crashed': crashed,
    }
    for phase_name in ['pre_failure', 'during_failure', 'post_failure']:
        m = metrics[phase_name]
        result[f'{phase_name}_mean_reward'] = np.mean(m['rewards']) if m['rewards'] else 0
        result[f'{phase_name}_mean_rt'] = np.mean(m['rts']) if m['rts'] else 0
        result[f'{phase_name}_errors'] = m['errors']

    # Degradation metrics
    pre_reward = result['pre_failure_mean_reward']
    during_reward = result['during_failure_mean_reward']
    post_reward = result['post_failure_mean_reward']

    result['reward_degradation_pct'] = ((pre_reward - during_reward) / abs(pre_reward) * 100) if pre_reward != 0 else 0
    result['recovery_pct'] = ((post_reward / pre_reward) * 100) if pre_reward != 0 else 0

    return result


def main():
    parser = argparse.ArgumentParser(description="MARLISE Chaos/Reliability Testing")
    parser.add_argument('--config', type=str, default='configs/benchmark_config.yaml')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['joint_ppo', 'threshold', 'llm_claude'])
    parser.add_argument('--chaos_types', type=str, nargs='+',
                        default=['cadvisor', 'k8s_api', 'llm_provider'])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--output', type=str, default='results/chaos')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    results = []
    for algorithm in args.algorithms:
        # Find model path
        model_path = None
        for alg_config in config.get('algorithms', []):
            if alg_config['name'] == algorithm:
                model_path = alg_config.get('model_path')
                break

        for chaos_type in args.chaos_types:
            # Skip LLM-specific chaos for non-LLM agents
            if chaos_type == 'llm_provider' and 'llm' not in algorithm:
                continue

            try:
                result = run_chaos_test(
                    algorithm=algorithm,
                    chaos_type=chaos_type,
                    steps=args.steps,
                    model_path=model_path,
                )
                results.append(result)

                print(f"    {algorithm} + {chaos_type}: "
                      f"degradation={result['reward_degradation_pct']:.1f}%, "
                      f"recovery={result['recovery_pct']:.1f}%, "
                      f"errors={result['during_failure_errors']}")
            except Exception as e:
                print(f"    {algorithm} + {chaos_type}: CRASHED - {e}")
                results.append({
                    'algorithm': algorithm,
                    'chaos_type': chaos_type,
                    'crashed': True,
                })

            # Reset
            set_container_cpu_values(cpus=100)
            time.sleep(3)

    # Save results
    if results:
        output_file = os.path.join(args.output, 'chaos_results.csv')
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")

        # Summary
        print("\n=== Chaos Test Summary ===")
        print(f"{'Algorithm':<15} {'Chaos':<15} {'Crashed':>8} {'Degrade%':>10} {'Recover%':>10} {'Errors':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.get('algorithm', '?'):<15} "
                  f"{r.get('chaos_type', '?'):<15} "
                  f"{str(r.get('crashed', '?')):>8} "
                  f"{r.get('reward_degradation_pct', 0):>10.1f} "
                  f"{r.get('recovery_pct', 0):>10.1f} "
                  f"{r.get('during_failure_errors', 0):>8}")


if __name__ == '__main__':
    main()
