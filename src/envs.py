import numpy as np
from gymnasium import Env, spaces

from pod_controller import patch_pod
from deployment_controller import scale_deployment, get_deployment_replicas, get_deployment_avg_cpu, resolve_pod_name
from utils import init_nodes, load_config


class BaseElasticityEnv(Env):
    '''
    BaseElasticityEnv is a custom environment for managing the elasticity of containerized applications.
    It can be inherited to create different types of environments with different action spaces and configurations.

    By default it has history of 6 states and uses the reward function 2, which empirically has shown to work the best.
    Each microservice has its own enivorment, and the environment is created with the pod_name of the microservice.
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__()
        config = load_config()
        # print(f"Loaded config: {config}")

        self.container_name = config['target_container_name']
        self.app_label = config['target_app_label']
        if pod_name:
            self.pod_name = pod_name
        else:
            self.pod_name = f'{self.container_name}{id}'
        self.deployment_name = self.pod_name
        self.pod_name = resolve_pod_name(self.deployment_name, debug=config['debug_deployment'])

        self.MAX_CPU_LIMIT = config['max_cpu']  # Dynamic, can change from outer scope
        self.MIN_CPU_LIMIT = config['min_cpu']
        self.INCREMENT = config['discrete_increment']  # Used for discrete action space
        self.scale_action = config['scale_action']  # Used for continuous action space

        self.AVAILABLE = 1000

        self.independent_state = independent_state

        self.debug_deployment = config['debug_deployment']
        nodes = init_nodes(debug=self.debug_deployment, custom_label=self.app_label)

        self.node, self.container_id = None, None
        self.id = id
        for node in nodes:
            for container_id, (pod_name, container_name, pod_ip) in list(node.get_containers().items()):
                if pod_name == f'{self.pod_name}':
                    # grab node object and container_id for agent
                    self.container_id = container_id
                    self.node = node
                    break
        if not self.node or not self.container_id:
            raise ValueError(f"Pod {self.pod_name} not found in the cluster")
        # print(f"Initialized Env {self.id} with {self.node}")

        # Initialized allocated resources of how much current does the pod have
        (cpu_limit, _, _), (_, _, _), (_, _), _ = self.node.get_container_usage(self.container_id)
        self.ALLOCATED = cpu_limit

        self.other_util = 0.0
        self.STATE_LENTGH = config['state_history']
        if self.independent_state:
            self.states_fifo = [[0, 0, 0, 0, 0] for _ in range(self.STATE_LENTGH)]
        else:
            self.states_fifo = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.STATE_LENTGH)]

        self.last_cpu_percentage = 0
        self.previous_cpu_percentage = 0
        self.priority = 1.0
        self.other_priorities = 0.0

        self.state = self.get_current_usage()
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1),
                                            shape=(self.STATE_LENTGH * len(self.state[0]),))

        self.steps = 0
        self.MAX_STEPS = config['max_steps']

        # 30% - 60%
        self.UPPER_CPU = config['upper_cpu']
        self.LOWER_CPU = config['lower_cpu']

        # Field for evaluation of how much resources have been changed
        self.cummulative_delta = 0

        # Patch debouncing: accumulate desired CPU deltas across steps and only call
        # patch_pod when they cross MIN_PATCH_DELTA. Reduces endpoint-slice churn
        # that triggers nginx-ingress 503 events.
        self.MIN_PATCH_DELTA = config.get('min_patch_delta', 50)
        self._pending_delta = 0

    def _apply_delta(self, scale_action):
        """Accumulate a desired CPU delta (millicores). Patch the pod only when
        cumulative pending delta is >= MIN_PATCH_DELTA. Honours AVAILABLE and
        MIN_CPU_LIMIT limits. Returns the actual delta applied (0 if accumulating)."""
        self._pending_delta += scale_action
        if abs(self._pending_delta) < self.MIN_PATCH_DELTA:
            return 0
        # Clip pending delta against headroom and floor
        proposed = self.ALLOCATED + self._pending_delta
        if self._pending_delta > 0:
            cap = min(self.ALLOCATED + max(self.AVAILABLE, 0), self.MAX_CPU_LIMIT)
            new_limit = int(max(min(proposed, cap), self.MIN_CPU_LIMIT))
        else:
            new_limit = int(max(proposed, self.MIN_CPU_LIMIT))
        applied = new_limit - self.ALLOCATED
        if new_limit != self.ALLOCATED:
            patch_pod(self.pod_name, cpu_request=f"{new_limit}m", cpu_limit=f"{new_limit}m",
                      container_name=self.container_name, debug=self.debug_deployment)
            self.ALLOCATED = new_limit
            self.cummulative_delta += abs(applied)
        self._pending_delta = 0
        return applied

    def norm_cpu(self, cpu_usage):
        return cpu_usage / self.MAX_CPU_LIMIT

    def get_current_usage(self):
        (cpu_limit, cpu, cpu_percentage), (_, _, _), (_, _), _ = self.node.get_container_usage(
            self.container_id)
        self.previous_cpu_percentage = self.last_cpu_percentage
        self.last_cpu_percentage = cpu_percentage
        n_cpu_limit, n_cpu = self.norm_cpu(cpu_limit), self.norm_cpu(cpu)

        available_normed = self.norm_cpu(self.AVAILABLE)
        if self.independent_state:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.priority]
        else:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.other_util / 100, self.priority,
                     self.other_priorities]

        self.states_fifo.append(state)
        self.states_fifo.pop(0)
        return self.states_fifo

    def reset(self):
        self.state = self.get_current_usage()
        self.steps = 0

        # Fill state with the last value
        self.state = [self.state[-1]] * self.STATE_LENTGH

        return self.state

    def set_last_limit(self):
        patch_pod(self.pod_name, cpu_request=f"{self.ALLOCATED}m", cpu_limit=f"{self.ALLOCATED}m",
                  container_name=self.container_name, debug=self.debug_deployment)

    def patch(self, limit):
        patch_pod(self.pod_name, cpu_request=f"{limit}m", cpu_limit=f"{limit}m", container_name=self.container_name,
                  debug=self.debug_deployment)
        self.ALLOCATED = limit

    def get_container_usage(self):
        return self.node.get_container_usage(self.container_id)

    def calculate_agent_reward(self, rf):
        reward = 0
        match rf:
            case 1:
                if self.last_cpu_percentage <= self.LOWER_CPU:
                    reward = -1
                return reward
            case 2 | 3 | 42:
                if self.LOWER_CPU <= self.last_cpu_percentage <= self.UPPER_CPU:
                    reward = 1 + (self.last_cpu_percentage / (self.UPPER_CPU - self.LOWER_CPU))
                elif self.last_cpu_percentage < self.LOWER_CPU:
                    # Last = current
                    delta = self.last_cpu_percentage - self.previous_cpu_percentage
                    reward = min(delta / 10, 1) if delta > 0 else max(delta / 10, -1)
                    # So, if from 10 to 15, reward is 0.5. and from 40 to 10, reward is -1
                # elif self.last_cpu_percentage > self.UPPER_CPU:
                #     curr_percentage = min(self.last_cpu_percentage, 100) # Can be higher than 100, so we avoid that
                #     reward = 1 - 2 * ((curr_percentage - self.UPPER_CPU) / (100 - self.UPPER_CPU))
                return reward
            case 4 | 5:
                if self.last_cpu_percentage < self.LOWER_CPU:
                    # Last = current
                    delta = self.last_cpu_percentage - self.previous_cpu_percentage
                    reward = min(delta / 10, 1) if delta > 0 else max(delta / 10, -1)
                    # So, if from 10 to 15, reward is 0.5. and from 40 to 10, reward is -1
                return reward
        return reward


class FiveDiscreteElasticityEnv(BaseElasticityEnv):
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Discrete(5)
        self.factors = [-1.0, -0.5, 0, 0.5, 1.0]
    
    def step(self, action, rf):
        factor = self.factors[action]
        if factor != 0:
            self.update_resources(factor)

        self.state = self.get_current_usage()

        reward = self.calculate_agent_reward(rf)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, {}

    def update_resources(self, factor):
        # updated_cpu_limit = int(max(min(self.ALLOCATED + factor * self.MAX_CPU_LIMIT, self.ALLOCATED 
                                        # + self.AVAILABLE), self.MIN_CPU_LIMIT))
        updated_cpu_limit = int(max(factor * self.MAX_CPU_LIMIT, self.MIN_CPU_LIMIT))
        if updated_cpu_limit - self.ALLOCATED > max(self.AVAILABLE, 0):
            updated_cpu_limit = int(self.ALLOCATED + self.AVAILABLE)

        if updated_cpu_limit != self.ALLOCATED:
            delta = abs(updated_cpu_limit - self.ALLOCATED)
            self.ALLOCATED = updated_cpu_limit
            patch_pod(self.pod_name, cpu_request=f"{updated_cpu_limit}m", cpu_limit=f"{updated_cpu_limit}m",
                      container_name=self.container_name, debug=self.debug_deployment)
            self.cummulative_delta += delta


class ElevenDiscrElasticityEnv(FiveDiscreteElasticityEnv):
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Discrete(11)
        self.factors = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        

class DiscreteElasticityEnv(BaseElasticityEnv):
    '''
    Discrete action space enviroment for a microservice.
    It works with 3 fixed increment actions: decrease resources, keep resources, increase resources.
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Discrete(3)

    def step(self, action, rf):
        if action == 0:
            self.decrease_resources()
        elif action == 2:
            self.increase_resources()

        self.state = self.get_current_usage()

        reward = self.calculate_agent_reward(rf)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0

    def increase_resources(self):
        self._apply_delta(self.INCREMENT)

    def decrease_resources(self):
        self._apply_delta(-self.INCREMENT)


class ContinuousElasticityEnv(BaseElasticityEnv):
    '''
    Continuous action space enviroment for a microservice.
    It works with a continuous action space, where the agent can increase or decrease the resources by a configurable scaling factor.
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action, rf):
        self.state = self.get_current_usage()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        scale_action = float(action[0]) * self.scale_action
        self._apply_delta(scale_action)

        reward = self.calculate_agent_reward(rf)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0

    def mimic_step(self):
        self.state = self.get_current_usage()

        reward = self.calculate_agent_reward(2)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, 0


class InstantContinuousElasticityEnv(BaseElasticityEnv):
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action, rf):
        self.state = self.get_current_usage()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scale_action = action[0] * self.MAX_CPU_LIMIT

        # if scale_action < self.ALLOCATED or (self.ALLOCATED - scale_action) <= max(self.AVAILABLE, 0):
        if (scale_action - self.ALLOCATED) <= max(self.AVAILABLE, 0):
            new_resource_limit = int(max(scale_action, self.MIN_CPU_LIMIT))
        else:
            new_resource_limit = int(max(self.ALLOCATED + self.AVAILABLE, self.MIN_CPU_LIMIT))
        
        if new_resource_limit != self.ALLOCATED:
            self.ALLOCATED = new_resource_limit
            patch_pod(
                self.pod_name,
                cpu_request=f"{new_resource_limit}m",
                cpu_limit=f"{new_resource_limit}m",
                container_name=self.container_name,
                debug=self.debug_deployment
            )
            self.cummulative_delta += abs(scale_action.item() - self.ALLOCATED)

        reward = self.calculate_agent_reward(rf)
        self.steps += 1
        done = self.steps >= self.MAX_STEPS
        return self.state, reward, done, 0


def set_available_resource(envs, initial_resources):
    max_group = initial_resources
    for env in envs:
        max_group -= env.ALLOCATED
    for env in envs:
        env.AVAILABLE = max_group


def set_other_utilization(env, other_envs):
    env.other_util = np.mean([o_env.last_cpu_percentage for o_env in other_envs]) if other_envs else 0.0


def set_other_priorities(env, other_envs):
    env.other_priorities = np.mean([o_env.priority for o_env in other_envs]) if other_envs else 0.0


class BaseJointElasticityEnv(BaseElasticityEnv):
    '''
    Base environment for joint HPA+VPA autoscaling.
    Extends the state with replica count, avg CPU across replicas, and request rate estimate.
    VPA executes every step; HPA only executes when cooldown has elapsed.
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        config = load_config()
        self.max_replicas = config.get('max_replicas', 5)
        self.min_replicas = config.get('min_replicas', 1)
        self.hpa_cooldown_steps = config.get('hpa_cooldown_steps', 30)
        from utils import get_deployment_name
        self.target_deployment = pod_name if pod_name else get_deployment_name(id, config)
        self.replica_change_penalty = config.get('replica_change_penalty', 0.3)

        # Initialize these BEFORE super().__init__() because it calls get_current_usage()
        # which is overridden and references these attributes
        self.last_network_rx = 0.0
        self.request_rate_estimate = 0.0
        self.current_replicas = 2  # Temporary default; updated below after super init
        self.hpa_cooldown_counter = 0
        self.previous_replicas = 2
        self.last_avg_cpu_replicas = 0.0  # Cached to avoid duplicate cAdvisor calls

        super().__init__(id, independent_state=independent_state, pod_name=pod_name)

        # Now that debug_deployment is set by parent, get the real replica count
        self.current_replicas = get_deployment_replicas(
            self.target_deployment, debug=self.debug_deployment) or 1
        self.previous_replicas = self.current_replicas

        # Override state FIFO with 10 dimensions (or 8 if independent)
        if self.independent_state:
            self.states_fifo = [[0] * 8 for _ in range(self.STATE_LENTGH)]
        else:
            self.states_fifo = [[0] * 10 for _ in range(self.STATE_LENTGH)]

        # Re-initialize observation space with new state dimensions
        self.state = self.get_current_usage()
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1),
                                            shape=(self.STATE_LENTGH * len(self.state[0]),))

    def get_current_usage(self):
        (cpu_limit, cpu, cpu_percentage), (_, _, _), (network_rx, _), _ = self.node.get_container_usage(
            self.container_id)
        self.previous_cpu_percentage = self.last_cpu_percentage
        self.last_cpu_percentage = cpu_percentage
        n_cpu_limit, n_cpu = self.norm_cpu(cpu_limit), self.norm_cpu(cpu)

        available_normed = self.norm_cpu(self.AVAILABLE)

        # Request rate estimate from network RX delta
        self.request_rate_estimate = max(0, network_rx - self.last_network_rx)
        self.last_network_rx = network_rx

        # Normalized replica info
        replica_norm = self.current_replicas / self.max_replicas
        avg_cpu_replicas_pct = get_deployment_avg_cpu(
            self.target_deployment, [self.node], debug=self.debug_deployment)
        self.last_avg_cpu_replicas = avg_cpu_replicas_pct  # Cache for calculate_joint_reward()
        avg_cpu_replicas = avg_cpu_replicas_pct / 100.0

        if self.independent_state:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.priority,
                     replica_norm, avg_cpu_replicas, min(self.request_rate_estimate, 1.0)]
        else:
            state = [n_cpu_limit, n_cpu, available_normed, cpu_percentage / 100, self.other_util / 100,
                     self.priority, self.other_priorities,
                     replica_norm, avg_cpu_replicas, min(self.request_rate_estimate, 1.0)]

        self.states_fifo.append(state)
        self.states_fifo.pop(0)
        return self.states_fifo

    def apply_hpa_action(self, hpa_action_value):
        '''Apply HPA action. Returns penalty if on cooldown.'''
        self.hpa_cooldown_counter += 1
        penalty = 0.0

        if hpa_action_value == 0:
            return penalty  # no-op

        if self.hpa_cooldown_counter < self.hpa_cooldown_steps:
            # Attempted HPA during cooldown
            penalty = -0.1
            return penalty

        # Cooldown elapsed, apply HPA
        self.hpa_cooldown_counter = 0
        self.previous_replicas = self.current_replicas
        new_replicas = int(np.clip(
            self.current_replicas + hpa_action_value,
            self.min_replicas, self.max_replicas))

        if new_replicas != self.current_replicas:
            scale_deployment(self.target_deployment, new_replicas, debug=self.debug_deployment)
            penalty = -self.replica_change_penalty * abs(new_replicas - self.current_replicas)
            self.current_replicas = new_replicas

        return penalty

    def calculate_joint_reward(self, rf, hpa_penalty):
        base_reward = self.calculate_agent_reward(rf)
        # Efficiency bonus: reward high per-replica utilization
        # Reuse avg CPU already fetched in get_current_usage() to avoid duplicate cAdvisor call
        if self.current_replicas > 0:
            avg_util = self.last_avg_cpu_replicas
            if self.LOWER_CPU <= avg_util <= self.UPPER_CPU:
                efficiency_bonus = 0.2
            else:
                efficiency_bonus = 0.0
        else:
            efficiency_bonus = 0.0

        return base_reward + hpa_penalty + efficiency_bonus


class JointDiscreteElasticityEnv(BaseJointElasticityEnv):
    '''
    Joint HPA+VPA with Discrete(9) action space: 3 VPA actions x 3 HPA actions.
    VPA: {decrease, maintain, increase} x HPA: {scale-down, maintain, scale-up}
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Discrete(9)

    def _decode_action(self, action):
        '''Decode flat action index into (vpa_action, hpa_action).
        VPA: 0=decrease, 1=maintain, 2=increase
        HPA: 0=scale-down, 1=maintain, 2=scale-up
        '''
        vpa_action = action % 3
        hpa_action = action // 3
        return vpa_action, hpa_action

    def step(self, action, rf):
        vpa_action, hpa_action = self._decode_action(action)

        # VPA: execute every step
        if vpa_action == 0:
            self._decrease_resources()
        elif vpa_action == 2:
            self._increase_resources()

        # HPA: map to delta (-1, 0, +1)
        hpa_delta = hpa_action - 1
        hpa_penalty = self.apply_hpa_action(hpa_delta)

        self.state = self.get_current_usage()
        reward = self.calculate_joint_reward(rf, hpa_penalty)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, {}

    def _increase_resources(self):
        self._apply_delta(self.INCREMENT)

    def _decrease_resources(self):
        self._apply_delta(-self.INCREMENT)


class JointContinuousElasticityEnv(BaseJointElasticityEnv):
    '''
    Joint HPA+VPA with continuous action space Box([-1,-1],[1,1]) shape (2,).
    action[0] = VPA (CPU scaling), action[1] = HPA (replica scaling).
    '''
    def __init__(self, id, independent_state=False, pod_name=None):
        super().__init__(id, independent_state=independent_state, pod_name=pod_name)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action, rf):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        vpa_action, hpa_action = float(action[0]), float(action[1])

        # VPA: accumulate scaled delta and patch only when threshold crossed
        scale_action = vpa_action * self.scale_action
        self._apply_delta(scale_action)

        # HPA: discretize continuous action to replica delta
        if abs(hpa_action) > 0.33:
            hpa_delta = int(np.sign(hpa_action) * max(1, int(abs(hpa_action) * 2)))
        else:
            hpa_delta = 0
        hpa_penalty = self.apply_hpa_action(hpa_delta)

        self.state = self.get_current_usage()
        reward = self.calculate_joint_reward(rf, hpa_penalty)

        self.steps += 1
        done = self.steps >= self.MAX_STEPS

        return self.state, reward, done, {}
