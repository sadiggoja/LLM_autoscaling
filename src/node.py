import time
from datetime import datetime

import requests
from kubernetes import client, config


class Node:
    """
    A class to represent a Kubernetes node and manage its containers.

    Attributes
    ----------
    name : str
        The name of the node.
    ip : str
        The IP address of the node.
    ca_ip : str
        The IP address of the container advisor (cAdvisor).
    containers : dict
        A dictionary to store container information with container IDs as keys.

    Methods
    -------
    __str__():
        Returns a string representation of the Node object.
    update_containers(debug=False, custom_label='app=localization'):
        Updates the container objects running on the node pointer based on the specified label.
    get_containers():
        Returns the dictionary of containers.
    get_container_usage(container_id):
        Retrieves the resource usage metrics for a specific container.
    """

    # Class-level circuit breaker state keyed by ca_ip so all Node instances
    # pointing at the same physical host share failure tracking.
    _circuit: dict = {}  # ca_ip -> {'available': bool, 'fails': int, 'last_fail': float}
    _sessions: dict = {}  # ca_ip -> requests.Session for keep-alive

    _CADVISOR_MAX_FAILS = 10
    _CADVISOR_RETRY_INTERVAL = 30.0
    _CADVISOR_INNER_RETRIES = 2
    _CADVISOR_INNER_BACKOFF = 0.5
    _CADVISOR_TIMEOUT = (10, 10)  # (connect, read) seconds — caps worst-case stall

    def __init__(self, name, ca_ip, ip):
        self.name = name
        self.ip = ip
        self.ca_ip = ca_ip
        self.containers = dict()
        self._subcontainers_cache = None
        self._subcontainers_cache_time = 0.0
        self._cache_ttl = 0.9  # just under housekeeping_interval=1s
        self._last_known_stats = None  # fallback when cAdvisor is temporarily unreachable
        self._last_known_stats_per_container = {}  # fallback when stats history is too short
        self._throughput_cache = None
        self._throughput_cache_time = 0.0
        self._last_known_throughput = None
        if ca_ip not in Node._circuit:
            Node._circuit[ca_ip] = {'available': True, 'fails': 0, 'last_fail': 0.0}
        if ca_ip not in Node._sessions:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
            session.mount('http://', adapter)
            Node._sessions[ca_ip] = session

    def __str__(self):
        return f"Node(name={self.name}, ip={self.ip}, ca_ip={self.ca_ip}, containers={self.containers})"

    def update_containers(self, debug=False, custom_label='type=ray', reset_containers=False):
        if reset_containers:
            self.containers = dict()

        config.load_kube_config() if debug else config.load_incluster_config()
        v1 = client.CoreV1Api()

        try:
            ret = v1.list_pod_for_all_namespaces(
                label_selector=custom_label, field_selector=f'spec.nodeName={self.name}'
            )
            for pod in ret.items:
                if pod.status.phase == "Running":
                    for container_status in pod.status.container_statuses:
                        # make sure it's the proper container
                        if container_status.name == custom_label.split("=")[-1]:
                            self.containers[container_status.container_id.split("//")[1]] = (
                                pod.metadata.name, container_status.name, pod.status.pod_ip)

        except Exception as e:
            print(f"Error: {e}")

    def get_containers(self):
        return self.containers

    def _cadvisor_get(self, url):
        session = Node._sessions[self.ca_ip]
        last_exc = None
        for attempt in range(Node._CADVISOR_INNER_RETRIES):
            try:
                return session.get(url, timeout=Node._CADVISOR_TIMEOUT)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt < Node._CADVISOR_INNER_RETRIES - 1:
                    time.sleep(Node._CADVISOR_INNER_BACKOFF * (attempt + 1))
        raise last_exc

    def _fetch_subcontainers(self):
        now = time.time()
        if self._subcontainers_cache is not None and (now - self._subcontainers_cache_time) < self._cache_ttl:
            return self._subcontainers_cache

        cb = Node._circuit[self.ca_ip]

        # If circuit is open, skip the network call entirely
        if not cb['available']:
            if (now - cb['last_fail']) < Node._CADVISOR_RETRY_INTERVAL:
                return self._last_known_stats
            # Retry window elapsed — half-open: try once more
            cb['available'] = True
            cb['fails'] = 0

        containers_stats_url = f"http://{self.ca_ip}:8080/api/v1.3/subcontainers/kubepods/"
        try:
            response = self._cadvisor_get(containers_stats_url)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            cb['fails'] += 1
            if cb['fails'] >= Node._CADVISOR_MAX_FAILS:
                print(f"cAdvisor at {self.ca_ip} unreachable after {cb['fails']} attempts — "
                      f"suspending retries for {int(Node._CADVISOR_RETRY_INTERVAL)}s")
                cb['available'] = False
                cb['last_fail'] = now
            else:
                print(f"cAdvisor timeout/connection error: {e}")
            return self._last_known_stats

        if response.status_code == 200:
            cb['fails'] = 0
            cb['available'] = True
            self._subcontainers_cache = response.json()
            self._subcontainers_cache_time = now
            self._last_known_stats = self._subcontainers_cache
            return self._subcontainers_cache

        return None

    def get_container_usage(self, container_id):
        containers_stats = self._fetch_subcontainers()
        if containers_stats is not None:
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                stats = container.get("stats") or []
                if len(stats) < 2:
                    last = self._last_known_stats_per_container.get(container_id)
                    if last is not None:
                        return last
                    cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                    memory_limit_bytes = container["spec"]["memory"]["limit"]
                    return (cpu_limit_mc, 0, 0), (memory_limit_bytes / (1024 * 1024), 0, 0), (0, 0), False

                current_cpu_usage_nanoseconds = stats[-1]["cpu"]["usage"]["total"]
                previous_cpu_usage_nanoseconds = stats[-2]["cpu"]["usage"]["total"]

                current_timestamp_str = stats[-1]["timestamp"].split('.')[0] + 'Z'
                previous_timestamp_str = stats[-2]["timestamp"].split('.')[0] + 'Z'

                current_timestamp = datetime.strptime(current_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                previous_timestamp = datetime.strptime(previous_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

                time_interval_seconds = (current_timestamp - previous_timestamp).total_seconds()
                if time_interval_seconds <= 0:
                    last = self._last_known_stats_per_container.get(container_id)
                    if last is not None:
                        return last
                    cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                    memory_limit_bytes = container["spec"]["memory"]["limit"]
                    return (cpu_limit_mc, 0, 0), (memory_limit_bytes / (1024 * 1024), 0, 0), (0, 0), False

                cpu_usage_delta_nanoseconds = current_cpu_usage_nanoseconds - previous_cpu_usage_nanoseconds
                cpu_usage_per_second = cpu_usage_delta_nanoseconds / time_interval_seconds

                cpu_usage_millicores = cpu_usage_per_second / 1000000
                cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                cpu_usage_percentage = (cpu_usage_per_second / (cpu_limit_mc * 1_000_000)) * 100

                current_memory_usage_bytes = stats[-1]["memory"]["usage"]

                memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                memory_usage_percentage = (current_memory_usage_bytes / memory_limit_bytes) * 100

                network_rx_per_second_mb, network_tx_per_second_mb = self.get_throughput(time_interval_seconds)

                throttled = stats[-1]['cpu']['cfs']['throttled_time'] > stats[-2]['cpu']['cfs']['throttled_time']

                result = ((cpu_limit_mc, cpu_usage_millicores, cpu_usage_percentage),
                          (memory_limit_bytes / (1024 * 1024), memory_usage_megabytes, memory_usage_percentage),
                          (network_rx_per_second_mb, network_tx_per_second_mb), throttled)
                self._last_known_stats_per_container[container_id] = result
                return result
            else:
                print(f"Container {container_id} not found")
                return (0, 0, 0), (0, 0, 0), (0, 0), False
        else:
            if Node._circuit[self.ca_ip]['available']:
                print("Failed to fetch containers stats")
            return (0, 0, 0), (0, 0, 0), (0, 0), False

    def get_container_limits(self, container_id):
        containers_stats = self._fetch_subcontainers()
        if containers_stats is not None:
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                cpu_limit_mc = container["spec"]["cpu"]["quota"] / 100
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                return cpu_limit_mc, memory_limit_bytes
            else:
                print(f"Container {container_id} not found")
                return 0, 0
        else:
            if Node._circuit[self.ca_ip]['available']:
                print("Failed to fetch containers stats")

    def get_container_usage_saving_data(self, container_id):
        containers_stats = self._fetch_subcontainers()
        if containers_stats is not None:
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                stats = container.get("stats") or []
                if len(stats) < 2:
                    print(f"Container {container_id}: cAdvisor returned <2 stat samples, skipping")
                    return None
                current_cpu_usage_nanoseconds = stats[-1]["cpu"]["usage"]["total"]
                previous_cpu_usage_nanoseconds = stats[-2]["cpu"]["usage"]["total"]

                current_timestamp_str = container["stats"][-1]["timestamp"].split('.')[0] + 'Z'
                previous_timestamp_str = container["stats"][-2]["timestamp"].split('.')[0] + 'Z'

                current_timestamp = datetime.strptime(current_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                previous_timestamp = datetime.strptime(previous_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

                time_interval = current_timestamp - previous_timestamp
                time_interval_seconds = time_interval.total_seconds()

                cpu_usage_delta_nanoseconds = current_cpu_usage_nanoseconds - previous_cpu_usage_nanoseconds
                cpu_usage_per_second = cpu_usage_delta_nanoseconds / time_interval_seconds

                cpu_usage_millicores = cpu_usage_per_second / 1000000
                cpu_limit_mc = container["spec"]["cpu"]["limit"]
                cpu_usage_percentage = (cpu_usage_per_second / (cpu_limit_mc * 1_000_000)) * 100

                current_memory_usage_bytes = container["stats"][-1]["memory"]["usage"]

                memory_usage_megabytes = current_memory_usage_bytes / (1024 * 1024)
                memory_limit_bytes = container["spec"]["memory"]["limit"]
                memory_usage_percentage = (current_memory_usage_bytes / memory_limit_bytes) * 100

                memory_usage_cache = container["stats"][-1]["memory"]["cache"]
                memory_usage_rss = container["stats"][-1]["memory"]["rss"]
                memory_usage_swap = container["stats"][-1]["memory"]["swap"]
                memory_usage_mapped_file = container["stats"][-1]["memory"]["mapped_file"]
                memory_usage_working_set = container["stats"][-1]["memory"]["working_set"]
                previous_memory_usage_cache = container["stats"][-2]["memory"]["cache"]
                previous_memory_usage_rss = container["stats"][-2]["memory"]["rss"]
                previous_memory_usage_swap = container["stats"][-2]["memory"]["swap"]
                previous_memory_usage_mapped_file = container["stats"][-2]["memory"]["mapped_file"]
                previous_memory_usage_working_set = container["stats"][-2]["memory"]["working_set"]
                memory_usage_cache_delta = memory_usage_cache - previous_memory_usage_cache
                memory_usage_rss_delta = memory_usage_rss - previous_memory_usage_rss
                memory_usage_swap_delta = memory_usage_swap - previous_memory_usage_swap
                memory_usage_mapped_file_delta = memory_usage_mapped_file - previous_memory_usage_mapped_file
                memory_usage_working_set_delta = memory_usage_working_set - previous_memory_usage_working_set

                mem_usage_cache = (memory_usage_cache_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_rss = (memory_usage_rss_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_swap = (memory_usage_swap_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_mapped_file = (memory_usage_mapped_file_delta / time_interval_seconds) / (1024 * 1024)
                mem_usage_working_set = (memory_usage_working_set_delta / time_interval_seconds) / (1024 * 1024)

                io_read_curr = container["stats"][-1]["diskio"]["io_service_bytes"][0]['stats']['Read']
                io_read_prev = container["stats"][-2]["diskio"]["io_service_bytes"][0]['stats']['Read']
                io_delta = io_read_curr - io_read_prev
                io_read_per_second = io_delta / time_interval_seconds

                io_write_curr = container["stats"][-1]["diskio"]["io_service_bytes"][0]['stats']['Write']
                io_write_prev = container["stats"][-2]["diskio"]["io_service_bytes"][0]['stats']['Write']
                io_delta = io_write_curr - io_write_prev
                io_write_per_second = io_delta / time_interval_seconds

                network_rx_per_second_mb, network_tx_per_second_mb = self.get_throughput(time_interval_seconds)

                return (cpu_limit_mc, cpu_usage_millicores, cpu_usage_percentage), (
                    memory_limit_bytes, memory_usage_megabytes, memory_usage_percentage), (
                    io_read_per_second, io_write_per_second), (network_rx_per_second_mb, network_tx_per_second_mb), (
                    mem_usage_cache, mem_usage_rss, mem_usage_swap, mem_usage_mapped_file, mem_usage_working_set)
            else:
                print(f"Container {container_id} not found")
                return (0, 0, 0), (0, 0, 0), (0, 0), (0, 0)
        else:
            if Node._circuit[self.ca_ip]['available']:
                print("Failed to fetch containers stats")

    def get_throughput(self, time_interval):
        now = time.time()
        if self._throughput_cache is not None and (now - self._throughput_cache_time) < self._cache_ttl:
            return self._throughput_cache

        cb = Node._circuit[self.ca_ip]
        if not cb['available']:
            return self._last_known_throughput if self._last_known_throughput is not None else (0, 0)

        containers_url = f"http://{self.ca_ip}:8080/api/v1.3/containers"
        try:
            response = self._cadvisor_get(containers_url)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if cb['available']:
                print(f"cAdvisor throughput timeout: {e}")
            return self._last_known_throughput if self._last_known_throughput is not None else (0, 0)
        if response.status_code == 200:
            containers_stats = response.json()
            stats = containers_stats.get("stats") or []
            if len(stats) < 2:
                return self._last_known_throughput if self._last_known_throughput is not None else (0, 0)
            current_network_rx_bytes = stats[-1]["network"]["interfaces"][-1]["rx_bytes"]
            previous_network_rx_bytes = stats[-2]["network"]["interfaces"][-1]["rx_bytes"]
            network_rx_delta = current_network_rx_bytes - previous_network_rx_bytes
            network_rx_per_second = network_rx_delta / time_interval

            current_network_tx_bytes = stats[-1]["network"]["interfaces"][-1]["tx_bytes"]
            previous_network_tx_bytes = stats[-2]["network"]["interfaces"][-1]["tx_bytes"]
            network_tx_delta = current_network_tx_bytes - previous_network_tx_bytes
            network_tx_per_second = network_tx_delta / time_interval

            result = (network_rx_per_second / (1024 * 1024)), (network_tx_per_second / (1024 * 1024))
            self._throughput_cache = result
            self._throughput_cache_time = now
            self._last_known_throughput = result
            return result
        else:
            if Node._circuit[self.ca_ip]['available']:
                print("Failed to fetch containers stats")
            return 0, 0

    def get_usage(self):
        summary_url = f"http://{self.ca_ip}:8080/api/v2.0/summary"
        total_cpu_capacity_millicores, total_memory_capacity = self.get_node_capacity()

        response = requests.get(summary_url)
        if response.status_code == 200:
            summary_data = response.json()

            current_cpu_usage = summary_data.get("/", {}).get("latest_usage", {}).get("cpu")
            memory_usage = summary_data.get("/", {}).get("latest_usage", {}).get("memory")

            cpu_usage_percentage = (current_cpu_usage / total_cpu_capacity_millicores) * 100
            memory_usage_percent = (memory_usage / total_memory_capacity) * 100

            memory_usage = memory_usage / (1024 * 1024)

            free_cpu = total_cpu_capacity_millicores - current_cpu_usage
            free_mem = total_memory_capacity - memory_usage
            free_mem = free_mem / (1024 * 1024)

            return current_cpu_usage, cpu_usage_percentage, memory_usage, memory_usage_percent, free_cpu, free_mem
        else:
            print("Failed to fetch resource usage.")

    def get_means(self):
        summary_url = f"http://{self.ca_ip}:8080/api/v2.0/summary"

        response = requests.get(summary_url)
        if response.status_code == 200:
            summary_data = response.json()

            minute_cpu = summary_data.get("/", {}).get("minute_usage", {}).get("cpu")['mean']
            hour_cpu = summary_data.get("/", {}).get("hour_usage", {}).get("cpu")['mean']
            day_cpu = summary_data.get("/", {}).get("day_usage", {}).get("cpu")['mean']
            minute_memory = summary_data.get("/", {}).get("minute_usage", {}).get("memory")['mean']
            hour_memory = summary_data.get("/", {}).get("hour_usage", {}).get("memory")['mean']
            day_memory = summary_data.get("/", {}).get("day_usage", {}).get("memory")['mean']

            return (minute_cpu, hour_cpu, day_cpu), (minute_memory, hour_memory, day_memory)
        else:
            print("Failed to fetch resource usage.")

    def get_node_capacity(self):
        limits_url = f"http://{self.ca_ip}:8080/api/v2.0/machine"
        response = requests.get(limits_url)
        if response.status_code == 200:
            machine_data = response.json()
            total_cpu_capacity_millicores = machine_data.get("num_cores") * 1000
            total_memory_capacity = machine_data.get("memory_capacity")  # in bytes
            return total_cpu_capacity_millicores, total_memory_capacity
        else:
            print("Failed to fetch machine information.")
            return 0, 0

    def get_allocated_resources(self):
        allocated_cpu = 0
        allocated_memory = 0
        containers_stats = self._fetch_subcontainers()
        if containers_stats is None:
            return allocated_cpu, allocated_memory
        for container_id, _ in list(self.get_containers().items()):
            container = next((c for c in containers_stats if container_id in c["name"]), None)
            if container:
                allocated_cpu += container['spec']['cpu']['limit']
                allocated_memory += container['spec']['memory']['limit']
        return allocated_cpu, allocated_memory

    # wip: calculates how much the pods from the application have allocated
    def get_unallocated_capacity(self):
        total_cpu_capacity, total_memory_capacity = self.get_node_capacity()
        allocated_cpu, allocated_memory = self.get_allocated_resources()

        unallocated_cpu = max(0, total_cpu_capacity - allocated_cpu)
        unallocated_memory = max(0, (total_memory_capacity - allocated_memory) / (1024 * 1024))

        return unallocated_cpu, unallocated_memory

    def get_root_storage(self):
        filesystem_url = f"http://{self.ca_ip}:8080/api/v2.0/storage"
        response = requests.get(filesystem_url)
        if response.status_code == 200:
            filesystem_stats = response.json()
            filesystem = next((fs for fs in filesystem_stats if fs['mountpoint'] == '/'), None)
            if filesystem:
                used = filesystem['usage']
                limit = filesystem['capacity']
                available = filesystem['available']
                percentage = (used / limit) * 100
                return (
                    (used / (1024 * 1024 * 1024)), (limit / (1024 * 1024 * 1024)), (available / (1024 * 1024 * 1024)),
                    percentage)
            else:
                print("Failed to fetch filesystem usage.")
        else:
            print("Failed to fetch filesystem usage.")
