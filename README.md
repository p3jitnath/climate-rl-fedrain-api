# **FedRAIN — Federated Reinforcement Learning for ClimateRL Environments**

FedRAIN is a focused Python package providing utilities and a minimal API for running **federated reinforcement learning (FedRL)** experiments in **climateRL** settings.
It enables scalable, multi-agent training for idealised climate models such as the **Budyko–Sellers Energy Balance Model (EBM)**, using **Flower** and **SmartRedis** for coordination and weight exchange.

FedRAIN provides:

* A user-facing façade `fedrain.api.FedRAIN` for constructing RL algorithms (e.g., DDPG, TD3, TQC)
* Lightweight SmartRedis helpers for exchanging flattened PyTorch actor weights (`fedrain.fedrl.FedRL`)
* A Flower-compatible client adapter (`fedrain.fedrl.client.FlowerClient`) for bridging RL agents to federated learning processes
* A server-side orchestration utility (`fedrain.fedrl.server.FLWRServer`) using Ray + Redis for local simulations
* Convenience tools for environment creation, seeding, and launching a local Redis instance (`fedrain.utils`)

---

## **Overview**

FedRAIN is intentionally **lightweight and modular**, designed for researchers or engineers who want to combine **RL experiments** with **federated learning (FL)** infrastructure without heavyweight dependencies.

It integrates:

* **Flower** for federated coordination
* **SmartRedis** for tensor-based actor weight exchange
* **Ray** for concurrent simulation and tuning

It was developed alongside the **FedRAIN-Lite** framework (Nath et al., *NeurIPS Workshop on Tackling Climate Change with ML, 2025*), which demonstrated faster convergence and geographically adaptive parameterisations in idealised EBM setups.

**Key design principles:**

* Minimal friction: Construct algorithms by name using a single API entry point.
* Redis-native exchange: Serialise flattened PyTorch actor weights to a SmartRedis store.
* Federation-ready: Implemented as a Flower `NumPyClient` for plug-and-play compatibility.
* Local-first development: Includes a Redis launcher for standalone testing.

---

## **Requirements**

FedRAIN is tested with dependencies listed in `environment.yml`. Core runtime stack:

* `numpy`, `torch`, `gymnasium`, `pygame`, `matplotlib`, `pandas`
* `climlab` - climateRL environments
* `flwr` — Flower framework for federated learning
* `smartredis` — Tensor operations and key–value communication
* `ray` — Parallel experiment orchestration

> **Note:** For RedisAI workflows, ensure your `redis-server` includes the `redisai.so` module. See the SmartSim installation guide linked below.

---

## **Installation (Conda)**

FedRAIN uses Conda for environment management.
To set up a development environment:

```bash
conda env create -f environment.yml
conda activate venv
pip install -e .
```

If you require a GPU-specific PyTorch build, install it manually following [PyTorch’s official guide](https://pytorch.org/) before running the `pip install -e .` command.

**SmartSim setup:**
Follow the installation notes here for Redis and RedisAI configuration:
👉 [https://gist.github.com/p3jitnath/aa790c560b2f71462c99f88f112815ef](https://gist.github.com/p3jitnath/aa790c560b2f71462c99f88f112815ef)

---

## **Environment Variables**

FedRAIN expects a running Redis instance accessible via:

```bash
export SSDB=127.0.0.1:6379
```

You can optionally launch a local server using:

```python
from fedrain.utils import RedisServer
RedisServer().start()
```

By default, it assumes the Redis binary is on your PATH and the RedisAI module is at `~/redisai/redisai.so`.

---

## **Quickstart Code Snippets**

### 1. Construct an RL Agent

```python
from fedrain.api import FedRAIN

...
api = FedRAIN()
agent = api.set_algorithm('DDPG', envs, seed=0)
```

### 2. Exchange Weights via SmartRedis

```python
from fedrain.fedrl import FedRL
fed = FedRL(actor=actor, cid=0)
step = ...
fed.save_weights(step)
...
fed.load_weights(step)
```

### 3. Run a Local Federated Simulation

```python
from fedrain.fedrl.server import FLWRServer

...
server = FLWRServer(num_clients=4, num_rounds=10)
server.generate_actor(env_class=Env, actor_class=Actor)
server.set_client(seed=1, fn=client_fn, num_steps=100)
server.serve(cpus_per_client=2)
...
```

---

## **Examples and Testing**

Example scripts for **climateRL** environments (e.g., `ebm-v2`, `ebm-v3`) are provided in the `examples/` directory.

Run the tests using:

```bash
pytest -q
```

> Integration tests requiring GPU support may be skipped automatically.

---

## **Troubleshooting**

* **Redis connection issues:**
  Confirm `redis-server` is reachable via `SSDB` and that RedisAI is loaded if required.
* **PyTorch/CUDA mismatch:**
  If training stalls or fails, verify PyTorch is installed for your CUDA version.
* **Dependency errors:**
  Ensure `flwr`, `ray`, and `smartredis` are installed and importable.

---

## **Contributing**

1. Fork the repository and create a feature branch
2. Add or update unit tests (`pytest`)
3. Submit a pull request with a short summary and test notes

If you extend the library (e.g., new algorithms or FedRL modes), update this README and include working examples.

---

## **Citations**

- Nath P, Schemm S, Moss H, Haynes P, Shuckburgh E, Webb M. FedRAIN-Lite: Federated reinforcement algorithms for improving idealised numerical weather and climate models. 2025 NeurIPS workshop on Tackling Climate Change with Machine Learning; 2025. arXiv:2508.14315 [cs]. Available from: https://arxiv.org/abs/2508.14315
- Nath P, Moss H, Shuckburgh E, Webb M. RAIN: Reinforcement algorithms for improving numerical weather and climate models. 2025 EGU General Assembly (Oral); 2025. EGU25-5159 (ITS1.4/CL0.10). Available from: http://arxiv.org/abs/2408.16118

---

## **Author**

**Pritthijit Nath** (University of Cambridge & Met Office) ✉️ [pn341@cam.ac.uk](mailto:pn341@cam.ac.uk)
