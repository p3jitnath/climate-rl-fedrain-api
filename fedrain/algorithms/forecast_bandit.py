"""Choose fixed correction profiles using one reward per complete forecast."""

import json
from pathlib import Path

import numpy as np
import torch

from fedrain.algorithms.base import BaseAlgorithm
from fedrain.utils import setup_logger


class ForecastBandit(BaseAlgorithm):
    """GP-UCB search over nine coefficients shared by all forecast ranks.

    Three Legendre modes define each field's vertical profile. Atmospheric
    observations only identify the field. MPI communication selects one
    common arm, and the caller supplies globally reduced reward totals at
    forecast completion. No policy update occurs between model exchanges.

    A positive calendar_case_count fits case-specific response deviations
    and predicts their calendar average. Zero retains the original GP fit.
    """

    def __init__(
        self,
        envs,
        seed,
        field_feature_index,
        field_values,
        episode_index,
        case_date,
        inference_only=False,
        communicator=None,
        calendar_case_count=0,
    ):
        self.envs = envs
        self.seed = seed
        self.field_feature_index = field_feature_index
        self.field_values = np.asarray(field_values)
        self.episode_index = episode_index
        self.case_date = case_date
        self.calendar_case_count = calendar_case_count
        self.train = not inference_only
        self.communicator = communicator
        self.rank = communicator.Get_rank() if communicator is not None else 0
        self.logger = setup_logger("forecast_bandit")
        self.levels = envs.single_action_space.shape[0]
        height = np.linspace(-1.0, 1.0, self.levels)
        self.basis = (
            np.stack(
                (
                    height,
                    (3.0 * height**2 - 1.0) / 2.0,
                    (5.0 * height**3 - 3.0 * height) / 2.0,
                )
            )
            / 3.0
        )
        self.coefficients = np.zeros(3 * len(self.field_values))
        self.history = []
        self.reward_totals = np.zeros((len(self.field_values), 2))
        self.exchanges = 0
        self.completed = False
        self.algorithm = {
            "name": "forecast_bandit",
            "train": self.train,
            "seed": seed,
            "basis": "Legendre P1, P2, P3 divided by 3",
            "field_values": self.field_values.tolist(),
            "coefficient_bound": 0.5,
            "initial_forecasts": 12,
            "kernel_length_scale": 0.75,
            "standardized_noise_std": 0.15 if calendar_case_count else 0.5,
            "reward_scale_floor": 1.0e-8 if calendar_case_count else 1.0e-3,
            "calendar_case_count": calendar_case_count,
            "ucb_multiplier": 2.0,
            "candidate_pool_size": 2048,
            "incumbent_repeat_interval": 6,
            "reward": "equal-field mean of complete-forecast nudge reduction",
        }
        self.obs, _ = envs.reset(seed=seed)
        self.logger.info("FORECAST_BANDIT configuration: %s", self.algorithm)

    def _field_indices(self, observations):
        columns = np.asarray(observations).reshape(
            len(observations),
            -1,
            self.levels,
        )
        identities = columns[:, self.field_feature_index, 0]
        distances = np.abs(identities[:, None] - self.field_values)
        if np.any(distances.min(axis=1) > 1.0e-6):
            raise ValueError("Unknown field identity in bandit observations")
        return distances.argmin(axis=1)

    def predict(self, observations):
        """Return the same field profile at every column and exchange."""
        profiles = self.coefficients.reshape(len(self.field_values), 3) @ self.basis
        return np.clip(
            profiles[self._field_indices(observations)],
            self.envs.single_action_space.low,
            self.envs.single_action_space.high,
        ).astype(np.float32)

    def update(self, actions, next_observations, rewards, *unused):
        """Accumulate field rewards without changing the selected arm."""
        del actions, unused
        fields = self._field_indices(next_observations)
        rewards = np.asarray(rewards, dtype=np.float64)
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Non-finite bandit reward")
        for field in range(len(self.field_values)):
            selected = fields == field
            self.reward_totals[field, 0] += rewards[selected].sum()
            self.reward_totals[field, 1] += np.count_nonzero(selected)
        self.exchanges += 1

    def finish_episode(self, global_reward_totals):
        """Record one observation after all ranks finish the forecast."""
        if not self.train or self.completed:
            raise RuntimeError("Bandit forecast is frozen or already recorded")
        if not np.all(np.isfinite(global_reward_totals)) or np.any(
            global_reward_totals[:, 1] <= 0
        ):
            raise ValueError("A complete forecast needs finite rewards for all fields")
        field_rewards = global_reward_totals[:, 0] / global_reward_totals[:, 1]
        self.history.append(
            {
                "episode": self.episode_index,
                "case_date": self.case_date,
                "coefficients": self.coefficients.tolist(),
                "field_rewards": field_rewards.tolist(),
                "field_samples": global_reward_totals[:, 1].astype(int).tolist(),
                "reward": float(field_rewards.mean()),
                "exchanges": self.exchanges,
            }
        )
        self.completed = True
        self.logger.info("FORECAST_BANDIT completed: %s", self.history[-1])

    def _posterior(self, candidates):
        coefficients = np.asarray([row["coefficients"] for row in self.history])
        rewards = np.asarray([row["reward"] for row in self.history])
        scale = max(float(rewards.std()), self.algorithm["reward_scale_floor"])
        centered = (rewards - rewards.mean()) / scale
        length_scale = self.algorithm["kernel_length_scale"]
        distances = ((coefficients[:, None] - coefficients[None, :]) ** 2).sum(axis=-1)
        covariance = np.exp(-0.5 * distances / length_scale**2)
        if self.calendar_case_count:
            dates = np.asarray([row["case_date"] for row in self.history])
            if len(np.unique(dates)) > self.calendar_case_count:
                raise ValueError(
                    "Bandit history exceeds its declared training calendar"
                )
            # Case-specific response deviations sum to zero over the calendar.
            # Querying the shared action kernel below predicts that average,
            # without conditioning the chosen arm on the next forecast date.
            case_kernel = (dates[:, None] == dates[None, :]).astype(float)
            case_kernel -= 1.0 / self.calendar_case_count
            covariance *= 1.0 + case_kernel
        covariance += np.eye(len(rewards)) * (
            self.algorithm["standardized_noise_std"] ** 2 + 1.0e-8
        )
        factor = np.linalg.cholesky(covariance)
        weights = np.linalg.solve(factor.T, np.linalg.solve(factor, centered))
        distances = ((candidates[:, None] - coefficients[None, :]) ** 2).sum(axis=-1)
        cross_covariance = np.exp(-0.5 * distances / length_scale**2)
        mean = rewards.mean() + scale * (cross_covariance @ weights)
        projected = np.linalg.solve(factor, cross_covariance.T)
        deviation = scale * np.sqrt(np.maximum(0.0, 1.0 - (projected**2).sum(axis=0)))
        return mean, deviation

    def _incumbent(self):
        candidates = np.unique(
            np.asarray([row["coefficients"] for row in self.history]),
            axis=0,
        )
        mean, _ = self._posterior(candidates)
        return candidates[mean.argmax()]

    def _suggest(self):
        count = len(self.history)
        dimensions = len(self.coefficients)
        if count == 0:
            return np.zeros(dimensions)
        if count < self.algorithm["initial_forecasts"]:
            # The first forecast is the zero-correction control. The other
            # initial arms cover every coefficient with a Latin hypercube.
            generator = np.random.default_rng(self.seed)
            design = np.column_stack(
                [
                    (generator.permutation(11) + generator.random(11)) / 11.0
                    for _ in range(dimensions)
                ]
            )
            return (design[count - 1] - 0.5) * 0.5
        incumbent = self._incumbent()
        if (count + 1) % self.algorithm["incumbent_repeat_interval"] == 0:
            return incumbent
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, count]))
        bound = self.algorithm["coefficient_bound"]
        half_pool = self.algorithm["candidate_pool_size"] // 2
        candidates = np.vstack(
            (
                generator.uniform(-bound, bound, (half_pool, dimensions)),
                np.clip(
                    generator.normal(incumbent, 0.15, (half_pool, dimensions)),
                    -bound,
                    bound,
                ),
                incumbent[None, :],
            )
        )
        mean, deviation = self._posterior(candidates)
        return candidates[
            (mean + self.algorithm["ucb_multiplier"] * deviation).argmax()
        ]

    def _shared_selection(self, explore):
        coefficients = None
        if self.rank == 0:
            coefficients = self._suggest() if explore else self._incumbent()
        if self.communicator is not None:
            coefficients = self.communicator.bcast(coefficients, root=0)
        return np.asarray(coefficients, dtype=np.float64)

    def save(self, folder, timestep_offset=-1, replay_buffer=True):
        """Save complete bandit history and the posterior-best observed arm."""
        del timestep_offset, replay_buffer
        if not self.completed:
            raise RuntimeError("Cannot save an incomplete bandit forecast")
        selected = self._shared_selection(explore=False)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        actor_path = folder / "actor.pt"
        torch.save(torch.as_tensor(selected), actor_path.with_suffix(".pt.partial"))
        actor_path.with_suffix(".pt.partial").replace(actor_path)
        state_path = folder / "bandit_state.json"
        state_path.with_suffix(".json.partial").write_text(
            json.dumps(
                {
                    "algorithm": self.algorithm,
                    "history": self.history,
                    "selected_coefficients": selected.tolist(),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        state_path.with_suffix(".json.partial").replace(state_path)
        self.logger.info("FORECAST_BANDIT incumbent: %s", selected.tolist())

    def load(self, folder, timestep_offset=1, replay_buffer=True, actor_only=False):
        """Resume the search or restore its frozen incumbent for evaluation."""
        del timestep_offset, replay_buffer
        folder = Path(folder)
        if actor_only:
            self.coefficients = torch.load(
                folder / "actor.pt",
                map_location="cpu",
                weights_only=True,
            ).numpy()
        else:
            saved = json.loads((folder / "bandit_state.json").read_text())
            self.history = saved["history"]
            if len(self.history) != self.episode_index - 1:
                raise ValueError(
                    "Bandit history must end immediately before this forecast"
                )
            self.coefficients = self._shared_selection(explore=True)
        self.logger.info(
            "FORECAST_BANDIT episode=%s date=%s frozen=%s coefficients=%s",
            self.episode_index,
            self.case_date,
            actor_only,
            self.coefficients.tolist(),
        )
