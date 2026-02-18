from pydantic import BaseModel, Field


# ── Registry ───────────────────────────────────────────────────────────

REWARD_REGISTRY: dict[str, type["BaseReward"]] = {}


def register_reward(cls: type["BaseReward"]) -> type["BaseReward"]:
    """Decorator that auto-registers a BaseReward subclass by its name field."""
    # Instantiate with defaults to read the name
    instance = cls()
    assert instance.name not in REWARD_REGISTRY, (
        f"Reward '{instance.name}' is already registered"
    )
    REWARD_REGISTRY[instance.name] = cls
    return cls


# ── Base class ─────────────────────────────────────────────────────────

class BaseReward(BaseModel):
    name: str = Field(..., description="The name of the reward")
    weight: float = Field(1.0, description="The weight of the reward")
    min_value: float = Field(0.0, description="Minimum raw score (for normalization)")
    max_value: float = Field(1.0, description="Maximum raw score (for normalization)")

    def score(self, prompt: str, completion: str, metadata: dict) -> float:
        raise NotImplementedError

    def normalized_score(self, prompt: str, completion: str, metadata: dict) -> float:
        """Score, clamp to [min_value, max_value], then normalize to [0, 1]."""
        raw = self.score(prompt, completion, metadata)
        clamped = max(self.min_value, min(self.max_value, raw))
        if self.max_value == self.min_value:
            return 0.0
        return (clamped - self.min_value) / (self.max_value - self.min_value)


# ── Built-in rewards (auto-registered on import) ──────────────────────

@register_reward
class LengthReward(BaseReward):
    name: str = "length"
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 500.0

    def score(self, prompt: str, completion: str, metadata: dict) -> float:
        return float(len(completion))


@register_reward
class ThinkFormatReward(BaseReward):
    """Rewards completions that use <think>...</think> format correctly."""
    name: str = "think_format"
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0

    def score(self, prompt: str, completion: str, metadata: dict) -> float:
        has_open = "<think>" in completion
        has_close = "</think>" in completion

        if not has_open and not has_close:
            return 0.0

        # Partial credit: has one tag but not the other
        if has_open != has_close:
            return 0.25

        # Both tags present — check ordering and that think comes before answer
        open_idx = completion.index("<think>")
        close_idx = completion.index("</think>")

        # Wrong order
        if close_idx <= open_idx:
            return 0.25

        # Correct structure: <think>...</think> with content inside
        think_content = completion[open_idx + len("<think>"):close_idx].strip()
        answer_after = completion[close_idx + len("</think>"):].strip()

        # Full marks: has thinking content AND answer after
        if think_content and answer_after:
            return 1.0

        # Has structure but empty thinking or no answer after
        if think_content or answer_after:
            return 0.75

        return 0.5


class Rewarder:
    def __init__(self, reward_names: list[str] | None = None, reward_weights: dict[str, float] | None = None):
        """
        Build a Rewarder from the registry.
        - reward_names: which rewards to use (default: all registered)
        - reward_weights: override weights by name (e.g. {"length": 2.0})
        """
        names = reward_names or list(REWARD_REGISTRY.keys())
        self.rewards: list[BaseReward] = []
        for name in names:
            assert name in REWARD_REGISTRY, (
                f"Reward '{name}' not found in registry. "
                f"Available: {list(REWARD_REGISTRY.keys())}"
            )
            weight = (reward_weights or {}).get(name, None)
            reward = REWARD_REGISTRY[name]() if weight is None else REWARD_REGISTRY[name](weight=weight)
            self.rewards.append(reward)

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        metadata: list[dict],
    ) -> list[float]:
        assert len(self.rewards) > 0, "No rewards registered"
        assert len(prompts) == len(completions) == len(metadata), (
            f"Batch size mismatch: {len(prompts)} prompts, "
            f"{len(completions)} completions, {len(metadata)} metadata"
        )

        batch_rewards = []
        for prompt, completion, meta in zip(prompts, completions, metadata):
            total = 0.0
            for reward in self.rewards:
                total += reward.normalized_score(prompt, completion, meta) * reward.weight
            batch_rewards.append(total)
        return batch_rewards
