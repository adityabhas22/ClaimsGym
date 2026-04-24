from claimsops_env.generator import ScenarioGenerator


def test_generator_is_deterministic_for_seed() -> None:
    generator = ScenarioGenerator()
    first = generator.generate(seed=42)
    second = generator.generate(seed=42)

    assert first.claim == second.claim
    assert first.hidden == second.hidden


def test_generator_covers_multiple_families() -> None:
    generator = ScenarioGenerator()
    families = {generator.generate(seed=seed).claim.family for seed in range(25)}

    assert len(families) >= 5


def test_family_generation_sets_hidden_labels() -> None:
    generator = ScenarioGenerator()
    spec = generator.generate_family("subrogation_opportunity", seed=1)

    assert spec.hidden.subrogation_expected is True
    assert spec.claim.difficulty == 5
