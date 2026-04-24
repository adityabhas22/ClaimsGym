from claimsops_env.generator import ScenarioGenerator
from claimsops_env.scenario_templates import SCENARIO_FAMILIES, SCENARIO_TEMPLATES


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
    assert len(SCENARIO_FAMILIES) >= 15


def test_family_generation_sets_hidden_labels() -> None:
    generator = ScenarioGenerator()
    spec = generator.generate_family("subrogation_opportunity", seed=1)

    assert spec.hidden.subrogation_expected is True
    assert spec.claim.difficulty == 5


def test_all_templates_generate_valid_specs() -> None:
    generator = ScenarioGenerator()

    for index, family in enumerate(SCENARIO_FAMILIES):
        spec = generator.generate_family(family, seed=index)
        template = SCENARIO_TEMPLATES[family]

        assert spec.claim.family == family
        assert spec.claim.difficulty == template.level
        assert spec.hidden.required_documents == set(template.required_documents)
        assert spec.claim.step_budget >= 14


def test_operational_template_adds_platform_work() -> None:
    generator = ScenarioGenerator()
    spec = generator.generate_family("rental_storage_leakage", seed=3)

    categories = {activity.category for activity in spec.platform_state.activities}
    event_types = {event.event_type for event in spec.platform_state.pending_events}

    assert "rental" in categories
    assert "towing_storage" in categories
    assert "rental_day_accrual" in event_types
    assert "storage_fee_accrual" in event_types
