import pytest
import torch

from src.distillation import (
    CategoryOverlapError,
    CategoryTeacherRouter,
    RoundRobinTeacherSampler,
    UnroutedCategoryError,
)


def test_router_routes_single_categories_and_uses_fallback():
    router = CategoryTeacherRouter(
        {
            "human_teacher": ["human", "fashion_tryon"],
            "animal_teacher": ["animal"],
        },
        fallback="general_teacher",
    )

    assert router.route("human") == "human_teacher"
    assert router.route("animal") == "animal_teacher"
    assert router.route("unknown") == "general_teacher"
    assert router.teachers == ("human_teacher", "animal_teacher", "general_teacher")
    with pytest.raises(TypeError):
        router.category_to_teacher["new"] = "teacher"  # type: ignore[index]


def test_router_groups_a_mixed_batch_with_stable_long_indices():
    router = CategoryTeacherRouter(
        {
            "human_teacher": ["human"],
            "animal_teacher": ["animal"],
        },
        fallback="general_teacher",
    )

    groups = router.route_groups(["animal", "human", "other", "animal", "other"])

    assert list(groups) == ["animal_teacher", "human_teacher", "general_teacher"]
    assert all(indices.dtype == torch.long for indices in groups.values())
    assert torch.equal(groups["animal_teacher"], torch.tensor([0, 3]))
    assert torch.equal(groups["human_teacher"], torch.tensor([1]))
    assert torch.equal(groups["general_teacher"], torch.tensor([2, 4]))


def test_router_detects_category_overlap_at_construction():
    with pytest.raises(CategoryOverlapError, match="shared"):
        CategoryTeacherRouter(
            [
                ("teacher_a", ["a", "shared"]),
                ("teacher_b", ["b", "shared"]),
            ]
        )

    with pytest.raises(CategoryOverlapError, match="declared more than once"):
        CategoryTeacherRouter([("teacher_a", ["duplicate", "duplicate"])])


def test_router_reports_all_unrouted_items_before_returning_groups():
    router = CategoryTeacherRouter({"human_teacher": ["human"]})

    with pytest.raises(UnroutedCategoryError, match=r"index 1: 'animal'.*index 2: 'unknown'"):
        router.route_groups(["human", "animal", "unknown"])
    with pytest.raises(UnroutedCategoryError, match="no teacher route"):
        router.route("animal")


def test_router_can_be_built_from_category_to_teacher_map():
    router = CategoryTeacherRouter.from_category_map(
        {"human": "human_teacher", "animal": "animal_teacher"},
        fallback="general_teacher",
    )

    assert router.route_batch(["human", "unknown", "animal"]) == (
        "human_teacher",
        "general_teacher",
        "animal_teacher",
    )
    assert router.category_to_teacher == {"human": "human_teacher", "animal": "animal_teacher"}


def test_router_rejects_ambiguous_or_invalid_inputs():
    with pytest.raises(ValueError, match="only one"):
        CategoryTeacherRouter(
            {"teacher": ["category"]},
            category_to_teacher={"category": "teacher"},
        )
    with pytest.raises(TypeError, match="batch iterable"):
        CategoryTeacherRouter({"teacher": ["category"]}).route_batch("category")
    with pytest.raises(TypeError, match="hashable"):
        CategoryTeacherRouter({"teacher": [["not", "hashable"]]})


def test_round_robin_sampler_balances_unequal_teacher_pools():
    dataset = type(
        "Dataset",
        (),
        {
            "samples": [
                {"category": "animal"},
                {"category": "animal"},
                {"category": "animal"},
                {"category": "human"},
            ]
        },
    )()
    router = CategoryTeacherRouter({"animal_teacher": ["animal"], "human_teacher": ["human"]})
    sampler = RoundRobinTeacherSampler(dataset, router, seed=5)

    indices = list(sampler)
    routed = [router.route(dataset.samples[index]["category"]) for index in indices]

    assert len(sampler) == 6
    assert routed == ["animal_teacher", "human_teacher"] * 3
    assert len({indices[index] for index in range(1, len(indices), 2)}) == 1


def test_round_robin_sampler_rejects_configured_teacher_without_samples():
    dataset = type("Dataset", (), {"samples": [{"category": "animal"}]})()
    router = CategoryTeacherRouter({"animal_teacher": ["animal"], "human_teacher": ["human"]})

    with pytest.raises(ValueError, match="no dataset samples"):
        RoundRobinTeacherSampler(dataset, router)

