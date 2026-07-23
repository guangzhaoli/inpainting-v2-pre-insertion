"""Strict category routing for task-specific teacher ensembles."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Dict, Generic, Optional, Tuple, TypeVar

import torch
from torch import Tensor
from torch.utils.data import Sampler


Category = TypeVar("Category", bound=Hashable)
Teacher = TypeVar("Teacher", bound=Hashable)


class CategoryRoutingError(ValueError):
    """Base class for invalid or incomplete category routing."""


class CategoryOverlapError(CategoryRoutingError):
    """Raised when a category is declared by more than one teacher route."""


class UnroutedCategoryError(CategoryRoutingError):
    """Raised when a batch category has neither a route nor a fallback."""


class CategoryTeacherRouter(Generic[Category, Teacher]):
    """Route every item in a possibly mixed batch to exactly one teacher.

    The primary constructor accepts ``teacher -> categories`` declarations so
    overlaps can be detected before training starts::

        router = CategoryTeacherRouter(
            {"human_teacher": ["human"], "animal_teacher": ["animal"]},
            fallback="general_teacher",
        )

    For already-normalized ``category -> teacher`` configuration, pass the
    explicit ``category_to_teacher`` keyword or use :meth:`from_category_map`.
    Routes are immutable after construction.
    """

    def __init__(
        self,
        teacher_categories: Optional[
            Iterable[Tuple[Teacher, Iterable[Category]]] | Mapping[Teacher, Iterable[Category]]
        ] = None,
        fallback: Optional[Teacher] = None,
        *,
        category_to_teacher: Optional[Mapping[Category, Teacher]] = None,
    ) -> None:
        if teacher_categories is not None and category_to_teacher is not None:
            raise ValueError("pass only one of teacher_categories or category_to_teacher")
        if teacher_categories is None and category_to_teacher is None:
            teacher_categories = {}

        routes: Dict[Category, Teacher] = {}
        teacher_order = []

        if category_to_teacher is not None:
            if not isinstance(category_to_teacher, Mapping):
                raise TypeError("category_to_teacher must be a mapping")
            declarations = ((teacher, (category,)) for category, teacher in category_to_teacher.items())
        else:
            if isinstance(teacher_categories, Mapping):
                declarations = iter(teacher_categories.items())
            else:
                try:
                    declarations = iter(teacher_categories)  # type: ignore[arg-type]
                except TypeError as exc:
                    raise TypeError(
                        "teacher_categories must be a mapping or iterable of (teacher, categories) pairs"
                    ) from exc

        for declaration in declarations:
            try:
                teacher, categories = declaration
            except (TypeError, ValueError) as exc:
                raise TypeError("each route declaration must be a (teacher, categories) pair") from exc
            self._require_hashable(teacher, name="teacher")
            if teacher not in teacher_order:
                teacher_order.append(teacher)

            # A string is one category, not an iterable of its characters.
            if isinstance(categories, (str, bytes)):
                normalized_categories = (categories,)
            else:
                try:
                    normalized_categories = iter(categories)
                except TypeError as exc:
                    raise TypeError(f"categories for teacher {teacher!r} must be iterable") from exc

            for category in normalized_categories:
                self._require_hashable(category, name="category")
                if category in routes:
                    previous = routes[category]
                    raise CategoryOverlapError(
                        f"category {category!r} is declared more than once " f"(teachers {previous!r} and {teacher!r})"
                    )
                routes[category] = teacher

        if fallback is not None:
            self._require_hashable(fallback, name="fallback")
            if fallback not in teacher_order:
                teacher_order.append(fallback)

        self._category_to_teacher = MappingProxyType(routes)
        self._fallback = fallback
        self._teachers = tuple(teacher_order)

    @staticmethod
    def _require_hashable(value: Hashable, *, name: str) -> None:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be hashable, got {value!r}") from exc

    @classmethod
    def from_category_map(
        cls,
        category_to_teacher: Mapping[Category, Teacher],
        fallback: Optional[Teacher] = None,
    ) -> "CategoryTeacherRouter[Category, Teacher]":
        """Build a router from an unambiguous ``category -> teacher`` map."""

        return cls(fallback=fallback, category_to_teacher=category_to_teacher)

    @property
    def category_to_teacher(self) -> Mapping[Category, Teacher]:
        return self._category_to_teacher

    @property
    def fallback(self) -> Optional[Teacher]:
        return self._fallback

    @property
    def teachers(self) -> Tuple[Teacher, ...]:
        return self._teachers

    def route(self, category: Category) -> Teacher:
        """Return the unique teacher for one category."""

        self._require_hashable(category, name="category")
        try:
            return self._category_to_teacher[category]
        except KeyError:
            if self._fallback is not None:
                return self._fallback
            raise UnroutedCategoryError(
                f"category {category!r} has no teacher route and no fallback is configured"
            ) from None

    def route_batch(self, categories: Sequence[Category] | Iterable[Category]) -> Tuple[Teacher, ...]:
        """Route a batch while preserving its original order."""

        if isinstance(categories, (str, bytes)):
            raise TypeError("categories must be a batch iterable, not a single string")
        batch = tuple(categories)
        unresolved = []
        routed = []
        for index, category in enumerate(batch):
            self._require_hashable(category, name=f"categories[{index}]")
            if category in self._category_to_teacher:
                routed.append(self._category_to_teacher[category])
            elif self._fallback is not None:
                routed.append(self._fallback)
            else:
                unresolved.append((index, category))

        if unresolved:
            details = ", ".join(f"index {index}: {category!r}" for index, category in unresolved)
            raise UnroutedCategoryError(
                f"batch contains unrouted categories ({details}); configure a fallback or route"
            )
        return tuple(routed)

    def route_groups(
        self,
        categories: Sequence[Category] | Iterable[Category],
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[Teacher, Tensor]:
        """Return ``teacher -> LongTensor[batch_indices]`` for a mixed batch.

        Dictionary order follows each teacher's first occurrence in the batch,
        and indices within a group retain batch order.  Routing is validated for
        the complete batch before any result is returned.
        """

        routed = self.route_batch(categories)
        groups: Dict[Teacher, list[int]] = {}
        for index, teacher in enumerate(routed):
            groups.setdefault(teacher, []).append(index)
        return {teacher: torch.tensor(indices, dtype=torch.long, device=device) for teacher, indices in groups.items()}


# A readable compatibility alias for callers that prefer the adjective form.
OverlappingCategoryError = CategoryOverlapError


class RoundRobinTeacherSampler(Sampler[int]):
    """Balance task teachers by round-robin oversampling their sample pools.

    The dataset must expose its lightweight sample metadata through a
    ``samples`` sequence, with a ``category`` field on every entry.  Each epoch
    shuffles within task pools, then cycles all teachers to the largest pool's
    length.  This mirrors OPD's full-task accumulation recipe and prevents a
    large domain from starving smaller teachers.
    """

    def __init__(self, dataset, router: CategoryTeacherRouter, seed: int = 0):
        self.dataset = dataset
        self.router = router
        self.seed = int(seed)
        self.epoch = 0

        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise TypeError("RoundRobinTeacherSampler requires dataset.samples category metadata")

        pools = {teacher: [] for teacher in router.teachers}
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or "category" not in sample:
                raise ValueError(f"dataset.samples[{index}] must be a mapping with category")
            teacher = router.route(sample["category"])
            pools.setdefault(teacher, []).append(index)

        empty = [teacher for teacher, indices in pools.items() if not indices]
        if empty:
            raise ValueError(f"OPD teacher routes have no dataset samples: {empty}")
        self._pools = pools
        self._teacher_order = tuple(pools)
        self._rounds = max(len(indices) for indices in pools.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._rounds * len(self._teacher_order)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled = {}
        for teacher, pool in self._pools.items():
            order = torch.randperm(len(pool), generator=generator).tolist()
            shuffled[teacher] = [pool[index] for index in order]

        for round_index in range(self._rounds):
            for teacher in self._teacher_order:
                pool = shuffled[teacher]
                yield pool[round_index % len(pool)]

