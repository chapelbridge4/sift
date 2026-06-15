"""Tests for the RAG failure taxonomy (app/triage/taxonomy.py).

These tests are intentionally written before the implementation (TDD).
They define the contract that taxonomy.py must satisfy.
"""

import pytest
from app.triage.taxonomy import RAGFailureType, STAGES, by_stage


class TestStages:
    def test_stages_exact_values(self):
        assert STAGES == ["chunking", "retrieval", "reranking", "generation"]

    def test_stages_is_list(self):
        assert isinstance(STAGES, list)


class TestRAGFailureTypeMembers:
    def test_exactly_16_members(self):
        members = list(RAGFailureType)
        assert len(members) == 16, f"Expected 16 members, got {len(members)}: {[m.name for m in members]}"

    def test_chunking_members_present(self):
        names = {m.name for m in RAGFailureType}
        expected = {"CHUNK_TOO_LARGE", "CHUNK_TOO_SMALL", "SEMANTIC_SPLIT", "CONTEXT_BOUNDARY_LOST"}
        assert expected.issubset(names), f"Missing chunking members: {expected - names}"

    def test_retrieval_members_present(self):
        names = {m.name for m in RAGFailureType}
        expected = {"RELEVANT_NOT_RETRIEVED", "IRRELEVANT_RETRIEVED", "EMBEDDING_MISMATCH", "QUERY_INTENT_MISPARSE"}
        assert expected.issubset(names), f"Missing retrieval members: {expected - names}"

    def test_reranking_members_present(self):
        names = {m.name for m in RAGFailureType}
        expected = {"RELEVANT_DEMOTED", "IRRELEVANT_PROMOTED", "RERANKER_NOOP", "DIVERSITY_COLLAPSE"}
        assert expected.issubset(names), f"Missing reranking members: {expected - names}"

    def test_generation_members_present(self):
        names = {m.name for m in RAGFailureType}
        expected = {"UNFAITHFUL", "INCOMPLETE", "CONTEXT_IGNORED", "FORMAT_ERROR"}
        assert expected.issubset(names), f"Missing generation members: {expected - names}"


class TestRAGFailureTypeMetadata:
    @pytest.mark.parametrize("member", list(RAGFailureType) if hasattr(RAGFailureType, '__iter__') else [])
    def test_description_non_empty(self, member):
        assert isinstance(member.description, str) and len(member.description.strip()) > 0, (
            f"{member.name}.description is empty or missing"
        )

    @pytest.mark.parametrize("member", list(RAGFailureType) if hasattr(RAGFailureType, '__iter__') else [])
    def test_fix_hint_non_empty(self, member):
        assert isinstance(member.fix_hint, str) and len(member.fix_hint.strip()) > 0, (
            f"{member.name}.fix_hint is empty or missing"
        )

    @pytest.mark.parametrize("member", list(RAGFailureType) if hasattr(RAGFailureType, '__iter__') else [])
    def test_stage_in_stages(self, member):
        assert member.stage in STAGES, (
            f"{member.name}.stage={member.stage!r} is not in STAGES={STAGES}"
        )


class TestByStage:
    def test_by_stage_returns_dict(self):
        result = by_stage()
        assert isinstance(result, dict)

    def test_by_stage_keyed_by_all_stages(self):
        result = by_stage()
        assert set(result.keys()) == set(STAGES), (
            f"by_stage() keys={set(result.keys())} != STAGES={set(STAGES)}"
        )

    def test_by_stage_values_are_lists(self):
        result = by_stage()
        for stage, members in result.items():
            assert isinstance(members, list), f"by_stage()[{stage!r}] is not a list"

    def test_by_stage_partitions_all_types(self):
        result = by_stage()
        all_from_dict = []
        for members in result.values():
            all_from_dict.extend(members)
        assert set(all_from_dict) == set(RAGFailureType), (
            "by_stage() values do not partition all RAGFailureType members"
        )

    def test_by_stage_no_overlap(self):
        result = by_stage()
        all_from_dict = []
        for members in result.values():
            all_from_dict.extend(members)
        assert len(all_from_dict) == len(set(all_from_dict)), (
            "by_stage() has duplicate members across stages"
        )

    def test_by_stage_4_per_stage(self):
        result = by_stage()
        for stage, members in result.items():
            assert len(members) == 4, (
                f"by_stage()[{stage!r}] has {len(members)} members, expected 4"
            )

    def test_by_stage_members_are_ragfailuretype(self):
        result = by_stage()
        for stage, members in result.items():
            for m in members:
                assert isinstance(m, RAGFailureType), (
                    f"by_stage()[{stage!r}] contains non-RAGFailureType value: {m!r}"
                )
