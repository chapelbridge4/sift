from app.knowledge.prompts import format_prompt, load_prompt


def test_load_paper_extract_prompt():
    contract = load_prompt("paper_extract")
    assert contract.name == "paper_extract"
    assert "topic_tags" in contract.template
    assert contract.output_schema["type"] == "object"
    assert "topic_tags" in contract.output_schema["properties"]


def test_load_topic_merge_prompt():
    contract = load_prompt("topic_merge")
    assert contract.name == "topic_merge"
    assert "Nuances" in contract.template
    assert "sources" in contract.output_schema["properties"]


def test_format_prompt_substitutes_placeholders():
    contract = load_prompt("paper_extract")
    rendered = format_prompt(
        contract,
        paper_id="2301.1",
        title="Test Paper",
        authors="Alice",
        source_file="papers/t.pdf",
        section_outline="- abstract: hello",
        claim_spans="- [abstract] claim one",
        max_output_tokens=800,
    )
    assert "2301.1" in rendered
    assert "Test Paper" in rendered
    assert "claim one" in rendered
    assert "800" in rendered


def test_unknown_prompt_fails_fast():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_prompt("does-not-exist")