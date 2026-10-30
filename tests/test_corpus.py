from pathlib import Path

import pytest

from synthwerk_mascot.corpus import inline_text, load_corpus, slugify, split_markdown


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("Decisions of 2026-10-29", "decisions-of-2026-10-29"),
        ("Email: Resend", "email-resend"),
        ("`code` & More!", "code--more"),
        ("Embedding model and mascot index", "embedding-model-and-mascot-index"),
        ("Größe und Ränder", "größe-und-ränder"),
        ("[Link](https://x.test) text", "link-text"),
        ("snake_case stays", "snake_case-stays"),
    ],
)
def test_slugify_heading_github_anchor(heading: str, anchor: str) -> None:
    assert slugify(heading) == anchor


def test_inline_text_markdown_marks_removed() -> None:
    assert inline_text("**Bold** `code` [label](u) <b>x</b>") == "Bold code label x"


def test_split_markdown_sections_get_anchor_and_title_path() -> None:
    chunks = split_markdown(
        "docs/a.md",
        "# Top\n\nIntro text that is long enough to count as a passage here.\n\n"
        "## Sub part\n\nBody text of the sub part that is long enough to keep it.\n",
    )
    assert [(c.anchor, c.title) for c in chunks] == [
        ("top", "Top"),
        ("sub-part", "Top > Sub part"),
    ]
    assert all(c.path == "docs/a.md" for c in chunks)


def test_split_markdown_duplicate_headings_numbered_like_github() -> None:
    text = "".join(
        f"## Notes\n\nSome words in section number {i} for the test of anchors.\n\n"
        for i in range(3)
    )
    assert [c.anchor for c in split_markdown("d.md", text)] == ["notes", "notes-1", "notes-2"]


def test_split_markdown_pictures_html_and_text_diagrams_left_out() -> None:
    text = (
        "# Page\n\n<picture><img alt='secret banner words' src='a.svg'></picture>\n"
        "<!-- hidden comment words -->\n"
        "```text\n+--+ diagram words +--+\n```\n"
        "```bash\ndocker compose up --wait\n```\n"
        "Plain sentence that stays in the chunk for sure.\n"
    )
    (chunk,) = split_markdown("p.md", text)
    assert "banner" not in chunk.text
    assert "hidden" not in chunk.text
    assert "diagram" not in chunk.text
    assert "docker compose up --wait" in chunk.text
    assert "Plain sentence" in chunk.text


def test_split_markdown_heading_inside_code_fence_is_not_a_heading() -> None:
    text = (
        "# Real\n\n```bash\n# a shell comment\necho hi there friend\n```\n"
        "More words here, enough.\n"
    )
    (chunk,) = split_markdown("c.md", text)
    assert chunk.anchor == "real"
    assert "# a shell comment" in chunk.text


def test_split_markdown_table_rows_to_text_rule_dropped() -> None:
    text = (
        "# T\n\n| Name | Role |\n|---|---|\n| `vision` | Image labels for pictures and photos |\n"
    )
    (chunk,) = split_markdown("t.md", text)
    assert chunk.text == "Name | Role\nvision | Image labels for pictures and photos"


def test_split_markdown_short_section_dropped() -> None:
    assert split_markdown("s.md", "# Only a title\n\nTiny.\n") == []


def test_split_markdown_long_section_split_below_max_chars() -> None:
    sentence = "This sentence has some words in it. "
    long_paragraph = sentence * 40
    text = f"# Long\n\n{long_paragraph}\n\nShort tail paragraph with words.\n\n{'x' * 500}\n"
    chunks = split_markdown("l.md", text, max_chars=300)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.anchor for c in chunks} == {"long"}


def test_split_markdown_text_before_first_heading_uses_file_name() -> None:
    (chunk,) = split_markdown(
        "docs/intro.md", "Words before any heading, and enough of them to count.\n"
    )
    assert (chunk.anchor, chunk.title) == ("", "intro")


def test_load_corpus_only_docs_markdown_in_sorted_order(corpus_dir: Path) -> None:
    chunks = load_corpus(corpus_dir)
    paths = [c.path for c in chunks]
    assert paths == sorted(paths)
    assert set(paths) == {"docs/design-system.md", "docs/infrastructure.md", "docs/roadmap.md"}
