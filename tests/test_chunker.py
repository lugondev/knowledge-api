from kbase.chunker import chunk_markdown


def test_empty_input_produces_nothing():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_short_document_is_one_chunk():
    pieces = chunk_markdown("Bảo hành 12 tháng kể từ ngày mua.")
    assert len(pieces) == 1
    assert pieces[0].text == "Bảo hành 12 tháng kể từ ngày mua."
    assert pieces[0].ordinal == 0


def test_heading_path_is_carried_on_each_piece():
    text = "# Sổ tay\n\nMở đầu.\n\n## Bảo hành\n\nMười hai tháng.\n\n### Đổi trả\n\nBảy ngày.\n"
    pieces = chunk_markdown(text)
    by_text = {p.text.strip(): p.heading for p in pieces}
    assert by_text["Mở đầu."] == "Sổ tay"
    assert by_text["Mười hai tháng."] == "Sổ tay > Bảo hành"
    assert by_text["Bảy ngày."] == "Sổ tay > Bảo hành > Đổi trả"


def test_deeper_heading_replaces_only_its_own_level():
    text = "## A\n\naaa\n\n### A1\n\nbbb\n\n## B\n\nccc\n"
    pieces = chunk_markdown(text)
    by_text = {p.text.strip(): p.heading for p in pieces}
    assert by_text["bbb"] == "A > A1"
    assert by_text["ccc"] == "B"


def test_long_section_splits_with_overlap():
    body = ". ".join(f"Câu số {i}" for i in range(400)) + "."
    pieces = chunk_markdown(body, max_chars=200, overlap=50)
    assert len(pieces) > 1
    assert all(len(p.text) <= 200 for p in pieces)
    # Overlap means consecutive pieces share text; without it a sentence cut in
    # half at a boundary is retrievable from neither side.
    assert pieces[0].text[-20:] in pieces[1].text


def test_ordinals_are_contiguous_across_the_whole_document():
    text = "## A\n\n" + ("x" * 500) + "\n\n## B\n\n" + ("y" * 500)
    pieces = chunk_markdown(text, max_chars=200, overlap=50)
    assert [p.ordinal for p in pieces] == list(range(len(pieces)))


def test_one_enormous_line_terminates():
    # No sentence boundary anywhere: the splitter must still make progress
    # rather than re-cutting the same window forever.
    pieces = chunk_markdown("x" * 50_000, max_chars=800, overlap=100)
    assert len(pieces) > 1
    assert all(len(p.text) <= 800 for p in pieces)
    assert sum(len(p.text) for p in pieces) < 200_000


def test_overlap_larger_than_max_chars_still_terminates():
    pieces = chunk_markdown("y" * 5_000, max_chars=100, overlap=500)
    assert len(pieces) > 1


def test_heading_with_no_body_produces_no_chunk():
    assert chunk_markdown("## Trống\n\n## Cũng trống\n") == []
