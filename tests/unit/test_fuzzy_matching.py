import pytest

from openoperator.perception import fuzzy


def test_levenshtein_distance_basic():
    assert fuzzy.levenshtein_distance("", "") == 0
    assert fuzzy.levenshtein_distance("a", "") == 1
    assert fuzzy.levenshtein_distance("kitten", "sitting") == 3


def test_similarity_ratio():
    assert fuzzy.similarity_ratio("", "") == 1.0
    assert pytest.approx(fuzzy.similarity_ratio("kitten", "sitting"), 0.01) == 1.0 - (3 / 7)
    assert fuzzy.similarity_ratio("hello", "hello") == 1.0


def test_token_set_ratio_identical_and_reordered():
    assert fuzzy.token_set_ratio("open operator", "operator open") == 1.0
    assert fuzzy.token_set_ratio("open operator demo", "operator demo") >= 0.5


def test_fuzzy_match_score_combines_methods():
    # close typo
    s1 = "notepad"
    s2 = "notepadd"
    score = fuzzy.fuzzy_match_score(s1, s2)
    assert score > 0.8
