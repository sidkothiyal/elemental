"""Unit tests for GSM8KVerifier and extract_last_number."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest

from rl.verifiers.gsm8k_verifier import GSM8KVerifier, extract_last_number


@pytest.mark.unit
class TestExtractLastNumber:
    def test_extracts_last_integer(self):
        assert extract_last_number("first 12 then 34") == "34"

    def test_handles_decimals(self):
        assert extract_last_number("price is 3.50") == "3.5"

    def test_handles_thousands_separator(self):
        assert extract_last_number("revenue 1,234,567") == "1234567"

    def test_handles_negatives(self):
        assert extract_last_number("delta -42") == "-42"

    def test_returns_none_when_absent(self):
        assert extract_last_number("no numbers here") is None


@pytest.mark.unit
class TestGSM8KVerifier:
    def test_match_returns_one(self):
        v = GSM8KVerifier()
        assert v.score("q", "the answer is 42", "#### 42") == 1.0

    def test_mismatch_returns_zero(self):
        v = GSM8KVerifier()
        assert v.score("q", "the answer is 99", "#### 42") == 0.0

    def test_no_number_returns_zero(self):
        v = GSM8KVerifier()
        assert v.score("q", "I don't know", "#### 42") == 0.0

    def test_format_bonus_applied_only_on_marker(self):
        v = GSM8KVerifier(format_bonus=0.1)
        assert v.score("q", "the answer is #### 42", "#### 42") == pytest.approx(1.1)
        # Wrong number but contains marker — bonus only, no full reward.
        assert v.score("q", "the answer is #### 99", "#### 42") == pytest.approx(0.1)
        # Correct without marker — no bonus.
        assert v.score("q", "the answer is 42", "#### 42") == pytest.approx(1.0)
