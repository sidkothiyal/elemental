"""Unit tests for DatasetServer helpers and message handling."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import time
from unittest.mock import MagicMock, patch

import pytest

from dataset.dataset_server import DatasetServer, _compute_hmac, _verify_hmac


@pytest.mark.unit
class TestHMACHelpers:
    def test_compute_hmac_deterministic(self):
        h1 = _compute_hmac("secret", "12345")
        h2 = _compute_hmac("secret", "12345")
        assert h1 == h2

    def test_different_tokens_different_hmac(self):
        h1 = _compute_hmac("secret1", "12345")
        h2 = _compute_hmac("secret2", "12345")
        assert h1 != h2

    def test_verify_hmac_valid(self):
        token = "test_token"
        ts = str(int(time.time()))
        h = _compute_hmac(token, ts)
        assert _verify_hmac(token, ts, h) is True

    def test_verify_hmac_invalid(self):
        assert _verify_hmac("token", str(int(time.time())), "wrong_hmac") is False

    def test_verify_hmac_stale_timestamp(self):
        token = "token"
        ts = str(int(time.time()) - 600)  # 10 minutes old
        h = _compute_hmac(token, ts)
        assert _verify_hmac(token, ts, h) is False


@pytest.mark.unit
class TestDatasetServerInit:
    @patch("dataset.dataset_server.zmq")
    def test_init_stores_dataset(self, _mock_zmq):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        server = DatasetServer(dataset=mock_dataset, auth_token="secret")
        assert server.dataset is mock_dataset
