import numpy as np
import pytest

from crm.automaton.machine import CountingRewardMachine


class TestEncoding:
    """Test the encoding functions."""

    @pytest.mark.parametrize(
        "u, expected_result",
        [
            (0, np.array([1, 0, 0])),
            (1, np.array([0, 1, 0])),
            (2, np.array([0, 0, 1])),
        ],
    )
    def test_encode_machine_state(
        self, u: int, expected_result: np.ndarray, crm: CountingRewardMachine
    ) -> None:
        """Test the encoding of machine states."""
        u_enc = crm.encode_machine_state(u)
        assert np.all(u_enc == expected_result)

    @pytest.mark.parametrize(
        "c, scale, expected_result",
        [
            ((1, 1), 1.0, np.array([1, 1])),
            ((1, 1), 2.0, np.array([0.5, 0.5])),
            ((1, 1, 2), 0.5, np.array([2, 2, 4])),
        ],
    )
    def test_encode_counter_configuration(
        self,
        c: tuple[int],
        scale: float,
        expected_result: np.ndarray,
        crm: CountingRewardMachine,
    ) -> None:
        """Test the encoding of counter configurations."""
        c_enc = crm.encode_counter_configuration(c, scale)
        assert np.all(c_enc == expected_result)

    @pytest.mark.parametrize(
        "c, expected_result",
        [
            ((0, 0), np.array([0, 0])),
            ((0, 1), np.array([0, 1])),
            ((1, 1, 2), np.array([1, 1, 1])),
            ((1, 0, 10), np.array([1, 0, 1])),
        ],
    )
    def test_encode_counter_state(
        self, c: tuple[int], expected_result: np.ndarray, crm: CountingRewardMachine
    ) -> None:
        """Test the encoding of counter states."""
        c_enc = crm.encode_counter_state(c)
        assert np.all(c_enc == expected_result)
