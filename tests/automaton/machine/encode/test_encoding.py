import numpy as np
import pytest

from pycrm.automaton import RmToCrmAdapter
from pycrm.automaton.machine import CountingRewardMachine


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


class TestAdapterEncoding:
    """Test encoding functionality of the RmToCrmAdapter."""

    def test_adapter_encode_machine_state(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that the adapter can encode machine states."""
        adapter = rm_to_crm_adapter

        # Test encoding different states
        u_enc_0 = adapter.encode_machine_state(0)
        u_enc_1 = adapter.encode_machine_state(1)
        u_enc_terminal = adapter.encode_machine_state(adapter.F[0])

        # Check that they're one-hot encoded
        assert u_enc_0[0] == 1
        assert np.sum(u_enc_0) == 1

        assert u_enc_1[1] == 1
        assert np.sum(u_enc_1) == 1

        # Terminal state should be encoded at the end
        assert u_enc_terminal[-1] == 1
        assert np.sum(u_enc_terminal) == 1

    def test_adapter_encode_counter_configuration(
        self,
        rm_to_crm_adapter: RmToCrmAdapter,
    ):
        """Test that the adapter can encode counter configurations."""
        adapter = rm_to_crm_adapter

        # Test with single counter (always 0 for RM adapter)
        c_enc = adapter.encode_counter_configuration((0,), 1.0)
        assert c_enc == [0.0]

        # Test with scaling
        c_enc_scaled = adapter.encode_counter_configuration((0,), 2.0)
        assert c_enc_scaled == [0.0]

    def test_adapter_encode_counter_state(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that the adapter can encode counter states."""
        adapter = rm_to_crm_adapter

        # Test with single counter at 0
        c_enc = adapter.encode_counter_state((0,))
        assert c_enc == [0]

        # Test with counter at 1 (though RM adapter should always use 0)
        c_enc = adapter.encode_counter_state((1,))
        assert c_enc == [1]

    def test_adapter_encoding_consistency(self, rm_to_crm_adapter: RmToCrmAdapter):
        """Test that adapter encoding is consistent."""
        adapter = rm_to_crm_adapter

        # Test that same inputs produce same outputs
        u_enc_1 = adapter.encode_machine_state(0)
        u_enc_2 = adapter.encode_machine_state(0)
        assert np.array_equal(u_enc_1, u_enc_2)

        c_enc_1 = adapter.encode_counter_configuration((0,))
        c_enc_2 = adapter.encode_counter_configuration((0,))
        assert np.array_equal(c_enc_1, c_enc_2)
