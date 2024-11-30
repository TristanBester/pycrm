import inspect


class TestTransitionFunctionInitialization:
    """Test the initialisation of the transition functions."""

    def test_state_transition_function(self, crm) -> None:
        """Test state transition function is initialised."""
        assert hasattr(crm, "delta_u")

        for state, transitions in crm.delta_u.items():
            assert isinstance(state, int)

            for transition_fn, next_state in transitions.items():
                assert isinstance(next_state, int)
                assert callable(transition_fn)

                signature = inspect.signature(transition_fn)
                assert len(signature.parameters) == 2
                assert list(signature.parameters.keys()) == ["props", "counter_states"]

    def test_counter_transition_function(self, crm) -> None:
        """Test counter transition function is initialised."""
        assert hasattr(crm, "delta_c")

        for state, transitions in crm.delta_c.items():
            assert isinstance(state, int)

            for transition_fn, counter_modifier in transitions.items():
                assert isinstance(counter_modifier, tuple)
                assert callable(transition_fn)

                signature = inspect.signature(transition_fn)
                assert len(signature.parameters) == 2
                assert list(signature.parameters.keys()) == ["props", "counter_states"]

    def test_reward_transition_function(self, crm) -> None:
        """Test reward transition function is initialised."""
        assert hasattr(crm, "delta_r")

        for state, transitions in crm.delta_r.items():
            assert isinstance(state, int)

            for transition_fn, reward_fn in transitions.items():
                assert callable(reward_fn)
                assert callable(transition_fn)

                signature = inspect.signature(transition_fn)
                assert len(signature.parameters) == 2
                assert list(signature.parameters.keys()) == ["props", "counter_states"]

                # Test the signature of the callable matches that of a reward function
                signature = inspect.signature(reward_fn)
                assert len(signature.parameters) == 3
                assert list(signature.parameters.keys()) == [
                    "obs",
                    "action",
                    "next_obs",
                ]
