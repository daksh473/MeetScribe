"""Tests for Refinement Guards."""

from refine.guards import reject_edit


def test_guard_rejects_hallucinated_token():
    reason = reject_edit(
        original_text="we have a meeting",
        final_text="we have a secret meeting",
        engine_alternatives=["we have a meeting", "we had a meeting"]
    )
    assert reason is not None
    assert "hallucinated token" in reason.lower() or "not present" in reason.lower()


def test_guard_accepts_valid_alternative():
    reason = reject_edit(
        original_text="we have a meeting",
        final_text="we had a meeting",
        engine_alternatives=["we have a meeting", "we had a meeting"]
    )
    assert reason is None


def test_guard_rejects_number_change():
    reason = reject_edit(
        original_text="we have 15 items",
        final_text="we have 50 items",
        engine_alternatives=["we have 15 items", "we have 50 items"]
    )
    # Even though 50 is in alternatives, number changes are rejected
    assert reason == "Changes a number."


def test_guard_rejects_negation_toggle():
    reason = reject_edit(
        original_text="I will go to the store",
        final_text="I will not go to the store",
        engine_alternatives=["I will go to the store", "I will not go to the store"]
    )
    assert reason == "Adds, removes, or toggles a negation word."


def test_guard_rejects_modal_change():
    reason = reject_edit(
        original_text="We must do this",
        final_text="We should do this",
        engine_alternatives=["We must do this", "We should do this"]
    )
    assert reason == "Changes a modal word."
