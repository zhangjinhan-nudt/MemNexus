import pytest

from agent_memory.evidence_policy import assertion_supported


@pytest.mark.parametrize('source,quote,expected', [
    ("The sensor was replaced, so I'm not sure if the noise persists.", "The sensor was replaced", True),
    ("I attended the seminar so I don't know whether I need the recording.", "I attended the seminar", True),
    ("The parcel arrived, so we are unsure whether another shipment is needed.", "The parcel arrived", True),
    ("I will replace the sensor if the noise persists.", "I will replace the sensor", False),
    ("If the noise persists, I will replace the sensor.", "I will replace the sensor", False),
    ("I did not attend the seminar, so I don't know if it was recorded.", "attend the seminar", False),
    ("Did I attend the seminar, so I can request the recording?", "attend the seminar", False),
    ("I will leave if it rains, so I might miss the session.", "I will leave", False),
    ("I adjusted the sensor so that if pressure rises it stops.", "I adjusted the sensor", False),
])
def test_consequential_scope_retains_independent_assertion(source, quote, expected):
    assert assertion_supported(source, quote) is expected
