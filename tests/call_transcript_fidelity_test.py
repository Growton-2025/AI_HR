"""A call transcript may only contain what was said.

A recruiter opened Nethranand P S's call and read a five-turn conversation
about his open-source contributions, with an AI summary quoting him on how
rewarding they had been. The recording is 9.4 seconds long and its only words
are "Thank you very much." Everything else was written by gpt-4o-mini.

The pipeline asked the model to "analyze a VoIP call" and to return a
`transcript` array. Handed four words, it produced the recruiting call it
expected to see, and that was stored as the record. The one guard in place
compared lengths in a single direction — it caught the model *dropping* half a
long call, and waved through a rewrite twenty times longer than the audio.

Across the database eight transcripts held more text than their recordings
could physically contain; the worst was 11,486 characters over 5.6 seconds.
"""

import pytest

from backend.integrations import plivo_service as ps

# Verbatim from call 908.
REAL_AUDIO_SAID = "Thank you very much."
WHAT_WAS_STORED = """Lead: I'm really excited about the potential for open source projects.
Recruiter: That's great to hear! Can you elaborate on your experience with open source?
Lead: I have contributed to several open source initiatives, and they have been very rewarding.
Recruiter: What do you find most rewarding about those experiences?
Lead: The collaboration with other developers and the learning opportunities are unmatched."""


# ── the fabrication itself ──────────────────────────────────────────────────

def test_the_fabricated_transcript_is_rejected():
    assert ps.transcript_is_faithful(REAL_AUDIO_SAID, WHAT_WAS_STORED) is False


def test_a_call_this_quiet_is_never_sent_for_analysis():
    # Cheaper and safer than catching the invention afterwards: with four words
    # there is nothing to summarise, so no model is asked to try.
    assert ps.too_little_speech_to_analyse(REAL_AUDIO_SAID) is True


def test_the_stored_summary_says_plainly_that_nothing_was_captured():
    summary = ps.no_conversation_summary(9)

    assert "No conversation was captured" in summary
    assert "9-second" in summary
    # The UI treats these phrasings as "still working" and keeps polling; this
    # is a final answer, so it must not read like one of them.
    for placeholder in ("not enough information", "insufficient information",
                        "transcript isn't fully provided", "please share the full"):
        assert placeholder not in summary.lower()


# ── what must still get through ─────────────────────────────────────────────

def test_genuine_speaker_labelling_survives():
    raw = ("Hi Ravi this is Admin calling from Growton about the account "
           "executive role are you open to a chat this week")
    labelled = (
        "Recruiter: Hi Ravi, this is Admin calling from Growton about the "
        "account executive role.\n"
        "Recruiter: Are you open to a chat this week?"
    )

    assert ps.transcript_is_faithful(raw, labelled) is True


def test_speaker_labels_are_not_counted_as_new_words():
    raw = "hello is that Priya yes speaking"
    labelled = "Recruiter: hello is that Priya\nLead: yes speaking"

    assert ps.transcript_is_faithful(raw, labelled) is True


def test_a_real_conversation_is_long_enough_to_analyse():
    raw = ("Hi Ravi this is Admin from Growton I wanted to talk about the "
           "account executive role you applied for last week")

    assert ps.too_little_speech_to_analyse(raw) is False


# ── the other failure direction, which the original guard existed for ───────

def test_an_abridged_long_call_is_still_rejected():
    raw = " ".join(f"word{i}" for i in range(500))
    digest = "Recruiter: word1 word2 and so on ..."

    assert ps.transcript_is_faithful(raw, digest) is False


# ── neither same-length paraphrase nor empty input passes ───────────────────

def test_a_same_length_paraphrase_is_rejected():
    # Length alone proves nothing — the words have to be the spoken ones.
    raw = "the candidate asked about the salary band for this role today"
    paraphrase = "Lead: he enquired regarding compensation ranges applicable here now okay"

    assert ps.transcript_is_faithful(raw, paraphrase) is False


@pytest.mark.parametrize("raw,labelled", [
    ("", "Lead: hello there friend"),      # no audio, invented words
    ("some words were spoken here", ""),   # nothing came back
    (None, None),
])
def test_missing_text_is_never_faithful(raw, labelled):
    assert ps.transcript_is_faithful(raw, labelled) is False


# ── the stored record is authoritative, so a fix can replace a fabrication ──

def test_the_transcript_write_is_not_longest_wins():
    import inspect

    source = inspect.getsource(ps._store_call_insights)

    # "Longest wins" made an invented transcript permanent: it is always longer
    # than the few words actually spoken, so no correct re-run could replace it.
    assert "LENGTH(%s) > LENGTH(COALESCE(transcript" not in source
    assert "transcript = %s" in source


def test_the_prompt_forbids_inventing_content():
    import inspect

    source = inspect.getsource(ps._generate_call_insights)

    assert "Use ONLY what appears in the transcript" in source
    assert "Do not add, infer or" in source
