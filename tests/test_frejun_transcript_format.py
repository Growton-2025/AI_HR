import unittest
from unittest.mock import patch

from backend.api.routes.calls import call_artifacts_need_repair, maybe_process_call_audio
from backend.services.frejun_calls import (
    SHORT_CALL_SUMMARY,
    extract_transcript_text,
    is_placeholder_summary,
    normalize_summary_text,
    prefer_better_summary,
    prefer_richer_text,
)


class FreJunTranscriptFormatTests(unittest.TestCase):
    def test_iterable_transcript_uses_recruiter_and_candidate_name(self):
        transcript = [
            {"speaker": "Speaker A", "text": "Hello Nethranand, can you hear me?"},
            {"speaker": "Speaker B", "text": "Yes, I can hear you clearly."},
            {"speaker": "Speaker A", "text": "Great, let's begin."},
        ]

        result = extract_transcript_text(transcript, candidate_name="Nethranand")

        self.assertEqual(
            result,
            "\n".join(
                [
                    "Recruiter: Hello Nethranand, can you hear me?",
                    "Nethranand: Yes, I can hear you clearly.",
                    "Recruiter: Great, let's begin.",
                ]
            ),
        )

    def test_string_transcript_rewrites_known_labels(self):
        transcript = "Speaker A: Good morning\nSpeaker B: Good morning sir"

        result = extract_transcript_text(transcript, candidate_name="Ashwin")

        self.assertEqual(
            result,
            "\n".join(
                [
                    "Recruiter: Good morning",
                    "Ashwin: Good morning sir",
                ]
            ),
        )

    def test_plain_string_transcript_guesses_alternating_speakers(self):
        transcript = "Hi. Hello. How are you? Hello."

        result = extract_transcript_text(transcript, candidate_name="Nethranand")

        self.assertEqual(
            result,
            "\n".join(
                [
                    "Recruiter: Hi.",
                    "Nethranand: Hello.",
                    "Recruiter: How are you?",
                    "Nethranand: Hello.",
                ]
            ),
        )

    def test_prefer_richer_text_replaces_short_snippet_with_full_transcript(self):
        existing = "Thank you."
        incoming = "\n".join(
            [
                "Recruiter: Hello, can you hear me?",
                "Nethranand: Yes, I can hear you.",
                "Recruiter: Thank you.",
            ]
        )

        result = prefer_richer_text(existing, incoming)

        self.assertEqual(result, incoming)

    def test_placeholder_summary_is_detected_and_sanitized(self):
        placeholder = (
            "The recruitment call transcript isn't fully provided. "
            "Please share the full or additional content for me to assist you appropriately."
        )

        self.assertTrue(is_placeholder_summary(placeholder))
        self.assertIsNone(normalize_summary_text(placeholder))

    def test_brief_transcript_placeholder_becomes_short_call_summary(self):
        placeholder = "Please share the full transcript so I can summarize it."
        transcript = "How are you? Very good. Thank you."

        result = normalize_summary_text(
            placeholder,
            transcript_text=transcript,
            short_call_fallback=True,
        )

        self.assertEqual(result, SHORT_CALL_SUMMARY)

    def test_prefer_better_summary_replaces_placeholder_summary(self):
        existing = "The recruitment call transcript isn't fully provided. Please share the full transcript."
        incoming = "Very brief exchange; no meaningful screening details captured."

        result = prefer_better_summary(existing, incoming)

        self.assertEqual(result, incoming)

    def test_call_artifacts_need_repair_for_placeholder_summary(self):
        call_data = {
            "transcript": "How are you? Very good. Thank you.",
            "summary": "Please share the full transcript.",
        }

        self.assertTrue(call_artifacts_need_repair(call_data))

    def test_maybe_process_call_audio_forces_repair_when_summary_is_placeholder(self):
        previous_call = {
            "id": 10,
            "recording_url": "https://example.com/audio.mp3",
            "transcript": "How are you? Very good. Thank you.",
            "summary": "Please share the full transcript.",
        }
        updated_call = dict(previous_call)

        with patch("backend.api.routes.calls.process_call_audio") as process_mock:
            maybe_process_call_audio(previous_call, updated_call, force_fallback=True)

        process_mock.assert_called_once_with(10, "https://example.com/audio.mp3")


if __name__ == "__main__":
    unittest.main()
