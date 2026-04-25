import unittest

from backend.services.frejun_calls import extract_transcript_text


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


if __name__ == "__main__":
    unittest.main()
