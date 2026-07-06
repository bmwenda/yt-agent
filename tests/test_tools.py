import unittest

from tools import extract_video_id


class ToolTests(unittest.TestCase):
    def test_extract_video_id_supports_shorts_urls(self):
        self.assertEqual(
            extract_video_id.invoke({"url": "https://www.youtube.com/shorts/dQw4w9WgXcQ"}),
            "dQw4w9WgXcQ",
        )

    def test_extract_video_id_rejects_invalid_urls(self):
        self.assertEqual(
            extract_video_id.invoke({"url": "https://example.com/video"}),
            "Error: Invalid YouTube URL",
        )


if __name__ == "__main__":
    unittest.main()
