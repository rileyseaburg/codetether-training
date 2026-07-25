"""Verify VRAM-based model selection."""

import unittest

from model_training.vram_plan import plan


class VramPlanTest(unittest.TestCase):
    """Device capacity must decide the trainable model size."""

    def test_h100_trains_thirty_billion(self) -> None:
        result = plan(80.0)
        self.assertIn('30B', str(result['recommended_model']))
        self.assertEqual(result['max_length'], 8192)

    def test_a100_forty_reduces_sequence_length(self) -> None:
        result = plan(42.0)
        self.assertIn('30B', str(result['recommended_model']))
        self.assertEqual(result['max_length'], 4096)

    def test_twenty_four_gigabytes_still_trains_thirty_billion(self) -> None:
        result = plan(24.0)
        self.assertIn('30B', str(result['recommended_model']))
        self.assertEqual(result['max_length'], 4096)

    def test_tiny_device_falls_back_to_four_billion(self) -> None:
        result = plan(15.0)
        self.assertIn('4B', str(result['recommended_model']))

    def test_plan_reports_memory_estimate_within_device(self) -> None:
        result = plan(40.0)
        estimated = result['estimated']
        self.assertIsInstance(estimated, dict)
        self.assertLess(float(estimated['total_gb']), 40.0)


if __name__ == '__main__':
    unittest.main()
