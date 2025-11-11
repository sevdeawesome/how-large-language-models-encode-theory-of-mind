#!/usr/bin/env python3
"""
Unit tests for ToM datasets and pipeline components.

Run with: pytest test_datasets.py -v
Or: python test_datasets.py
"""

import os
import json
import sys
from pathlib import Path

def test_tom_training_data():
    """Test original ToM training dataset structure."""
    path = "./tom_training_data.json"

    assert os.path.exists(path), f"Missing {path}"

    with open(path, 'r') as f:
        data = json.load(f)

    assert isinstance(data, list), "Should be a list"
    assert len(data) > 0, "Should not be empty"

    # Check first entry
    assert 'txt' in data[0], "Each entry should have 'txt' field"
    assert isinstance(data[0]['txt'], str), "'txt' should be string"
    assert len(data[0]['txt']) > 0, "'txt' should not be empty"

    print(f"✓ ToM training data: {len(data)} samples")
    return True


def test_simpletom_contrast_pairs():
    """Test SimpleTOM contrast pair dataset structure."""
    path = "./tom_dataset/simpletom_contrast_pairs.json"

    assert os.path.exists(path), f"Missing {path}"

    with open(path, 'r') as f:
        data = json.load(f)

    assert isinstance(data, list), "Should be a list"
    assert len(data) > 0, "Should not be empty"

    # Check required fields
    required_fields = [
        'id', 'category', 'scenario',
        'high_tom_prompt', 'high_tom_completion', 'high_tom_combined',
        'low_tom_prompt', 'low_tom_completion', 'low_tom_combined',
        'metadata'
    ]

    for field in required_fields:
        assert field in data[0], f"Missing required field: {field}"

    # Check metadata
    assert 'requires_false_belief' in data[0]['metadata']
    assert 'question_type' in data[0]['metadata']

    # Check that contrast pairs have similar length
    high_len = len(data[0]['high_tom_combined'])
    low_len = len(data[0]['low_tom_combined'])
    length_diff = abs(high_len - low_len)

    # Allow up to 30% length difference
    assert length_diff / max(high_len, low_len) < 0.3, \
        f"High/low ToM samples have very different lengths: {high_len} vs {low_len}"

    print(f"✓ SimpleTOM contrast pairs: {len(data)} pairs")
    print(f"  Categories: {set(item['category'] for item in data[:100])}")

    # Count false belief scenarios
    false_belief_count = sum(1 for item in data if item['metadata']['requires_false_belief'])
    print(f"  False belief scenarios: {false_belief_count}/{len(data)}")

    return True


def test_self_other_dataset():
    """Test self/other contrast pair dataset structure."""
    path = "./self_other_dataset/self_other.json"

    assert os.path.exists(path), f"Missing {path}"

    with open(path, 'r') as f:
        data = json.load(f)

    assert isinstance(data, list), "Should be a list"
    assert len(data) > 0, "Should not be empty"

    # Check required fields
    assert 'self_subject' in data[0], "Missing 'self_subject'"
    assert 'other_subject' in data[0], "Missing 'other_subject'"

    assert isinstance(data[0]['self_subject'], str)
    assert isinstance(data[0]['other_subject'], str)

    # Check that pairs have similar structure
    self_len = len(data[0]['self_subject'])
    other_len = len(data[0]['other_subject'])
    length_diff = abs(self_len - other_len)

    # These should be very similar in length (just pronoun changes)
    assert length_diff / max(self_len, other_len) < 0.2, \
        f"Self/other samples have very different lengths: {self_len} vs {other_len}"

    print(f"✓ Self/other contrast pairs: {len(data)} pairs")

    # Sample a few to show variety
    print(f"  Sample self-reference: {data[0]['self_subject'][:100]}...")

    return True


def test_dataset_balance():
    """Test that contrast pairs are well-balanced for gradient computation."""

    # Load SimpleTOM
    with open("./tom_dataset/simpletom_contrast_pairs.json", 'r') as f:
        simpletom = json.load(f)

    # Sample 100 pairs
    sample = simpletom[:100]

    high_lengths = [len(item['high_tom_combined']) for item in sample]
    low_lengths = [len(item['low_tom_combined']) for item in sample]

    import numpy as np

    high_mean = np.mean(high_lengths)
    low_mean = np.mean(low_lengths)

    # Check that means are within 20% of each other
    diff_pct = abs(high_mean - low_mean) / max(high_mean, low_mean) * 100

    print(f"\n✓ SimpleTOM balance check:")
    print(f"  High ToM mean length: {high_mean:.1f} chars")
    print(f"  Low ToM mean length:  {low_mean:.1f} chars")
    print(f"  Difference: {diff_pct:.1f}%")

    assert diff_pct < 20, f"High/low ToM samples poorly balanced: {diff_pct:.1f}% difference"

    # Same for self/other
    with open("./self_other_dataset/self_other.json", 'r') as f:
        self_other = json.load(f)

    sample = self_other[:100]

    self_lengths = [len(item['self_subject']) for item in sample]
    other_lengths = [len(item['other_subject']) for item in sample]

    self_mean = np.mean(self_lengths)
    other_mean = np.mean(other_lengths)

    diff_pct = abs(self_mean - other_mean) / max(self_mean, other_mean) * 100

    print(f"\n✓ Self/Other balance check:")
    print(f"  Self mean length:  {self_mean:.1f} chars")
    print(f"  Other mean length: {other_mean:.1f} chars")
    print(f"  Difference: {diff_pct:.1f}%")

    assert diff_pct < 20, f"Self/other samples poorly balanced: {diff_pct:.1f}% difference"

    return True


def test_existing_scripts():
    """Test that required Python scripts exist."""
    required_scripts = [
        'create_gradient.py',
        'chunk_gradient.py',
        'ToM_and_perplexity_evaluation.py',
        'summarize.py',
        'ToM_tasks.py'
    ]

    for script in required_scripts:
        assert os.path.exists(script), f"Missing required script: {script}"

    print(f"✓ All required scripts present: {', '.join(required_scripts)}")
    return True


def test_notebooks_exist():
    """Test that notebooks were created."""
    notebooks = [
        '01_paper_original_methodology.ipynb',
        '02_contrast_pair_methodology.ipynb'
    ]

    for nb in notebooks:
        assert os.path.exists(nb), f"Missing notebook: {nb}"

    print(f"✓ All notebooks created: {', '.join(notebooks)}")
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        ("ToM training data", test_tom_training_data),
        ("SimpleTOM contrast pairs", test_simpletom_contrast_pairs),
        ("Self/Other dataset", test_self_other_dataset),
        ("Dataset balance", test_dataset_balance),
        ("Existing scripts", test_existing_scripts),
        ("Notebooks", test_notebooks_exist),
    ]

    print("="*70)
    print("Running ToM Dataset Tests")
    print("="*70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[Test] {name}")
        print("-"*70)
        try:
            test_func()
            passed += 1
            print(f"✓ PASSED: {name}")
        except AssertionError as e:
            failed += 1
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {name}")
            print(f"  Error: {e}")

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
