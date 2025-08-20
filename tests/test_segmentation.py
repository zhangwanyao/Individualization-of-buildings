import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from segmentation import _reindex_instances

def test_reindex_instances_relabels_positive_ids_and_preserves_unassigned():
    original = np.array([5, -1, 3, 3, 0, 2, 5, -4])
    result = _reindex_instances(original)
    expected = np.array([3, 0, 2, 2, 0, 1, 3, 0])

    assert np.array_equal(result, expected)

    positives = result[result > 0]
    assert np.array_equal(np.unique(positives), np.arange(1, positives.max() + 1))
    assert np.all(result[original <= 0] == 0)
