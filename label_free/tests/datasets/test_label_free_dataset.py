#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path

import numpy as np

from label_free.datasets.label_free_dataset import LabelFreeDataset


def test_dataset(data_dir: Path) -> None:
    # This is a dataset with a single OME-TIFF stored remotely
    # The provided URI is an S3 URI
    ds = LabelFreeDataset(
        base_dataset=str(data_dir / "single-remote-ome-tiff-dataset.csv"),
        patches_per_image=10,
        image_reading_kwargs=dict(fs_kwargs=dict(anon=True)),
    )

    # Check length
    assert len(ds) == ds.patches_per_image

    # Check each patch is different from the last
    last_signal_patch, last_target_patch = ds[0]
    for i in range(1, len(ds)):
        # Get new patch
        current_signal_patch, current_target_patch = ds[i]

        # Check that we have different signal from target
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                current_signal_patch,
                current_target_patch,
            )

        # Check that we have different signal from last iter
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                last_signal_patch,
                current_signal_patch,
            )

        # Check that we have different target from last iter
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                last_target_patch,
                current_target_patch,
            )

        # Update last to current
        last_signal_patch = current_signal_patch
        last_target_patch = current_target_patch
