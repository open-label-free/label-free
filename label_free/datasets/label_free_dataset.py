#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import lru_cache, partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.transform import resize
from torch.utils.data import Dataset

from ..types import Dimensions, NormalizationConfig

###############################################################################


class LabelFreeDatasetColumns:
    signal_image_uri = "signal_image_uri"
    signal_image_channel = "signal_image_channel"
    target_image_uri = "target_image_uri"
    target_image_channel = "target_image_channel"


###############################################################################


class LabelFreeDataset(Dataset):
    """
    A basic dataset for Label Free training. This object can handle local or remote
    files. Loads the images provided with AICSImageIO. Loads, normalization, and
    downsampling are only done on a cache miss (cached on the combination of image
    URI and requested channel).

    Parameters
    ----------
    base_dataset: Union[str, pd.DataFrame]
        The base dataset to build off of. Provide either a string path to a CSV file or
        provide a loaded pandas DataFrame. The DataFrame should have the columns:
        'signal_image_uri', 'signal_image_channel', 'target_image_uri',
        and 'target_image_channel'
    normalization_config: Optional[NormalizationConfig]
        Optional normalization parameters.
        Default: None (normalize each requested image channel independently)
    downsample_dims: Dimensions
        The size to downsample the selected Z stack to.
        Default: Z: 32, Y: 128, X: 128
    patch_dims: Dimensions
        The size of the patches.
        Default: Z: 16, Y: 64, X: 64
    patches_per_image: int
        The number of patches to create for each image.
        Default: 1000
    image_reading_kwargs: Dict[str, Any]
        Any extra keyword arguments to pass to the AICSImageIO constructor.
    """

    def __init__(
        self,
        base_dataset: Union[str, pd.DataFrame],
        normalization_config: Optional[NormalizationConfig] = None,
        downsample_dims: Dimensions = Dimensions(32, 128, 128),
        patch_dims: Dimensions = Dimensions(16, 64, 64),
        patches_per_image: int = 1000,
        image_reading_kwargs: Dict[str, Any] = {},
    ):
        # Load dataset
        if isinstance(base_dataset, str):
            base_dataset = pd.read_csv(base_dataset)

        # Store general
        self.base_dataset = base_dataset
        self.normalization_config = normalization_config
        self.downsample_dims = downsample_dims
        self.patch_dims = patch_dims
        self.patches_per_image = patches_per_image

        # Dicts aren't hashable so make a partial of the image reader
        self.aicsimage_loader = partial(AICSImage, **image_reading_kwargs)

        # Store state
        self.current_dataset_index = 0
        self.current_patch_index = 0

    @staticmethod
    def _normalize_arr(
        arr: np.ndarray,
        clip_min: float,
        clip_max: float,
        subtract_all: Union[float, Callable],
        div_all: Union[float, Callable],
    ) -> np.ndarray:
        # Convert to float64 for "safe" math
        arr = arr.astype(np.float64)

        # Clip upper bound
        arr = np.clip(
            arr,
            a_min=np.percentile(arr, clip_min),
            a_max=np.percentile(arr, clip_max),
        )

        # Normalize grayscale values to floats between zero and one
        if isinstance(subtract_all, float):
            arr = arr - subtract_all
        else:
            arr = arr - subtract_all(arr)
        if isinstance(div_all, float):
            arr = arr / div_all
        else:
            arr = arr / div_all(arr)

        return arr

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_image(
        aicsimage_loader: Callable,
        uri: str,
        channel: Union[str, int],
        normalization_config: Optional[NormalizationConfig] = None,
        downsample_dims: Dimensions = Dimensions(32, 128, 128),
    ) -> np.ndarray:
        """
        Cached image loader. Because normalization config and dimensions should
        not change the entire training process, this is really caching on the image
        URI and the requested channel pairing.

        It will only cache 4 pairs which usually means the current image and the last
        image.
        """
        # Load AICSImage object
        img = aicsimage_loader(uri)

        # Handle channel by name
        if isinstance(channel, str):
            channel = img.channel_names.index(channel)

        # Pull channel specific Z
        z_stack = img.get_image_dask_data("ZYX", C=channel).compute()

        # Optional normalization
        if normalization_config:
            z_stack = LabelFreeDataset._normalize_arr(
                z_stack,
                clip_min=normalization_config.clip_min,
                clip_max=normalization_config.clip_max,
                subtract_all=normalization_config.min_,
                div_all=normalization_config.max_,
            )
        else:
            z_stack = LabelFreeDataset._normalize_arr(
                z_stack,
                clip_min=0.02,
                clip_max=99.8,
                subtract_all=np.min,
                div_all=np.max,
            )

        # Downsample
        z_stack = resize(
            z_stack,
            (downsample_dims.Z, downsample_dims.Y, downsample_dims.X),
        )

        # Cast the floats to integers
        imax = np.iinfo(np.uint16).max + 1  # eg imax = 256 for uint8
        z_stack = z_stack * imax
        z_stack[z_stack == imax] = imax - 1
        z_stack = z_stack.astype(np.int32)

        return z_stack

    def __len__(self) -> int:
        return len(self.base_dataset) * self.patches_per_image

    @staticmethod
    def _generate_random_index_selection(
        full_dims: Dimensions,
        patch_dims: Dimensions,
    ) -> Dimensions:
        return Dimensions(
            Z=np.random.randint(0, full_dims.Z - patch_dims.Z),
            Y=np.random.randint(0, full_dims.Y - patch_dims.Y),
            X=np.random.randint(0, full_dims.X - patch_dims.X),
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Calc requested dataset row index
        dataset_idx = idx // self.patches_per_image

        # Get dataset row
        details = self.base_dataset.iloc[dataset_idx]

        # Load, norm, and downsample zstacks
        z_stack_signal = LabelFreeDataset._load_image(
            aicsimage_loader=self.aicsimage_loader,
            uri=details[LabelFreeDatasetColumns.signal_image_uri],
            channel=details[LabelFreeDatasetColumns.signal_image_channel],
            normalization_config=self.normalization_config,
            downsample_dims=self.downsample_dims,
        )
        z_stack_target = LabelFreeDataset._load_image(
            aicsimage_loader=self.aicsimage_loader,
            uri=details[LabelFreeDatasetColumns.target_image_uri],
            channel=details[LabelFreeDatasetColumns.target_image_channel],
            normalization_config=self.normalization_config,
            downsample_dims=self.downsample_dims,
        )

        # Get random patches
        random_patch_start_dims = LabelFreeDataset._generate_random_index_selection(
            full_dims=self.downsample_dims,
            patch_dims=self.patch_dims,
        )
        z_stack_signal_patch = z_stack_signal[
            random_patch_start_dims.Z : random_patch_start_dims.Z + self.patch_dims.Z,
            random_patch_start_dims.Y : random_patch_start_dims.Y + self.patch_dims.Y,
            random_patch_start_dims.X : random_patch_start_dims.X + self.patch_dims.X,
        ]
        z_stack_target_patch = z_stack_target[
            random_patch_start_dims.Z : random_patch_start_dims.Z + self.patch_dims.Z,
            random_patch_start_dims.Y : random_patch_start_dims.Y + self.patch_dims.Y,
            random_patch_start_dims.X : random_patch_start_dims.X + self.patch_dims.X,
        ]
        return (z_stack_signal_patch, z_stack_target_patch)
