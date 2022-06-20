#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass

from dataclasses_json import dataclass_json

###############################################################################


@dataclass_json
@dataclass(frozen=True)
class NormalizationConfig:
    clip_min: float
    clip_max: float
    min_: float
    max_: float


@dataclass_json
@dataclass(frozen=True)
class Dimensions:
    Z: int
    Y: int
    X: int
