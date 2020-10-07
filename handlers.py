# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence
from functools import partial

import torch

from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from monai.utils import MetricReduction, exact_version, optional_import

NotComputableError, _ = optional_import("ignite.exceptions", "0.4.2", exact_version, "NotComputableError")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")
sync_all_reduce, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "sync_all_reduce")


class HausdorffDistance(Metric):
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
            self,
            label_idx: int = 1,
            distance_metric: str = "euclidean",
            percentile: Optional[float] = None,
            directed: bool = False,
            output_transform: Callable = lambda x: x,
            device: Optional[torch.device] = None,
    ) -> None:
        """

        Args:
            label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: calculate directed Hausdorff distance. Defaults to ``False``.

        See also:
            :py:meth:`monai.metrics.hausdorff_distance.compute_hausdorff_distance`
        """
        super().__init__(output_transform, device=device)
        self.hd = partial(compute_hausdorff_distance, label_idx=label_idx, distance_metric=distance_metric, percentile=percentile, directed=directed)
        self._sum = 0.0
        self._num_examples = 0


    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. HausdorffDistance metric can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        score = self.hd(y_pred, y)

        # add all items in current batch
        self._sum += score
        self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self._num_examples == 0:
            raise NotComputableError("HausdorffDistance must have at least one example before it can be computed.")
        return self._sum / self._num_examples


class AvgSurfaceDistance(Metric):
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
            self,
            label_idx: int = 1,
            distance_metric: str = "euclidean",
            symmetric: bool = False,
            output_transform: Callable = lambda x: x,
            device: Optional[torch.device] = None,
    ) -> None:
        """

        Args:
            label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: calculate directed Hausdorff distance. Defaults to ``False``.

        See also:
            :py:meth:`monai.metrics.hausdorff_distance.compute_hausdorff_distance`
        """
        super().__init__(output_transform, device=device)
        self.sd = partial(compute_average_surface_distance, label_idx=label_idx, distance_metric=distance_metric,
                          symmetric=symmetric)
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. HausdorffDistance metric can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        score = self.sd(y_pred, y)

        # add all items in current batch
        self._sum += score
        self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self._num_examples == 0:
            raise NotComputableError("HausdorffDistance must have at least one example before it can be computed.")
        return self._sum / self._num_examples
