# colintern_models.py
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from mteb.models.colpali_models import ColPaliEngineWrapper
from mteb.requires_package import requires_package


class ColInternWrapper(ColPaliEngineWrapper):
    """
    Thin wrapper around ColPaliEngineWrapper that:
      - carries runtime knobs (batch_size/num_workers/pin_memory)
      - sets processor device kwargs for MTEB
      - caps visual tokens safely (valid size dict)
    """

    def __init__(self, model_name="vidore/colintern3.5", revision=None, device=None, **kwargs):
        batch_size = kwargs.pop("batch_size", 1)
        num_workers = kwargs.pop("num_workers", 0)  # force single-process
        pin_memory = kwargs.pop("pin_memory", False)
        max_num_visual_tokens = kwargs.pop("max_num_visual_tokens", None)

        requires_package(self, "colpali_engine", model_name, "pip install mteb[colpali_engine]")
        from colpali_engine.models import ColIntern3_5, ColIntern3_5_Processor

        super().__init__(
            model_name=model_name,
            model_class=ColIntern3_5,
            processor_class=ColIntern3_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )

        # Ensure processor kwargs include device for downstream calls
        if not hasattr(self, "processor_kwargs") or self.processor_kwargs is None:
            self.processor_kwargs = {}
        try:
            import torch
            if device is not None:
                self.processor_kwargs.setdefault(
                    "device", torch.device(device) if isinstance(device, str) else device
                )
            else:
                self.processor_kwargs.setdefault("device", next(self.model.parameters()).device)
        except Exception:
            pass

        # Runtime knobs used by base wrapper when building DataLoaders
        self.batch_size = batch_size
        self.num_workers = 0  # harden to 0
        self.pin_memory = False if pin_memory is None else bool(pin_memory)