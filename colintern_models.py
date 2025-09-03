from __future__ import annotations

import logging
from functools import partial

logger = logging.getLogger(__name__)
from mteb.models.colpali_models import ColPaliEngineWrapper
from mteb.requires_package import requires_package

class ColInternWrapper(ColPaliEngineWrapper):
    """Wrapper for ColIntern3.5 models (InternVL3.5-based multi-modal model)."""

    def __init__(
        self,
        model_name: str = "vidore/colintern3.5",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )

        # Import the model and processor classes for ColIntern3.5
        from colpali_engine.models import ColIntern3_5, ColIntern3_5_Processor

        super().__init__(
            model_name=model_name,
            model_class=ColIntern3_5,
            processor_class=ColIntern3_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )

        # --- MTEB compatibility: similarity() forwards **self.processor_kwargs
        if not hasattr(self, "processor_kwargs") or self.processor_kwargs is None:
            self.processor_kwargs = {}
        # ensure MaxSim scoring runs on the right device
        try:
            import torch
            if device is not None:
                self.processor_kwargs.setdefault(
                    "device", torch.device(device) if isinstance(device, str) else device
                )
            else:
                self.processor_kwargs.setdefault(
                    "device", next(self.model.parameters()).device
                )
        except Exception:
            pass
