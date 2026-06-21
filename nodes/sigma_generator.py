"""
nodes/sigma_generator.py
============================
Internal pipeline
-----------------
  KSamplerSelect  ─────────────────────────────────────► SAMPLER output
  BasicScheduler  (model, scheduler, steps, denoise) ──► SIGMAS
  SplitSigmas     (sigmas, skip_start_step)           ──► low_sigmas ► SIGMAS output

Inputs
------
  model            MODEL   – diffusion model
  sampler_name     COMBO   – sampler algorithm
  scheduler        COMBO   – noise schedule
  steps            INT     – total steps  (default 4)
  skip_start_step  INT     – SplitSigmas step; skips that many leading sigmas (default 0)
  denoise          FLOAT   – denoising strength 0-1 (default 1.0)

Outputs
-------
  SIGMAS   – low_sigmas portion after the split
  SAMPLER  – the selected sampler object
"""

import comfy.samplers


# ---------------------------------------------------------------------------
# Helper – mirrors BasicScheduler behaviour
# ---------------------------------------------------------------------------

def _run_basic_scheduler(model, scheduler: str, steps: int, denoise: float):
    """Return a full sigma schedule tensor (same logic as the core node)."""
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (
                model.get_model_object("model_sampling")
                .sigma(
                    model.get_model_object("model_sampling").timestep(
                        model.get_model_object("model_sampling").sigma_max
                    )
                )
                .unsqueeze(0)
                * 0
            )
        total_steps = max(int(round(steps / denoise)), 1)

    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"),
        scheduler,
        total_steps,
    )

    # Keep only the last `steps + 1` entries (matches denoise trimming)
    sigmas = sigmas[-(steps + 1):]
    return sigmas


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SigmaGenerator:
    """
    Sigma Generator (SG)

    Replicates the subgraph:
      KSamplerSelect → BasicScheduler → SplitSigmas

    Returns the *low_sigmas* half (steps after skip_start_step)
    plus the sampler object, ready to feed into a SamplerCustom node.
    """

    CATEGORY = "YSNodes/sampling"
    FUNCTION = "generate"
    RETURN_TYPES = ("SIGMAS", "SAMPLER")
    RETURN_NAMES = ("SIGMAS", "SAMPLER")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps": (
                    "INT",
                    {"default": 4, "min": 1, "max": 10000, "step": 1},
                ),
                "skip_start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": (
                            "How many leading sigma steps to skip "
                            "(SplitSigmas 'step' input). "
                            "0 = keep all sigmas."
                        ),
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Denoising strength — values < 1.0 reduce effective steps (e.g., 0.5 with 20 steps → 10 steps)",
                    },
                ),
            }
        }

    def generate(
        self,
        model,
        sampler_name: str,
        scheduler: str,
        steps: int,
        skip_start_step: int,
        denoise: float,
    ):
        # ── KSamplerSelect ──────────────────────────────────────────────
        sampler = comfy.samplers.sampler_object(sampler_name)

        # ── BasicScheduler ──────────────────────────────────────────────
        sigmas = _run_basic_scheduler(model, scheduler, steps, denoise)

        # ── SplitSigmas ─────────────────────────────────────────────────
        # high_sigmas = sigmas[:skip_start_step + 1]  (discarded)
        low_sigmas = sigmas[skip_start_step:]

        return (low_sigmas, sampler)


# ---------------------------------------------------------------------------
# Registration — picked up automatically by comfyui-ysnodes/__init__.py
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SigmaGenerator": SigmaGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmaGenerator": "Sigma Generator",
}
