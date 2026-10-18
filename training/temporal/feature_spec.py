"""Combine authoritative public encoder/preprocessing recipes into one binding."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from training.guards import HEX64, canonical_sha256, require


def build_feature_spec(encoder_recipe: dict[str, Any], *, decoder_recipe_sha256: str,
                       sampling_recipe_sha256: str, crop_recipe_sha256: str) -> tuple[dict[str, Any], str]:
    """Keep A15's complete encoder recipe verbatim; hash every source/view policy.

    This is the private cache/HeadIdentity.feature_spec_sha256 binding, not a
    prediction schema or a substitute for A15's encoder artifact verification.
    No source names, per-video identities or labels belong in a generic recipe.
    """
    require(isinstance(encoder_recipe, dict) and bool(encoder_recipe), "encoder_feature_recipe_missing")
    require(all(isinstance(value, str) and HEX64.fullmatch(value) for value in
                (decoder_recipe_sha256, sampling_recipe_sha256, crop_recipe_sha256)),
            "feature_recipe_policy_hash_invalid")
    spec = {"schema_version": "forte.frozen-feature-spec.v1", "encoder_recipe": deepcopy(encoder_recipe),
            "encoder_recipe_sha256": canonical_sha256(encoder_recipe),
            "decoder_recipe_sha256": decoder_recipe_sha256, "sampling_recipe_sha256": sampling_recipe_sha256,
            "crop_recipe_sha256": crop_recipe_sha256, "view_layout": "TVD", "global_view_required": True,
            "frame_identity_protocol": "content_sha256:source_pts:canonical_rational_time_base",
            "time_units": "original_source_seconds", "feature_coverage_implies_visibility": False}
    return spec, canonical_sha256(spec)
