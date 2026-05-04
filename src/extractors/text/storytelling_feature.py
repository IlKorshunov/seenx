import pandas as pd
from ._base import get_segments_and_duration, logger, skip_if_exists
from .common import collect_valid_segments, score_regex_then_ensemble, spread_scores_over_timeline
from .constants import STORYTELLING_COLS, STORYTELLING_ENSEMBLE_THRESHOLD, STORYTELLING_PATTERN, STORYTELLING_REGEX_SCORE, STORYTELLING_TASK


def extract_storytelling(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(STORYTELLING_COLS, existing_features, "storytelling"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segments(segments, duration)
    scores, regex_hits = score_regex_then_ensemble(
        valid, STORYTELLING_PATTERN, STORYTELLING_REGEX_SCORE, STORYTELLING_TASK, config, ensemble_threshold=STORYTELLING_ENSEMBLE_THRESHOLD
    )
    out = spread_scores_over_timeline(valid, scores, duration)
    n_strong = int((scores > 0).sum())
    logger.info(
        "storytelling: %d/%d segments active (%d regex, %d ensemble >= %.2f), mean_active=%.3f",
        n_strong,
        len(valid),
        regex_hits,
        n_strong - regex_hits,
        STORYTELLING_ENSEMBLE_THRESHOLD,
        float(scores[scores > 0].mean()) if n_strong > 0 else 0.0,
    )
    return pd.DataFrame({"storytelling": out})
