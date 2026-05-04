import pandas as pd
from ._base import get_segments_and_duration, logger, skip_if_exists
from .common import collect_valid_segments, score_regex_then_ensemble, spread_scores_over_timeline
from .constants import VIEWER_ENGAGEMENT_COLS, VIEWER_ENGAGEMENT_PATTERN, VIEWER_ENGAGEMENT_REGEX_SCORE, VIEWER_ENGAGEMENT_TASK


def extract_viewer_engagement(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(VIEWER_ENGAGEMENT_COLS, existing_features, "viewer_engagement"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segments(segments, duration)
    scores, regex_hits = score_regex_then_ensemble(valid, VIEWER_ENGAGEMENT_PATTERN, VIEWER_ENGAGEMENT_REGEX_SCORE, VIEWER_ENGAGEMENT_TASK, config)
    out = spread_scores_over_timeline(valid, scores, duration)

    n_strong = int((scores > 0.5).sum())
    logger.info("viewer_engagement: %d/%d segments > 0.5 (%d regex, %d ensemble), mean=%.3f", n_strong, len(valid), regex_hits, n_strong - regex_hits, float(scores.mean()))
    return pd.DataFrame({"viewer_engagement": out})
