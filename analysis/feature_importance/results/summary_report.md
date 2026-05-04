# Feature Importance Analysis — Summary Report
**Target:** `target_avg_retention`
**Results directory:** `/home/kolya/ilya/seenx-ml/analysis/feature_importance/results`

## Pipelines Run
- ✅ **catboost** (1.1s)
- ✅ **correlation** (1.6s)
- ✅ **transformer** (2.1s)

## Top 20 Features (Consensus Ranking)

| Rank | Feature | Group | Avg Rank | Methods |
|------|---------|-------|----------|---------|
| 1 | `motion_speed` | visual_motion | 8.0 | 8 |
| 2 | `pause_rate` | audio_speech | 12.0 | 8 |
| 3 | `speech_rate_cv` | audio_speech | 14.4 | 8 |
| 4 | `rolloff` | audio_basic | 15.8 | 8 |
| 5 | `viewer_address` | text_content | 18.9 | 8 |
| 6 | `centroid` | audio_basic | 21.1 | 8 |
| 7 | `sentiment_intensity` | unknown | 22.5 | 8 |
| 8 | `radial_ratio` | visual_motion | 22.5 | 8 |
| 9 | `text_prob` | visual_content | 23.0 | 8 |
| 10 | `hook_score` | hook | 23.1 | 8 |
| 11 | `scene_novelty` | visual_motion | 23.2 | 8 |
| 12 | `frame` | visual_motion | 25.1 | 8 |
| 13 | `syntactic_depth` | text_complexity | 25.1 | 8 |
| 14 | `topic_shift` | unknown | 26.8 | 8 |
| 15 | `edit_pace` | visual_motion | 27.0 | 8 |
| 16 | `hook_score_x_time_pct` | interaction | 27.6 | 8 |
| 17 | `sentiment_polarity` | unknown | 28.4 | 8 |
| 18 | `cinematic` | visual_quality | 28.5 | 8 |
| 19 | `beat_sync` | audio_music | 28.8 | 8 |
| 20 | `vocal_zcr` | audio_vocal | 29.1 | 8 |

## Feature Groups (by consensus importance)

| Group | Avg Rank (lower = more important) |
|-------|-----------------------------------|
| text_content | 18.9 |
| audio_basic | 27.7 |
| visual_motion | 28.1 |
| unknown | 30.5 |
| audio_vocal | 33.9 |
| visual_quality | 34.5 |
| audio_speech | 35.9 |
| interaction | 36.8 |
| emotion | 37.3 |
| hook | 38.1 |
| audio_music | 38.6 |
| audio_loudness | 40.2 |
| ad | 40.2 |
| visual_content | 40.8 |
| text_complexity | 41.8 |
| temporal | 45.6 |

## Output Files

### `catboost/`
- `importance_consensus.csv`
- `importance_groups.csv`
- `importance_groups.png`
- `importance_lfc.csv`
- `importance_lfc.png`
- `importance_pvc.csv`
- `importance_pvc.png`
- `importance_shap.csv`
- `importance_shap.png`
- `shap_beeswarm.png`

### `correlation/`
- `combined_ranking.csv`
- `feature_corr_heatmap.png`
- `feature_corr_matrix.csv`
- `group_corr_heatmap.png`
- `mutual_information.csv`
- `mutual_information.png`
- `pearson_agg.csv`
- `pearson_agg.png`
- `redundant_features.txt`
- `redundant_pairs.csv`
- `spearman_agg.csv`
- `spearman_agg.png`
- `spearman_timeseries.csv`
- `spearman_timeseries.png`

### `permutation/`
- `perm_importance_comparison.png`
- `perm_importance_rf.csv`
- `perm_importance_rf.png`
- `perm_importance_ridge.csv`
- `perm_importance_ridge.png`

### `shap/`
- `dependence_plots`
- `shap_group_bar.png`
- `shap_importance.csv`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_values.csv`

### `transformer/`
- `transformer_attention_heatmap.png`
- `transformer_attention_importance.csv`
- `transformer_attention_importance.png`
- `transformer_attention_matrix.csv`
- `transformer_combined_importance.png`
- `transformer_gradient_importance.csv`
- `transformer_gradient_importance.png`
- `transformer_training_curve.png`

