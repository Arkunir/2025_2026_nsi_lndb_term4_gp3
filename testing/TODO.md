# TODO: Remove Home/Away Probability Advantage

## Tasks
- [x] Modify add_features in main.py to compute neutral combined features (e.g., average of home and away stats) instead of separate home/away features.
- [x] Update feature_cols in main.py to use the new neutral features.
- [x] Update feature_cols in predict_matches.py to match the new neutral features.
- [x] Verify home_adv=0 in compute_elo calls (already set).
- [ ] Test the changes by running the prediction script.
