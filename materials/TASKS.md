# ML Portfolio Project 1: Materials Property Prediction (Regression)

## File Provided
- `data/alloy_properties.csv` — 1,215 synthetic aluminum alloy samples

## Dataset Description
Each row represents one experimental/processed alloy sample with:
- **Composition** (wt %): Al_pct, Mg_pct, Si_pct, Cu_pct, Zn_pct, Fe_pct
- **Processing parameters**: aging_temp_C, aging_time_hr, cooling_rate_Cps
- **Microstructure descriptor**: grain_size_um
- **Target properties** (what you'll predict): tensile_strength_MPa, elongation_pct, hardness_HV

The data follows a real materials-science logic: strength increases with solute content (Mg, Cu, Zn) and grain refinement (Hall-Petch-like effect), peaks at an optimal aging temperature (overaging reduces strength), and trades off against elongation (classic strength-ductility tradeoff). This is loosely modeled on precipitation-hardened aluminum alloys (6xxx/7xxx series), but the exact relationship is synthetic — built so a model can genuinely learn it, with realistic noise on top.

The dataset deliberately includes messiness: missing `grain_size_um` values (microscopy wasn't done on every sample), duplicate rows, and a few corrupted tensile readings (instrument glitch) that read roughly 2.5x too high. Cleaning these is part of the exercise.

---

## Task List (Beginner → Advanced)

### Level 1 — Data Exploration and Cleaning
1. Load the CSV and inspect shape, dtypes, and summary statistics (`.describe()`).
2. Identify and handle missing values in `grain_size_um` (try at least two approaches: drop rows vs. impute with median/group-mean, and compare).
3. Detect the duplicate rows and remove them.
4. Detect the outlier tensile_strength_MPa readings (hint: look for values far outside the IQR or more than 3 standard deviations from the mean) and decide how to handle them — remove, cap, or flag.
5. Plot histograms of each composition variable and each target property. Note which targets look roughly normal vs. skewed.

### Level 2 — Exploratory Analysis and Feature Relationships
6. Create a correlation heatmap between all numeric features and `tensile_strength_MPa`. Which features correlate most strongly?
7. Plot `tensile_strength_MPa` vs `aging_temp_C` and visually check whether there's a peak (optimal aging temperature) rather than a straight line.
8. Plot `tensile_strength_MPa` vs `elongation_pct` to confirm the strength-ductility tradeoff visually.
9. Engineer a new feature: total solute content (Mg_pct + Si_pct + Cu_pct + Zn_pct) and check its correlation with strength.
10. Split the data into train/test sets (80/20), with cleaning applied beforehand.

### Level 3 — Baseline Regression Models
11. Train a Linear Regression model to predict `tensile_strength_MPa` from composition and processing features. Report R² and RMSE on the test set.
12. Train a Random Forest Regressor on the same target and compare R²/RMSE to the linear model. Which performs better, and why might that be (hint: think about the aging-temperature peak, which isn't a linear relationship)?
13. Use the Random Forest's `feature_importances_` to identify which features matter most for predicting strength. Do they match what you'd expect from materials science intuition?
14. Repeat the modeling process for `hardness_HV` and `elongation_pct` as separate targets.

### Level 4 — Model Improvement and Validation
15. Try a Gradient Boosting Regressor (e.g. XGBoost or scikit-learn's GradientBoostingRegressor) and compare performance to Random Forest.
16. Use k-fold cross-validation (5-fold) instead of a single train/test split to get a more robust performance estimate.
17. Engineer a quadratic feature for aging_temp_C (e.g. (aging_temp_C - 175)^2) to help a linear model capture the non-linear peak, and see if it closes the performance gap with the tree-based models.
18. Build a simple feed-forward neural network (using scikit-learn's MLPRegressor or PyTorch/TensorFlow if you want the deep-learning angle for your portfolio) to predict tensile_strength_MPa, and compare its performance to the tree-based models.
19. Create a residual plot (predicted vs. actual, and predicted vs. residual) for your best model. Are there particular ranges of strength where the model performs worse?
20. Write a one-page summary (this is the deliverable that matters most for a portfolio): which model you'd recommend for a client, what its expected error is in MPa, which 3 features matter most, and one actionable insight a materials engineer could use (e.g. "samples aged above 200°C show a sharp drop in predicted strength, consistent with overaging").

---

## Why This Project Matters For Your Portfolio
This directly demonstrates the Materials Engineer + ML combination from your Fiverr niche strategy — you're not just fitting a model, you're showing you understand *why* the model's findings make physical sense (the aging-temperature peak, the strength-ductility tradeoff, the Hall-Petch grain-size effect). That interpretive layer is exactly what a generic ML freelancer can't offer, and it's the single best portfolio piece you can show a materials-science or R&D client.
