import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "scikit-learn", "pandas", "numpy", "matplotlib"])
# ----------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)
sns.set_theme(style="whitegrid")

# ==========================================
# SETUP: SYNTHETIC DATA GENERATION FOR PORTFOLIO
# ==========================================
def generate_portfolio_data(n_samples=1215):
    """Generates synthetic aluminum alloy dataset with realistic physical anomalies."""
    # Compositions (wt %)
    Al_pct = np.random.uniform(88.0, 95.0, n_samples)
    Mg_pct = np.random.uniform(0.5, 3.0, n_samples)
    Si_pct = np.random.uniform(0.3, 1.5, n_samples)
    Cu_pct = np.random.uniform(0.1, 2.0, n_samples)
    Zn_pct = np.random.uniform(0.0, 5.0, n_samples)
    Fe_pct = np.random.uniform(0.1, 0.5, n_samples)
    
    # Processing parameters
    aging_temp_C = np.random.uniform(140, 220, n_samples)
    aging_time_hr = np.random.uniform(2, 24, n_samples)
    cooling_rate_Cps = np.random.uniform(0.5, 50.0, n_samples)
    
    # Microstructure (Hall-Petch effect)
    grain_size_um = np.random.lognormal(mean=2.5, sigma=0.4, size=n_samples)
    
    # Physical logic formulas with noise
    solute_total = Mg_pct + Si_pct + Cu_pct + Zn_pct
    
    # Non-linear relationship for aging temperature (peak around 175C)
    aging_effect = -0.05 * (aging_temp_C - 175)**2 + 50
    grain_effect = 150 / np.sqrt(grain_size_um)
    
    # Targets
    tensile_strength_MPa = 150 + (solute_total * 35) + aging_effect + grain_effect + np.random.normal(0, 15, n_samples)
    hardness_HV = tensile_strength_MPa * 0.3 + np.random.normal(0, 5, n_samples)
    elongation_pct = 45 - (tensile_strength_MPa * 0.08) + np.random.normal(0, 2, n_samples)
    elongation_pct = np.clip(elongation_pct, 2, 35) # keep realistic physical boundaries
    
    df = pd.DataFrame({
        'Al_pct': Al_pct, 'Mg_pct': Mg_pct, 'Si_pct': Si_pct, 'Cu_pct': Cu_pct, 'Zn_pct': Zn_pct, 'Fe_pct': Fe_pct,
        'aging_temp_C': aging_temp_C, 'aging_time_hr': aging_time_hr, 'cooling_rate_Cps': cooling_rate_Cps,
        'grain_size_um': grain_size_um,
        'tensile_strength_MPa': tensile_strength_MPa, 'elongation_pct': elongation_pct, 'hardness_HV': hardness_HV
    })
    
    # Inject Messiness intentionally
    # 1. Missing values in grain size (~10%)
    df.loc[df.sample(frac=0.10).index, 'grain_size_um'] = np.nan
    # 2. Duplicates (~25 rows)
    dup_indices = df.sample(25).index
    df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)
    # 3. Corrupted tensile strength readings (instrument glitch: 2.5x spike)
    glitch_indices = df.sample(12).index
    df.loc[glitch_indices, 'tensile_strength_MPa'] *= 2.5
    
    return df

# Create data directory and save to simulate project structure
import os
os.makedirs('data', exist_ok=True)
df_raw = generate_portfolio_data()
df_raw.to_csv('data/alloy_properties.csv', index=False)

print("⚡ Synthetic dataset successfully created and saved to 'data/alloy_properties.csv'!")

# ==========================================
# LEVEL 1 — DATA EXPLORATION AND CLEANING
# ==========================================
print("\n--- LEVEL 1: Data Exploration and Cleaning ---")
# 1. Load and Inspect
df = pd.read_csv(r'D:\Automation\materials\alloy_properties.csv')
print(df.info())
print("\nSummary Statistics:\n", df.describe().T)

# 2. Handle missing values in grain_size_um
print(f"\nInitial Missing Grain Size Values: {df['grain_size_um'].isna().sum()}")
# Approach A: Drop rows
df_dropped = df.dropna(subset=['grain_size_um'])
# Approach B: Impute with median (Preferred for data preservation)
median_grain_size = df['grain_size_um'].median()
df['grain_size_um'] = df['grain_size_um'].fillna(median_grain_size)
print(f"Missing values remaining after Imputation: {df['grain_size_um'].isna().sum()}")

# 3. Detect and remove duplicate rows
print(f"Duplicate count before removal: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate count after removal: {df.duplicated().sum()}")

# 4. Detect and handle tensile strength outlier glitches
Q1 = df['tensile_strength_MPa'].quantile(0.25)
Q3 = df['tensile_strength_MPa'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
print(f"IQR Outlier upper bound: {upper_bound:.2f} MPa")

outliers = df[df['tensile_strength_MPa'] > upper_bound]
print(f"Detected {len(outliers)} instrument glitch outliers.")
# Filter out the instrument glitches
df = df[df['tensile_strength_MPa'] <= upper_bound].reset_index(drop=True)

# 5. Plot distributions of composition and target variables
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
features_to_plot = ['Al_pct', 'Mg_pct', 'Si_pct', 'Cu_pct', 'Zn_pct', 'Fe_pct', 
                    'aging_temp_C', 'grain_size_um', 'tensile_strength_MPa', 'elongation_pct', 'hardness_HV']
axes = axes.flatten()

for i, col in enumerate(features_to_plot):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}', fontsize=10)
    
# Delete unused axis
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig('distributions.png', dpi=300)
plt.close()
print("Saved feature distributions to 'distributions.png'")

# ==========================================
# LEVEL 2 — EXPLORATORY ANALYSIS & FEATURE RELATIONSHIPS
# ==========================================
print("\n--- LEVEL 2: Exploratory Analysis ---")

# 6. Heatmap matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Processing & Material Properties")
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

strong_corr = corr_matrix['tensile_strength_MPa'].sort_values(ascending=False)
print("\nCorrelations with Tensile Strength:\n", strong_corr)

# 7. Non-linear relationship check: Strength vs Aging Temp
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='aging_temp_C', y='tensile_strength_MPa', alpha=0.6, color='crimson')
plt.title("Tensile Strength vs. Aging Temperature (Thermal Processing)")
plt.tight_layout()
plt.savefig('strength_vs_aging_temp.png', dpi=300)
plt.close()

# 8. Tradeoff visualization: Strength vs Ductility
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='elongation_pct', y='tensile_strength_MPa', hue='hardness_HV', palette='viridis', alpha=0.7)
plt.title("The Strength-Ductility Trade-off Curve")
plt.tight_layout()
plt.savefig('strength_ductility_tradeoff.png', dpi=300)
plt.close()

# 9. Feature Engineering: Total solute content
df['total_solute_pct'] = df['Mg_pct'] + df['Si_pct'] + df['Cu_pct'] + df['Zn_pct']
new_corr = df['total_solute_pct'].corr(df['tensile_strength_MPa'])
print(f"\nCorrelation between Engineered 'total_solute_pct' and Strength: {new_corr:.4f}")

# 10. Split Data into Train/Test (80/20)
X = df.drop(columns=['tensile_strength_MPa', 'elongation_pct', 'hardness_HV'])
y_ts = df['tensile_strength_MPa']

X_train, X_test, y_train, y_test = train_test_split(X, y_ts, test_size=0.2, random_state=42)
print(f"Data Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# LEVEL 3 — BASELINE REGRESSION MODELS
# ==========================================
print("\n--- LEVEL 3: Baseline Regression Models ---")

# Helper function to print evaluation stats
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)
    return rmse, r2, preds

# 11. Linear Regression
lr_model = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
lr_rmse, lr_r2, _ = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
print(f"Linear Regression -> R²: {lr_r2:.4f}, RMSE: {lr_rmse:.2f} MPa")

# 12. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rmse, rf_r2, rf_preds = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
print(f"Random Forest Regressor -> R²: {rf_r2:.4f}, RMSE: {rf_rmse:.2f} MPa")
print("Insight: Random Forest vastly outperforms Linear Regression due to non-linear physical constraints (like aging-temperature peaks).")

# 13. Feature Importances via Random Forest
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='magma')
plt.title('Random Forest Feature Importance Analysis')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# 14. Repeat for other properties (Hardness and Elongation)
y_hard = df['hardness_HV']
y_elon = df['elongation_pct']

_, X_test_h, _, y_test_h = train_test_split(X, y_hard, test_size=0.2, random_state=42)
_, X_test_e, _, y_test_e = train_test_split(X, y_elon, test_size=0.2, random_state=42)

_, rf_r2_h, _ = evaluate_model(RandomForestRegressor(random_state=42), X_train, X_test_h, y_train, y_test_h)
_, rf_r2_e, _ = evaluate_model(RandomForestRegressor(random_state=42), X_train, X_test_e, y_train, y_test_e)
print(f"RF Model Multi-Target Performance -> Hardness (HV) R²: {rf_r2_h:.4f} | Elongation (%) R²: {rf_r2_e:.4f}")

# ==========================================
# LEVEL 4 — MODEL IMPROVEMENT AND VALIDATION
# ==========================================
print("\n--- LEVEL 4: Model Improvement and Validation ---")

# 15. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, random_state=42)
gb_rmse, gb_r2, gb_preds = evaluate_model(gb_model, X_train, X_test, y_train, y_test)
print(f"Gradient Boosting Regressor -> R²: {gb_r2:.4f}, RMSE: {gb_rmse:.2f} MPa")

# 16. K-Fold Cross Validation (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(GradientBoostingRegressor(random_state=42), X, y_ts, cv=kf, scoring='r2')
print(f"5-Fold Cross-Validated R² Score for Gradient Boosting: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 17. Feature Engineering Quadratic Term for Linear Model
X_train_quad = X_train.copy()
X_test_quad = X_test.copy()
X_train_quad['aging_temp_quad'] = (X_train_quad['aging_temp_C'] - 175) ** 2
X_test_quad['aging_temp_quad'] = (X_test_quad['aging_temp_C'] - 175) ** 2

lr_quad_model = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
lr_q_rmse, lr_q_r2, _ = evaluate_model(lr_quad_model, X_train_quad, X_test_quad, y_train, y_test)
print(f"Linear Regression with Quadratic Processing Feature -> R²: {lr_q_r2:.4f}, RMSE: {lr_q_rmse:.2f} MPa")

# 18. Multi-Layer Perceptron (Neural Network)
mlp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True))
])
mlp_rmse, mlp_r2, _ = evaluate_model(mlp_model, X_train, X_test, y_train, y_test)
print(f"Neural Network (MLP Regressor) -> R²: {mlp_r2:.4f}, RMSE: {mlp_rmse:.2f} MPa")

# 19. Residual Plots (For the top model: Gradient Boosting)
residuals = y_test - gb_preds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
sns.scatterplot(x=y_test, y=gb_preds, alpha=0.6, ax=axes[0], color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Strength (MPa)')
axes[0].set_ylabel('Predicted Strength (MPa)')
axes[0].set_title('Actual vs. Predicted Performance')

# Residuals Plot
sns.scatterplot(x=gb_preds, y=residuals, alpha=0.6, ax=axes[1], color='purple')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Strength (MPa)')
axes[1].set_ylabel('Residual Error (MPa)')
axes[1].set_title('Residual Variation Analysis')

plt.tight_layout()
plt.savefig('model_residuals.png', dpi=300)
plt.close()
print("Saved evaluation residual plots to 'model_residuals.png'")