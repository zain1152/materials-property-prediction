import os
import sys
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# LEVEL 1 — DATA EXPLORATION AND CLEANING
# =====================================================================
def clean_sensor_data(file_path):
    print("--- Level 1: Loading & Cleaning Data ---")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found at: {file_path}")
        sys.exit()
    
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    if 'day' in df.columns:
        df = df.sort_values(['machine_id', 'day']).reset_index(drop=True)
    
    for machine_id, group in df.groupby('machine_id'):
        rolling_mean = group['vibration_mm_s'].rolling(window=7, min_periods=1, center=True).mean()
        spike_mask = (group['vibration_mm_s'] > 8 * rolling_mean)
        df.loc[group.index[spike_mask], 'vibration_mm_s'] = np.nan
        
    sensor_cols = ['vibration_mm_s', 'temperature_C', 'pressure_psi']
    for col in sensor_cols:
        if col in df.columns:
            df[col] = df.groupby('machine_id')[col].ffill().groupby(df['machine_id']).bfill()
        
    print("Data cleaning completed.")
    return df

# =====================================================================
# LEVEL 2 — FEATURE ENGINEERING FOR TIME-SERIES SENSOR DATA
# =====================================================================
def engineer_features(df):
    print("\n--- Level 2: Engineering Features ---")
    features_df = df.copy()
    sensor_cols = ['vibration_mm_s', 'temperature_C', 'pressure_psi']
    
    for col in sensor_cols:
        features_df[f'{col}_roll_mean_7d'] = features_df.groupby('machine_id')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        features_df[f'{col}_roll_std_7d'] = features_df.groupby('machine_id')[col].transform(lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
        features_df[f'{col}_diff_1d'] = features_df.groupby('machine_id')[col].transform(lambda x: x.diff().fillna(0))
        
    features_df['total_age_factor'] = features_df['install_age_years'] + (features_df['operating_hours'] / 8760.0)
    features_df = pd.get_dummies(features_df, columns=['machine_type'], drop_first=True)
    return features_df

# =====================================================================
# LEVEL 3 & 4 — MODELING, SPLITTING & EVALUATION
# =====================================================================
def train_and_evaluate_pipeline(features_df):
    print("\n--- Levels 3 & 4: Model Training & Evaluation ---", flush=True)
    exclude_cols = ['machine_id', 'failure_event', 'failure_within_7_days', 'day', 'date', 'timestamp']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(features_df[c])]
    
    X = features_df[feature_cols]
    y = features_df['failure_within_7_days']
    groups = features_df['machine_id']
    
    unique_machines = groups.unique()
    train_machines = unique_machines[:30] if len(unique_machines) >= 30 else unique_machines[:int(len(unique_machines)*0.75)]
    test_machines = [m for m in unique_machines if m not in train_machines]
    
    X_train, y_train = X[groups.isin(train_machines)], y[groups.isin(train_machines)]
    X_test, y_test = X[groups.isin(test_machines)], y[groups.isin(test_machines)]
    features_df_test = features_df[groups.isin(test_machines)].copy()
    
    print("Training models...", flush=True)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # This line ensures the variables are defined correctly for the plots
    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_probs)
    
    chosen_threshold = 0.30
    custom_preds = (rf_probs >= chosen_threshold).astype(int)
    
    print(f"\n=== RANDOM FOREST PERFORMANCE (Threshold = {chosen_threshold}) ===", flush=True)
    print(classification_report(y_test, custom_preds), flush=True)
    
    features_df_test['pred_risk'] = custom_preds
    print("\n=== MAINTENANCE LEAD TIME METRICS ===", flush=True)
    for m_id in test_machines:
        m_data = features_df_test[features_df_test['machine_id'] == m_id].reset_index(drop=True)
        if m_data['failure_event'].sum() > 0:
            fail_idx = m_data[m_data['failure_event'] == 1].index[0]
            flagged_days = m_data[(m_data['pred_risk'] == 1) & (m_data.index <= fail_idx)]
            if not flagged_days.empty:
                print(f"Machine {m_id}: Lead time generated = {fail_idx - flagged_days.index[0]} days.", flush=True)
                
    # Return statement matching the variables above
    return lr_pipeline, rf_model, precisions, recalls, thresholds

# Ensure this is the last line of your train_and_evaluate_pipeline function before moving on:
    return lr_pipeline, rf_model, precisions, recalls, thresholds

# =====================================================================
# 5. VISUALIZATION FUNCTION (SMART COLUMN DETECTION)
# =====================================================================
def plot_portfolio_visuals(df, precisions, recalls, thresholds):
    print("\n📈 Generating portfolio illustrations...", flush=True)
    import matplotlib.pyplot as plt
    
    # Automatically detect the timeline column name (day, Day, DATE, etc.)
    possible_time_cols = ['day', 'Day', 'DAY', 'date', 'Date', 'DATE', 'timestamp']
    time_col = next((c for c in possible_time_cols if c in df.columns), None)
    
    # Fallback to index if no time column found
    if time_col is None:
        df = df.copy()
        df['derived_day'] = df.groupby('machine_id').cumcount() + 1
        time_col = 'derived_day'

    # --- ILLUSTRATION 1: SENSOR DEGRADATION CURVE ---
    plt.figure(figsize=(12, 5))
    failed_machines = df[df['failure_event'] == 1]['machine_id'].unique()
    healthy_machines = [m for m in df['machine_id'].unique() if m not in failed_machines]
    
    if len(failed_machines) > 0 and len(healthy_machines) > 0:
        m_fail = failed_machines[0]
        fail_data = df[df['machine_id'] == m_fail].sort_values(time_col)
        plt.plot(fail_data[time_col], fail_data['vibration_mm_s'], label=f'Machine {m_fail} (Failed)', color='crimson', lw=2)
        
        fail_day = fail_data[fail_data['failure_event'] == 1][time_col].values[0]
        plt.axvline(x=fail_day, color='black', linestyle='--', label='Actual Failure Event')
        
        m_healthy = healthy_machines[0]
        healthy_data = df[df['machine_id'] == m_healthy].sort_values(time_col)
        plt.plot(healthy_data[time_col], healthy_data['vibration_mm_s'], label=f'Machine {m_healthy} (Healthy)', color='teal', alpha=0.7)
        
        plt.title("Sensor Degradation Over Time: Healthy vs. Failing Machine", fontsize=14, fontweight='bold')
        plt.xlabel("Timeline", fontsize=12)
        plt.ylabel("Vibration (mm/s)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sensor_degradation_chart.png', dpi=300)
        print("💾 Saved image file: 'sensor_degradation_chart.png'", flush=True)
        plt.close()

    # --- ILLUSTRATION 2: PRECISION-RECALL FRONTIER ---
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='darkblue', lw=2)
    plt.plot(thresholds, recalls[:-1], label='Recall', color='darkorange', lw=2)
    plt.axvline(x=0.30, color='red', linestyle=':', label='Chosen Threshold (0.30)')
    
    plt.title("Optimizing the Maintenance Alert Threshold", fontsize=14, fontweight='bold')
    plt.xlabel("Probability Threshold Decision Boundary", fontsize=12)
    plt.ylabel("Score Metric", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_threshold_optimization.png', dpi=300)
    print("💾 Saved image file: 'model_threshold_optimization.png'", flush=True)
    plt.close()

# =====================================================================
# 6. THE LAUNCH PAD (The true bottom of the file)
# =====================================================================
if __name__ == "__main__":
    print("🚀 Script initialized. Starting pipeline...")
    
    csv_path = r"D:\Machine Learning Projects\project2_predictive_maintenance\data\machine_sensor_logs.csv"
    
    try:
        cleaned_df = clean_sensor_data(csv_path)
        engineered_df = engineer_features(cleaned_df)
        
        # Run the model and grab the plot metrics
        _, _, prec, rec, thres = train_and_evaluate_pipeline(engineered_df)
        
        # Generate the illustrations
        plot_portfolio_visuals(engineered_df, prec, rec, thres)
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED DURING EXECUTION:")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*50)
    input("Press ENTER to exit and close this window...")