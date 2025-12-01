import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 5000 samples
n_samples = 5000

# Feature generation with realistic distributions
data = {
    # Demographics
    'age': np.random.normal(28, 5, n_samples).clip(18, 45),
    'bmi': np.random.normal(26, 5, n_samples).clip(16, 45),
    
    # Medical History
    'family_history': np.random.binomial(1, 0.3, n_samples),  # 30% have family history
    'pcos': np.random.binomial(1, 0.15, n_samples),  # 15% have PCOS
    'previous_gdm': np.random.binomial(1, 0.12, n_samples),  # 12% had GDM before
    
    # Lifestyle
    'physical_activity': np.random.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.35, 0.35, 0.15]),  # 0=none, 3=high
    'diet_quality': np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.50, 0.25]),  # 0=poor, 2=excellent
    
    # Pregnancy Related
    'gestational_week': np.random.randint(8, 32, n_samples),
    'weight_gain': np.random.normal(8, 4, n_samples).clip(0, 25),
    'previous_pregnancies': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.30, 0.30, 0.20, 0.15, 0.05]),
    
    # Clinical (simple measurements)
    'blood_pressure_systolic': np.random.normal(118, 12, n_samples).clip(90, 160),
    'resting_heart_rate': np.random.normal(75, 10, n_samples).clip(55, 100),
}

df = pd.DataFrame(data)

# Calculate risk score based on medical logic
risk_score = (
    (df['age'] - 25) * 0.08 +  # Age effect
    (df['bmi'] - 22) * 0.15 +  # BMI strongly predicts GDM
    df['family_history'] * 2.5 +  # Strong genetic component
    df['pcos'] * 2.0 +  # PCOS increases risk
    df['previous_gdm'] * 3.5 +  # Previous GDM is strongest predictor
    (3 - df['physical_activity']) * 0.8 +  # Less activity = higher risk
    (2 - df['diet_quality']) * 0.6 +  # Poor diet increases risk
    (df['weight_gain'] - 8) * 0.25 +  # Excessive weight gain
    (df['blood_pressure_systolic'] - 115) * 0.05 +  # BP effect
    df['previous_pregnancies'] * 0.4 +  # Parity effect
    np.random.normal(0, 1.5, n_samples)  # Random variation
)

# Convert to probability using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x/5))

probabilities = sigmoid(risk_score - 5)  # Shift to get ~25% GDM rate

# Generate labels
df['gdm_risk'] = (probabilities > 0.5).astype(int)

# Add some realistic noise and edge cases
# Even with normal values, some high-risk cases
high_risk_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
df.loc[high_risk_indices, 'gdm_risk'] = 1

# Round values for cleaner data
df['age'] = df['age'].round(0).astype(int)
df['bmi'] = df['bmi'].round(1)
df['weight_gain'] = df['weight_gain'].round(1)
df['blood_pressure_systolic'] = df['blood_pressure_systolic'].round(0).astype(int)
df['resting_heart_rate'] = df['resting_heart_rate'].round(0).astype(int)
df['gestational_week'] = df['gestational_week'].astype(int)

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)

print("Dataset created successfully!")
print(f"Total samples: {len(df)}")
print(f"GDM cases: {df['gdm_risk'].sum()} ({df['gdm_risk'].mean()*100:.1f}%)")
print(f"\nFeature correlations with GDM risk:")
for col in df.columns[:-1]:
    corr = df[col].corr(df['gdm_risk'])
    print(f"{col}: {corr:.3f}")