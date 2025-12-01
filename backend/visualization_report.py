# visualization_report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('../data/synthetic_data.csv')  # Adjust path if needed

# -------------------------------
# Define feature importance (from your model logic)
# -------------------------------
feature_importance = {
    'previous_gdm': 5.0,
    'bmi': 4.5,
    'family_history': 4.0,
    'pcos': 3.5,
    'age': 3.0,
    'weight_gain': 2.5,
    'physical_activity': 2.0,
    'diet_quality': 2.0,
    'blood_pressure_systolic': 1.5,
    'previous_pregnancies': 1.2,
    'resting_heart_rate': 0.8,
    'gestational_week': 0.5
}

# -------------------------------
# 1. Feature Distributions
# -------------------------------
numerical_features = ['age', 'bmi', 'weight_gain', 'blood_pressure_systolic', 'resting_heart_rate', 'gestational_week']

for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], bins=10, kde=True, color='skyblue')
    plt.title(f'{feature.title()} Distribution')
    plt.xlabel(feature.title())
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'graphs/{feature}_histogram.png', dpi=300)
    plt.close()

    plt.figure(figsize=(4,4))
    sns.boxplot(y=df[feature], color='lightgreen')
    plt.title(f'{feature.title()} Boxplot')
    plt.tight_layout()
    plt.savefig(f'graphs/{feature}_boxplot.png', dpi=300)
    plt.close()

# -------------------------------
# 2. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('graphs/correlation_heatmap.png', dpi=300)
plt.close()

# -------------------------------
# 3. Risk Level Distribution (if risk_level column exists)
# -------------------------------
if 'risk_level' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='risk_level', data=df, order=['Low','Moderate','High'], palette=['green','orange','red'])
    plt.title('Distribution of Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('graphs/risk_level_distribution.png', dpi=300)
    plt.close()

# -------------------------------
# 4. Feature Importance / Contribution
# -------------------------------
features = list(feature_importance.keys())
weights = list(feature_importance.values())

plt.figure(figsize=(10,6))
sns.barplot(x=weights, y=features, palette='viridis')
plt.title('Feature Contribution to GDM Risk')
plt.xlabel('Weight / Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('graphs/feature_importance.png', dpi=300)
plt.close()

# -------------------------------
# 5. Pair Plot of Key Features
# -------------------------------
key_features = ['age','bmi','weight_gain','blood_pressure_systolic']
if 'risk_level' in df.columns:
    sns.pairplot(df, vars=key_features, hue='risk_level', palette=['green','orange','red'])
else:
    sns.pairplot(df, vars=key_features)
plt.savefig('graphs/pairplot.png', dpi=300)
plt.close()

# -------------------------------
# 6. ROC Curve and Confusion Matrix
# -------------------------------
# Only if actual labels exist
if 'actual_gdm' in df.columns and 'predicted_prob' in df.columns:
    y_true = df['actual_gdm']
    y_scores = df['predicted_prob']

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('graphs/roc_curve.png', dpi=300)
    plt.close()

    # Confusion Matrix (threshold 0.5)
    y_pred = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No GDM','GDM'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('graphs/confusion_matrix.png', dpi=300)
    plt.close()

print("All graphs generated and saved in the 'graphs' folder!")
