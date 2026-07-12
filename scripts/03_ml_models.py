import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

df = pd.read_csv('/home/claude/project/data/cleaned_transactions.csv', parse_dates=['Transaction_Date'])

# Step 1: model-ready subset -- drop rows missing target or key numeric features
df_model = df.dropna(subset=['Transaction_Status', 'Price', 'Quantity']).copy()
df_model = df_model.drop(columns=['Transaction_ID', 'Transaction_Date', 'Customer_ID'])
print(f"Model dataset rows: {len(df_model)}")

# Step 2: encode categoricals
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Transaction_Status']
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

target_le = LabelEncoder()
df_model['Transaction_Status'] = target_le.fit_transform(df_model['Transaction_Status'])
class_names = target_le.classes_

# fill remaining NaNs in Year/Month (rows w/o valid date) with a sentinel
df_model['Year'] = df_model['Year'].fillna(0)
df_model['Month'] = df_model['Month'].fillna(0)

# Step 3: train/test split
X = df_model.drop(columns=['Transaction_Status'])
y = df_model['Transaction_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: scaling (for logistic regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print(f"Training samples: {len(X_train)}  Test samples: {len(X_test)}")

def plot_cm(y_true, y_pred, model_name, fname):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

results_log = []

# Step 5: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
acc_lr = accuracy_score(y_test, y_pred_lr)
plot_cm(y_test, y_pred_lr, 'Logistic Regression', '/home/claude/project/dashboard/cm_logistic_regression.png')
results_log.append(('Logistic Regression', acc_lr, classification_report(y_test, y_pred_lr, target_names=class_names)))

# Step 6: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
plot_cm(y_test, y_pred_dt, 'Decision Tree', '/home/claude/project/dashboard/cm_decision_tree.png')
results_log.append(('Decision Tree', acc_dt, classification_report(y_test, y_pred_dt, target_names=class_names)))

# Step 7: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
plot_cm(y_test, y_pred_rf, 'Random Forest', '/home/claude/project/dashboard/cm_random_forest.png')
results_log.append(('Random Forest', acc_rf, classification_report(y_test, y_pred_rf, target_names=class_names)))

# Step 8: Feature importance (Random Forest)
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feat_imp.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importances — Random Forest')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/project/dashboard/feature_importance.png', dpi=150)
plt.close()

# Step 9: Model comparison chart
comp = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [acc_lr, acc_dt, acc_rf]
}).sort_values('Accuracy', ascending=False)
plt.figure(figsize=(7,4))
plt.bar(comp['Model'], comp['Accuracy'], color=['#4C72B0','#55A868','#C44E52'], edgecolor='black')
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.savefig('/home/claude/project/dashboard/model_comparison.png', dpi=150)
plt.close()

with open('/home/claude/project/models/model_results.txt', 'w') as f:
    f.write(f"Model dataset rows: {len(df_model)}\n")
    f.write(f"Training samples: {len(X_train)}  Test samples: {len(X_test)}\n\n")
    for name, acc, report in results_log:
        f.write(f"{'='*50}\n{name}\nAccuracy: {acc:.4f}\n\n{report}\n\n")
    f.write("Feature Importances (Random Forest):\n")
    f.write(feat_imp.to_string())

print(comp)
print("Saved model results and charts.")
