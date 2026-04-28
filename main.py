import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

train_path = "KDDTRAIN+short"
test_path = "KDDTEST+short"

train_df = pd.read_csv(train_path, names=cols)
test_df = pd.read_csv(test_path, names=cols)

df = pd.concat([train_df, test_df], ignore_index=True)

df = df.drop(columns=['difficulty'], errors='ignore')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

y_raw = df['label'].copy()
X = df.drop(columns=['label'])

binary_y = (y_raw != 'normal').astype(int)

cat_cols = ['protocol_type', 'service', 'flag']
X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, binary_y, test_size=0.25, random_state=42, stratify=binary_y
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=100,
    n_jobs=-1
)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

selected_features = importances[importances > 0.05].index.tolist()
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

models = {
    "SVM": SVC(kernel='linear', C=10, gamma='scale', random_state=42, probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train_sel, y_train)
    scores = cross_val_score(model, X_train_sel, y_train, cv=10)
    pred = model.predict(X_test_sel)

    results.append({
    "Model": name,
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, model.predict_proba(X_test_sel)[:, 1])
})

results_df = pd.DataFrame(results)

print("Top 9 cech:")
print(selected_features)

print("\nRezultaty:")
print(results_df.sort_values("Accuracy", ascending=False).to_string(index=False))

for name, model in models.items():
    pred = model.predict(X_test_sel)
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name} confusion matrix:\n{cm}")