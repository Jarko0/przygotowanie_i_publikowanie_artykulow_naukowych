# import pandas as pd
# import numpy as np

# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# # =========================
# # 1. Wczytanie danych
# # =========================
# train = pd.read_csv("KDDTRAIN+", header=None, sep=",", engine="python")
# test = pd.read_csv("KDDTEST+", header=None, sep=",", engine="python")

# # =========================
# # 2. Kolumny (43)
# # =========================
# cols = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
# train.columns = cols
# test.columns = cols

# # =========================
# # 3. Usunięcie difficulty
# # =========================
# train = train.drop("difficulty", axis=1)
# test = test.drop("difficulty", axis=1)

# # =========================
# # 4. Label encoding
# # =========================
# train["label"] = train["label"].apply(lambda x: 0 if x == "normal" else 1)
# test["label"] = test["label"].apply(lambda x: 0 if x == "normal" else 1)

# # =========================
# # 5. One-hot encoding
# # =========================
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)

# # wyrównanie kolumn
# train, test = train.align(test, join='left', axis=1, fill_value=0)

# # =========================
# # 6. X / y
# # =========================
# X_train = train.drop("label", axis=1)
# y_train = train["label"]

# X_test = test.drop("label", axis=1)
# y_test = test["label"]

# # =========================
# # 7. SCALING (KLUCZOWE FIX)
# # =========================
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # =========================
# # 8. MODELE
# # =========================

# # SVM (lepsza konfiguracja)
# svm = SVC(kernel='rbf')
# svm.fit(X_train, y_train)
# svm_pred = svm.predict(X_test)

# # KNN
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)

# # Logistic Regression
# lr = LogisticRegression(max_iter=3000)
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)

# # Random Forest (nie potrzebuje scalingu, ale OK)
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# # XGBoost (lepsze parametry)
# xgb = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric='logloss'
# )

# xgb.fit(X_train, y_train)
# xgb_pred = xgb.predict(X_test)

# # =========================
# # 9. WYNIKI
# # =========================
# print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
# print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
# print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
# print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# NSL-KDD standard columns
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

train_path = "KDDTRAIN+"
test_path = "KDDTEST+"

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
    X_scaled, binary_y, test_size=0.30, random_state=42, stratify=binary_y
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=100,
    n_jobs=-1
)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Jak w artykule: wybór najlepszych cech
selected_features = importances.head(9).index.tolist()
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

models = {
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=None, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    model.fit(X_train_sel, y_train)
    pred = model.predict(X_test_sel)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_sel)[:, 1]
    else:
        proba = model.decision_function(X_test_sel)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba)
    })

results_df = pd.DataFrame(results)
print("Top 9 features:")
print(selected_features)
print("\nResults:")
print(results_df.sort_values("Accuracy", ascending=False).to_string(index=False))

# Confusion matrices
for name, model in models.items():
    pred = model.predict(X_test_sel)
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name} confusion matrix:\n{cm}")