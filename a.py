import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# サンプルデータの生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=5, random_state=42)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# モデルの訓練
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 予測
y_pred = clf.predict(X_test)

# 評価
print(f"正解率: {accuracy_score(y_test, y_pred):.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred))
