import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import zipfile

# 1. Chargement du dataset réel NBI-IoT depuis le fichier ZIP
zip_path = "dataset.zip"
csv_filename = "nbaiot_combined.csv"

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Le fichier {zip_path} est introuvable.")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    if csv_filename not in zip_ref.namelist():
        raise FileNotFoundError(f"{csv_filename} n'est pas présent dans {zip_path}.")
    with zip_ref.open(csv_filename) as file:
        df = pd.read_csv(file)

# Vérification de la colonne cible
if 'label' not in df.columns:
    raise ValueError("Le fichier CSV doit contenir une colonne 'label' pour la classe cible.")

X = df.drop('label', axis=1)
y = df['label']

X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

# 2. Équilibrage avec SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Modèles à tester
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# 6. Entraînement et évaluation
f1_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='binary')
    f1_scores[name] = f1
    print(f"\n{name} - F1 Score: {f1:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Matrice de confusion - {name}")
    plt.savefig(f"matrice_confusion_{name.replace(' ', '_')}.png")
    plt.show()

# 7. Choix du meilleur modèle
best_model_name = max(f1_scores, key=f1_scores.get)
print(f"\nMeilleur modèle : {best_model_name}")

# 8. Optimisation du meilleur modèle
if best_model_name == "Random Forest":
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=5)
elif best_model_name == "SVM":
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, scoring='f1', cv=5)
elif best_model_name == "Logistic Regression":
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, scoring='f1', cv=5)

grid.fit(X_train, y_train)
print("\nMeilleurs hyperparamètres trouvés :", grid.best_params_)
best_model = grid.best_estimator_

# 9. Évaluation du modèle optimisé
y_pred_optimized = best_model.predict(X_test)
f1_optimized = f1_score(y_test, y_pred_optimized, average='binary')
print(f"Nouveau F1 Score après optimisation : {f1_optimized:.4f}")
cm_opt = confusion_matrix(y_test, y_pred_optimized)
ConfusionMatrixDisplay(confusion_matrix=cm_opt).plot()
plt.title(f"Matrice de confusion - {best_model_name} Optimisé")
plt.savefig(f"matrice_confusion_{best_model_name.replace(' ', '_')}_optimise.png")
plt.show()

# 10. Résumé final
print("\nRésumé des F1 Scores avant optimisation :")
for name, score in f1_scores.items():
    print(f"- {name}: {score:.4f}")
print(f"\nModèle final retenu : {best_model_name} avec F1 optimisé de {f1_optimized:.4f}")

with open("resultats_f1.txt", "w") as f:
    f.write("Résumé des F1 Scores avant optimisation :\n")
    for name, score in f1_scores.items():
        f.write(f"- {name}: {score:.4f}\n")
    f.write(f"\nModèle final retenu : {best_model_name} avec F1 optimisé de {f1_optimized:.4f}\n")
