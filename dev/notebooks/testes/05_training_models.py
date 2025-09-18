import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.datasets import make_classification
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================
# Modelos de machine learning utilizados
# ==========================
MODELS = {
    'LR': LogisticRegression(max_iter=1000),
    'RFC': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'DTC': DecisionTreeClassifier(),
    'GBC': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'GNB': GaussianNB(),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'ABC': AdaBoostClassifier(),
    'LGBM': LGBMClassifier()
}

# ==========================
# Cria estrutura de pastas para o experimento
# ==========================
def create_experiment_folders(base_path):
    paths = {
        'base': base_path,
        'metrics': os.path.join(base_path, 'metrics'),
        'plots': os.path.join(base_path, 'plots'),
        'models': os.path.join(base_path, 'models')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

# ==========================
# Plota curvas ROC e PRC e salva como imagem
# ==========================
def plot_roc_prc(y_test, y_probs, model_name, plot_path):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.figure(figsize=(10, 4))

    # Curva ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_probs):.2f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend()

    # Curva PRC
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{model_name} (AUC={auc(recall, precision):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva PRC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{model_name}_roc_prc.png'))
    plt.close()

# ==========================
# Plota matriz de confusão e salva como imagem
# ==========================
def plot_confusion_matrix(y_true, y_pred, model_name, plot_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Matriz de Confusão - {model_name}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

# ==========================
# Calcula métricas de avaliação
# ==========================
def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': np.mean(y_true == y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'dice': 2 * precision_score(y_true, y_pred, zero_division=0) * recall_score(y_true, y_pred, zero_division=0) / (precision_score(y_true, y_pred, zero_division=0) + recall_score(y_true, y_pred, zero_division=0) + 1e-6),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

# ==========================
# Seleção de atributos com RFECV
# ==========================
def apply_rfecv(X, y, estimator=None):
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5), scoring='f1')
    rfecv.fit(X, y)
    X_selected = X.iloc[:, rfecv.support_]
    return X_selected, rfecv.support_, rfecv.grid_scores_

# ==========================
# Pipeline principal do experimento
# ==========================
def run_experiment(df, target_column, experiment_name, use_rfecv=True):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    base_dir = os.path.join('experimentos', experiment_name)
    paths = create_experiment_folders(base_dir)

    if use_rfecv:
        X_train_sel, support, grid_scores = apply_rfecv(X_train, y_train)
        X_test_sel = X_test.iloc[:, support]

        # Salva suporte e curva de desempenho
        np.save(os.path.join(paths['base'], 'selected_features.npy'), support)
        plt.figure()
        plt.xlabel("Número de features selecionadas")
        plt.ylabel("F1 Score")
        plt.plot(range(1, len(grid_scores) + 1), grid_scores)
        plt.savefig(os.path.join(paths['plots'], 'rfecv_scores.png'))
        plt.close()
    else:
        X_train_sel, X_test_sel = X_train, X_test

    results = []

    for name, model in tqdm(MODELS.items(), desc="Treinando modelos"):
        print(f'Treinando modelo: {name}')
        start = time.time()
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_sel)
        elapsed = time.time() - start

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['model'] = name
        metrics['training_time'] = elapsed
        results.append(metrics)

        # Gráficos e relatório
        plot_roc_prc(y_test, y_prob, name, paths['plots'])
        plot_confusion_matrix(y_test, y_pred, name, paths['plots'])
        with open(os.path.join(paths['metrics'], f'{name}_classification_report.txt'), 'w') as f:
            f.write(classification_report(y_test, y_pred))

        joblib.dump(model, os.path.join(paths['models'], f'{name}.pkl'))

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(os.path.join(paths['metrics'], 'resultados.csv'), index=False)

# ==========================
# Exemplo de uso
# ==========================
if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_classes=2, weights=[0.8, 0.2], random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    df['target'] = y

    run_experiment(df, target_column='target', experiment_name='teste_rfecv')


'''   

class ModelContainer:
    def __init__(self, models=None):
        self.models = models or {
            'logistic_regression': LogisticRegression(),
            'svm': SVC(),
            'random_forest': RandomForestClassifier(),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier()
        }
        self.trained_models = {}

    def train(self, X, y, test_size=0.2, random_state=42):
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
         for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = {'model': model, 'X_test': X_test, 'y_test': y_test}

    def predict(self, model_name, X):
        if model_name in self.trained_models:
            return self.trained_models[model_name]['model'].predict(X)
        else:
            raise ValueError(f"Model '{model_name}' not trained yet.")
    
    def evaluate(self):
        results = {}
        for name, trained_model in self.trained_models.items():
            y_pred = trained_model['model'].predict(trained_model['X_test'])
            accuracy = accuracy_score(trained_model['y_test'], y_pred)
            report = classification_report(trained_model['y_test'], y_pred)
            results[name] = {'accuracy': accuracy, 'classification_report': report}
        return results

'''