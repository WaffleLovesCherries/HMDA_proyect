from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import newaxis, argmax, array
from matplotlib.colors import ListedColormap
from pandas import DataFrame

def evaluate_model(model, X_test, y_test, best_threshold=None):

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        if best_threshold is None:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            optimal_threshold_index = argmax(tpr - fpr)
            best_threshold = thresholds[optimal_threshold_index]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    }

def get_predictions(models, X_test, y_test):
    predictions = []
    for model in models:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            optimal_threshold_index = argmax(tpr - fpr)
            best_threshold = thresholds[optimal_threshold_index]
            y_pred = (y_pred_proba >= best_threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
        predictions.append(y_pred)
    return array(predictions)

def plot_predictions( models, X_test, y_test, model_names, figsize=(10, 8) ):
    predictions = get_predictions( models, X_test, y_test )
    predictions = DataFrame(predictions.T, columns=model_names)
    true_1 = predictions.loc[y_test.reset_index(drop=True) == 1]
    true_0 = predictions.loc[y_test.reset_index(drop=True) == 0]

    cmap = ListedColormap(['#2D3047', '#FBFEF9'])
    plt.figure(figsize=figsize)
    
    plt.subplot(2, 1, 1)
    sns.heatmap(true_1.values, annot=False, cmap=cmap, cbar=True, xticklabels=[model for model in model_names])
    plt.xlabel('Models')
    plt.ylabel('Observations')
    plt.title(f'True 1s')
    
    plt.subplot(2, 1, 2)
    sns.heatmap(true_0.values, annot=False, cmap=cmap, cbar=True, xticklabels=[model for model in model_names])
    plt.xlabel('Models')
    plt.ylabel('Observations')
    plt.title(f'True 0s')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, model_name, best_threshold=None):
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        if best_threshold is None:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            optimal_threshold_index = argmax(tpr - fpr)
            best_threshold = thresholds[optimal_threshold_index]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, newaxis] * 100

    plt.figure()
    sns.heatmap(cm_norm, annot=True, cmap="Blues", fmt=".2f", cbar=False, annot_kws={"size": 14})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of {model_name}')
    plt.show()

def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of {model_name}')
    plt.legend(loc='lower right')

    optimal_threshold_index = argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]
    plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], s=100, c='black', marker='o', label=f'Best Threshold = {optimal_threshold:.2f}')
    plt.legend(loc='lower right')

    plt.show()

def plot_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(6, 10))
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title(f'Feature Importances of {model_name}')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()