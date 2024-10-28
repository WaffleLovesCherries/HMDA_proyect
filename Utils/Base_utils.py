from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from numpy import argmax

def model_plot( name: str ):
    return html.Div([
        dbc.Row([ dbc.Card(dbc.CardBody( dbc.InputGroup([
            dbc.InputGroupText('Tipo de gráfico', className='bg-secondary text-light' ),
            dbc.Select(
                options=[
                    { 'label': 'Matriz de confusión', 'value': 'conf_matrix' },
                    { 'label': 'Curva ROC', 'value': 'roc_curve' }
                ],
                value='conf_matrix',
                id={ 'type':'selected-plot', 'index':name }
            ),
        ]) ), style={'backgroundColor': '#f5f3ff'} ) ], className='g-0 m-2'),
        dbc.Row([ dbc.Card(dbc.CardBody(
            dbc.Spinner(children=[ dcc.Graph( id={ 'type':'model-plot', 'index':name } ) ], color="primary"), class_name='m-5'
        ), style={'backgroundColor': '#f5f3ff'} ) ], className='g-0 m-2')
    ])


def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    optimal_threshold_index = argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='lightsalmon', width=4),
        name=f'ROC curve (AUC = {roc_auc:.2f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='purple', width=2, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_threshold_index]], y=[tpr[optimal_threshold_index]],
        mode='markers',
        marker=dict(color='black', size=10, symbol='circle'),
        name=f'Mejor Umbral = {optimal_threshold:.2f}'
    ))

    fig.update_layout(
        title={'text':f'Curva ROC del modelo {model_name}', 'font':{'size':24}},
        xaxis_title={'text':'Proporción de Falsos Positivos', 'font':{'size':18}},
        yaxis_title={'text':'Proporción de Verdaderos Positivos', 'font':{'size':18}},
        legend=dict(x=0.8, y=0.1),
        template='pulse',
        height=800
    )

    return fig

import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np

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
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['True Negative', 'True Positive'],
        colorscale='Purples',
        text=cm_norm, 
        hoverinfo='text', 
        colorbar=None
    ))

    fig.update_layout(
        title={'text':f'Matriz de confusión del modelo {model_name}', 'font':{'size':24}},
        xaxis_title={'text':'Predicho', 'font':{'size':18}},
        yaxis_title={'text':'Verdadero', 'font':{'size':18}},
        template='pulse',
        height=800
    )

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            fig.add_annotation(
                x=j, y=i,
                text=f'{cm_norm[i, j]:.2f}%', 
                showarrow=False,
                font=dict(size=14)
            )

    return fig 


