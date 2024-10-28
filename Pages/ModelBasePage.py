from dash import Dash, Input, Output, html, dcc, ALL, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc 
from pandas import DataFrame
from pickle import load
from Utils.Base_utils import model_plot, plot_roc_curve, plot_confusion_matrix
from Utils.models_testing import evaluate_model


with open('Data/data.pkl','rb') as file:
    _ = load( file )
    X_test, y_test = load( file )
X_test, y_test = X_test.copy(), y_test.copy()

with open( 'Models/dt_trained.pkl', 'rb' ) as file:
    dt = load(file)
with open( 'Models/dt_ada_trained.pkl', 'rb' ) as file:
    dt_ada = load(file)
with open( 'Models/knn_trained.pkl', 'rb' ) as file:
    knn = load(file)
with open( 'Models/knn_ada_trained.pkl', 'rb' ) as file:
    knn_ada = load(file)

def get_params( *models ):
    return DataFrame([
        { 'Nombre': model['name'] } | 
        model['model'].best_params_ | 
        { 'Puntaje AUC': model['model'].best_score_, 'Tiempo de CPU (s)': model['cpu_time'] } 
        for model in models
    ])

dt_tab = dbc.Tab(
    dbc.Card( dbc.CardBody([ 
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe( get_params( dt ), color='secondary', bordered=True, striped=True )
            ], title='Parámetros del modelo'),
            dbc.AccordionItem([
                model_plot( 'dt' )
            ], title='Gráficas del modelo')
        ])
    ])), label='Desicion Tree'
)

dt_ada_tab = dbc.Tab(
    dbc.Card( dbc.CardBody([ 
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe( get_params( dt_ada ), color='secondary', bordered=True, striped=True )
            ], title='Parámetros del modelo'),
            dbc.AccordionItem([
                model_plot( 'dt_ada' )
            ], title='Gráficas del modelo')
        ])
    ])), label='Desicion Tree Balanceado'
)

knn_tab = dbc.Tab(
    dbc.Card( dbc.CardBody([ 
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe( get_params( knn ), color='secondary', bordered=True, striped=True )
            ], title='Parámetros del modelo'),
            dbc.AccordionItem([
                model_plot( 'knn' )
            ], title='Gráficas del modelo')
        ])
    ])), label='k-Nearest Neighbors'
)

knn_ada_tab = dbc.Tab(
    dbc.Card( dbc.CardBody([ 
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe( get_params( knn_ada ), color='secondary', bordered=True, striped=True )
            ], title='Parámetros del modelo'),
            dbc.AccordionItem([
                model_plot( 'knn_ada' )
            ], title='Gráficas del modelo')
        ])
    ])), label='k-Nearest Neighbors'
)

def render_base_page():
    return html.Div([ 
        html.H1('Modelos básicos', className='text-center' ),
        html.Hr(),
        dbc.Tabs([ dt_tab, dt_ada_tab, knn_tab, knn_ada_tab ])
    ], className='px-4 py-3')

def load_base_callbacks( app: Dash ):
    for m in [ dt, dt_ada, knn, knn_ada ]:
        @app.callback(
            Output({'type': 'model-plot', 'index': m['name']}, 'figure'),
            Input({'type': 'selected-plot', 'index': m['name'] }, 'value')
        )
        def update_plot( value, model=m ):
            if value == 'conf_matrix':
                return plot_confusion_matrix( model['model'], X_test, y_test, model['name'] )
            else: 
                return plot_roc_curve( model['model'], X_test, y_test, model['name'] )
