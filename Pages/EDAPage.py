from dash import Dash, Input, Output, html, dcc
import dash_bootstrap_components as dbc 
from pandas import concat, Series
from Utils.Eda_utils import identifiers_control, variables_control, eda_multi_plot, eda_multi_table
from pickle import load

with open( 'Data/data_eda.pkl', 'rb' ) as file:
    train = load( file )
    test = load( file )

train = concat( [ train.reset_index(), Series( ['Train'] * train.shape[0], name='source' ) ], axis=1 )
test = concat( [ test.reset_index(), Series( ['Test'] * test.shape[0], name='source' ) ], axis=1 )
both = concat( [ train, test ], axis=0 )

col_selected = []

def render_eda_page():

    return html.Div([ 
        html.H1('Análisis exploratorio de los datos', className='text-center' ),
        html.P('La siguiente tabla a continuación permite tener un mayor entendimiento de las variables usadas en los modelos.'),
        html.Hr(),
        dbc.Card([
            dbc.Row([ dbc.Card(dbc.CardBody( identifiers_control ), style={'backgroundColor': '#f5f3ff'} ) ], className='g-0 m-2'),
            dbc.Row([
                dbc.Col( dbc.Card(dbc.CardBody(id='multitable'), style={'backgroundColor': '#f5f3ff', 'height': '100%'}), width=3, className='px-2 pb-2' ),
                dbc.Col( dbc.Card(dbc.CardBody(
                    dbc.Spinner(children=[ dcc.Graph( id='dynamic-plot', style={'height': '600px'} ) ], color="primary")
                ), style={'backgroundColor': '#f5f3ff'}), width=7, class_name='pb-2' ),
                dbc.Col( dbc.Card(dbc.CardBody(variables_control), style={'backgroundColor': '#f5f3ff', 'height': '100%'}), width=2, className='px-2 pb-2' )
            ], className='g-0')
        ])
    ], className='px-4 py-3')

def load_eda_callbacks( app: Dash ):
    @app.callback(
        [
            Output('var-plot-num', 'value'),
            Output('var-plot-cat', 'value'),
            Output('dynamic-plot', 'figure'),
            Output('multitable', 'children')
        ],
        [
            Input('var-plot-num', 'value'),
            Input('var-plot-cat', 'value'),
            Input('data-selected', 'value'),
            Input('separator', 'value')
        ]
    )
    def update_eda(num_selected, cat_selected, source, separator):

        if num_selected or cat_selected:
            curr = [ x for x in num_selected + cat_selected if x not in col_selected ]
            if len(curr) == 1:
                if len(col_selected) > 1: col_selected.pop(0)
                col_selected.append( curr[0] )
                num_selected = [ x for x in num_selected if x in col_selected ]
                cat_selected = [ x for x in cat_selected if x in col_selected ]
            elif len(curr) == 0:
                extra = [ x for x in col_selected if x not in num_selected + cat_selected ]                
                for item in extra:
                    col_selected.remove(item)
            else:
                while len( col_selected ) > 0: col_selected.pop(0)
                col_selected.extend( curr )

        if source == 'train': data = train
        elif source == 'test': data = test
        else: data = both

        if len(col_selected) == 2: fig = eda_multi_plot( data, col_selected, separator )
        else: fig = {}

        table = eda_multi_table( data, col_selected + [ separator ] )

        return num_selected, cat_selected, fig, table

