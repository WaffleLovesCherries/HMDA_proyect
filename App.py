from dash import Dash, Input, Output, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from Components.Navbar import navbar, load_navbar_callbacks

from Pages.EDAPage import render_eda_page, load_eda_callbacks
from Pages.ModelBasePage import render_base_page, load_base_callbacks
from Pages.ModelSelPage import *

from pickle import load, dump

with open( 'Models/model_paths.txt', 'r' ) as file:
    model_paths = file.readlines()

# App
app = Dash( 
    __name__, 
    suppress_callback_exceptions = True, 
    external_stylesheets = [ dbc.themes.PULSE, dbc.icons.BOOTSTRAP ] 
)
load_figure_template( ['pulse'] )

# Layout
app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={'backgroundColor': '#f5f3ff', 'height': '100vh'} )

# Callbacks
load_navbar_callbacks( app=app, page_dict={
    '/eda': render_eda_page,
    '/models-base': render_base_page,
    '/selection': render_eda_page
})
load_eda_callbacks( app=app )
load_base_callbacks( app=app )

# Server
if __name__ == '__main__':
    app.run_server( debug=True )