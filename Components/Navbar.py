from dash import Dash, Input, Output
import dash_bootstrap_components as dbc
from typing import Callable

# Custom Navbar
class Navbar( dbc.Navbar ):
    
    def __init__( self, app: Dash, page_dict: dict[ str, Callable ] ):
        nav_content = dbc.Container([
            dbc.NavbarBrand( 'HMDA', href="/models-base" ),
            dbc.Nav([
                    dbc.NavItem( dbc.NavLink( 'EDA', href='/eda' ) ),
                    dbc.DropdownMenu([ 
                        dbc.DropdownMenuItem( 'Modelos base', href='/models-base' )
                    ], label = 'Modelos', nav=True ),
                    dbc.NavItem( dbc.NavLink( 'Selección de modelos', href='/selection' ) )
                ],
                className="ml-auto",
                navbar=True
            )
        ])
        super().__init__( nav_content, color = 'primary', dark = True )

        self.load_callbacks( app=app, page_dict=page_dict )

    def load_callbacks( self, app: Dash, page_dict: dict[ str, Callable ] ):
        @app.callback(
            Output('page-content', 'children'),
            Input('url', 'pathname')
        )
        def display_page(pathname):
            if pathname in (page_dict.keys()): return page_dict[pathname]()
            return page_dict['/']()
