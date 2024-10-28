from dash import Dash, Input, Output
import dash_bootstrap_components as dbc

# Navbar definition
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand( 'HMDA', href="/models-base" ),
            dbc.Nav(
                [
                    dbc.NavItem( dbc.NavLink( 'EDA', href='/eda' ) ),
                    dbc.DropdownMenu([
                        dbc.DropdownMenuItem( 'Modelos base', href='/models-base' )
                    ], label = 'Modelos', nav=True ),
                    dbc.NavItem( dbc.NavLink( 'Selección de modelos', href='/selection' ) )
                ],
                className="ml-auto",
                navbar=True
            )
        ]
    ),
    color = 'primary',
    dark = True
)

# Callback loader
def load_navbar_callbacks( app: Dash, page_dict: dict[ str, callable ] ):

    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        return page_dict[ pathname ]()