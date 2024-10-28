from pandas.api.types import is_numeric_dtype
from pandas import DataFrame, concat
from plotly.express import scatter, box, bar
from dash import html
import dash_bootstrap_components as dbc
from scipy.stats import kstest, spearmanr, chi2_contingency, mannwhitneyu, kruskal

identifiers_control = dbc.InputGroup([
    dbc.InputGroupText('Fuente de los datos', className='bg-secondary text-light' ),
    dbc.Select(
        options=[
            { 'label': 'Conjunto de entrenamiento', 'value': 'train' },
            { 'label': 'Conjunto de prueba', 'value': 'test' },
            { 'label': 'Entrenamiento y prueba', 'value': 'both' }
        ],
        value='both',
        id='data-selected'
    ),
    dbc.InputGroupText('Separar los datos según', className='bg-secondary text-light'),
    dbc.Select(
        options=[
            { 'label': 'Variable Respuesta', 'value': 'action_taken' },
            { 'label': 'Conjunto de datos', 'value': 'source' }
        ],
        value='action_taken',
        id='separator'
    ),
    dbc.InputGroupText([dbc.Checkbox(), 'Mostrar visualización por PCA (0.85)'], className='bg-secondary text-light')
])

variables_control = [
    dbc.Alert('Variables Numéricas', color='primary' ),
    dbc.Checklist(
        options=[
            { 'label': 'Monto del préstamo', 'value': 'loan_amount' },
            { 'label': 'Ingresos', 'value': 'income' },
            { 'label': 'Mediana de ingresos del sector', 'value': 'med_fam_income' }
        ],
        value=['loan_amount'],
        id='var-plot-num',
        switch=True,
        className='mx-3 mb-3'
    ),
    dbc.Alert('Variables Categóricas', color='primary'),
    dbc.Checklist(
        options=[
            { 'label': 'Prioridad del Lien', 'value': 'lien_status' },
            { 'label': 'AUS', 'value': 'aus_GUS' },
            { 'label': 'Conforme al Límite', 'value': 'conforming_loan_limit' },
            { 'label': 'Petición de Preapruebo', 'value': 'preapproval' },
            { 'label': 'Unidades Totales', 'value': 'total_units' }
        ],
        value=['lien_status'],
        id='var-plot-cat',
        switch=True,
        className='mx-3 mb-3'
    )
]

def _test_relation( x, y ):
    is_x_num = is_numeric_dtype(x)
    is_y_num = is_numeric_dtype(y)

    if is_x_num and is_y_num: 
        _, p_val = spearmanr( x, y )
        if p_val is None: return None
        return { 'test':'Spearman', 'var':x.name + ', ' + y.name, 'pval':round(p_val,2) }

    elif not( is_x_num or is_y_num ):
        _data = concat([x,y], axis=1).groupby([x.name]).value_counts().unstack().reset_index()
        _data.index.name, _data.columns.name = None, None
        _data.index = _data[x.name].values
        _data = _data.drop( columns=[x.name] )
        _, p_val, _, _ = chi2_contingency( _data )
        if p_val is None: return None
        return { 'test':'Chisq', 'var':x.name + ', ' + y.name, 'pval':round(p_val,2) }
    
    if is_y_num: x, y = y, x
    values = y.unique()
    if len( values ) == 2:
        group1, group2 = x[ y == values[0] ], x[ y == values[1] ]
        if len(group1) * len(group2) == 0: return None
        _, p_val = mannwhitneyu( group1, group2 )
        if p_val is None: return None
        return { 'test':'Mann Whitney', 'var':x.name + ', ' + y.name, 'pval':round(p_val,2) }
    elif len(values) < 2: return None
    else:
        groups = [ x[ y == values[i] ] for i in range(len(values)) ]
        _, p_val = kruskal( *groups )
        if p_val is None: return None
        return { 'test':'Kruskal Wallis', 'var':x.name + ', ' + y.name, 'pval':round(p_val,2) }

def _make_stat_tests( data: DataFrame, vars: list ):
    results = []
    for var in vars:
        if is_numeric_dtype(data[var]):
            _, p_val = kstest( data[var], 'norm', args=( data[var].mean(), data[var].std() ) )
            if p_val is not None: results.append( { 'test':'Kolmogorov Smirnov', 'var':var, 'pval':round(p_val,2) } )
    results.append( _test_relation( data[vars[0]], data[vars[1]] )) 
    if len( vars ) == 3: 
        results.append( _test_relation( data[vars[0]], data[vars[2]] )) 
        results.append( _test_relation( data[vars[1]], data[vars[2]] ))
    return [
        html.Tr([
            html.Td( row['test'] ),
            html.Td( row['var'] ),
            html.Td( row['pval'] if row['pval'] > 0 else '<0.05' )
        ]) 
        for row in results if row is not None
    ]

def _analyze_column(data: DataFrame, column: str):
    if is_numeric_dtype(data[column]):
        stats = {
            'Variable': column,
            'Media': data[column].mean(),
            'SD': data[column].std(),
            'Mediana': data[column].median(),
            'RIC': data[column].quantile(0.75) - data[column].quantile(0.25)
        }
        return stats, True
    else:
        value_counts = data[column].value_counts()
        largest_category = value_counts.idxmax()
        largest_category_count = value_counts.max()
        total_count = data[column].count()
        proportion = largest_category_count / total_count if total_count > 0 else 0
        
        return {
            'Variable': column,
            'Moda': largest_category,
            'Proporcion': proportion
        }, False

def eda_multi_plot( data: DataFrame, col_selected: list, separator: str ):

    is_x_num = is_numeric_dtype( data[col_selected[0]] )
    is_y_num = is_numeric_dtype( data[col_selected[1]] )

    if is_x_num and is_y_num: 
        _data = data.drop_duplicates(subset=[col_selected[0], col_selected[1]])
        fig = scatter(
            _data, 
            x=col_selected[0], 
            y=col_selected[1], 
            color=separator, 
            title=f'Gráfico de dispersión de {col_selected[0]} y {col_selected[1]}',
            opacity=0.6
        )
    elif not( is_x_num or is_y_num ):
        _data = DataFrame(data=data.groupby(col_selected[0])[col_selected[1]].value_counts()).reset_index()
        fig = bar(
            _data,
            x=col_selected[0],
            y='count',
            color=col_selected[1], 
            barmode='stack',
            title=f'Diagrama de barras de {col_selected[0]} y {col_selected[1]}'
        )
    else: 
        if is_y_num: col_selected.reverse()
        fig = box(
            data,
            x=col_selected[0], 
            y=col_selected[1], 
            color=separator, 
            title=f'Diagrama de cajas de {col_selected[0]} y {col_selected[1]}'
        )

    return fig

def eda_multi_table( data: DataFrame, vars: list ):

    nums = []
    cats = []

    for var in vars:
        result, is_num = _analyze_column( data, var )
        if is_num: nums.append(result)
        else: cats.append(result)

    return html.Div([
        html.H5('Variables numéricas', className='text-small'),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th('Variable'),
                    html.Th('Media'),
                    html.Th('SD'),
                    html.Th('Mediana'),
                    html.Th('RIC')
                ]), className='table-primary'
            ),
            html.Tbody([
                html.Tr([ 
                    html.Td(row['Variable']),
                    html.Td(round(row['Media'], 2)),
                    html.Td(round(row['SD'], 2)),
                    html.Td(round(row['Mediana'])),
                    html.Td(round(row['RIC']))
                ]) for row in nums
            ])
        ], bordered=True, style={'fontSize': '14px'}),
        html.H5('Variables categóricas'),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th('Variable'),
                    html.Th('Moda'),
                    html.Th('Proporción'),
                ]), className='table-primary'
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row['Variable']),
                    html.Td( row['Moda'] ), 
                    html.Td( round(row['Proporcion'],3) ) 
                ]) for row in cats
            ])
        ], bordered=True, style={'fontSize': '14px'}),
        html.H5('Pruebas de Hipótesis'),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th('Prueba'),
                    html.Th('Variable(s)'),
                    html.Th('P valor')
                ]), className='table-primary'
            ),
            html.Tbody( _make_stat_tests( data, vars ) )
        ], bordered=True, style={'fontSize': '14px'})
    ], className='text-center')
