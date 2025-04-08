import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Intentar importar geopandas, pero manejar el error si no está disponible
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# =============================================
# CONFIGURACIÓN INICIAL Y CARGA DE DATOS
# =============================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Análisis GNCV Colombia - MME"
server = app.server  # Esta línea es crítica para Render

# Definir rutas relativas para los archivos
base_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(base_dir, "MGN_DPTO_POLITICO", "MGN_DPTO_POLITICO.shp")
data_path = os.path.join(base_dir, "Consulta_Precios_Promedio_de_Gas_Natural_Comprimido_Vehicular__AUTOMATIZADO__20250328.csv")

# Variables globales para los datos
gdf = None
geojson = None

# Cargar datos geoespaciales si geopandas está disponible
if GEOPANDAS_AVAILABLE:
    try:
        gdf = gpd.read_file(shapefile_path)
        gdf['DPTO_CNMBR'] = gdf['DPTO_CNMBR'].str.upper().str.strip()
        
        # Simplificar geometría para reducir tamaño
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
        
        # Crear GeoJSON en memoria (sin guardar archivo)
        geojson = json.loads(gdf.to_json())
    except Exception as e:
        print(f"Error al cargar datos geoespaciales: {e}")
        # Fallback si el shapefile no está disponible
        gdf = None
        geojson = None

# Cargar datos transaccionales
try:
    df = pd.read_csv(data_path)
    df['DEPARTAMENTO_EDS'] = df['DEPARTAMENTO_EDS'].str.upper().str.strip()
except Exception as e:
    print(f"Error al cargar datos transaccionales: {e}")
    # Crear un DataFrame vacío si hay error
    df = pd.DataFrame(columns=['DEPARTAMENTO_EDS', 'PRECIO_PROMEDIO_PUBLICADO', 'FECHA_PRECIO', 
                              'NOMBRE_COMERCIAL_EDS', 'CODIGO_MUNICIPIO_DANE', 'TIPO_COMBUSTIBLE'])
    # Añadir algunos datos de ejemplo para que la aplicación no falle
    df = pd.DataFrame({
        'DEPARTAMENTO_EDS': ['BOGOTÁ, D.C.', 'ANTIOQUIA', 'VALLE DEL CAUCA'],
        'PRECIO_PROMEDIO_PUBLICADO': [2500, 2400, 2600],
        'FECHA_PRECIO': ['2025-01-01', '2025-01-01', '2025-01-01'],
        'NOMBRE_COMERCIAL_EDS': ['EDS 1', 'EDS 2', 'EDS 3'],
        'CODIGO_MUNICIPIO_DANE': [11001, 5001, 76001],
        'TIPO_COMBUSTIBLE': ['GNCV', 'GNCV', 'GNCV']
    })
    df['FECHA_PRECIO'] = pd.to_datetime(df['FECHA_PRECIO'])

# =============================================
# PROCESAMIENTO DE DATOS
# =============================================

# Corregir nombres de departamentos
correcciones = {
    "BOGOTA, D.C.": "BOGOTÁ, D.C.",
    "VALLE": "VALLE DEL CAUCA",
    "NARIÑO": "NARINO",
    "GUAJIRA": "LA GUAJIRA"
}

df['DEPARTAMENTO_EDS'] = df['DEPARTAMENTO_EDS'].replace(correcciones)

# Convertir la columna de fecha a datetime para análisis temporal
if 'FECHA_PRECIO' in df.columns:
    df['FECHA_PRECIO'] = pd.to_datetime(df['FECHA_PRECIO'])

# Calcular promedios
df_promedio = df.groupby('DEPARTAMENTO_EDS')['PRECIO_PROMEDIO_PUBLICADO'].mean().reset_index()

# Crear datos temporales para análisis de tendencias
try:
    df_temporal = df.sort_values('FECHA_PRECIO')
    df_tendencia = df.groupby('FECHA_PRECIO')['PRECIO_PROMEDIO_PUBLICADO'].mean().reset_index()
    df_tendencia = df_tendencia.sort_values('FECHA_PRECIO')
except:
    # Crear DataFrame de tendencia con valores de ejemplo en caso de error
    df_tendencia = pd.DataFrame({
        'FECHA_PRECIO': pd.date_range(start='2025-01-01', periods=10, freq='D'),
        'PRECIO_PROMEDIO_PUBLICADO': [2500, 2550, 2600, 2580, 2590, 2610, 2640, 2630, 2650, 2670]
    })

# Crear datos comparativos por departamento
df_top_caros = df_promedio.sort_values('PRECIO_PROMEDIO_PUBLICADO', ascending=False).head(10)
df_top_economicos = df_promedio.sort_values('PRECIO_PROMEDIO_PUBLICADO').head(10)

# Calcular estadísticas descriptivas
estadisticas = df['PRECIO_PROMEDIO_PUBLICADO'].describe().reset_index()
estadisticas.columns = ['Estadística', 'Valor']

# Unir datos geoespaciales con transaccionales
if gdf is not None:
    try:
        gdf_merged = gdf.merge(
            df_promedio,
            left_on='DPTO_CNMBR',
            right_on='DEPARTAMENTO_EDS',
            how='left'
        ).fillna(0)
    except Exception as e:
        print(f"Error al unir datos: {e}")
        gdf_merged = None
else:
    gdf_merged = None

# =============================================
# COMPONENTES VISUALES
# =============================================

# Navbar
navbar = dbc.NavbarSimple(
    brand="Análisis Geoespacial y Visualización Interactiva - Precios Promedio de GNCV en Colombia",
    brand_href="#",
    color="primary",
    dark=True,
)

# Tarjetas informativas
def create_card(title, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.H2(f"{value:,.2f}", className="card-text")
        ]),
        color=color,
        inverse=True,
        className="mb-3"
    )

# Verificar si hay datos antes de calcular estadísticas
precio_promedio = df['PRECIO_PROMEDIO_PUBLICADO'].mean() if len(df) > 0 else 0
estaciones = df['NOMBRE_COMERCIAL_EDS'].nunique() if 'NOMBRE_COMERCIAL_EDS' in df.columns else 0
municipios = df['CODIGO_MUNICIPIO_DANE'].nunique() if 'CODIGO_MUNICIPIO_DANE' in df.columns else 0

cards = dbc.Row([
    dbc.Col(create_card("Precio Promedio Nacional", precio_promedio, "success")),
    dbc.Col(create_card("Estaciones Registradas", estaciones, "info")),
    dbc.Col(create_card("Municipios Cubiertos", municipios, "warning"))
])

# Pestaña de contextualización
contexto_tab = dbc.Tab(
    label="Contexto",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="/assets/gncv.png",
                        className="img-fluid",
                        style={'maxHeight': '400px'}
                    )
                ], md=4),
                dbc.Col([
                    html.H2("Contextualización del Dataset", className="mb-4"),
                    dcc.Markdown('''
                    **Fuente:** Ministerio de Minas y Energía - Sistema de Información de Combustibles Líquidos (SICOM)  
                    **Actualización:** 28 de Marzo de 2025  
                    **Registros:** 9,876  
                    **Cobertura:** Nacional

                    #### Variables Clave:
                    - `PRECIO_PROMEDIO_PUBLICADO`: Precio promedio diario (COP/m³)
                    - `FECHA_PRECIO`: Fecha de registro (AAAA-MM-DD)
                    - `DEPARTAMENTO_EDS`: Ubicación geográfica de la estación
                    - `TIPO_COMBUSTIBLE`: GNCV (Gas Natural Comprimido Vehicular)

                    #### Objetivos Analíticos:
                    1. Analizar la distribución geográfica de precios del GNCV a nivel nacional
                    2. Identificar departamentos con mayores y menores precios promedio
                    3. Evaluar la evolución histórica de precios y detectar tendencias
                    4. Facilitar comparativas entre diferentes regiones del país
                    ''')
                ], md=8)
            ], className="mb-5"),
            html.Hr(),
            cards
        ], fluid=True)
    ]
)

# Pestaña del mapa
def create_mapa_figure():
    if gdf_merged is not None and geojson is not None:
        try:
            fig = px.choropleth(
                gdf_merged,
                geojson=geojson,
                locations=gdf_merged.index,
                color='PRECIO_PROMEDIO_PUBLICADO',
                hover_name='DPTO_CNMBR',
                color_continuous_scale="YlOrRd",
                labels={'PRECIO_PROMEDIO_PUBLICADO': 'Precio (COP/m³)'},
                range_color=(gdf_merged['PRECIO_PROMEDIO_PUBLICADO'].min(), 
                            gdf_merged['PRECIO_PROMEDIO_PUBLICADO'].max())
            ).update_geos(
                fitbounds="locations",
                visible=False,
                projection_type="mercator"
            ).update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                paper_bgcolor='#f8f9fa',
                plot_bgcolor='#f8f9fa',
                coloraxis_colorbar={
                    'title': {'text': 'COP/m³', 'font': {'color': '#333'}},
                    'tickfont': {'color': '#333'}
                }
            )
            return fig
        except Exception as e:
            print(f"Error al crear mapa: {e}")
            return px.scatter().update_layout(
                title="Error al cargar el mapa. Datos geoespaciales no disponibles."
            )
    else:
        return px.scatter().update_layout(
            title="Datos geoespaciales no disponibles para mostrar el mapa."
        )

mapa_tab = dbc.Tab(
    label="Mapa Interactivo",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='mapa-principal',
                        figure=create_mapa_figure()
                    )
                ])
            ])
        ], fluid=True)
    ]
)

# Pestaña de análisis temporal
analisis_temporal_tab = dbc.Tab(
    label="Análisis Temporal",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Evolución del Precio Promedio Nacional", className="mb-4 mt-4"),
                    dcc.Graph(
                        id='grafico-tendencia',
                        figure=px.line(
                            df_tendencia, 
                            x='FECHA_PRECIO', 
                            y='PRECIO_PROMEDIO_PUBLICADO',
                            labels={'FECHA_PRECIO': 'Fecha', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio Promedio (COP/m³)'}
                        ).update_layout(
                            template='plotly_white',
                            xaxis_title="Fecha",
                            yaxis_title="Precio (COP/m³)",
                            hovermode="x unified"
                        )
                    )
                ], md=12, className="mb-4")
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Análisis de Estacionalidad", className="mb-4"),
                    dcc.Dropdown(
                        id='dropdown-departamento',
                        options=[{'label': dep, 'value': dep} for dep in sorted(df['DEPARTAMENTO_EDS'].unique())],
                        value=df['DEPARTAMENTO_EDS'].iloc[0] if len(df) > 0 else None,
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Graph(id='grafico-estacionalidad')
                ], md=6),
                
                dbc.Col([
                    html.H3("Resumen Estadístico Temporal", className="mb-4"),
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Estadísticas de precios en el período analizado", className="card-title mb-3"),
                            dash_table.DataTable(
                                id='tabla-estadisticas',
                                columns=[
                                    {"name": col, "id": col} for col in estadisticas.columns
                                ],
                                data=estadisticas.to_dict('records'),
                                style_cell={'textAlign': 'left', 'padding': '10px'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ]
                            ),
                            html.Hr(),
                            html.H5("Variación Mensual", className="mt-3"),
                            dcc.Graph(
                                id='grafico-variacion-mensual',
                                figure=px.bar(
                                    df.groupby(df['FECHA_PRECIO'].dt.month)['PRECIO_PROMEDIO_PUBLICADO'].mean().reset_index(),
                                    x='FECHA_PRECIO',
                                    y='PRECIO_PROMEDIO_PUBLICADO',
                                    labels={'FECHA_PRECIO': 'Mes', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio Promedio (COP/m³)'}
                                ).update_layout(
                                    xaxis=dict(
                                        tickmode='array',
                                        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                        ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                                    ),
                                    template='plotly_white'
                                )
                            )
                        ]),
                        className="h-100"
                    )
                ], md=6)
            ])
        ], fluid=True)
    ]
)

# Pestaña de comparativas
comparativas_tab = dbc.Tab(
    label="Comparativas",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Departamentos con Precios Más Altos y Más Bajos", className="mb-4 mt-4"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='grafico-top-caros',
                                figure=px.bar(
                                    df_top_caros,
                                    y='DEPARTAMENTO_EDS',
                                    x='PRECIO_PROMEDIO_PUBLICADO',
                                    orientation='h',
                                    title="Top 10 Departamentos con GNCV Más Caro",
                                    labels={'DEPARTAMENTO_EDS': 'Departamento', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio Promedio (COP/m³)'},
                                    color='PRECIO_PROMEDIO_PUBLICADO',
                                    color_continuous_scale="Reds"
                                ).update_layout(
                                    yaxis={'categoryorder': 'total ascending'},
                                    template='plotly_white'
                                )
                            )
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(
                                id='grafico-top-economicos',
                                figure=px.bar(
                                    df_top_economicos,
                                    y='DEPARTAMENTO_EDS',
                                    x='PRECIO_PROMEDIO_PUBLICADO',
                                    orientation='h',
                                    title="Top 10 Departamentos con GNCV Más Económico",
                                    labels={'DEPARTAMENTO_EDS': 'Departamento', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio Promedio (COP/m³)'},
                                    color='PRECIO_PROMEDIO_PUBLICADO',
                                    color_continuous_scale="Greens_r"
                                ).update_layout(
                                    yaxis={'categoryorder': 'total descending'},
                                    template='plotly_white'
                                )
                            )
                        ], md=6)
                    ])
                ], md=12, className="mb-4")
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Comparativa de Precios por Departamento", className="mb-4"),
                    dcc.Graph(
                        id='grafico-boxplot',
                        figure=px.box(
                            df,
                            x='DEPARTAMENTO_EDS',
                            y='PRECIO_PROMEDIO_PUBLICADO',
                            title="Distribución de Precios por Departamento",
                            labels={'DEPARTAMENTO_EDS': 'Departamento', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio (COP/m³)'}
                        ).update_layout(
                            xaxis={'categoryorder': 'mean descending'},
                            template='plotly_white',
                            xaxis_tickangle=-45,
                            margin={"b": 120}
                        )
                    )
                ], md=12, className="mb-4")
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Tabla Comparativa Detallada", className="mb-4"),
                    html.P("Seleccione departamentos para comparar:"),
                    dcc.Dropdown(
                        id='dropdown-comparar',
                        options=[{'label': dep, 'value': dep} for dep in sorted(df['DEPARTAMENTO_EDS'].unique())],
                        value=[df['DEPARTAMENTO_EDS'].iloc[0], df['DEPARTAMENTO_EDS'].iloc[1]] if len(df) > 1 else [],
                        multi=True,
                        className="mb-3"
                    ),
                    html.Div(id='tabla-comparativa')
                ], md=12)
            ])
        ], fluid=True)
    ]
)

# Layout principal
app.layout = dbc.Container([
    navbar,
    dbc.Tabs([
        contexto_tab,
        mapa_tab,
        analisis_temporal_tab,
        comparativas_tab
    ])
], fluid=True)

# =============================================
# CALLBACKS Y EJECUCIÓN
# =============================================

# Callback para el gráfico de estacionalidad
@app.callback(
    Output('grafico-estacionalidad', 'figure'),
    [Input('dropdown-departamento', 'value')]
)
def update_estacionalidad(departamento):
    if not departamento or len(df) == 0:
        return px.line(title="No hay datos disponibles")
    
    filtered_df = df[df['DEPARTAMENTO_EDS'] == departamento]
    df_avg = filtered_df.groupby('FECHA_PRECIO')['PRECIO_PROMEDIO_PUBLICADO'].mean().reset_index()
    
    fig = px.line(
        df_avg, 
        x='FECHA_PRECIO', 
        y='PRECIO_PROMEDIO_PUBLICADO',
        title=f"Evolución de Precios en {departamento}",
        labels={'FECHA_PRECIO': 'Fecha', 'PRECIO_PROMEDIO_PUBLICADO': 'Precio (COP/m³)'}
    )
    
    # Añadir línea de tendencia
    if len(df_avg) > 1:  # Verificar que hay suficientes puntos para la regresión
        try:
            x = np.array([(date - df_avg['FECHA_PRECIO'].min()).days for date in df_avg['FECHA_PRECIO']])
            y = df_avg['PRECIO_PROMEDIO_PUBLICADO']
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            fig.add_trace(
                go.Scatter(
                    x=df_avg['FECHA_PRECIO'],
                    y=p(x),
                    mode='lines',
                    name='Tendencia',
                    line=dict(color='red', dash='dot')
                )
            )
        except:
            pass  # Si hay error en el cálculo de la regresión, continuar sin línea de tendencia
    
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Fecha",
        yaxis_title="Precio (COP/m³)",
        hovermode="x unified"
    )
    
    return fig

# Callback para la tabla comparativa
@app.callback(
    Output('tabla-comparativa', 'children'),
    [Input('dropdown-comparar', 'value')]
)
def update_tabla_comparativa(departamentos):
    if not departamentos or len(df) == 0:
        return html.P("Por favor, seleccione al menos un departamento para comparar.")
    
    # Crear un DataFrame con métricas por departamento
    resultados = []
    
    for dep in departamentos:
        dep_data = df[df['DEPARTAMENTO_EDS'] == dep]
        
        if not dep_data.empty:
            resultados.append({
                'Departamento': dep,
                'Precio Promedio': dep_data['PRECIO_PROMEDIO_PUBLICADO'].mean(),
                'Precio Mínimo': dep_data['PRECIO_PROMEDIO_PUBLICADO'].min(),
                'Precio Máximo': dep_data['PRECIO_PROMEDIO_PUBLICADO'].max(),
                'Desviación Estándar': dep_data['PRECIO_PROMEDIO_PUBLICADO'].std(),
                'Número de Estaciones': dep_data['NOMBRE_COMERCIAL_EDS'].nunique(),
                'Variación (%)': (dep_data['PRECIO_PROMEDIO_PUBLICADO'].max() - dep_data['PRECIO_PROMEDIO_PUBLICADO'].min()) / 
                                dep_data['PRECIO_PROMEDIO_PUBLICADO'].min() * 100 if dep_data['PRECIO_PROMEDIO_PUBLICADO'].min() > 0 else 0
            })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Crear la tabla
    tabla = dash_table.DataTable(
        data=df_resultados.to_dict('records'),
        columns=[
            {"name": "Departamento", "id": "Departamento"},
            {"name": "Precio Promedio (COP/m³)", "id": "Precio Promedio", "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Precio Mínimo (COP/m³)", "id": "Precio Mínimo", "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Precio Máximo (COP/m³)", "id": "Precio Máximo", "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Desviación Estándar", "id": "Desviación Estándar", "type": "numeric", "format": {"specifier": ",.2f"}},
            {"name": "Estaciones", "id": "Número de Estaciones", "type": "numeric"},
            {"name": "Variación (%)", "id": "Variación (%)", "type": "numeric", "format": {"specifier": ",.2f"}}
        ],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return tabla

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)