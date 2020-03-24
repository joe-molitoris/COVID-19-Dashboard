import pandas as pd
import numpy as np
import pathlib
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from matplotlib.ticker import ScalarFormatter
import requests
from bs4 import BeautifulSoup

#################################################################################################
#Download data, clean it, and construct features.
#################################################################################################

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

europecdc = requests.get("https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide")
europedata = europecdc.text
soup = BeautifulSoup(europedata)
links = []
for link in soup.find_all('a'):
    links.append(link.get('href'))
data = [i for i in links if str(i).endswith('xlsx')]

resp = requests.get(data[0])
df = pd.read_excel(data[0])

df = df.drop(['Day', 'Month', 'Year'], axis=1)
df.columns = ['DateRep', 'NewCases', 'Deaths', 'Country', 'GeoId']

iso = pd.read_csv(DATA_PATH.joinpath('ISO-codes.csv'), encoding='latin1')
totalpop =  pd.read_csv(DATA_PATH.joinpath('World Population.csv'), encoding='latin1')
total_pop_iso = pd.read_csv(DATA_PATH.joinpath('Linked ISO.csv'), encoding='latin1')

totalpop = pd.merge(totalpop, total_pop_iso, how='inner', left_on='Country', right_on='Country or Area')
totalpop = totalpop.drop(['Country', 'year', 'Country or Area'], axis=1)
totalpop.columns = ['TotalPopulation', 'iso3']

df['Country'] = df['Country'].str.replace("_", " ")

df.loc[df['Country']=='switzerland', 'Country']='Switzerland'
df.loc[df['Country']=='United States of America', 'Country']='USA'
df.loc[df['Country']=='South Korea', 'Country']='S. Korea'

unique_dates = df['DateRep'].unique().tolist()
unique_countries = df['Country'].unique().tolist()

full_dates_df = pd.DataFrame({'DateRep':unique_dates})
full_dates_df2 = full_dates_df.copy()
x=0
while x<len(unique_countries)-1:
    full_dates_df2 = full_dates_df2.append(full_dates_df)
    x+=1

indexes = list(range(0,len(full_dates_df2), len(unique_dates)))
stopindexes = [i+len(unique_dates) for i in indexes]
stopindexes[-1]=stopindexes[-1]-1

full_dates_df2['Country']=""

full_dates_df2 = full_dates_df2.reset_index(drop=True)

x = 0
for i in indexes:
    full_dates_df2.iloc[i:, 1] = unique_countries[x]
    x+=1

df['DateRep'] = pd.to_datetime(df['DateRep'])

full_dates_df2['DateRep'] = pd.to_datetime(full_dates_df2['DateRep'])

df = df.sort_values(by=['Country', 'DateRep'])
df = df.reset_index(drop=True)

df = pd.merge(df, full_dates_df2, how='right', on=['Country', 'DateRep'])

df = df.sort_values(by=['Country', 'DateRep'])
df = df.reset_index(drop=True)

df['NewCases'] = df['NewCases'].fillna(0)
df['Deaths'] = df['Deaths'].fillna(0)

geoids = pd.DataFrame(df.groupby(['Country'])['GeoId'].value_counts())
geoids = geoids.rename(columns={'GeoId':'counts'}).reset_index()

geodict = dict(zip(geoids['Country'].tolist(), geoids['GeoId'].tolist()))

df['GeoId'] = df['Country'].map(geodict)

df['TotalCases'] = df.groupby('Country')['NewCases'].transform('cumsum')
df['TotalDeaths'] = df.groupby('Country')['Deaths'].transform('cumsum')
df['CaseFatalityRate'] = df['TotalDeaths']/df['TotalCases']*100

df['AnyCases'] = df['TotalCases']>0
df['FirstCase']=df.groupby(['Country', 'AnyCases'])['DateRep'].transform('min')
df['FirstCase']=df.groupby(['Country'])['FirstCase'].transform('max')

df['DaysSinceFirstCase'] = (df['DateRep']-df['FirstCase']).dt.days

for i in ['TotalCases', 'TotalDeaths', 'CaseFatalityRate']:
    df.loc[df[i]==0, i]=np.nan

#Since Onset statistics
onset_df = df.copy()
onset_df = onset_df[onset_df['DaysSinceFirstCase']>=0].copy()
onset_df['pct_change'] = onset_df['TotalCases'].pct_change()*100

total_df = df.groupby('DateRep')['NewCases', 'Deaths'].sum().reset_index()
total_df['Country'] = 'World'
total_df['TotalCases'] = total_df['NewCases'].cumsum()
total_df['TotalDeaths'] = total_df['Deaths'].cumsum()
total_df['CaseFatalityRate'] = total_df['TotalDeaths']/total_df['TotalCases']*100
total_df['AnyCases'] = True
total_df['FirstCase'] = total_df['DateRep'].min()
total_df['DaysSinceFirstCase']= (total_df['DateRep']-total_df['FirstCase']).dt.days

onset_df = onset_df.append(total_df).reset_index(drop=True)

onset_df = pd.merge(onset_df, iso[['iso2', 'iso3']], how='left', left_on='GeoId', right_on='iso2')
onset_df.loc[onset_df['Country']=='United Kingdom', 'iso3']='GBR'
onset_df.loc[onset_df['Country']=='Greece', 'iso3']='GRC'
onset_df.loc[onset_df['Country']=='Kosovo', 'iso3']='RKS'
onset_df.loc[onset_df['Country']=='French Polynesia', 'iso3']='PYF'

onset_df = pd.merge(onset_df, totalpop, how= 'left', on='iso3')
onset_df['TotalPopulation'] = onset_df['TotalPopulation']*1000
onset_df['CasesPerMillion'] = onset_df['TotalCases']/onset_df['TotalPopulation']*1000000

onset_df.to_csv(DATA_PATH.joinpath('COVID-19 Since Onset.csv'), index=False)

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

data_collected=str(date.today())

#Read newly constructed data
df = pd.read_csv(DATA_PATH.joinpath('COVID-19 Since Onset.csv'))

#Defines function to add spaces between capital letters
def spacify(s):
    space_indexes=[s.index(i) for i in s if i.isupper()]
    parts = [s[i:j] for i,j in zip(space_indexes, space_indexes[1:]+[None])]
    new_string=" ".join(parts)
    return new_string

#Defines function to create trace
def create_trace(y, x='DaysSinceFirstCase', country=['World']):
    if y!='CaseFatalityRate':
        hover_template = spacify(y) + ": %{y}"
    else:
        hover_template = spacify(y) + ": %{y: .1f}"
    color_cycler=['#FF4F01', '#F5CF00', '#A30003', '#EF8616', '#FD0000']
    trace = [
        {
        'x':df[df['Country']==i][x],
        'y':df[df['Country']==i][y],
        'mode':'lines',
        'name':i,
        'showlegend':False,
        'marker':{'opacity':0.8, 'color':c},
        'hovertemplate': hover_template
        } for i, c in zip(country, color_cycler)
    ] 
    return trace

def create_layout(y, x='DaysSinceFirstCase', country=['World'], logscale=False):
    axis_font_style={'size':12, 'family':'Franklin Gothic Medium', 'color':'#CECECE'}
    hover_font_style={'size':11, 'family':'Franklin Gothic Medium', 'color':'rgb(255,255,255)'}
    graph_bg_color = '#2e2e30'
    grid_color = '#3a3a3b'
    lower_bound = min([df[df['Country']==i][x].min() for i in country])
    upper_bound = max([df[df['Country']==i][x].max() for i in country])

    xaxis_range=[lower_bound, upper_bound]
    if logscale==False or y=='CaseFatalityRate':
        yaxis_options = {
            'gridcolor':grid_color,
            'gridwidth':0.1
            }
    else:
        yaxis_options = {
            'gridcolor':grid_color,
            'gridwidth':0.1,
            'type':'log',
            'dtick':1
            }
    layout = dict(
        xaxis={
            'title':{'text':spacify(x)},
            'range':xaxis_range,
            'gridcolor': grid_color, 
            'gridwidth':0.1
        },
        yaxis=yaxis_options,
        title=spacify(y),
        font=axis_font_style,
        hoverlabel={'font':hover_font_style},
        paper_bgcolor=graph_bg_color,
        plot_bgcolor=graph_bg_color,
        margin={'l':40,'b':40, 't':30, 'r':10}
    )
    return layout

def create_map(color_value='TotalCases'):
    temp_df=df.groupby('Country').tail(1).reset_index(drop=True)
    temp_df = temp_df[temp_df['Country']!='World']
    temp_df['rank'] = pd.qcut(temp_df[color_value], q=20, duplicates='drop', labels=False)
    trace=[go.Choropleth(
        locations=temp_df['iso3'],
        name='{}'.format(spacify(color_value)),
        z=temp_df['rank'],
        text=["{0}: {1:.0f}".format(i,j) if color_value not in ['CaseFatalityRate', 'CasesPerMillion'] else "{0}: {1:.1f}".format(i,j) for i,j in zip(temp_df['Country'], temp_df[color_value])],
        hovertemplate="%{text}",
        autocolorscale=False,
        colorscale='YlOrRd',
        showscale=False,
        marker_line_width=0.5,
        unselected={'marker':{'opacity': 0.3}}
        )
        ]
    return trace

#Create maps
def map_layout():
    hover_font_style={'size':11, 'family':'Franklin Gothic Medium', 'color':'rgb(255,255,255)'}
    layout=dict(
        geo={'showframe':False,
                'showcoastlines':False,
                'projection':{'type':'miller'},
                'showland':False,
                'showcountries':True,
                'visible':True,
                'countrycolor':'#2e2e30',
                'showocean':True,
                'oceancolor':'#2e2e30',
                'lataxis':{'range':[-40, 90]}},
        annotations=[go.layout.Annotation(
            x=0.05,
            y=0.98,
            xref='paper',
            yref='paper',
            text= "Reported as of: {} 08:00:00 GMT+1".format(data_collected),
            font={'family':'Franklin Gothic Medium', 'size':12, 'color':'#CECECE'},
            showarrow=False
        )],
        margin= {'l':0, 'b':0, 't':0, 'r':0},
        paper_bgcolor='#2e2e30',
        plot_bgcolor='#2e2e30',
        hoverlabel={'font':hover_font_style}
    )
    return layout

#Create DataTable
def create_table():
    infected_countries=df['Country'].nunique()

    latest_df = df.groupby('Country').tail(1)

    total_deaths = int(latest_df[latest_df['Country']=='World']['TotalDeaths'])
    new_deaths = int(latest_df[latest_df['Country']=='World']['Deaths'])

    total_cases = int(latest_df[latest_df['Country']=='World']['TotalCases'])
    new_cases = int(latest_df[latest_df['Country']=='World']['NewCases'])

    total_cfr = float(latest_df[latest_df['Country']=='World']['CaseFatalityRate'])

    latest_df = latest_df[latest_df['Country']!='World'].copy()

    new_countries = len(latest_df[latest_df['FirstCase']==latest_df['DateRep']])

    latest_df['TotalPopulation'] = latest_df['TotalPopulation']*1000
    latest_df['CasesPerMillion'] = latest_df['TotalCases']/latest_df['TotalPopulation']*1000000

    average_rate_per_million = latest_df['CasesPerMillion'].mean()

    highest_CFR = int(latest_df['CaseFatalityRate'].max())
    highest_CFR_country = latest_df[latest_df['CaseFatalityRate']==latest_df['CaseFatalityRate'].max()]['Country'].values[0]

    highest_number_of_new_cases = int(latest_df['NewCases'].max())
    highest_number_new_cases_country = latest_df[latest_df['NewCases']==latest_df['NewCases'].max()]['Country'].values[0]

    highest_number_of_new_deaths = int(latest_df['Deaths'].max())
    highest_number_new_deaths_country = latest_df[latest_df['Deaths']==latest_df['Deaths'].max()]['Country'].values[0]

    highest_cases_per_million = int(latest_df['CasesPerMillion'].max())
    highest_cases_per_million_country =latest_df[latest_df['CasesPerMillion']==latest_df['CasesPerMillion'].max()]['Country'].values[0]

    countries = "{0} ({1})".format(infected_countries, new_countries)
    deaths = "{0:.0f} ({1:.0f})".format(total_deaths, new_deaths)
    cases = "{0:.0f} ({1:.0f})".format(total_cases, new_cases)
    cfr = "{0:.1f}%".format(total_cfr)
    most_new_cases = "{0:.0f} ({1})".format(highest_number_of_new_cases, highest_number_new_cases_country)
    most_new_deaths = "{0:.0f} ({1})".format(highest_number_of_new_deaths, highest_number_new_deaths_country)
    most_casespermill = "{0:.1f} ({1})".format(highest_cases_per_million, highest_cases_per_million_country)
    most_cfr = "{0:.1f}% ({1})".format(highest_CFR, highest_CFR_country)

    summary_df = pd.DataFrame({
                            "Infected Countries":[countries],
                            "Total Deaths":[deaths],
                            "Total Cases":[cases],
                            "Case Fatality Rate" : [cfr],
                            "Most New Cases": [most_new_cases],
                            "Most New Deaths":[most_new_deaths],
                            "Most Cases/Million":[most_casespermill],
                            "Highest Fatality Rate":[most_cfr]
                            })

    summary_df = summary_df.T.reset_index()
    summary_df.columns=["", "Value (new)"]
    return summary_df
    
#Create traces and layouts
trace1=create_trace(y='TotalCases')
layout1=create_layout(y='TotalCases')

trace2=create_trace(y='TotalDeaths')
layout2=create_layout(y='TotalDeaths')

trace3=create_trace(y='NewCases')
layout3=create_layout(y='NewCases')

trace4=create_trace(y='CaseFatalityRate')
layout4=create_layout(y='CaseFatalityRate')

trace_map = create_map()
layout_map= map_layout()

datatable=create_table()

trace_dimensions = {'height':350, 'width':400, 'box-shadow':'1px 1px 1px darkgrey'}

page_style = {'backgroundColor':'#1C1A1A', 'color':'#CECECE'}
header_style = {'backgroundColor':'#1C1A1A'}
#Creates app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/jmolitoris/pen/BaNpwVy.css'], meta_tags=[{'name':'viewport', 'content':"width=device-width, initial-scale=1"}])

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(
                children='COVID-19 Situation Report',
                style={
                    'fontSize':26,
                    'font-family':'Franklin Gothic Medium',
                    'marginLeft':15,
                    'marginTop':0,
                    'color':'#CECECE'
                },
                className='six columns'
            ),
            html.Div([
                html.A(children="Follow me on Twitter", 
                        className='twitter-link',
                        href="https://twitter.com/JoeMolitoris?ref_src=twsrc%5Etfw"),
                html.Br(),
                html.A(children="Follow me on ResearchGate", 
                        className='twitter-link',
                        href="https://www.researchgate.net/profile/Joseph_Molitoris")
             ], className='two columns', style={'marginLeft':50}),
            dcc.Dropdown(id='dropdown', 
                        className='three columns',
                        options=[{'label':i, 'value':i} for i in df['Country'].unique()],
                        multi=True,
                        value=['World'],
                        placeholder='Select countries...',
                        style={'backgroundColor':'#DDDDDD'})
        ], style={'marginTop':20, 'font-family':'Franklin Gothic Medium', 'color':'#454545', 'backgroundColor':'#1C1A1A'}, 
            className='row'),
        html.Div([
            html.H4(
                children='''
                This page contains the most recent data published by the 
                European Centre for Disease Prevention and Control.\n
                For more information, visit https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide.
                ''',
                style={
                    'fontSize':12,
                    'font-family':'Franklin Gothic Medium',
                    'marginLeft':15,
                    'marginRight':180
                },
                className='six columns'
            ),
            html.Div(children='Time:', 
                    className='one column'),
            dcc.RadioItems(
                id='radio-button1',
                options=[
                    {'label': 'Days since first case', 'value':'DaysSinceFirstCase'},
                    {'label': 'Calendar date', 'value':'DateRep'}
                ],
                value='DaysSinceFirstCase',
                className='two columns',
                style={'marginLeft':-50}
            ),
            html.Div(children='Units:', 
                    className='one column'),
            dcc.RadioItems(
                id='radio-button2',
                options=[
                    {'label': 'Linear', 'value':False},
                    {'label': 'Logarithmic', 'value':True}
                ],
                value=False,
                className='one columns',
                style={'marginLeft':-50}
            )
        ], className='row', style=header_style), 

        html.Div([
            html.Div(children='Current Situation:',
                    className='two columns',
                    style={'marginLeft':15}    
            ),
            dcc.RadioItems(
                id='radio-button3',
                options=[
                    {'label':'Total Cases', 'value':'TotalCases'},
                    {'label':'Total Deaths', 'value':'TotalDeaths'},
                    {'label':'New Cases', 'value':'NewCases'},
                    {'label':'Case Fatality Rate', 'value':'CaseFatalityRate'},
                    {'label':'Cases per Million', 'value':'CasesPerMillion'}
                ],
                value='TotalCases',
                labelStyle={'display':'inline-block'},
                className='nine columns',
                style={'marginLeft':15}
            )
        ], style=header_style, className='row'),

        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{'name':i, 'id':i} for i in datatable.columns],
                    data=datatable.to_dict('records'),
                    style_as_list_view=True,
                    style_cell={
                        'fontSize':12, 
                        'font-family':'Franklin Gothic Medium',
                        'minWidth':'100px',
                        'maxWidth':'130px',
                        'backgroundColor':'rgb(50,50,50)',
                        'color':'white',
                        'textAlign':'left'
                    },
                    style_data={'height':40},
                    style_data_conditional=[
                        {
                        'if': {'row_index':'odd'},
                        'backgroundColor': 'rgb(100,100,100)'
                    }
                    ],
                    style_header={
                        'backgroundColor':'rgb(76,76,76)',
                        'fontWeight':'bold'
                    },
                    style_table={'overflowX': 'scroll'}                    
                )
            ], className='three columns', style={'width':200, 'height':350, 'marginLeft':15}),
            html.Div([
                dcc.Graph(
                    id='map',
                    style={'height':350, 'width':590,
                    'box-shadow':'1px 1px 1px darkgrey'
                    },
                    figure={
                        'data':trace_map,
                        'layout':layout_map
                    }
                )
            ],className='six columns', style={'marginLeft':20}),
            html.Div([
                dcc.Graph(
                    id='graph-1',
                    style=trace_dimensions,
                    figure={
                        'data':trace1,
                        'layout':layout1
                    })
            ],className='three columns', style={'marginLeft':-20})
        ], style=page_style, className='row'),

        html.Br(),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='graph-2',
                    style=trace_dimensions,
                    figure={
                        'data':trace2,
                        'layout':layout2
                    }
                )
            ],className='four columns'),
            html.Div([
                dcc.Graph(
                    id='graph-3',
                    style=trace_dimensions,
                    figure={
                        'data':trace3,
                        'layout':layout3
                    })
            ],className='four columns'),
            html.Div([
                dcc.Graph(
                    id='graph-4',
                    style=trace_dimensions,
                    figure={
                        'data':trace4,
                        'layout':layout4
                    }
                )
            ], className='four columns'),
        ], style=page_style, className='row'),
        html.Br()        
        ], style=page_style, className='twelve columns')
    ], style=page_style)

@app.callback(
    [Output(component_id='graph-1', component_property='figure'),
    Output(component_id='graph-2', component_property='figure'),
    Output(component_id='graph-3', component_property='figure'),
    Output(component_id='graph-4', component_property='figure')],
    [Input(component_id='dropdown', component_property='value'),
    Input(component_id='radio-button1', component_property='value'),
    Input(component_id='radio-button2', component_property='value')]
)

def update_graphs(country_value=["World"], time_option='DaysSinceFirstCase', logscale=False):
    if country_value is None:
        raise PreventUpdate

    new_trace1 = create_trace(y='TotalCases', x=time_option, country=country_value)
    new_trace2 = create_trace(y='TotalDeaths', x=time_option, country=country_value)
    new_trace3 = create_trace(y='NewCases', x=time_option, country=country_value)
    new_trace4 = create_trace(y='CaseFatalityRate', x=time_option, country=country_value)

    new_layout1 = create_layout(y='TotalCases', x=time_option, country=country_value, logscale=logscale)
    new_layout2 = create_layout(y='TotalDeaths', x=time_option, country=country_value, logscale=logscale)
    new_layout3 = create_layout(y='NewCases', x=time_option, country=country_value, logscale=logscale)
    new_layout4 = create_layout(y='CaseFatalityRate', x=time_option, country=country_value, logscale=logscale)

    return {'data':new_trace1, 'layout': new_layout1}, {'data':new_trace2, 'layout': new_layout2}, {'data':new_trace3, 'layout': new_layout3}, {'data':new_trace4, 'layout': new_layout4}

@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input(component_id='radio-button3', component_property='value')]
    )

def update_map(new_map_value):
    if new_map_value is None:
        raise PreventUpdate

    new_map = create_map(color_value=new_map_value)
    new_map_layout = map_layout()
    return {'data':new_map, 'layout':new_map_layout}

if __name__=='__main__':
    app.run_server()
