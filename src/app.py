from dash import Dash, dcc, html, Input, Output
import colorlover as cl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

app = Dash(__name__)
server = app.server
colorscale = cl.scales['9']['qual']['Paired']

#Data 
df_stocks = pd.read_csv(
    'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/'
    'DataVisualizationRPython/Lectures/Python/PythonDataSets/dash-stock-ticker-demo.csv'
)
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])

df_bees_raw = pd.read_csv(
    'https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/'
    'DataVisualizationRPython/Lectures/Python/PythonDataSets/intro_bees.csv'
)
df_bees = (
    df_bees_raw
    .groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']]
    .mean()
    .reset_index()
)

# Helper functions

def calc_bbands(price, window=20, num_std=2):
    m = price.rolling(window).mean()
    s = price.rolling(window).std()
    return m, m + num_std * s, m - num_std * s

def calc_macd(price, fast=12, slow=26, signal=9):
    ema_fast    = price.ewm(span=fast,   adjust=False).mean()
    ema_slow    = price.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_rsi(price, period=14):
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def calc_ad(high, low, close, volume):
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv.fillna(0) * volume).cumsum()

def calc_stoch(high, low, close, k_period=14, d_period=3):
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def calc_aroon(high, low, period=25):
    aroon_up   = high.rolling(period + 1).apply(lambda x: float(np.argmax(x)), raw=True) / period * 100
    aroon_down = low.rolling(period + 1).apply(lambda x: float(np.argmin(x)),  raw=True) / period * 100
    return aroon_up, aroon_down

def calc_adx(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(period).mean()
    dm_plus  = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
    dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.rolling(period).mean() / atr.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.rolling(period).mean()
    return adx, di_plus, di_minus

# Design attributes

BG      = '#0a0a14'
BG2     = '#12122a'
BG3     = '#1a1a2e'
BORDER  = '#252545'
TEXT    = '#d0d0f0'
MUTED   = '#606090'
ACCENT1 = '#7c83fd'
ACCENT2 = '#f5c842'
GREEN   = '#27e8a7'
RED     = '#ff4d6d'
FONT    = '"DM Mono", monospace'

TAB_STYLE = dict(
    backgroundColor=BG3, color=MUTED,
    padding='12px 28px', border='none',
    fontFamily=FONT, fontSize='12px',
    letterSpacing='0.09em', textTransform='uppercase',
)
TAB_SEL = {**TAB_STYLE,
           'backgroundColor': BG2, 'color': TEXT,
           'borderTop': f'2px solid {ACCENT1}', 'fontWeight': '700'}

DD = dict(backgroundColor=BG2, color=TEXT)

def lbl(txt):
    return html.Div(txt, style={
        'color': MUTED, 'fontSize': '10px',
        'letterSpacing': '0.12em', 'textTransform': 'uppercase',
        'marginBottom': '6px',
    })

def sec(txt, color=ACCENT1):
    return html.Div(txt, style={
        'color': color, 'fontSize': '13px', 'fontWeight': '700',
        'letterSpacing': '0.1em', 'textTransform': 'uppercase',
        'marginBottom': '16px', 'paddingBottom': '6px',
        'borderBottom': f'1px solid {BORDER}',
    })

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, family=FONT, size=11),
    xaxis=dict(gridcolor=BORDER, color=MUTED, showgrid=True),
    yaxis=dict(gridcolor=BORDER, color=MUTED, showgrid=True),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    margin=dict(l=55, r=20, t=35, b=40),
)


#Finance explorer definition 

INDICATORS = [
    {'label': 'Bollinger Bands',                    'value': 'bbands'},
    {'label': 'MACD',                               'value': 'macd'},
    {'label': 'RSI (14)',                            'value': 'rsi'},
    {'label': 'On-Balance Volume (OBV)',             'value': 'obv'},
    {'label': 'Accumulation/Distribution (A/D)',    'value': 'ad'},
    {'label': 'Stochastic Oscillator',              'value': 'stoch'},
    {'label': 'Aroon Oscillator',                   'value': 'aroon'},
    {'label': 'Average Directional Index (ADX)',    'value': 'adx'},
]
IND_IDS = ['bbands','macd','rsi','obv','ad','stoch','aroon','adx']

def param_panel():
    return html.Div([
        html.Div(id='params-bbands', children=[
            html.Div([
                html.Div([lbl('BB Window'),
                          dcc.Dropdown(id='bb-window',
                                       options=[{'label': str(n), 'value': n} for n in [5,10,14,20,26,50]],
                                       value=20, clearable=False, style=DD)],
                         style={'width':'150px','marginRight':'16px'}),
                html.Div([lbl('BB Std Dev'),
                          dcc.Dropdown(id='bb-std',
                                       options=[{'label': str(n), 'value': n} for n in [1,1.5,2,2.5,3]],
                                       value=2, clearable=False, style=DD)],
                         style={'width':'150px'}),
            ], style={'display':'flex'}),
        ], style={'display':'none'}),

        html.Div(id='params-macd', children=[
            html.Div([
                html.Div([lbl('Fast'),
                          dcc.Dropdown(id='macd-fast',
                                       options=[{'label':str(n),'value':n} for n in [5,8,12,15]],
                                       value=12, clearable=False, style=DD)],
                         style={'width':'120px','marginRight':'16px'}),
                html.Div([lbl('Slow'),
                          dcc.Dropdown(id='macd-slow',
                                       options=[{'label':str(n),'value':n} for n in [20,26,30,35]],
                                       value=26, clearable=False, style=DD)],
                         style={'width':'120px','marginRight':'16px'}),
                html.Div([lbl('Signal'),
                          dcc.Dropdown(id='macd-signal',
                                       options=[{'label':str(n),'value':n} for n in [5,7,9,12]],
                                       value=9, clearable=False, style=DD)],
                         style={'width':'120px'}),
            ], style={'display':'flex'}),
        ], style={'display':'none'}),

        html.Div(id='params-rsi', children=[
            html.Div([lbl('RSI Period'),
                      dcc.Dropdown(id='rsi-period',
                                   options=[{'label':str(n),'value':n} for n in [7,9,14,21,28]],
                                   value=14, clearable=False, style=DD)],
                     style={'width':'150px'}),
        ], style={'display':'none'}),

        html.Div(id='params-obv', style={'display':'none'}),
        html.Div(id='params-ad',  style={'display':'none'}),

        html.Div(id='params-stoch', children=[
            html.Div([
                html.Div([lbl('%K Period'),
                          dcc.Dropdown(id='stoch-k',
                                       options=[{'label':str(n),'value':n} for n in [5,9,14,21]],
                                       value=14, clearable=False, style=DD)],
                         style={'width':'150px','marginRight':'16px'}),
                html.Div([lbl('%D Period'),
                          dcc.Dropdown(id='stoch-d',
                                       options=[{'label':str(n),'value':n} for n in [3,5,7]],
                                       value=3, clearable=False, style=DD)],
                         style={'width':'150px'}),
            ], style={'display':'flex'}),
        ], style={'display':'none'}),

        html.Div(id='params-aroon', children=[
            html.Div([lbl('Aroon Period'),
                      dcc.Dropdown(id='aroon-period',
                                   options=[{'label':str(n),'value':n} for n in [14,25,50]],
                                   value=25, clearable=False, style=DD)],
                     style={'width':'150px'}),
        ], style={'display':'none'}),

        html.Div(id='params-adx', children=[
            html.Div([lbl('ADX Period'),
                      dcc.Dropdown(id='adx-period',
                                   options=[{'label':str(n),'value':n} for n in [7,10,14,21]],
                                   value=14, clearable=False, style=DD)],
                     style={'width':'150px'}),
        ], style={'display':'none'}),
    ])

tab1_layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Span('TradeDesk', style={'color':ACCENT1,'fontSize':'2em','fontWeight':'800'}),
                html.Span(' Trading Dashboard', style={'color':TEXT,'fontSize':'2em','fontWeight':'300'}),
            ]),
            html.Div('Multi-indicator Technical Analysis · Candlestick · Long/Short Signals',
                     style={'color':MUTED,'fontSize':'10px','letterSpacing':'0.14em',
                            'textTransform':'uppercase','marginTop':'4px'}),
        ]),
        html.Div([
            html.Div('📊', style={'fontSize':'3em'}),
            html.Div('TRADEDESK', style={'color':ACCENT1,'fontSize':'11px',
                                        'letterSpacing':'0.2em','fontWeight':'700'}),
        ], style={'textAlign':'right'}),
    ], style={'display':'flex','justifyContent':'space-between','alignItems':'center',
              'padding':'24px 36px 20px','borderBottom':f'1px solid {BORDER}'}),

    html.Div([
        html.Div([
            lbl('Stocks (multi-select)'),
            dcc.Dropdown(id='stock-ticker-input',
                         options=[{'label':s,'value':s} for s in df_stocks.Stock.unique()],
                         value=['YHOO','GOOGL'], multi=True, style=DD),
        ], style={'flex':'2','marginRight':'24px'}),
        html.Div([
            lbl('Technical Indicators (up to 4)'),
            dcc.Dropdown(id='indicator-select', options=INDICATORS,
                         value=['bbands','macd'], multi=True, style=DD),
        ], style={'flex':'3'}),
    ], style={'display':'flex','padding':'20px 36px 8px'}),

    html.Div(param_panel(), style={'padding':'8px 36px 16px'}),
    html.Div(id='graphs', style={'padding':'0 36px 36px'}),
], style={'backgroundColor':BG})


# bee colony tab definition

AFFECTED_OPTIONS = [{'label':v,'value':v} for v in sorted(df_bees['Affected by'].unique())]
YEAR_OPTIONS     = [{'label':str(y),'value':y} for y in sorted(df_bees['Year'].unique())]

tab2_layout = html.Div([
    html.Div([
        html.Div([
            html.Span('BeeWatch', style={'color':ACCENT2,'fontSize':'2em','fontWeight':'800'}),
            html.Span(' Colony Analytics', style={'color':TEXT,'fontSize':'2em','fontWeight':'300'}),
            html.Div('US Bee-Colony Impact · State / Stressor / Year Analysis',
                     style={'color':MUTED,'fontSize':'10px','letterSpacing':'0.14em',
                            'textTransform':'uppercase','marginTop':'4px'}),
        ]),
        html.Div('🐝', style={'fontSize':'4em','opacity':'0.5'}),
    ], style={'display':'flex','justifyContent':'space-between','alignItems':'center',
              'padding':'24px 36px 20px','borderBottom':f'1px solid {BORDER}'}),

    html.Div([
        html.Div([
            lbl('Year'),
            dcc.Dropdown(id='slct_year', options=YEAR_OPTIONS, value=2015,
                         clearable=False, style={**DD,'width':'180px'}),
        ], style={'marginRight':'32px'}),
        html.Div([
            lbl('Affected by'),
            dcc.Dropdown(id='slct_affected', options=AFFECTED_OPTIONS,
                         value='Varroa_mites', clearable=False,
                         style={**DD,'width':'280px'}),
        ]),
    ], style={'display':'flex','alignItems':'flex-end','padding':'20px 36px 4px'}),

    html.Div(id='output_container',
             style={'padding':'6px 36px 16px','color':MUTED,'fontSize':'11px'}),

    html.Div([
        sec('Geographic Distribution – % Colonies Impacted', ACCENT2),
        dcc.Graph(id='my_bee_map', figure={}, style={'height':'460px'}),
    ], style={'padding':'16px 36px 0'}),

    html.Div([
        html.Div([
            html.Div([sec('Avg Impact by Year', ACCENT2),
                      dcc.Graph(id='bee-by-year', figure={}, style={'height':'280px'})],
                     style={'flex':'1','marginRight':'16px','backgroundColor':BG2,
                            'padding':'16px','borderRadius':'6px'}),
            html.Div([sec('Impact by Stressor (selected year)', ACCENT2),
                      dcc.Graph(id='bee-by-affected', figure={}, style={'height':'280px'})],
                     style={'flex':'1','backgroundColor':BG2,'padding':'16px','borderRadius':'6px'}),
        ], style={'display':'flex','marginBottom':'16px'}),

        html.Div([
            html.Div([sec('Top 10 States by Impact', ACCENT2),
                      dcc.Graph(id='bee-top-states', figure={}, style={'height':'300px'})],
                     style={'flex':'1','marginRight':'16px','backgroundColor':BG2,
                            'padding':'16px','borderRadius':'6px'}),
            html.Div([sec('Stressor Trend Over Years', ACCENT2),
                      dcc.Graph(id='bee-trend', figure={}, style={'height':'300px'})],
                     style={'flex':'1','backgroundColor':BG2,'padding':'16px','borderRadius':'6px'}),
        ], style={'display':'flex','marginBottom':'16px'}),

        html.Div([
            sec('Heatmap: State × Stressor (selected year)', ACCENT2),
            dcc.Graph(id='bee-heatmap', figure={}, style={'height':'420px'}),
        ], style={'backgroundColor':BG2,'padding':'16px','borderRadius':'6px','marginBottom':'36px'}),
    ], style={'padding':'16px 36px 0'}),
], style={'backgroundColor':BG})


# Layout 

app.layout = html.Div([
    html.Link(rel='stylesheet',
              href='https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500;700&display=swap'),

    dcc.Tabs(
        id='main-tabs', value='tab-finance',
        children=[
            dcc.Tab(label='📈  AlphaDesk',          value='tab-finance',
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label='🐝  BeeWatch Analytics',  value='tab-bees',
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ],
        style={'backgroundColor':BG3,'borderBottom':f'1px solid {BORDER}'},
        colors={'border':BORDER,'primary':ACCENT1,'background':BG3},
    ),

    html.Div(tab1_layout, id='tab-finance-content'),
    html.Div(tab2_layout, id='tab-bees-content', style={'display':'none'}),
], style={'backgroundColor':BG,'minHeight':'100vh','fontFamily':FONT})


# Finance Explorer Callback functions

@app.callback(
    Output('tab-finance-content','style'),
    Output('tab-bees-content','style'),
    Input('main-tabs','value'),
)
def toggle_tabs(tab):
    show, hide = {}, {'display':'none'}
    return (show, hide) if tab == 'tab-finance' else (hide, show)


@app.callback(
    [Output(f'params-{i}','style') for i in IND_IDS],
    Input('indicator-select','value'),
)
def show_params(selected):
    selected = selected or []
    return [({} if i in selected else {'display':'none'}) for i in IND_IDS]


@app.callback(
    Output('graphs','children'),
    Input('stock-ticker-input','value'),
    Input('indicator-select',  'value'),
    Input('bb-window',   'value'),
    Input('bb-std',      'value'),
    Input('macd-fast',   'value'),
    Input('macd-slow',   'value'),
    Input('macd-signal', 'value'),
    Input('rsi-period',  'value'),
    Input('stoch-k',     'value'),
    Input('stoch-d',     'value'),
    Input('aroon-period','value'),
    Input('adx-period',  'value'),
)
def update_stock_graph(tickers, indicators,
                       bb_win, bb_std,
                       macd_fast, macd_slow, macd_sig,
                       rsi_per, stoch_k, stoch_d,
                       aroon_per, adx_per):

    if not tickers:
        return html.Div('Select at least one stock ticker.',
                        style={'color':MUTED,'marginTop':'60px',
                               'textAlign':'center','fontSize':'14px'})

    indicators = [i for i in (indicators or []) if i][:4]
    SUB_PANEL  = {'macd','rsi','obv','ad','stoch','aroon','adx'}
    overlay    = [i for i in indicators if i not in SUB_PANEL]
    panels     = [i for i in indicators if i in SUB_PANEL]
    n_rows     = 1 + len(panels)
    row_heights = [0.50] + [0.50/len(panels)]*len(panels) if panels else [1.0]

    graphs = []
    tick_colors = [ACCENT1, GREEN, '#ff9f43', '#a29bfe', '#fd79a8', '#ee5a24']

    for ti, ticker in enumerate(tickers):
        dff = df_stocks[df_stocks['Stock']==ticker].sort_values('Date').reset_index(drop=True)

        panel_names = [p.upper() for p in panels]
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=row_heights,
            subplot_titles=[f'{ticker}  ·  Price + Overlays'] + panel_names,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dff['Date'],
            open=dff['Open'], high=dff['High'],
            low=dff['Low'],   close=dff['Close'],
            name=ticker,
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
        ), row=1, col=1)

        # Overlay: Bollinger Bands
        if 'bbands' in overlay:
            mid, upper, lower = calc_bbands(dff['Close'], bb_win or 20, bb_std or 2)
            for y, name, dash in [(upper,'BB Upper','dot'),(mid,'BB Mid','dash'),(lower,'BB Lower','dot')]:
                fig.add_trace(go.Scatter(
                    x=dff['Date'], y=y, mode='lines', name=name,
                    line=dict(color=ACCENT1, width=1, dash=dash),
                    hoverinfo='skip', showlegend=(name=='BB Mid'),
                ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pd.concat([dff['Date'], dff['Date'][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(124,131,253,0.07)',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip', showlegend=False, name='BB Fill',
            ), row=1, col=1)

        # Sub-panels
        for pi, ind in enumerate(panels):
            row = pi + 2

            if ind == 'macd':
                ml, sl2, hist = calc_macd(dff['Close'], macd_fast or 12, macd_slow or 26, macd_sig or 9)
                bar_colors = [GREEN if v >= 0 else RED for v in hist.fillna(0)]
                fig.add_trace(go.Bar(x=dff['Date'], y=hist, name='Histogram',
                                     marker_color=bar_colors, opacity=0.55), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=ml, mode='lines', name='MACD',
                                         line=dict(color=ACCENT1, width=1.5)), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=sl2, mode='lines', name='Signal',
                                         line=dict(color=ACCENT2, width=1.5)), row=row, col=1)

            elif ind == 'rsi':
                rsi = calc_rsi(dff['Close'], rsi_per or 14)
                fig.add_trace(go.Scatter(x=dff['Date'], y=rsi, mode='lines', name='RSI',
                                         line=dict(color='#fd79a8', width=1.5)), row=row, col=1)
                for lvl, col in [(70, RED),(30, GREEN)]:
                    fig.add_hline(y=lvl, line_dash='dot', line_color=col, opacity=0.5, row=row, col=1)

            elif ind == 'obv':
                obv = calc_obv(dff['Close'], dff['Volume'])
                fig.add_trace(go.Scatter(x=dff['Date'], y=obv, mode='lines', name='OBV',
                                         line=dict(color='#00cec9', width=1.5)), row=row, col=1)

            elif ind == 'ad':
                ad = calc_ad(dff['High'], dff['Low'], dff['Close'], dff['Volume'])
                fig.add_trace(go.Scatter(x=dff['Date'], y=ad, mode='lines', name='A/D Line',
                                         line=dict(color='#fdcb6e', width=1.5)), row=row, col=1)

            elif ind == 'stoch':
                k, d = calc_stoch(dff['High'], dff['Low'], dff['Close'], stoch_k or 14, stoch_d or 3)
                fig.add_trace(go.Scatter(x=dff['Date'], y=k, mode='lines', name='%K',
                                         line=dict(color='#a29bfe', width=1.5)), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=d, mode='lines', name='%D',
                                         line=dict(color='#fd79a8', width=1.5, dash='dot')), row=row, col=1)
                for lvl, col in [(80, RED),(20, GREEN)]:
                    fig.add_hline(y=lvl, line_dash='dot', line_color=col, opacity=0.5, row=row, col=1)

            elif ind == 'aroon':
                aup, adn = calc_aroon(dff['High'], dff['Low'], aroon_per or 25)
                fig.add_trace(go.Scatter(x=dff['Date'], y=aup, mode='lines', name='Aroon Up',
                                         line=dict(color=GREEN, width=1.5)), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=adn, mode='lines', name='Aroon Down',
                                         line=dict(color=RED, width=1.5)), row=row, col=1)

            elif ind == 'adx':
                adx, dip, dim = calc_adx(dff['High'], dff['Low'], dff['Close'], adx_per or 14)
                fig.add_trace(go.Scatter(x=dff['Date'], y=adx, mode='lines', name='ADX',
                                         line=dict(color=ACCENT2, width=2)), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=dip, mode='lines', name='+DI',
                                         line=dict(color=GREEN, width=1, dash='dot')), row=row, col=1)
                fig.add_trace(go.Scatter(x=dff['Date'], y=dim, mode='lines', name='-DI',
                                         line=dict(color=RED, width=1, dash='dot')), row=row, col=1)
                fig.add_hline(y=25, line_dash='dot', line_color=MUTED, opacity=0.4, row=row, col=1)

        total_h = 400 + 200 * len(panels)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, family=FONT, size=11),
            height=total_h,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', y=1.02, x=0,
                        bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            margin=dict(l=60, r=20, t=40, b=30),
        )
        for r in range(1, n_rows + 1):
            fig.update_xaxes(gridcolor=BORDER, color=MUTED, row=r, col=1)
            fig.update_yaxes(gridcolor=BORDER, color=MUTED, row=r, col=1)

        graphs.append(html.Div([
            html.Div(ticker, style={
                'color': tick_colors[ti % len(tick_colors)],
                'fontSize':'13px','fontWeight':'700',
                'letterSpacing':'0.1em','padding':'8px 0 4px',
            }),
            dcc.Graph(figure=fig, config={'displayModeBar':True}),
        ], style={'backgroundColor':BG2,'borderRadius':'8px',
                  'padding':'8px 16px 12px','marginBottom':'20px',
                  'border':f'1px solid {BORDER}'}))

    return graphs



# Bee colony callback functions

def _bl():
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, family=FONT, size=11),
        xaxis=dict(gridcolor=BORDER, color=MUTED),
        yaxis=dict(gridcolor=BORDER, color=MUTED),
        margin=dict(l=55, r=20, t=20, b=50),
    )

@app.callback(
    Output('output_container','children'),
    Output('my_bee_map','figure'),
    Output('bee-by-year','figure'),
    Output('bee-by-affected','figure'),
    Output('bee-top-states','figure'),
    Output('bee-trend','figure'),
    Output('bee-heatmap','figure'),
    Input('slct_year','value'),
    Input('slct_affected','value'),
)
def update_bee_graphs(year, affected):
    dff_map = df_bees[(df_bees['Year']==year) & (df_bees['Affected by']==affected)]

    # Choropleth
    fig_map = px.choropleth(
        dff_map, locationmode='USA-states', locations='state_code', scope='usa',
        color='Pct of Colonies Impacted',
        hover_data=['State','Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted':'% Impacted'},
        template='plotly_dark',
    )
    fig_map.update_layout(
        paper_bgcolor=BG, geo_bgcolor=BG,
        font=dict(family=FONT, color=TEXT),
        margin=dict(r=0, t=0, l=0, b=0),
    )

    # Avg by year (line)
    by_year = (df_bees[df_bees['Affected by']==affected]
               .groupby('Year')['Pct of Colonies Impacted'].mean().reset_index())
    fig_yr = go.Figure(go.Scatter(
        x=by_year['Year'], y=by_year['Pct of Colonies Impacted'],
        mode='lines+markers',
        line=dict(color=ACCENT2, width=2),
        marker=dict(size=7, color=ACCENT2),
        fill='tozeroy', fillcolor='rgba(245,200,66,0.08)',
    ))
    fig_yr.update_layout(**_bl())

    # By stressor bar
    by_aff = (df_bees[df_bees['Year']==year]
              .groupby('Affected by')['Pct of Colonies Impacted'].mean()
              .sort_values(ascending=False).reset_index())
    fig_aff = go.Figure(go.Bar(
        x=by_aff['Affected by'], y=by_aff['Pct of Colonies Impacted'],
        marker_color=[ACCENT2 if v==affected else BORDER for v in by_aff['Affected by']],
        marker_line_width=0,
    ))
    fig_aff.update_layout(**_bl(), xaxis_tickangle=-30)

    # Top 10 states
    top10 = dff_map.nlargest(10,'Pct of Colonies Impacted').sort_values('Pct of Colonies Impacted')
    fig_top = go.Figure(go.Bar(
        x=top10['Pct of Colonies Impacted'], y=top10['State'],
        orientation='h',
        marker_color=ACCENT2, marker_line_width=0,
        text=top10['Pct of Colonies Impacted'].round(1).astype(str)+'%',
        textposition='outside',
        textfont=dict(color=MUTED, size=9),
    ))
    fig_top.update_layout(**_bl())

    # Trend lines
    trend = (df_bees.groupby(['Year','Affected by'])['Pct of Colonies Impacted']
             .mean().reset_index())
    pal = px.colors.qualitative.Vivid
    fig_trend = go.Figure()
    for k, aff in enumerate(sorted(trend['Affected by'].unique())):
        sub = trend[trend['Affected by']==aff]
        fig_trend.add_trace(go.Scatter(
            x=sub['Year'], y=sub['Pct of Colonies Impacted'],
            mode='lines+markers', name=aff,
            line=dict(color=pal[k % len(pal)], width=1.5),
            marker=dict(size=5),
        ))
    fig_trend.update_layout(**_bl(),
                            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)))

    # Heatmap
    pivot = (df_bees[df_bees['Year']==year]
             .groupby(['State','Affected by'])['Pct of Colonies Impacted'].mean()
             .reset_index()
             .pivot(index='State', columns='Affected by', values='Pct of Colonies Impacted'))
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale='YlOrRd',
        hovertemplate='State: %{y}<br>Stressor: %{x}<br>Impact: %{z:.1f}%<extra></extra>',
    ))
    fig_heat.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, family=FONT, size=10),
        xaxis=dict(color=MUTED, tickangle=-35),
        yaxis=dict(color=MUTED, autorange='reversed'),
        margin=dict(l=130, r=20, t=10, b=100),
    )

    container = (f'📅 Year: {year}  ·  🔬 Stressor: {affected}  ·  '
                 f'📊 States with data: {len(dff_map)}')
    return container, fig_map, fig_yr, fig_aff, fig_top, fig_trend, fig_heat


if __name__ == '__main__':
    app.run(debug=True)