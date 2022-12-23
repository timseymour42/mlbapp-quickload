import datetime
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import plotly
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from dash import Dash
import plotly.express as px
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import json
from dash.exceptions import PreventUpdate
import MySQLdb
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template('LUX')

def date_to_str(date):
    '''
    Args:
        date (datetime): datetime object for the day of the season

    Returns:
        str: string representation of the given date

    '''
    month = str(date.month)
    day = str(date.day)
    if date.day <= 9:
        day = str(0) + day
    if date.month <= 9:
        month = str(0) + month
    return str(date.year) + '-' + month + '-' + day

# turning every value in the dataframe into a float
def string_to_num(string):
    if(string == 'NA'):
        return 'NA'
    elif(type(string) == str):
        if('%' in string):
            string = string.replace('%', '')
    return float(string)



class pitcher:
    hr = 0
    war = 0
    # parameterized constructor
    def __init__(self, name, seasons, ip):
        pitch_df = ui_pitch_df.set_index(['Name', 'Season'])
        pitcher = pitch_df.loc[name]
        #eliminating seasons that the player did not play in the given range
        played = list()
        for seas in seasons:
            if seas in pitcher.index:
                played.append(seas)
        pitcher = pitcher.loc[played]
        #finds pitcher HR allowed per inning pitched
        self.hr = pitcher['HR'].sum() / pitcher['IP'].sum()
        #finds pitcher WAR per inning pitched
        self.war = pitcher['WAR'].sum() / pitcher['IP'].sum()
        #Number of innings to be pitched by player (hypothetically)
        self.ip = ip
        self.name = name
    
    def display(self):
        print("HR/9: " + str(self.hr / 9))
        print("WAR/9 " + str(self.war / 9))

    def setIP(self, ip):
        self.ip = ip

    def scale(self, innings):
        self.hr = self.hr * innings
        self.war = self.war * innings

    def getHR(self):
        return self.hr

    def getWAR(self):
        return self.war


class hitter:
    wrcp = 0
    bsr = 0
    defn = 0
    slg = 0
    # parameterized constructor
    def __init__(self, name, seasons, games):
        hit_df = ui_hit_df.set_index(['Name', 'Season'])
        hitter = hit_df.loc[name]
        #eliminating seasons that the player did not play in the given range
        played = list()
        for seas in seasons:
            if seas in hitter.index:
                played.append(seas)
        hitter = hitter.loc[played]
        #wRC+ is normalized season by season, so average is taken across inputted season range; drawback is smaller sample sizes may have greater effect than hoped
        #Users will be able to see season by season stats for the player, so they can use their own intuition to evaluate validity of using a given season for a player
        self.wrcp = hitter['wRC+'].mean()
        #finds slugging percentage
        self.slg = hitter['TB'].sum() / hitter['AB'].sum()
        #finds defense per game
        self.defn = hitter['Def'].sum() / hitter['G'].sum()
        #finds baserunning per game
        self.bsr = hitter['BsR'].sum() / hitter['G'].sum()
        #Number of games to be played by player (hypothetically)
        self.games = games
        self.name = name
    
    def display(self):
        # displays statistics at a 162 game pace
        print("wRC+: " + str(self.wrcp))
        print("BsR: " + str(self.bsr * 162))
        print("Def: " + str(self.defn * 162))
        print("SLG: " + str(self.slg))

    def setGames(self, games):
        self.games = games

    def scale(self, games):
        self.wrcp = self.wrcp * games
        self.bsr = self.bsr * games
        self.defn = self.defn * games
        self.slg = self.slg * games

    def getWRC(self):
        return self.wrcp

    def getBsR(self):
        return self.bsr

    def getDef(self):
        return self.defn

    def getSLG(self):
        return self.slg


def pitcher_df(rotation):
    #Accumulating dictionaries representing players into a list
    ps = list()
    for p in rotation:
        row = {'Name': p.name, 'IP': p.ip, 'HR': p.hr, 'WAR': p.war}
        ps.append(row)
    return pd.DataFrame(ps)

def hitter_df(lineup):
    #Accumulating dictionaries representing players into a list
    hs = list()
    for batter in lineup:
        row = {'Name': batter.name, 'G': batter.games, 'wRC+': batter.wrcp, 'BsR': batter.bsr, 'Def': batter.defn, 'SLG': batter.slg}
        hs.append(row)
    return pd.DataFrame(hs)

def wins_for_team(lineup, rotation, model='standard'):
    '''
    lineup (list) consists of hitter objects
    pitchers (list) consists of pitcher objects
    model (Sklearn object) model trained on team data to be used to classify customized team (using predict_proba)
    '''

    #Scaling each players statistics to have their contribution correspond to their designated innings pitched or games played
    for batter in lineup:
        batter.display()
        #multiplies each batting statistic by the games they play
        batter.scale(batter.games)
        batter.display()
    for p in rotation:
        p.display()
        #multiplies each pitching statistic by the innings they pitch
        p.scale(p.ip)
        p.display()

    #Creating dataframes for simpler aggregation
    lineup = hitter_df(lineup)
    rotation = pitcher_df(rotation)

    #ensuring that there are 1458 games played by position players and innings thrown by pitchers
    games = lineup['G'].sum()
    if games != 1458:
        raise Exception(f'Total Games inputted: {games}, must be 1458')
    ip = rotation['IP'].sum()
    if ip != 1458:
        raise Exception(f'Total IP inputted: {ip}, must be 1458')
    
    stats = dict()
    #when scaled wrc is multiplied by games played for each player to ensure proportionate contribution
    stats['wRC+'] = lineup['wRC+'].sum() / 1458
    #when scaled slg is multiplied by games played for each player to ensure proportionate contribution
    stats['SLG'] = lineup['SLG'].sum() / 1458
    #The equivalent of a single team's defense metric is the sum of their entire lineup's Def (which is why the denominator is 162)
    stats['Def'] = lineup['Def'].sum() / 162
    stats['BsR'] = lineup['BsR'].sum() / 162
    stats['HR/9'] = rotation['HR'].sum() / (9 * 1458)
    stats['WAR'] = rotation['WAR'].sum() / (162)
    reg_stats = stats.copy()
    #Stored normalization factors from team data
    metrics = ['wRC+', 'HR/9', 'BsR', 'WAR', 'Def', 'SLG']
    for stat in metrics:
        stats[stat] = (stats[stat] - scales.loc[stat]['Mean']) / scales.loc[stat]['Unit Variance']
    if (type(model) == str):
        logReg = LogisticRegression().fit(X, y)
        adaBoost = AdaBoostClassifier(learning_rate = .3, n_estimators = 30).fit(X, y)
        wins = logReg.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1] * 2
        wins += adaBoost.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1]
        wins /= 3
        wins *= 162
        return wins, reg_stats
    else:
        return model.predict_proba([[stats['wRC+'], stats['HR/9'], stats['BsR'], stats['WAR'], stats['Def'], stats['SLG']]])[0][1] * 162, reg_stats






def clean_game_data(all_stats):
    all_stats = all_stats[all_stats.GS == 1]
    # These columns have only null values for single games
    all_stats.drop(columns = ['xwOBA', 'xERA', 'vFA (pi)'], inplace = True)
    # applying the function to each column to ensure all data points are numerical
    for col in all_stats.columns:
        if col not in ['Team', 'Date', 'GB']:
            all_stats[col] = all_stats[col].apply(string_to_num)
    all_stats = all_stats.drop(columns = ['Team', 'G', 'PA', 'R',
           'Date', 'L', 'SV', 'GS', 'IP', 'RBI'])
    # Only ~100 columns with null values
    all_stats.dropna(inplace = True)
    X = all_stats.drop(columns = ['W'])
    X = X[['wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG']]
    cols = X.columns
    y = all_stats['W']
    #Scaling each column to be 
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X, columns = cols)
    #Storing values used to scale each feature for manual normalization in later step
    feat_names = ['wRC+', 'HR/9', 'BsR', 'WAR', 'Def', 'SLG']
    scales = pd.DataFrame({'Feature': feat_names, 'Unit Variance': scaler.scale_, 'Mean': scaler.mean_})
    scales.set_index('Feature', inplace = True)
    return X, y, scales



def clean_team_data(team_data):
    # applying the function to each column to ensure all data points are numerical
    for col in team_data.columns:
        if col not in ['Team', 'Season', 'GB']:
            team_data[col] = team_data[col].apply(string_to_num)
    # Saving a copy of the scraped data 
    return team_data


def clean_player_data(hit_df, pitch_df):
    '''
    function intended to make statistics numerical, manually calculate statistics, and set the indices to Name and Season

    Args:
    wrc (pd.DataFrame) contains individual player data by season
    pitch (pd.DataFrame) contains individual pitcher data by season

    Returns wrc, pitch as clean datasets for use in App'''
    
    hit_df = hit_df[hit_df['wRC+'] != None]
    # applying the function to each column to ensure all data points are numerical
    for col in hit_df.columns:
        if col not in ['Name', 'Team', 'GB', 'Pos']:
            hit_df[col] = hit_df[col].apply(string_to_num)
    for col in pitch_df.columns:
        if col not in ['Name', 'Team', 'GB']:
            pitch_df[col] = pitch_df[col].apply(string_to_num)
    #Determining home runs allowed for each player for easier calculation
    pitch_df['HR'] = pitch_df['HR/9'] * pitch_df['IP'] * 9
    #Determining total bases for each player for more accurate slugging percentage calculation
    # First must find at bats by subtracting walks using walk percentage
    # Calculation ignores HBP
    hit_df['AB'] = hit_df['PA'] * (1 - (hit_df['BB%'] * .01))
    # Calculation necessary for determining slugging percentage over multiple seasons
    hit_df['TB'] = hit_df['SLG'] * hit_df['AB']
    return hit_df, pitch_df

# scale WAR, BsR, Def
def scale_stat(df, stats):
    for stat in stats:
        df[stat] = (162 / df['GS']) * df[stat]
    return df

def refresh_data():
    '''
    This section will be replaced to read from github csv for deployment purposes

    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='mlb_db')
    tblchk = db.cursor()
    # The year of the latest record in the data table
    sql_game_data = pd.read_sql('SELECT * FROM game_data', con = db)
    sql_team_data = pd.read_sql('SELECT * FROM team_data', con = db)
    sql_hitter_data = pd.read_sql('SELECT * FROM hitter_data', con = db)
    sql_pitcher_data = pd.read_sql('SELECT * FROM pitcher_data', con = db)
    '''
    sql_game_data = pd.read_csv('https://github.com/timseymour42/mlbapp/blob/main/game_data.csv?raw=true')
    sql_team_data = pd.read_csv('https://github.com/timseymour42/mlbapp/blob/main/team_data.csv?raw=true')
    sql_hitter_data = pd.read_csv('https://github.com/timseymour42/mlbapp/blob/main/hitter_data.csv?raw=true')
    sql_pitcher_data = pd.read_csv('https://github.com/timseymour42/mlbapp/blob/main/pitcher_data.csv?raw=true')
    sql_col_mapping = {'BB%': 'BB_pct', 'K%': 'K_pct', 'wRC+': 'wRC_plus', 'K/9': 'K_per_9',
        'BB/9': 'BB_per_9', 'HR/9': 'HR_per_9', 'LOB%': 'LOB_pct', 'GB%': 'GB_pct', 'HR/FB': 'HR_per_FB', 'vFA (pi)': 'vFA'}
    python_col_mapping = {v: k for k, v in sql_col_mapping.items()}
    sql_game_data.rename(columns = python_col_mapping, inplace = True)
    sql_team_data.rename(columns = python_col_mapping, inplace = True)
    sql_hitter_data.rename(columns = python_col_mapping, inplace = True)
    sql_pitcher_data.rename(columns = python_col_mapping, inplace = True)
    X, y, scales = clean_game_data(sql_game_data)
    ui_hit_df, ui_pitch_df = clean_player_data(sql_hitter_data, sql_pitcher_data)
    team_history = clean_team_data(sql_team_data)
    team_history = scale_stat(team_history, ['BsR', 'WAR_y', 'Def'])
    team_history = team_history[['Team', 'Season', 'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG', 'W']]
    team_history['Season'] = team_history['Season'].apply(int)
    team_history = team_history.sort_values(by='Season', ascending=False).round(decimals=3)
    team_history['W'] = team_history['W'].round()
    ui_hit_df.reset_index(inplace=True)
    ui_pitch_df.reset_index(inplace=True)
    #hitters selected
    hit_sel = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
    #pitchers selected
    pit_sel = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
    #current year
    curr_year = datetime.datetime.now().year
    first_year = int(ui_hit_df['Season'].min())
    games = 1458
    innings = 1458
    ui_hit_df = ui_hit_df.round(decimals=3).sort_values(by=['Season', 'HR'], ascending=False)
    ui_pitch_df = ui_pitch_df.round(decimals=3).sort_values(by=['Season', 'WAR'], ascending=False)
    return team_history, ui_hit_df, ui_pitch_df, X, y, scales, hit_sel, pit_sel, curr_year, first_year, games, innings

team_history, ui_hit_df, ui_pitch_df, X, y, scales, hit_sel, pit_sel, curr_year, first_year, games, innings = refresh_data()
ui_hit_df = ui_hit_df.drop(columns=['index', 'hitter_id'])
ui_pitch_df = ui_pitch_df.drop(columns=['index', 'pitcher_id'])
ui_hit_df = ui_hit_df[ui_hit_df.Season >= 2019]
ui_pitch_df = ui_pitch_df[ui_pitch_df.Season >= 2019]

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
#server for render
server=app.server
def generate_table(dataframe, id):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))], id = id
    )



app.layout = html.Div(children=[
    #Storing the hitters selected in a df
    dcc.Store(id = 'sel_tbl', data = [], storage_type = 'memory'),
    #Storing the remaining number of games
    dcc.Store(id = 'games_rem', data = [], storage_type = 'memory'),
    #Storing the pitchers selected in a df
    dcc.Store(id = 'psel_tbl', data = [], storage_type = 'memory'),
    #Storing the remaining number of innings
    dcc.Store(id = 'inn_rem', data = [], storage_type = 'memory'), 
    #Storing team stats to be displayed in scatter plot  
    dcc.Store(id = 'team_stats', data = [], storage_type = 'memory'),
    html.Div(html.Label("MLB Build a Team"), style = {'text-align': 'center', 'font-size': '25px', 'vertical-align': 'top',
                                           'align': 'center', 'width': '100%', 'margin-top': '40px'}),              
    #PLAYER SELECTION
    #Multi DropDown for hitters
    html.Div(children = [
        html.Label(['Build your roster!', html.Br(),
        '1. Select any player since 1900', html.Br(), 
        '2. Choose a range of seasons that they played in', html.Br(),
        '3. Specify their games played or innings pitched', html.Br(),
        '4. Make sure to use all 1458 games and innings', html.Br(),
        '5. See how your team stacks up against MLB history', html.Br()], style={"border":"10px black solid", 'margin-bottom': '20px'}),
        html.Div(children=[
            #HITTER, START YEAR, END YEAR
            html.Div(children = [
                html.Label([
                    "Hitter",
                    dcc.Dropdown(
                        #Dropdown with players to be inputted into algo
                        id='hitter-dd-calc', clearable=True,
                        multi=False,
                        style={'width':'200px'},
                        value=[], options=[
                            {'label': c, 'value': c}
                            for c in ui_hit_df['Name'].unique()
                        ])
                ]),
                #Start year
                html.Label(['Start Year',
                dcc.Dropdown(
                        id='start-year-dropdown', clearable=False,
                        style={'width':'100px'},
                        value=[curr_year], options=[
                            {'label': c, 'value': c}
                            for c in reversed(range(first_year, curr_year + 1, 1))
                        ])]), 
                #End year
                html.Label(['End Year',
                dcc.Dropdown(
                        id='end-year-dropdown', clearable=False,
                        style={'width':'100px'},
                        value=[curr_year], options=[
                            {'label': c, 'value': c}
                            for c in reversed(range(first_year, curr_year + 1, 1))
                ])])], style={'display': 'block', 'verticalAlign':'top'}),
            #HITTER, START YEAR, END YEAR

            #GAME INPUT, SUBMIT HITTER, CLEAR PLAYER INFO
            #Input Box for games
            html.Div(children=[
                html.Label(['Games', dcc.Input(id='game_input', type='number', min=1, max=games, step=1)], style={'width':'100px'}),
                #Add Player Button
                html.Button('Submit Hitter', id='submit-hitter', n_clicks=None, type = 'submit'),
                #Clear Player info Button
                html.Button('Clear Player Info', id='clear-player', n_clicks=None)], style={'display': 'block', 'verticalAlign':'top', 'width':'100%'}),
            #GAME INPUT,SUBMIT HITTER, CLEAR PLAYER INFO

            #CLEAR LINEUP        
            #Clear lineup button
            html.Button('Clear Lineup', id='clear-lineup', n_clicks=None, style = {'text-align': 'center'}),
            #CLEAR LINEUP

            #GAMES REMAINING LABEL
            #Label for Games Remaining
            #HTML Table populated by DropDown; (Player, Years, Games)
            html.Div(children = [f'Hitters Selected; Games Remaining: {games}'], id = 'game'),
            #GAMES REMAINING LABEL

            #TABLE
            html.Div(children = [generate_table(hit_sel, 'hit_sel')], id = 'hit_sel_tbl')], style={'display':'block', 'width': '500px'}),
            #TABLE
        
        #PITCHER, START YEAR, END YEAR
        html.Div(children=[
            #Multi DropDown for pitchers
            html.Label([
                "Pitcher",
                dcc.Dropdown(
                    id='pitcher-dd-calc', clearable=True,
                    multi = False,
                    style={'width':'200px'},
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in ui_pitch_df['Name'].unique()
                    ])
            ]),
            #Start year
            html.Label(['Start Year',
            dcc.Dropdown(
                    id='start-year-dropdown-p', clearable=False,
                    style={'width':'100px'},
                    value=[curr_year], options=[
                        {'label': c, 'value': c}
                        for c in reversed(range(first_year, curr_year + 1, 1))
                    ])]), 
            #End year
            html.Label(['End Year',
            dcc.Dropdown(
                    id='end-year-dropdown-p', clearable=False,
                    style={'width':'100px'},
                    value=[curr_year], options=[
                        {'label': c, 'value': c}
                        for c in reversed(range(first_year, curr_year + 1, 1))
            ])])], style={'display': 'block', 'verticalAlign':'top', 'width':'100%'}),
            #PITCHER, START YEAR, END YEAR

        html.Div(children=[
            #INNINGS INPUT, SUBMIT PICTHER, CLEAR PLAYER INFO
            #Input Box for innings
            html.Label(['Innings', dcc.Input(id='inn_input', type='number', min=1, max=innings, step=1)], style={'width':'100px'}),
            #Add Pitcher Button
            html.Button('Submit Pitcher', id='submit-pitcher', n_clicks=None),
            #Clear Pitcher info Button
            html.Button('Clear Pitcher Info', id='clear-pitcher', n_clicks=None)], style={'display': 'block', 'verticalAlign':'top', 'width':'100%'}),
            #INNINGS INPUT, SUBMIT PICTHER, CLEAR PLAYER INFO

        #CLEAR ROTATION
        #Clear rotation button
        html.Button('Clear Rotation', id='clear-rotation', n_clicks=None, style = {'text-align': 'center'}),
        #CLEAR ROTATION

        #INNINGS REMAINING LABEL
        #Label for Innings Remaining
        #HTML Table populated by DropDown; (Player, Years, Innings)
        html.Div(children = [f'Pitchers Selected; Games Remaining: {innings}'], id = 'inn'),
        #INNINGS REMAINING LABEL

        #TABLE
        html.Div(children = [generate_table(pit_sel, 'pit_sel')], id = 'pit_sel_tbl'),
        #TABLE


        #Submit Buttom that is only clickable when innings and games remaining are 0
        html.Button('Submit Team', id='sub-team', n_clicks=None, style = {'margin-left': '55px'}),
        html.Div(children = ['Wins: '], id = 'team-wins-prediction', style={'margin-top': '20px', 'margin-left': '10px'}),
        ], style = {'display': 'inline-block', 'margin-top': '100px', 'vertical-align': 'top', 'margin-left':'100px'}),
    #Creative Lineup Comparison Graph
    html.Div(children = [
        
        dcc.Graph(id = 'team-wins', style={'width': '90vh', 'height': '90vh', 'text-align': 'center'}),
        #Dropdowns for querying team wins graph
        #Start year
        html.Div(children = [
        html.Div(
            html.Label(['Start Year', 
            dcc.Dropdown(
                    id='start-year-dropdown-g', clearable=False,
                    style={'width':'160px'},
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in range(first_year, curr_year + 1, 1)
                    ])]), style = {'display': 'inline-block', 'width': '20%', 'margin-left': '55px'}), 
        #End year
        html.Div(
            html.Label(['End Year',
            dcc.Dropdown(
                    id='end-year-dropdown-g', clearable=False,
                    style={'width':'160px'},
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in range(first_year, curr_year + 1, 1)
            ])]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        html.Div(
            html.Label([
                "Team",
                dcc.Dropdown(
                    id='team-graph', clearable=True,
                    style={'width':'160px'},
                    multi=True,
                    value=[], options=[
                        {'label': c, 'value': c}
                        for c in ui_hit_df['Team'].unique()
                    ])
            ]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        html.Div(
            html.Label([ 'Stat',
            dcc.Dropdown(
                    id='stat-dd', clearable=False,
                    style={'width':'160px'},
                    value=[], options=[
                        'wRC+', 'HR/9', 'BsR', 'WAR_y', 'Def', 'SLG'
                    ])]), style = {'display': 'inline-block', "margin-left": "15px", 'width': '20%'}),
        ], style = {'display': 'inline-block', 'width': '100%'})

        ], style = {'display': 'inline-block', "margin-left": "50px"}),
    html.Label(['Search through a database of every player since 1900 (>, <, = may be helpful in filter row)'], 
    style={'margin-left': '500px', 'margin-top': '60px', 'font-size':'20px', 'font-weight': 'bold'}),
    #HITTER SECTION
    # Team Dropdown for hitter table
    html.Div(children = [
        html.Label([
            "Team",
            dcc.Dropdown(
                id='team-hit', clearable=True,
                style={'width':'100px'},
                multi=True,
                value=[], options=[
                    {'label': c, 'value': c}
                    for c in ui_hit_df['Team'].unique()
                ])
        ]),
        # Hitter Dropdown
        html.Label([
            "Hitter",
            dcc.Dropdown(
                id='hitter-dropdown', clearable=False,
                style={'width':'200px'},
                multi = True,
                value=['Aaron Judge'], options=[
                    {'label': c, 'value': c}
                    for c in ui_hit_df['Name'].unique()
                ])
        ]),
        # hitter research table
        dash_table.DataTable(
        data=ui_hit_df.loc[ui_hit_df.Name == 'Aaron Judge'].to_dict('records'), ####### inserted line
        columns = [{'id': c, 'name': c} for c in ui_hit_df.columns], ####### inserted line
            id='htable',
            filter_action='native',
            row_selectable='single',
            editable=False,
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current= 0,
            virtualization=False,
            page_size= 20,
            hidden_columns = ['AB', 'TB'], fill_width=False
        )], style = {'display': 'inline-block', 'margin-left':'50px'}),
    html.Div([
    #PITCHER SECTION
    # Team Dropdown for pitcher table
    html.Label([
        "Team",
        dcc.Dropdown(
            id='team-pitch', clearable=True,
            style={'width':'100px'},
            multi=True,
            value=[], options=[
                {'label': c, 'value': c}
                for c in ui_pitch_df['Team'].unique()
            ])
    ]),
    # Pitcher Dropdown
    html.Label([
        "Pitcher",
        dcc.Dropdown(
            id='pitcher-dropdown', clearable=False,
            style={'width':'200px'},
            multi = True,
            value=['Corbin Burnes'], options=[
                {'label': c, 'value': c}
                for c in ui_pitch_df['Name'].unique()
            ])
    ]),
    # pitcher research table
    dash_table.DataTable(
       data=ui_pitch_df.loc[ui_pitch_df.Name == 'Corbin Burnes'].to_dict('records'), ####### inserted line
       columns = [{'id': c, 'name': c} for c in ui_pitch_df.columns], ####### inserted line
        id='ptable',
        filter_action='native',
        row_selectable='single',
        editable=False,
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        virtualization=False,
        page_current= 0,
        page_size= 20,
        hidden_columns = ['HR']
    )], style={'margin-left':'50px', 'margin-bottom': '200px'})])
    

#HITTER RESEARCH SECTION CALLBACKS
@app.callback(Output('htable', 'columns'), [Input('team-hit', 'value'), 
                                           Input('hitter-dropdown', 'value')])
def update_columns(teams, hitters):
    return [{"name": i, "id": i} for i in ui_hit_df.columns]
    
@app.callback(Output('htable', 'data'), [Input('team-hit', 'value'),
                                        Input('hitter-dropdown', 'value')])
def update_data(teams, hitters):
  '''
  Args: 
    teams: selected teams
    htiters: selected hitters
  '''
  if teams and hitters:
    a = ui_hit_df.loc[(ui_hit_df.Team.isin(teams)) & (ui_hit_df.Name.isin(hitters))]
    return a.to_dict('records')
  elif hitters:
    a = ui_hit_df.loc[(ui_hit_df.Name.isin(hitters))]
    return a.to_dict('records')
  return pd.DataFrame().to_dict('records')

#PITCHER RESEARCH SECTION CALLBACKS
@app.callback(Output('ptable', 'columns'), [Input('team-pitch', 'value'), 
              Input('pitcher-dropdown', 'value')])
def update_columns(teams, pitchers):
    return [{"name": i, "id": i} for i in ui_pitch_df.columns]
    
@app.callback(Output('ptable', 'data'), [Input('team-pitch', 'value'),
                                        Input('pitcher-dropdown', 'value')])
def update_data(teams, pitchers):
  '''
  Args: 
    teams: selected teams
    htiters: selected hitters
  '''
  if teams and pitchers:
    a = ui_pitch_df.loc[(ui_pitch_df.Team.isin(teams)) & (ui_pitch_df.Name.isin(pitchers))]
    return a.to_dict('records')
  elif pitchers:
    a = ui_pitch_df.loc[(ui_pitch_df.Name.isin(pitchers))]
    return a.to_dict('records')
  return pd.DataFrame().to_dict('records')

#CALLBACK FOR PLAYER SUBMISSION
@app.callback([Output('hit_sel_tbl', 'children'), Output('game', 'children'),
               Output('sel_tbl', 'data'), Output('games_rem', 'data'),
               Output('submit-hitter', 'n_clicks'), Output('clear-lineup', 'n_clicks'), 
               Output('game_input', 'max')],
              [Input('hitter-dd-calc', 'value'), Input('start-year-dropdown', 'value'),
              Input('end-year-dropdown', 'value'), Input('game_input', 'value'),
              Input('submit-hitter', 'n_clicks'), Input('clear-lineup', 'n_clicks'),
              State('sel_tbl', 'data'), State('games_rem', 'data')])
def update_lineup(hitter, start_year, end_year, game_input, button, cl_button, sel_tbl, gs):
    #clearing the lineup
    if (cl_button):
        hitters = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
        return generate_table(hitters, 'hit_sel'), 'Hitters Selected; Games Remaining: 1458', [], 1458, None, None, 1458 
    if len(sel_tbl) == 0:
        hitters = pd.DataFrame(columns = ['Name', 'Years', 'Games'])
    else:
        hitters = pd.DataFrame(sel_tbl['data-frame'])
    if type(gs) == list:
        gms = 1458
    else:
        gms = gs
    if (hitter and start_year and end_year and game_input and button and (gms - game_input >= 0)):
        years = f'{start_year} - {end_year}'
        hitters = hitters.append({'Name': hitter, 'Years': years, 'Games': game_input}, ignore_index = True)
        gms = gms - game_input
    table = generate_table(hitters, 'hit_sel')
    new_text = 'Hitters Selected; Games Remaining: ' + str(gms)
    df = {'data-frame': hitters.to_dict('records')}
    return (table, new_text, df, gms, None, None, gms)


#CLEARING DROPDOWNS UPON PLAYER SUBMISSION
@app.callback([Output('hitter-dd-calc', 'value'), Output('start-year-dropdown', 'value'),
              Output('end-year-dropdown', 'value'), Output('game_input', 'value')],
              [Input('clear-player', 'n_clicks')])
def reset_dropdowns(button):
    return None, None, None, None  

#PITCHER SELECTION

#CALLBACK FOR PLAYER SUBMISSION
@app.callback([Output('pit_sel_tbl', 'children'), Output('inn', 'children'),
               Output('psel_tbl', 'data'), Output('inn_rem', 'data'),
               Output('submit-pitcher', 'n_clicks'), Output('clear-rotation', 'n_clicks'), 
               Output('inn_input', 'max')],
              [Input('pitcher-dd-calc', 'value'), Input('start-year-dropdown-p', 'value'),
              Input('end-year-dropdown-p', 'value'), Input('inn_input', 'value'),
              Input('submit-pitcher', 'n_clicks'), Input('clear-rotation', 'n_clicks'),
              State('psel_tbl', 'data'), State('inn_rem', 'data')])
def update_rotation(pitcher, start_year, end_year, inn_input, button, cl_button, psel_tbl, inn):
    #clearing the rotation
    if (cl_button):
        pitchers = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
        return generate_table(pitchers, 'hit_sel'), 'Pitchers Selected; Innings Remaining: 1458', [], 1458, None, None, 1458 
    if len(psel_tbl) == 0:
        pitchers = pd.DataFrame(columns = ['Name', 'Years', 'Innings'])
    else:
        pitchers = pd.DataFrame(psel_tbl['data-frame'])
    if type(inn) == list:
        inns = 1458
    else:
        inns = inn
    if (pitcher and start_year and end_year and inn_input and button and (inns - inn_input >= 0)):
        years = f'{start_year} - {end_year}'
        pitchers = pitchers.append({'Name': pitcher, 'Years': years, 'Innings': inn_input}, ignore_index = True)
        inns = inns - inn_input
    table = generate_table(pitchers, 'pit_sel')
    new_text = 'Pitchers Selected; Innings Remaining: ' + str(inns)
    df = {'data-frame': pitchers.to_dict('records')}
    return (table, new_text, df, inns, None, None, inns)


#CLEARING DROPDOWNS UPON PLAYER SUBMISSION
@app.callback([Output('pitcher-dd-calc', 'value'), Output('start-year-dropdown-p', 'value'),
              Output('end-year-dropdown-p', 'value'), Output('inn_input', 'value'),
              Output('clear-pitcher', 'n_clicks')],
              [Input('clear-pitcher', 'n_clicks')])
def reset_p_dropdowns(button):
      return None, None, None, None, None 

#UPDATING THE GRAPH BASED ON WHICH STAT TO DISPLAY, WHICH TEAM, AND WHICH YEAR
@app.callback(Output('team-wins', 'figure'),
              [Input('team-graph', 'value'), Input('start-year-dropdown-g', 'value'),
              Input('end-year-dropdown-g', 'value'), Input('stat-dd', 'value'),
              Input('team_stats', 'data'), Input('sub-team', 'n_clicks')])
def update_figure(team, sy, ey, stat, team_stats, sub_team):
    a = team_history.copy()
    s = 'wRC+'
    if team and sy and ey:
        a = a.loc[(a.Team.isin(team)) & (a.Season >= sy) & (a.Season <= ey)]
    elif sy and ey:
        a = a.loc[(a.Season >= sy) & (a.Season <= ey)]
    elif team:
        a = a.loc[(a.Team.isin(team))]
    if stat:
        s = stat
    a['my_team'] = False
    if sub_team and team_stats:
        team_stats['my_team'] = True
        a = a.append(team_stats, ignore_index = True)
    
    fig = px.scatter(a, x = s, y = 'W', hover_data={'Team':':%s', 'Season':':.0f', 'wRC+':':.2f', 'HR/9':':.2f', 'BsR':':.2f', 'WAR_y':':.2f', 'Def':':.2f', 'SLG':':.2f'}, 
                    color = 'my_team', title='MLB Team Seasons', labels={'W': 'Wins'}).update_layout(title_x=.5)
    return fig

#SUBMITTING A LINEUP
@app.callback(Output('team_stats', 'data'), Output('team-wins-prediction', 'children'),
              Input('sub-team', 'n_clicks'), State('psel_tbl', 'data'),
              State('sel_tbl', 'data'), State('games_rem', 'data'),
              State('inn_rem', 'data')
)
def submit_team(submit, pit_sel_tbl, hit_sel_tbl, gs, inn):
    if submit is None:
        raise PreventUpdate
    reg_stats = {}
    wins = ''
    if submit and (gs == 0) and (inn == 0):
        # convert hitters in hitter objects
        hitters = np.array([])
        for h in hit_sel_tbl['data-frame']:
            # Parsing seasons string from hit_sel_tbl
            seasons = h['Years']
            first = int(seasons[:4])
            second = int(seasons[-4:])
            ##########
            # checking to see that first is less than second
            if (first <= second):
                yr_range = list(range(first, second + 1, 1))
            else:
                yr_range = list(range(second, first + 1, 1))
            player = hitter(h['Name'], yr_range, h['Games'])
            hitters = np.append(hitters, player)
        pitchers = np.array([])
        for p in pit_sel_tbl['data-frame']:
            # Parsing seasons string from hit_sel_tbl
            seasons = p['Years']
            first = int(seasons[:4])
            second = int(seasons[-4:])
            # checking to see that first is less than second
            if (first <= second):
                yr_range = list(range(first, second + 1, 1))
            else:
                yr_range = list(range(second, first + 1, 1))
            player = pitcher(p['Name'], yr_range, p['Innings'])
            pitchers = np.append(pitchers, player)

        # hitters and pitchers should be full and contain enough info to make predictions
        # make predictions
        wins, reg_stats = wins_for_team(hitters, pitchers)
        reg_stats['Team'] = 'my_team'
        reg_stats['Season'] = 2022
        reg_stats['W'] = wins
        reg_stats['WAR_y'] = reg_stats.pop('WAR') * 162
        reg_stats['Def'] = reg_stats['Def'] * 162
        reg_stats['BsR'] = reg_stats['BsR'] * 162
    return reg_stats, f'Wins: {wins}'

if __name__ == '__main__':
    app.run_server(debug=True)