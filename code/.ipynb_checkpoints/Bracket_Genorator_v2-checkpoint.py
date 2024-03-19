import streamlit as st

from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error, classification_report 
import random 
from random import sample

# to solve problems that I am encounterinbg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt #plotting for visualization purposes of story telling 
import numpy as np

#from tqdm import tqdm

import datetime
import gc
import seaborn as sns
sns.set_palette(sns.color_palette('hls', 7))

from statistics import mean
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import plotly.express as px

from scipy import linalg as la
import networkx as nx
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import pickle 
import time
import json 

#Import things from other file 
# teams that could make the tournament all D1 teams 
# teams_df = pd.read_csv("../data/MTeams.csv")
# #conference tournament games detailed results 
# conf_tour = pd.read_csv("../data/MConferenceTourneyGames.csv")
# #regular season games from 2003 onwards 
# reg_deets = pd.read_csv("../data/MRegularSeasonDetailedResults.csv")
power_seeds_df = pd.read_csv("../data/PowerSeeds_copy.csv")

# json_file_path = "total_dict.json"
# with open(json_file_path, 'r') as json_file:
#     total_dict = json.load(json_file)

st.set_page_config(

page_title = "March Madness Machine Learning Powered Bracket Genorator"
)

st.image("../viz/mm.jpeg")

st.write("# To select the drop first for games, and then you can run the algorithm.")

st.write("Sko Cougs!!!")

sixteen_1 =  st.selectbox(

    'First 16 seed play in ',

    ('TX A&M Commerce','SE Missouri St'))

sixteen_2 =  st.selectbox(

    'Second 16 play in game',

    ('F Dickinson','TX Southern'))

eleven_1 =  st.selectbox(

    'First 11 seed play in game',

    ('Arizona St','Nevada'))

eleven_2 =  st.selectbox(

    'Last play in game',

    ('Mississippi St','Pittsburgh'))


year_ = 2023

# generate =  st.selectbox(

#     'Pre built or run new?',

#     ('Use a pre-built model, about 30 seconds faster - uses pre-built model',
#     "Create my own bracket"))




def build_second_round(result):
    '''
    building a function that takes in the result of the first round and 
    delivers a list of tuples containing the power seeds of the following
    round, we will then iteritevly build the rounds as the 
    tournament progresses 
    
    '''

    second = []
   
    if result[0] ==1:
        if result[31] == 1:
            second.append((first_round[0][0], first_round[31][0]))
         
        else:
            second.append((first_round[0][0], first_round[31][1]))
            st.write("inside first couble else ")
    else:
        if result[31] == 1:
            second.append((first_round[0][1], first_round[31][0]))
        else:
            second.append((first_round[0][1], first_round[31][1]))
    
    if result[1] ==1:
        if result[30] == 1:
            second.append((first_round[1][0], first_round[30][0]))
        else:
            second.append((first_round[1][0], first_round[30][1]))

    else:
        if result[30] == 1:
            second.append((first_round[1][1], first_round[30][0]))
        else:
            second.append((first_round[1][1], first_round[30][1]))
    
    if result[2] ==1:
        if result[29] == 1:
            second.append((first_round[2][0], first_round[29][0]))
        else:
            second.append((first_round[2][0], first_round[29][1]))

    else:
        if result[29] == 1:
            second.append((first_round[2][1], first_round[29][0]))
        else:
            second.append((first_round[2][1], first_round[29][1]))
    
    if result[3] ==1:
        if result[28] == 1:
            second.append((first_round[3][0], first_round[28][0]))
        else:
            second.append((first_round[3][0], first_round[28][1]))

    else:
        if result[28] == 1:
            second.append((first_round[3][1], first_round[28][0]))
        else:
            second.append((first_round[3][1], first_round[28][1]))
########## 
    if result[4] ==1:
        if result[24] == 1:
            second.append((first_round[4][0], first_round[24][0]))
        else:
            second.append((first_round[4][0], first_round[24][1]))

    else:
        if result[24] == 1:
            second.append((first_round[4][1], first_round[24][0]))
        else:
            second.append((first_round[4][1], first_round[24][1]))
    
    if result[5] ==1:
        if result[25] == 1:
            second.append((first_round[5][0], first_round[25][0]))
        else:
            second.append((first_round[5][0], first_round[25][1]))
    else:
        if result[25] == 1:
            second.append((first_round[5][1], first_round[25][0]))
        else:
            second.append((first_round[5][1], first_round[25][1]))
    
    if result[6] ==1:
        if result[26] == 1:
            second.append((first_round[6][0], first_round[26][0]))
        else:
            second.append((first_round[6][0], first_round[26][1]))

    else:
        if result[26] == 1:
            second.append((first_round[6][1], first_round[26][0]))
        else:
            second.append((first_round[6][1], first_round[26][1]))
    
    if result[7] ==1:
        if result[27] == 1:
            second.append((first_round[7][0], first_round[27][0]))
        else:
            second.append((first_round[7][0], first_round[27][1]))

    else:
        if result[27] == 1:
            second.append((first_round[7][1], first_round[27][0]))
        else:
            second.append((first_round[7][1], first_round[27][1]))
            
    if result[11] ==1:
        if result[23] == 1:
            second.append((first_round[11][0], first_round[23][0]))
        else:
            second.append((first_round[11][0], first_round[23][1]))
    else:
        if result[23] == 1:
            second.append((first_round[11][1], first_round[23][0]))
        else:
            second.append((first_round[11][1], first_round[23][1]))
            
    if result[10] ==1:
        if result[22] == 1:
            second.append((first_round[10][0], first_round[22][0]))
        else:
            second.append((first_round[10][0], first_round[22][1]))
    else:
        if result[22] == 1:
            second.append((first_round[10][1], first_round[22][0]))
        else:
            second.append((first_round[10][1], first_round[22][1]))

            
    if result[9] ==1:
        if result[21] == 1:
            second.append((first_round[9][0], first_round[21][0]))
        else:
            second.append((first_round[9][0], first_round[21][1]))
    else:
        if result[21] == 1:
            second.append((first_round[9][1], first_round[21][0]))
        else:
            second.append((first_round[9][1], first_round[21][1]))       
            
            
    if result[8] ==1:
        if result[20] == 1:
            second.append((first_round[8][0], first_round[20][0]))
        else:
            second.append((first_round[8][0], first_round[20][1]))
    else:
        if result[20] == 1:
            second.append((first_round[8][1], first_round[20][0]))
        else:
            second.append((first_round[8][1], first_round[20][1]))
        
            
######

    if result[12] ==1:
        if result[16] == 1:
            second.append((first_round[12][0], first_round[16][0]))
        else:
            second.append((first_round[12][0], first_round[16][1]))
    else:
        if result[16] == 1:
            second.append((first_round[12][1], first_round[16][0]))
        else:
            second.append((first_round[12][1], first_round[16][1]))
            
    
            
            
    if result[13] ==1:
        if result[17] == 1:
            second.append((first_round[13][0], first_round[17][0]))
        else:
            second.append((first_round[13][0], first_round[17][1]))
    else:
        if result[17] == 1:
            second.append((first_round[13][1], first_round[17][0]))
        else:
            second.append((first_round[13][1], first_round[17][1]))
            
            
            
            
    if result[14] ==1:
        if result[18] == 1:
            second.append((first_round[14][0], first_round[18][0]))
        else:
            second.append((first_round[14][0], first_round[18][1]))
    else:
        if result[18] == 1:
            second.append((first_round[14][1], first_round[18][0]))
        else:
            second.append((first_round[14][1], first_round[18][1]))
            
            
                        
            
    if result[15] ==1:
        if result[19] == 1:
            second.append((first_round[15][0], first_round[19][0]))
        else:
            second.append((first_round[15][0], first_round[19][1]))
    else:
        if result[19] == 1:
            second.append((first_round[15][1], first_round[19][0]))
        else:
            second.append((first_round[15][1], first_round[19][1]))         
            
            
            
    
    return second 


def get_names(matchups):
    pyear = power_seeds[power_seeds['Season'] ==year_]
    
    yd = dict(zip(pyear['PowerSeed '], pyear['TeamName']))
    for i in range(len(matchups)):
        st.write(yd[matchups[i][0]]+ " v " +yd[matchups[i][1]])

def get_champ_names(matchups):
    pyear = power_seeds[power_seeds['Season'] ==year_]
    
    yd = dict(zip(pyear['PowerSeed '], pyear['TeamName']))
    for i in range(len(matchups)):
        return (yd[matchups[i][0]]+ " v " +yd[matchups[i][1]])
        
def get_next(n_games, prev_round, result):
    current_round = []
    for i in range(0,n_games):
        if result[i] ==1:
            if result[-i-1] == 1:
            
                current_round.append((prev_round[i][0], prev_round[-i-1][0]))
                #print(first_round[-i][0])
            else:
                current_round.append((prev_round[i][0], prev_round[-i-1][1]))

        else:
            if result[-i-1] == 1:
                current_round.append((prev_round[i][1], prev_round[-i-1][0]))
            else:
                current_round.append((prev_round[i][1], prev_round[-i-1][1]))
                
    current_round = list(pd.Series(current_round).apply(lambda x:x if x[0]<x[1] else (x[1], x[0])))
    return current_round

def full_bracket(df, xgc, loaded_model = False):
    
    if loaded_model == False:
        xgc = xgc
    else:
        xgc = xgb_model_loaded
    
    
    r1 =  build_rounds(df, first_round, 'first')
    r1 = r1[list(train.columns)]
    r1 = r1.drop(columns = 'target')
    
    
    r1_preds = xgc.predict(r1)
    
    
    second_match = build_second_round(r1_preds)
    st.write("Second Round Matchups: ")
    get_names(second_match)
    st.write(" ")
    
    r2 = build_rounds(df, second_match, 'second')
    r2 = r2[train.columns]
    r2 = r2.drop(columns = ['target'])
    
    r2_preds = xgc.predict(r2)
    
    
    third_match = get_next(8, second_match, r2_preds)
    st.write("Sweet 16 Matchups: ")
    get_names(third_match)
    st.write(" ")
    
    r3 = build_rounds(df, third_match, 'third')
    r3 = r3[train.columns]
    r3 = r3.drop(columns = ['target'])
    
    r3_preds = xgc.predict(r3)
    
    
    fourth_match = get_next(4, third_match, r3_preds)
    st.write("Elite 8 Matchups: ") 
    get_names(fourth_match)
    st.write(" ")
    
    r4 = build_rounds(df, fourth_match, 'fourth')
    r4 = r4[train.columns]
    r4 = r4.drop(columns = ['target'])
    
    r4_preds = xgc.predict(r4)
    
    
    fifth_match = get_next(2, fourth_match, r4_preds)
    st.write("Final Four Matchups: ")
    get_names(fifth_match)
    st.write(" ")
    
    r5 = build_rounds(df, fifth_match, 'fifth')
    r5 = r5[train.columns]
    r5 = r5.drop(columns = ['target'])
    
    r5_preds = xgc.predict(r5)
    
    
    sixth_match = get_next(1, fifth_match, r5_preds)
    st.write("Championship Matchups: ")
    get_names(sixth_match)
    champ_names = get_champ_names(sixth_match)
    champ_matchup = get_champ_names(sixth_match)
    #print(type(champ_matchup))
    st.write(" ")
    
    r6 = build_rounds(df, sixth_match, 'sixth')
    #print('champ seeds', sixth_match)
    r6 = r6[train.columns]
    r6 = r6.drop(columns = ['target'])
    
    r6_preds = xgc.predict(r6)
    
    ##Get seeds and names for champion 
    def get_last_name(champ_matchup, sixth_match,r6_preds):
        
        champ_matchup = champ_matchup.split(" v ")
   
    
        cham_ = dict(zip(champ_matchup, sixth_match[0]))
        
        st.write("Champion")
        if r6_preds[0] == 1:
            
            #find min of this dict 
            seed_1 = cham_[champ_matchup[0]]

            seed_2 = cham_[champ_matchup[1]]

            if seed_1 < seed_2:
                st.write(champ_matchup[0])
            else:
                st.write(champ_matchup[1])
                
        else:
            #find max of this dict 
            seed_1 = cham_[champ_matchup[0]]

            seed_2 = cham_[champ_matchup[1]]

            if seed_1 > seed_2:
                st.write(champ_matchup[0])
            else:
                st.write(champ_matchup[1])
    
    st.write(" ")
    st.write("Good luck!") 
    champ = get_last_name(champ_matchup, sixth_match, r6_preds)
         


if st.button('Generate Bracket'):
    st.write('Good Luck! I do not endorse using this to place bets, and am not responsible for any lost earnings, but I do expect a kick back should you win anything :) Happy March')
    
    def put_in_first_four(df):
    
        eleven_1_index = df.loc[(df['Season'] == 2023)&(df['PowerSeed '] == 42)].index[0]
        eleven_2_index = df.loc[(df['Season'] == 2023)&(df['PowerSeed '] == 43)].index[0]

        sixteen_1_index = df.loc[(df['Season'] == 2023)&(df['PowerSeed '] == 64)].index[0]
        sixteen_2_index = df.loc[(df['Season'] == 2023)&(df['PowerSeed '] == 61)].index[0]

        df['TeamName'][eleven_1_index] = eleven_1
        df['TeamName'][eleven_2_index] = eleven_2
        
        df['TeamName'][sixteen_1_index] = sixteen_1
        df['TeamName'][sixteen_2_index] = sixteen_2
        
        return df 
    
    power_seeds = put_in_first_four(power_seeds_df)

    
    
    
    input_str = sixteen_1+"_"+sixteen_2+"_"+eleven_1+"_"+eleven_2+".csv"
    
    train = pd.read_csv('../train_data/'+input_str)
    t_copy = train.copy()
    val = pd.read_csv('../val_data/'+input_str)
    
    train = train.drop(columns = ["HSTeamID","LSTeamID",])
    val = val.drop(columns = ["HSTeamID","LSTeamID",])
    
    def train_predict(t_df, v_df):
        t_df = t_df.astype(float)
        v_df = v_df.astype(float)

        X_train = t_df.drop(columns = ['target'])
        y_train = t_df['target']
        X_val = v_df.drop(columns =['target'])
        y_val = v_df['target']

    #     #splitting into train_test_split

        xclass = xgb.XGBClassifier()
        
        st.write("Training model") 
        
        parameters = {
            'learning_rate':  np.arange(.01, 1, .01), 
            'max_depth': np.arange(5, 50, 3), #
            'subsample': np.arange(.3, .7, .1),
            'colsample_bytree': np.arange(.1, 1, .1),
            'n_estimators' :np.arange(50, 1200, 50), 
            #'objective': ['f1'],  
            }

        gs = RandomizedSearchCV(xclass, parameters, cv = 5)
        #gs = GridSearchCV(xclass, parameters, cv = 5)
        gs.fit(X_train, y_train)

        #save trained model in pickle file
        timestr = time.strftime("%Y%m%d-%H%M")



        boost_preds = gs.predict(X_val)
        return boost_preds, y_val, gs  

   
    bp,y_a, xgc = train_predict(train, val)
    
    #st.write(classification_report(y_a, bp))
    #st.write(xgc.best_params_)


    total_dict = pickle.load(open('../data/total_dict.pickle', 'rb'))
    psd = pickle.load(open('../dictionaries/psd.pickle', 'rb'))
    conf_affil = pickle.load(open('../dictionaries/conf_affil.pickle', 'rb'))
    conf_champs = pickle.load(open('../dictionaries/conf_champs.pickle', 'rb'))
    page_rank = pickle.load(open('../dictionaries/page_rank.pickle', 'rb'))
    tw = pickle.load(open('../dictionaries/tw.pickle', 'rb'))
    
    
    ### building a list of tuples for the first round match ups 
    first_round = [(1, 64), 
                (2, 63), 
                (3, 62), 
                (4, 61), 
                (5, 57), 
                (6, 58), 
                (7, 59),
                (8, 60), 
                (9, 53), 
                (10,54), 
                (11,55), 
                (12,56), 
                (13,49), 
                (14,50), 
                (15,51), 
                (16,52), 
                (17,45), 
                (18,46), 
                (19,47), 
                (20,48),
                (21,41), 
                (22,42), 
                (23,43), 
                (24,44), 
                (25,37),
                (26,38), 
                (27,39), 
                (28,40),
                (29,33),
                (30,34),
                (31,35),
                (32,36),]
    
    
    
    
    def build_rounds(df, match_ups, tourney_round):
        '''
        Building the test data set from the total_dictionary columns to mirror 
        our testing / validation data set 

        '''
        zero_data = np.zeros(shape=(len(match_ups),len(list(df.columns))))
        df  = pd.DataFrame(data = zero_data, columns = list(df.columns))
        #st.write(df.columns.tolist())
        df['Season'] = year_
        #print(len(testd))

        year = psd[year_]

        seed_team = dict([(value, key) for key, value in psd[year_].items()])

        for i in range(len(df)):

            
            df['HS_power_seed'][i] = match_ups[i][0]
            df['LS_power_seed'][i] = match_ups[i][1]

            df['HSTeamID'][i] = seed_team[df['HS_power_seed'][i]]
            df['LSTeamID'][i] = seed_team[df['LS_power_seed'][i]]



        df['HSFGM'] = df.apply(lambda x: total_dict[x['HSTeamID']][year_]['FGM'], axis = 1)
        df['HSFGA'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['FGA'], axis = 1)
        df['HSFGM3'] = df.apply(lambda x: total_dict[x['HSTeamID']][year_]['FGM3'], axis = 1)
        df['HSFGA3'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['FGA3'], axis = 1)
        df['HSFTM'] = df.apply(lambda x: total_dict[x['HSTeamID']][year_]['FTM'], axis = 1)
        df['HSFTA'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['FTA'], axis = 1)
        df['HSOR'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['OR'], axis = 1)
        df['HSDR'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['DR'], axis = 1)
        df['HSAst'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Ast'], axis = 1)
        df['HSTO'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['TO'], axis = 1)
        df['HSStl'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Stl'], axis = 1)
        df['HSBlk'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Blk'], axis = 1)   
        df['HSPF'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['PF'], axis = 1)



        df['LSFGM'] = df.apply(lambda x: total_dict[x['LSTeamID']][year_]['FGM'], axis = 1)
        df['LSFGA'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['FGA'], axis = 1)
        df['LSFGM3'] = df.apply(lambda x: total_dict[x['LSTeamID']][year_]['FGM3'], axis = 1)
        df['LSFGA3'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['FGA3'], axis = 1)
        df['LSFTM'] = df.apply(lambda x: total_dict[x['LSTeamID']][year_]['FTM'], axis = 1)
        df['LSFTA'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['FTA'], axis = 1)
        df['LSOR'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['OR'], axis = 1)
        df['LSDR'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['DR'], axis = 1)
        df['LSAst'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Ast'], axis = 1)
        df['LSTO'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['TO'], axis = 1)
        df['LSStl'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Stl'], axis = 1)
        df['LSBlk'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Blk'], axis = 1)   
        df['LSPF'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['PF'], axis = 1)


        #psd[df['Season'][i]][df['LSTeamID'][i]]
        df['LS_power_seed'] = df.apply(lambda x: psd[year_][x['LSTeamID']], axis = 1)
        df['HS_power_seed'] = df.apply(lambda x: psd[year_][x['HSTeamID']], axis = 1)

        if tourney_round == 'first':
            df['first_round'] = 1
            df['second_round'] = np.zeros(len(df))
            df['sweet_16'] = np.zeros(len(df))
            df['elite_8'] = np.zeros(len(df))
            df['final_four'] = np.zeros(len(df))
            df['championship'] = np.zeros(len(df))

        if tourney_round == 'second':
            df['first_round'] = np.zeros(len(df))
            df['second_round'] = 1
            df['sweet_16'] = np.zeros(len(df))
            df['elite_8'] = np.zeros(len(df))
            df['final_four'] = np.zeros(len(df))
            df['championship'] = np.zeros(len(df))

        if tourney_round == 'third':
            df['first_round'] = np.zeros(len(df))
            df['second_round'] = np.zeros(len(df))
            df['sweet_16'] = 1
            df['elite_8'] = np.zeros(len(df))
            df['final_four'] = np.zeros(len(df))
            df['championship'] = np.zeros(len(df))

        if tourney_round == 'fourth':
            df['first_round'] = np.zeros(len(df))
            df['second_round'] = np.zeros(len(df))
            df['sweet_16'] = np.zeros(len(df))
            df['elite_8'] = 1
            df['final_four'] = np.zeros(len(df))
            df['championship'] = np.zeros(len(df)) 

        if tourney_round == 'fifth':
            df['first_round'] = np.zeros(len(df))
            df['second_round'] = np.zeros(len(df))
            df['sweet_16'] = np.zeros(len(df))
            df['elite_8'] = np.zeros(len(df))
            df['final_four'] = 1
            df['championship'] = np.zeros(len(df)) 

        if tourney_round == 'sixth':
            df['first_round'] = np.zeros(len(df))
            df['second_round'] = np.zeros(len(df))
            df['sweet_16'] = np.zeros(len(df))
            df['elite_8'] = np.zeros(len(df))
            df['final_four'] = 1
            df['championship'] = np.zeros(len(df)) 


        df['HS_avg_score'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Score'], axis = 1)
        df['HS_avg_against'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_score'], axis = 1)
        df['LS_avg_score'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Score'], axis = 1)
        df['LS_avg_against'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_score'], axis = 1)


        df['HS_wins'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['num_wins'], axis = 1)
        df['HS_loss'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['num_loss'], axis = 1)
        df['LS_wins'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['num_wins'], axis = 1)
        df['LS_loss'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['num_loss'], axis = 1)


        df['HS_op_FGM'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_FGM'], axis = 1)
        df['HS_op_FGA'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_FGA'], axis = 1)
        df['HS_op_FMG3'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_FGM3'], axis = 1)
        df['HS_op_FGA3'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_FGA3'], axis = 1)
        df['HS_op_OR'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_OR'], axis = 1)
        df['HS_op_DR'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_DR'], axis = 1)
        df['HS_op_To'] = df.apply(lambda x: total_dict[x['HSTeamID']][x['Season']]['Op_TO'], axis = 1)



        df['LS_op_FGM'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_FGM'], axis = 1)
        df['LS_op_FGA'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_FGA'], axis = 1)
        df['LS_op_FMG3'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_FGM3'], axis = 1)
        df['LS_op_FGA3'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_FGA3'], axis = 1)
        df['LS_op_OR'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_OR'], axis = 1)
        df['LS_op_DR'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_DR'], axis = 1)
        df['LS_op_To'] = df.apply(lambda x: total_dict[x['LSTeamID']][x['Season']]['Op_TO'], axis = 1)





        df['ls_conf'] = df['LSTeamID'].apply(lambda x:  conf_affil[year_][x])
        df['hs_conf'] = df['HSTeamID'].apply(lambda x:  conf_affil[year_][x])

        power_conf = ["acc", 'sec', "big_ten", 
                      "big_twelve", "pac_ten", "pac_twelve"]

        mid_major = ['cusa', 'aac', 'mwc', 'sun_belt', 'ivy', 
                     'mac', 'big_sky', 'meac' ,'southland', 
                     'summit', 'wac', 'wcc',]

        low_major = ['aec', 'a_ten', 'big_south', 'caa', 
                     'nec', 'patriot', 'southern', 'swac', 
                     'mvc', 'a_sun', 'ovc', 'horizon', 
                     'maac', 'swac']    


        df['HS_power_conf']=df['hs_conf'].apply(lambda x: 1 if x in power_conf else 0)
        df['HS_mid_conf'] = df['hs_conf'].apply(lambda x: 1 if x in mid_major else 0)
        df['HS_low_conf'] = df['hs_conf'].apply(lambda x: 1 if x in low_major else 0)

        df['LS_power_conf']=df['ls_conf'].apply(lambda x: 1 if x in power_conf else 0)
        df['LS_mid_conf'] = df['ls_conf'].apply(lambda x: 1 if x in mid_major else 0)
        df['LS_low_conf'] = df['ls_conf'].apply(lambda x: 1 if x in low_major else 0)


        df['HS_conf_champ'] = df['HSTeamID'].apply(lambda x: 1 if x in list(conf_champs[2023].keys()) else 0)
        df['LS_conf_champ'] = df['LSTeamID'].apply(lambda x: 1 if x in list(conf_champs[2023].keys()) else 0)


        df = df.drop(columns = ['ls_conf', 'hs_conf'])

        df['HS_page_rank'] = df.apply(lambda x: page_rank[x['Season']][str(int(x['HSTeamID']))], axis = 1)
        df['LS_page_rank'] = df.apply(lambda x: page_rank[x['Season']][str(int(x['LSTeamID']))], axis = 1)

        df['HS_historical_tournament_win%'] = df.apply(lambda x: tw[x['Season']][x['HSTeamID']] if x['HSTeamID'] in tw[x['Season']].keys() else 0 , axis = 1)
        df['LS_historical_tournament_win%'] = df.apply(lambda x: tw[x['Season']][x['LSTeamID']] if x['LSTeamID'] in tw[x['Season']].keys() else 0, axis = 1)
        
        
        df = df.drop(columns = ["HSTeamID","LSTeamID",])
   
        return df
    
    
    bp,y_a, xgc = train_predict(train, val)
    feat_imp = xgc.best_estimator_.feature_importances_
    feat_names2 = xgc.feature_names_in_
    fig = px.bar(x=feat_names2, y=feat_imp,
                 labels={'y':'importance'}, height=500)

    #fig.write_html("../viz/feat_imp.html")  
    ### remember how to generatae in line 
    #fig.show()
    st.write(classification_report(y_a, bp))
    st.write(" ") 
    #st.write(total_dict['1386'])
    full_bracket(t_copy, xgc, loaded_model = False)
    

        
