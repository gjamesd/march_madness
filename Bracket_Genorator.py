import streamlit as st

#import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# to solve problems that I am encounterinbg
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt #plotting for visualization purposes of story telling
import numpy as np

#from tqdm import tqdm

#import seaborn as sns
#sns.set_palette(sns.color_palette('hls', 7))

#from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
#import plotly.express as px


#import warnings
#warnings.filterwarnings('ignore')
#from tqdm import tqdm

import pickle
#import time


cols = ['LS_power_conf','HSDR','sweet_16','first_round','HSTO','HS_power_seed',
 'LSFTA','HS_op_To','LS_avg_score','LS_op_FMG3','HSOR','LS_loss','LS_op_FGM',
 'HS_wins','LSPF','HS_low_conf','elite_8','LSDR','HS_op_FGM','LSFGM','HS_mid_conf',
 'LS_op_OR','LSTO','HS_op_FGA','HSBlk','HS_op_FGA3','LS_low_conf','LS_op_DR',
 'LS_op_To','HSFGM3','HSFGM','HS_avg_against','HS_op_FMG3','HSStl','HSAst',
 'LSAst','HSPF','HS_op_OR','HSFGA3','HSFTM','LS_wins','LSFGA3','LS_mid_conf',
 'HS_avg_score', 'LSStl','HS_loss','LS_op_FGA3','LSFTM','LSFGA','HSFGA',
 'LS_avg_against','HS_conf_champ','final_four','championship','LSOR','HSFTA',
 'HS_power_conf','LSFGM3','second_round','LS_power_seed','LSBlk','LS_op_FGA',
 'HS_op_DR']

ticker_data = pd.read_csv('data/odds_.csv',  encoding='latin-1')
#https://www.sportingnews.com/us/ncaa-basketball/news/march-madness-odds-2024-updated-betting-every-team-win-ncaa-tournament/9b72561b3b4ba1707192901d
teams_df = pd.read_csv("data/MTeams.csv")
power_seeds_df = pd.read_csv("data/PowerSeeds.csv")

page_names = pd.read_csv("dictionaries/page_rank_names.csv")


total_dict = pickle.load(open('dictionaries/total_dict.pickle', 'rb'))
#psd = pickle.load(open('dictionaries/psd.pickle', 'rb'))
conf_affil = pickle.load(open('dictionaries/conf_affil.pickle', 'rb'))
conf_champs = pickle.load(open('dictionaries/conf_champs.pickle', 'rb'))
page_rank = pickle.load(open('dictionaries/page_rank.pickle', 'rb'))
tw = pickle.load(open('dictionaries/tw.pickle', 'rb'))



str_test = ''
for i in range(len(ticker_data)):
    str_test += ticker_data['School'][i] +": + " +str(+ticker_data['Odds'][i])+", "

st.set_page_config(

page_title = "March Madness Machine Learning Powered Bracket Genorator"
)


def get_page_names(df):

    page_str = ''

    for i in range(len(df)):
        page_str += df['TeamName'][i]+":  " +str(df['PR'][i])+",    "

    return page_str

ranked_str = get_page_names(page_names)





st.image("viz/mm.jpeg")
st.write("# March Madness Machine Learning Generated Bracket :) ")

ticker_html = f"""
<div class="ticker-wrap">
<div class="ticker">
  <div class="ticker__item">{str_test}</div>
  <div class="ticker__item">{str_test}</div> <!-- Duplicate content for a seamless loop -->
</div>
</div>
<style>
@keyframes ticker {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }} /* Move only half of the total width for looping */
}}
.ticker-wrap {{
  width: 100%;
  overflow: hidden;
  background-color: #333;
  padding: 10px 0;
  color: #FFF;
  font-size: 20px;
  box-sizing: border-box;

}}
.ticker {{
  display: flex;
  width: fit-content;
  animation: ticker 750s linear infinite;
}}
.ticker__item {{
  white-space: nowrap;
  padding-right: 75px; /* Space between items */
}}
</style>
"""

st.write("Draft Kings Betting Odds")
st.markdown(ticker_html, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


ticker_html2 = f"""
<div class="ticker-wrap">
<div class="ticker">
  <div class="ticker__item">{ranked_str}</div>
  <div class="ticker__item">{ranked_str}</div> <!-- Duplicate content for a seamless loop -->
</div>
</div>
<style>
@keyframes ticker {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }} /* Move only half of the total width for looping */
}}
.ticker-wrap {{
  width: 100%;
  overflow: hidden;
  background-color: #333;
  padding: 10px 0;
  color: #FFF;
  font-size: 20px;
  box-sizing: border-box;

}}
.ticker {{
  display: flex;
  width: fit-content;
  animation: ticker 250s linear infinite;
}}
.ticker__item {{
  white-space: nowrap;
  padding-right: 75px; /* Space between items */
}}
</style>
"""

st.write("Page Ranking For Tournament Teams")
st.markdown(ticker_html2, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)



# Example usage
#st.markdown(scrolling_marquee(str_test, scroll_speed=15))

st.write("### To run the algorithm, please select winners of the 'First Four' games and then hit 'Generate Bracket'.")

st.write("Sko Cougs!!!")

#Howard v Wagner
sixteen_1 =  st.selectbox(

    'First 16 seed play in ',

    ('Howard','Wagner'))

#Montana St v Gambling
sixteen_2 =  st.selectbox(

    'Second 16 play in game',

    ('Montana St','Grambling'))


#Virginia v Colorado St
eleven_1 =  st.selectbox(

    'First 10 seed play in game',

    ('Virginia','Colorado St'))

#BSU V Colorado
eleven_2 =  st.selectbox(

    'Last play in game',

    ('Boise St','Colorado'))


year_ = 2024





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

    st.write("inside full bracket")

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

        eleven_1_index = df.loc[(df['Season'] == year_)&(df['PowerSeed '] == 38)].index[0]
        eleven_2_index = df.loc[(df['Season'] == year_)&(df['PowerSeed '] == 39)].index[0]

        sixteen_1_index = df.loc[(df['Season'] == year_)&(df['PowerSeed '] == 61)].index[0]
        sixteen_2_index = df.loc[(df['Season'] == year_)&(df['PowerSeed '] == 62)].index[0]

        df['TeamName'][eleven_1_index] = eleven_1
        df['TeamName'][eleven_2_index] = eleven_2

        df['TeamName'][sixteen_1_index] = sixteen_1
        df['TeamName'][sixteen_2_index] = sixteen_2

        return df

    power_seeds = put_in_first_four(power_seeds_df)


    def create_power_seeds_dict(df):

    #first add team id to TeamNamein power seeds

        teams2 = teams_df[['TeamID', 'TeamName']]

        df2 = pd.merge(left = df, right = teams2, on = 'TeamName')

        df2.to_csv('data/power_test.csv')
        ps1 = pd.DataFrame(df2.groupby(by = "Season"))
        empty = {}
        #print(ps1.columns.tolist())
        # for i in range(len(ps1)):
        #     empty.update({ps1[0][i]: dict(zip(ps1[1][i]['TeamID'], ps1[1][i]['PowerSeed ']))})
        result_dict = {}
        for season, season_group in df2.groupby('Season'):
            season_dict = {}
            for index, row in season_group.iterrows():
                team_name = row['TeamID']
                power_seed = row['PowerSeed ']
                season_dict[team_name] = power_seed
            result_dict[season] = season_dict


        return result_dict

    psd  = create_power_seeds_dict(power_seeds)
    #st.write(psd)



    input_str = sixteen_1+"_"+sixteen_2+"_"+eleven_1+"_"+eleven_2+".csv"

    train = pd.read_csv('train_data/'+input_str)
    t_copy = train.copy()

    y_train = train['target']

    trian = train[cols]
    t_copy = t_copy[cols]

    val = pd.read_csv('val_data/'+input_str)
    y_val = val['target']
    val = val[cols]

    #t_copy = t_copy.drop(columns = ["Unnamed: 0"])
    #train = train.drop(columns = ["Unnamed: 0", "HSTeamID","LSTeamID",])
    #val = val.drop(columns = ["Unnamed: 0", "HSTeamID","LSTeamID",])

    def train_predict(t_df, v_df):

        t_df = t_df.astype(float)
        v_df = v_df.astype(float)

        X_train = t_df#.drop(columns = ['target'])
        #y_train = y_train #t_df['target']
        X_val = v_df#.drop(columns =['target'])
        #y_val = y_val#v_df['target']

        # y_train = t_df['target']
        # y_val = v_df['target']
        #
        # X_train = t_df[cols]
        # X_val = v_df[cols]
        #
        # X_train = X_train.astype(float)
        # X_val = X_val.astype(float)


    #     #splitting into train_test_split

        xclass = xgb.XGBClassifier()

        st.write("Training model")

        parameters = {
            'learning_rate':  np.arange(.01, .1, .01),
            'max_depth': np.arange(15, 75, 10), #
            'subsample': np.arange(.3, .7, .1),
            'colsample_bytree': np.arange(.1, 1, .1),
            'n_estimators' :np.arange(100, 1000, 50),
            #'objective': ['f1'],
            }

        gs = RandomizedSearchCV(xclass, parameters, cv = 4)
        st.write("Cross Validating")
        #gs = GridSearchCV(xclass, parameters, cv = 5)
        gs.fit(X_train, y_train)
        st.write("Model fitted")
        #save trained model in pickle file

        #timestr = time.strftime("%Y%m%d-%H%M")


        st.write("Predicting")
        boost_preds = gs.predict(X_val)
        return boost_preds, y_val, gs


    #bp,y_a, xgc = train_predict(train, val)

    #st.write(classification_report(y_a, bp))
    #st.write(xgc.best_params_)


    # total_dict = pickle.load(open('dictionaries/total_dict.pickle', 'rb'))
    # #psd = pickle.load(open('dictionaries/psd.pickle', 'rb'))
    # conf_affil = pickle.load(open('dictionaries/conf_affil.pickle', 'rb'))
    # conf_champs = pickle.load(open('dictionaries/conf_champs.pickle', 'rb'))
    # page_rank = pickle.load(open('dictionaries/page_rank.pickle', 'rb'))
    # tw = pickle.load(open('dictionaries/tw.pickle', 'rb'))




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
        #st.write('inside build rounds')
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


        #df['Season'] = str(year)



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





        df['ls_conf'] = df['LSTeamID'].apply(lambda x:  conf_affil[2023][x])
        df['hs_conf'] = df['HSTeamID'].apply(lambda x:  conf_affil[2023][x])

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


        df['HS_conf_champ'] = df['HSTeamID'].apply(lambda x: 1 if x in list(conf_champs[year_].keys()) else 0)
        df['LS_conf_champ'] = df['LSTeamID'].apply(lambda x: 1 if x in list(conf_champs[year_].keys()) else 0)


        df = df.drop(columns = ['ls_conf', 'hs_conf'])

        df['HS_page_rank'] = df.apply(lambda x: page_rank[x['Season']][str(int(x['HSTeamID']))], axis = 1)
        df['LS_page_rank'] = df.apply(lambda x: page_rank[x['Season']][str(int(x['LSTeamID']))], axis = 1)

        df['HS_historical_tournament_win%'] = df.apply(lambda x: tw[x['Season']][x['HSTeamID']] if x['HSTeamID'] in tw[x['Season']].keys() else 0 , axis = 1)
        df['LS_historical_tournament_win%'] = df.apply(lambda x: tw[x['Season']][x['LSTeamID']] if x['LSTeamID'] in tw[x['Season']].keys() else 0, axis = 1)

        #df['Season'] = year_
        df = df.drop(columns = ["HSTeamID","LSTeamID",])


        df = df[cols]


        return df


    bp,y_a, xgc = train_predict(train, val)
    feat_imp = xgc.best_estimator_.feature_importances_
    feat_names2 = xgc.feature_names_in_
    #fig = px.bar(x=feat_names2, y=feat_imp,
    #             labels={'y':'importance'}, height=500)

    #fig.write_html("viz/feat_imp.html")
    ### remember how to generatae in line
    #fig.show()
    #st.write(classification_report(y_a, bp))
    st.write(" ")
    #st.write(total_dict['1386'])
    full_bracket(t_copy, xgc, loaded_model = False)
