
# coding: utf-8

# In[71]:


# Import all the necessary libraries

import numpy as np
import pandas as pd
import itertools

#Ignore the Deprication warnings in the code
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


# Using pandas to load csv file into dataframe

data_1 = pd.read_csv('EPL_0809.csv')
data_2 = pd.read_csv('EPL_0910.csv')
data_3 = pd.read_csv('EPL_1011.csv')
data_4 = pd.read_csv('EPL_1112.csv')
data_5 = pd.read_csv('EPL_1213.csv')
data_6 = pd.read_csv('EPL_1314.csv')
data_7 = pd.read_csv('EPL_1415.csv')
data_8 = pd.read_csv('EPL_1516.csv')
data_9 = pd.read_csv('EPL_1617.csv')
data_10 = pd.read_csv('EPL_1718.csv')


# In[73]:


data_1.keys()


# In[74]:


# Gets all the statistics related to gameplay
# Removed 'HTR' and 'Referee'. 
                      
columns_req = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
       'HY', 'AY', 'HR', 'AR']

playing_stats_1 = data_1[columns_req]                      
playing_stats_2 = data_2[columns_req]
playing_stats_3 = data_3[columns_req]
playing_stats_4 = data_4[columns_req]
playing_stats_5 = data_5[columns_req]
playing_stats_6 = data_6[columns_req]
playing_stats_7 = data_7[columns_req]
playing_stats_8 = data_8[columns_req]
playing_stats_9 = data_9[columns_req]
playing_stats_10 = data_10[columns_req]


# In[75]:


# Goals scored agg arranged by teams and matchweek
def goals_scored(playing_stat):
    # dictionary with team names as keys
    teams = {}
    for team in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[team] = []
    
    # Value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    # Dataframe for goals scored where rows are teams and columns are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsScored[0] = 0
    # Aggregate to get up until that point
    for i in range(2,39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored


# In[76]:


# Total amount of goals scored cumulatively each matchweek
goals_scored(playing_stats_10).head()


# In[77]:


# Goals conceded agg arranged by teams and matchweek
def goals_conceded(playing_stat):
    # Dictionary with team names as keys
    teams = {}
    for team in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[team] = []
    
    # Value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    
    # Dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsConceded[0] = 0
    # Aggregate to get up until that point
    for i in range(2,39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded


# In[78]:


# Total amount of goal conceded cumulatively each matchweek
goals_conceded(playing_stats_10).head()


# In[79]:


# Adding goals scored and goals conceded to dataset
def add_gs_stats(playing_stat):
    GC = goals_conceded(playing_stat)
    GS = goals_scored(playing_stat)
    # Initial variables
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        if ((i + 1)% 10) == 0:
            j += 1
        
    playing_stat['HTGS'] = HTGS
#   writing in loc or iloc:  playing_stat.loc[row_name,HTGS] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat


# In[80]:


# Goals scored and conceded at the end of matchweek, arrangeed by teams and matchweek.
# Initial week is indexed at 0
#add_gs_stats(playing_stats_10).tail()


# In[81]:


# Apply to each dataset
playing_stats_1 = add_gs_stats(playing_stats_1)
playing_stats_2 = add_gs_stats(playing_stats_2)
playing_stats_3 = add_gs_stats(playing_stats_3)
playing_stats_4 = add_gs_stats(playing_stats_4)
playing_stats_5 = add_gs_stats(playing_stats_5)
playing_stats_6 = add_gs_stats(playing_stats_6)
playing_stats_7 = add_gs_stats(playing_stats_7)
playing_stats_8 = add_gs_stats(playing_stats_8)
playing_stats_9 = add_gs_stats(playing_stats_9)
playing_stats_10 = add_gs_stats(playing_stats_10)


# In[82]:


# Get Match points
def points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


# In[83]:


# Add up points
def cumulative_points(matches):
    match_points = matches.applymap(points)
    
    for i in range(2,39):
        match_points[i] = match_points[i] + match_points[i-1]
        
    match_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return match_points

def matches_played(playing_stat):
    # Dictionary with team names as keys
    teams = {}
    for team in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[team] = []

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            
    return pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T


# In[84]:


matches_played(playing_stats_10).head()


# In[85]:


# creating Home Team Points and Away Team Points column to be added to dataset  
def agg_points(playing_stat):
    matches = matches_played(playing_stat)
    cum_pts = cumulative_points(matches)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
            
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat


# In[86]:


agg_points(playing_stats_10).tail()


# In[87]:


# Apply to each dataset
playing_stats_1 = agg_points(playing_stats_1)
playing_stats_2 = agg_points(playing_stats_2)
playing_stats_3 = agg_points(playing_stats_3)
playing_stats_4 = agg_points(playing_stats_4)
playing_stats_5 = agg_points(playing_stats_5)
playing_stats_6 = agg_points(playing_stats_6)
playing_stats_7 = agg_points(playing_stats_7)
playing_stats_8 = agg_points(playing_stats_8)
playing_stats_9 = agg_points(playing_stats_9)
playing_stats_10 = agg_points(playing_stats_10)


# In[88]:


# Check form based on previous match results throughout the matchweek
def current_form(playing_stat,num):
    form = matches_played(playing_stat)
    form_final = form.copy()
    
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final


# In[89]:


current_form(playing_stats_10, 3).head()


# In[90]:


def add_form(playing_stat,num):
    form = current_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]               # get past n results
        h.append(past[num-1])                    # 0 index is most recent
        
        past = form.loc[at][j]               # get past n results.
        a.append(past[num-1])                   # 0 index is most recent
        
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    
    return playing_stat


# In[91]:


add_form(playing_stats_10, 4).tail()


# In[92]:


def add_form_df(playing_stats):
    playing_stats = add_form(playing_stats,1)
    playing_stats = add_form(playing_stats,2)
    playing_stats = add_form(playing_stats,3)
    playing_stats = add_form(playing_stats,4)
    playing_stats = add_form(playing_stats,5)
    return playing_stats    


# In[93]:


#add_form_df(playing_stats_10)


# In[94]:


# Make changes to df
playing_stats_1 = add_form_df(playing_stats_1)
playing_stats_2 = add_form_df(playing_stats_2)
playing_stats_3 = add_form_df(playing_stats_3)
playing_stats_4 = add_form_df(playing_stats_4)
playing_stats_5 = add_form_df(playing_stats_5)
playing_stats_6 = add_form_df(playing_stats_6)
playing_stats_7 = add_form_df(playing_stats_7)
playing_stats_8 = add_form_df(playing_stats_8)
playing_stats_9 = add_form_df(playing_stats_9)
playing_stats_10 = add_form_df(playing_stats_10)


# In[95]:


# Position in the league throughout the season
standings_data = pd.read_csv('EPLstandings.csv')


# In[96]:


# league standing archive
standings_data.head()


# In[97]:


# Remove columns not in use
standings_main = standings_data.drop(['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007'], axis=1)


# In[98]:


standings_main.head(10)


# In[99]:


# Setting 'Team' as index
standings_set = standings_main.set_index(['Team'])


# In[100]:


# NaN indicates team did not participate in the league that year
# Replacing all NaN values with 20
standings_set.fillna(20).head(10)


# In[101]:


standings_set.shape


# In[102]:


# Chech league points
# Add HTLP and ATLP to dataset 
def previousLP(playing_stat, standings, year):
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HomeTeamLP.append(standings_set.loc[ht][year])
        AwayTeamLP.append(standings_set.loc[at][year])
    playing_stat['HomeTeamLP'] = HomeTeamLP
    playing_stat['AwayTeamLP'] = AwayTeamLP
    return playing_stat


# In[103]:


previousLP(playing_stats_10, standings_set, 1).tail()
# NaN indicating team was not present in the league previous year.


# In[104]:


playing_stats_1 = previousLP(playing_stats_1, standings_set, 0)
playing_stats_2 = previousLP(playing_stats_2, standings_set, 1)
playing_stats_3 = previousLP(playing_stats_3, standings_set, 2)
playing_stats_4 = previousLP(playing_stats_4, standings_set, 3)
playing_stats_5 = previousLP(playing_stats_5, standings_set, 4)
playing_stats_6 = previousLP(playing_stats_6, standings_set, 5)
playing_stats_7 = previousLP(playing_stats_7, standings_set, 6)
playing_stats_8 = previousLP(playing_stats_8, standings_set, 7)
playing_stats_9 = previousLP(playing_stats_9, standings_set, 8)
playing_stats_10 = previousLP(playing_stats_10, standings_set, 9)


# In[105]:


# Get matchweek
def m_week(playing_stat):
    week = 1
    match_week = []
    for i in range(380):
        match_week.append(week)
        if ((i + 1)% 10) == 0:
            week = week + 1
    playing_stat['MW'] = match_week
    return playing_stat


# In[106]:


playing_stats_1 = m_week(playing_stats_1)
playing_stats_2 = m_week(playing_stats_2)
playing_stats_3 = m_week(playing_stats_3)
playing_stats_4 = m_week(playing_stats_4)
playing_stats_5 = m_week(playing_stats_5)
playing_stats_6 = m_week(playing_stats_6)
playing_stats_7 = m_week(playing_stats_7)
playing_stats_8 = m_week(playing_stats_8)
playing_stats_9 = m_week(playing_stats_9)
playing_stats_10 = m_week(playing_stats_10)


# In[107]:


playing_stat = pd.concat([playing_stats_1,
                          playing_stats_2,
                          playing_stats_3,
                          playing_stats_4,
                          playing_stats_5,
                          playing_stats_6,
                          playing_stats_7,
                          playing_stats_8,
                          playing_stats_9,
                          playing_stats_10])


# In[108]:


playing_stat.shape


# In[109]:


# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += points(letter)
    return sum


# In[110]:


# Chech results in recents games as column features
playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']

# Points accumulated in recent games as features
playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)


# In[111]:


playing_stat['HTFormPtsStr'].tail()


# In[112]:


# Win streaks if present
def win_streak_3g(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def win_streak_5g(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0


# In[113]:


# Loss streaks if any exists
def l_streak_3g(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def l_streak_5g(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0


# In[114]:


# Add Win/Loss streak for Home and Away games to our dataframe
playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(win_streak_3g)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(win_streak_5g)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(l_streak_3g)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(l_streak_5g)

playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(win_streak_3g)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(win_streak_5g)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(l_streak_3g)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(l_streak_5g)


# In[115]:


playing_stat.tail()


# In[116]:


# Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Points diff.
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']


# In[117]:


def add_form(playing_stat,num):
    form = current_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]               # get past n results
        h.append(past[num-1])                    # 0 index is most recent
        
        past = form.loc[at][j]               # get past n results.
        a.append(past[num-1])                   # 0 index is most recent
        
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    
    return playing_stat


# In[118]:


# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
columns = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float)

for column in columns:
    playing_stat[column] = playing_stat[column] / playing_stat.MW


# In[119]:


playing_stat.tail()


# In[120]:


#Check feature names and order
playing_stat.keys()


# In[121]:


playing_stat[['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
       'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5']].tail()


# In[122]:


# Intuitive removal of feature
# Removing non influential features

# Those features are: 

playing_stat = playing_stat[[
                              'Date', 'HomeTeam', 'AwayTeam', 'FTHG','FTAG', 'FTR','HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
                              'HC', 'AC','HY', 'AY', 'HR', 'AR', 'HTGS', 'ATGS', 'HTGC', 'ATGC','HTGD','ATGD', 'HTP', 'ATP',
                              'HomeTeamLP', 'AwayTeamLP','HTFormPts', 'ATFormPts', 'HTFormPtsStr', 'ATFormPtsStr',
                              'HTWinStreak3','ATWinStreak3','HTWinStreak5','ATWinStreak5','HTLossStreak3','ATLossStreak3',
                              'HTLossStreak5','ATLossStreak5','DiffPts', 'DiffFormPts'
                            ]]


# In[123]:


playing_stat.shape


# In[124]:


playing_stat.loc[:, 'HomeTeamLP': 'AwayTeamLP'].head()


# In[125]:


# replacing NaN with 20's
playing_stat[['HomeTeamLP', 'AwayTeamLP']] = playing_stat[['HomeTeamLP', 'AwayTeamLP']].fillna(20)


# In[126]:


# Last year position difference
playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']


# In[127]:


playing_stat.loc[:, 'DiffPts': 'DiffLP'].head()


# In[128]:


playing_stat.to_csv("final_dataset.csv")

