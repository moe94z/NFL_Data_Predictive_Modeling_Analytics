# %% [markdown]
# # Abstract
# 
# Global Sport Market has increased its total revenues in about 35% in the last 15 years. Total revenue of all 32 National Football League (NFL) teams has risen from about 4 billion U.S. dollars in 2001 to over 15 billion U.S. dollars in 2019, the highest figure to date [1]. The uses of data analysis and statistics in sports helps coaches, players, fans, advertising industry, etc., it helps not only to win games but to improve players performances, prevent injuries, fun for fans, and so on. 
# Legendary football coach Paul “Bear” Bryant famously said, “Offense sells tickets. Defense wins championships”, if this is true, predicting what defense scheme is going to result in stopping the yard gain of the other team, this prediction will give you the championship. 
# 
# In 2002 Oakland Athletics baseball team made people realize the serious effect the use of data analytics could have on the success of a team, the first team in NFL to lead the data analytics technique was the Philadelphia Eagles. Beginning in 2014, Eagles head coach Doug Pederson made it clear that all decisions made by the organization were going to be informed by analytics. Ryan Paganetti started in the Eagles’ analytics department in 2014 [2].
# 
# The NFL team dedicates their time to set defenses strategies according with a list of possible offense plays, based on this concept, prediction of 0-yard gains taking in consideration features like, type of play, yards to gain, offense formation, and others, could give the coaches an insight of how to play the next game.
# 
# 
# 
# # Problem Statement: 
# 
# Since the inception of the National Football League (NFL) in 1920, defensive coordinators have been aggressively seeking any advantage over opposing offenses. The desire for an advantage has led to numerous reports of cheating, none more notable than deflate-gate. The controversial event gave the New England Patriots defense a huge advantage, given they deflated the footballs during the practices leading up to the American Football Conference (AFC) Championship Game against the Indianapolis Colts in 2014 [3]. The New England Patriots were disciplined by the NFL resulting in the case going to the Supreme Court. The Patriots were found guilty and as a result, lost two draft selections in the 2016 NFL draft, fined $1 Million, and Tom Brady, the Quarterback, was suspended for four games for his involvement in the scandal [3]. 
# 
# Despite this scandal, there remains a need for defensive coordinators to have an ethical means of gaining an advantage. With this project, we aim to create a model to aid defensive coordinators in their game planning. The proposed project will give the defensive coordinators an advantage by using data science modeling techniques to provide the defensive coordinators insights into which of their schemes will best defend against their opponent’s offense. This method will not only be more ethical but also will ultimately prove more effective than deflate-gate.  
# 
# This project will be delivered in the milestones set by Bellevue University Department of Science and Technology. We will work towards our problem stated, analyzing the data set, applying Exploratory Data Analysis (EDA) techniques and finally creating, evaluating and selecting a model. Assumptions to get to this model will be explained and detailed in future steps of our project based on the analysis of the data set and features of interest.
# 
# The remainder of this document will serve to document the proposed process for building a model that will solve the problem statement detailed above. The data and the techniques that will be implemented to clean the data will be outlined along with the steps we will use to build the model. We will discuss how we intend to evaluate our model’s efficacy and any risks associated with this model. As this is just a preliminary proposal, all of this is subject to change based on the results we observe through our work. 
# 
# The three models that we are going to create in the case study:
# 
# Logistic Regression Model
# 
# K-Nearest Neighbors (KNN)
# 
# Random Forest
# 
# Optimizing technique: RandomizedSearch

# %% [markdown]
# # Methods
# 
# 
# ## Exploratory Data Analysis

# %% [markdown]
# Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
from scipy import stats
!pip install missingno
import missingno as msno 
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
!pip install pandas_profiling
import pandas_profiling

# %% [markdown]
# # Data Frame Analysis
# 
# The following data set was extratect from: https://www.kaggle.com/c/nfl-big-data-bowl-2021/data
# This data frame contains play-level information for each game.
# Variables description can be found using the link provided
# Following section looks to import data set in Pandas Data frame as well as describe the principal variables that we have defined.

# %%
df = pd.read_csv("plays.csv")

# %%
df.profile_report() 

# %%
df.head()

# %%
print("The shape of the data:\n",
      df.shape, 
      "\nThe first 5 rows are:\n", 
      df.head(), 
      "\nThe last 5 rows are:\n",
      df.tail(), 
      "\nThe column names are:\n",
      df.columns)
      

# %% [markdown]
# ### Questions the research will tackle
# 1. What does play_type_unknown represent? 
# 2. Are most of the 10 yards to go on first down?
# 3. Defenders in the box == more pass rushers? what about sacks? 
# 4. Do shotgun plays == 5 or more DBs defensive formation? Defenders in the box == more pass      rushers? what about sacks?
# 5. 
# 

# %% [markdown]
# ## Part 1: Variable Analysis
# In an effort to begin to get familiar with the variables in our data while also begining to identify relationships between variables we have done some very quick variable analysis using value counts. 

# %%
print(
    "Yards To Go :\n",df["yardsToGo"].value_counts(),
    "Play Type:\n", df["playType"].value_counts(),
    "Offense Formation:\n", df["offenseFormation"].value_counts(),
    "Offensive Personnel:\n", df["personnelO"].value_counts(),
    "Defenders In The Box:\n", df["defendersInTheBox"].value_counts(),
    "Number Of Pass Rushers:\n", df["numberOfPassRushers"].value_counts(),
    "Defensive personnel:\n", df["personnelD"].value_counts(),
    "The Result of Pass Play:\n", df["passResult"].value_counts(),
    "Offense Play Result:\n", df["offensePlayResult"].value_counts(),
    "Play Results:\n", df["playResult"].value_counts(),
     )

# %% [markdown]
# #### Quick Variable analysis: 
# We made a few key discoveries running through the values of each feature(features that matter for the research). One key discovery comes from the playType feature. We can see that all the plays are passing plays and the data does not contain any running plays. I have to review the "play_type_unknown" value and ensure that it is a pass play. The other value, "play_type_sack" is a negative result of the pass play and it is still classified as a pass play. This was not known before and will change the overall problem statement. 
# 
# KEY TAKEAWAYS:
# 1. No Running Plays
# 
# 2. Shotgun most common play
# 
# 3. Most common offensive formation: 1 RB, 1 TE, 3 WR, 2nd most common: 1 RB, 2 TE, 2 WR
# 
# 4. There usually 6-7 defenders in the box but this varies -- 4 to 8 is rather consistent.
# 
# 5. Usually 4 defenders rush the passer
# 
# 6. Most common defensive formation: 4 DL, 2 LB, 5 DB, with 3 DL, 3 LB, 5 DB as the runner up
# - defense formations deviate more than offensive formations.
# 
# 7. Defense usually wins the battle with the offense on most plays, the most common play result ==0
# 
# 8. More completed passes than incomplete. 

# %% [markdown]
# #### Evaluating what "play_type_unknown" represents in the data
# 
# We have a significant number of unknown play types - in the below analysis we aim to identify what that group is comprised of. 

# %%
df_copy = df.copy()
df_copy.loc[(df_copy['playType'] == "play_type_unknown")]
df_copy.loc[(df_copy['playType'] == "play_type_unknown")]


# %% [markdown]
# These plays are just pass plays that are either short or incompete. Confirms our orginal conclusion that the data is made up of only pass plays(success + unsuccessful pass plays). What needs to be done with the play type variable is convert to only one value, pass play. This will be much easier to work with and containing those other 2 values is only going to slow down the model and analysis. For the purpose of this study as such is an unsuccessful pass play. 

# %% [markdown]
# ### Check empty for empty or Nan values

# %%
msno.bar(df)
plt.figure(figsize=(10,10))
# Show the figure

# %%
print("There are: ", df["yardsToGo"].isnull().values.sum(), "empty values in the Yards to go feature",
      "\nThere are: ", df["playType"].isnull().values.sum(), "empty values in the Play Type feature",
      "\nThere are: ", df["offenseFormation"].isnull().values.sum(), "empty values in the offense Formation feature",
      "\nThere are: ", df["personnelO"].isnull().values.sum(), "empty values in the Offensive personnel feature",
      "\nThere are: ", df["defendersInTheBox"].isnull().values.sum(), "empty values in the defenders in the box feature",
      "\nThere are: ", df["numberOfPassRushers"].isnull().values.sum(), "empty values in the number Of Pass Rushers feature",
      "\nThere are: ", df["personnelD"].isnull().values.sum(), "empty values in the Defensive personnel feature",
      "\nThere are: ", df["passResult"].isnull().values.sum(), "empty values in the Result of the play feature",
      "\nThere are: ", df["offensePlayResult"].isnull().values.sum(), "empty values in the Result of the offensive Play feature",
      "\nThere are: ", df["playResult"].isnull().values.sum(), "empty values in the play result feature",
      "\nThere are: ", df["down"].isnull().values.sum(),  "empty values in the downs feature")

# %% [markdown]
# There are missing values in the data frame and we have review the offensive formation, offensive personnel, defenders in the box, pass rushers column, Defensive Personnel column and result of the play column. The result of the play column maybe one that we may need to remove from the data because it may lead to issues with the model. 

# %% [markdown]
# ###  Feature 1: Offensive Personnel (with missing values) analysis and distributions
# 
# After identifying the missing values we take a closer look at what those are made up of.

# %%
df_copy1 = df.copy()
off_per = df_copy["personnelO"]
off_per

# %%
off_per.isnull().values.sum()
# mising values, we already knew from earlier.
df1 = df_copy1[off_per.isna()]
df1

# %%
df1.columns

# %%
plt.figure(figsize=(12,10))
df_copy1.groupby("personnelO").size().plot(kind='bar')
# Label the axes
plt.xlabel("Formation")
plt.ylabel("Frequency")
plt.title("Offensive Formation Distribution Chart")
# Show the figure
plt.show()

# %% [markdown]
# ### Analysis on Offensive personnell
# The issue with the feature is that *there* is no big variation in the data, most of the time the offense is going to line up in the standard formation(1RB:1TE:3WR) with the distant second being the 2 TE formation(1RB:2TE:2WR). This is something we must keep in mind moving forward. 

# %% [markdown]
# ### Feature 2: Yards To Go (with missing values) analysis and distributions

# %%
df_copy1["yardsToGo"].value_counts()


# %% [markdown]
# The distribution between 1-10 are the most common, the other are not as common. The data is more evenly distrubted compared to 

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("yardsToGo").size().plot(kind='bar')
# Label the axes
plt.xlabel("Formation")
plt.ylabel("Frequency")
plt.title("Offensive Formation Distribution Chart")
# Show the figure
plt.show()

# %% [markdown]
# Now reviewing the graph, we can that the most common is 10. This makes the most sense because anytime you get a first down, you are back to having 10 yards to go. 
# 

# %% [markdown]
# ### Are most of the 10 yards to go on first down?
# 
# We will use the play result to figure what down it usual is when there are 10 yards to go. This feature may not even be included in our models.

# %%
down = df_copy1.loc[(df_copy1['down'] == 1)]
down_10 = df_copy1.loc[(df_copy1['down'] == 1)& (df_copy1['yardsToGo'] == 10)]
print(down.down.value_counts())
to_go = down_10.down.value_counts().sum()/ down.down.value_counts().sum() 
print("%.2f%% of all plays with 10 yards to go are first downs." % (to_go *100))
# We plan to do more analysis here, how many yards to go are there on 2nd down or 3rd down usually? 
# There is some additional work to be done here. 

# %% [markdown]
# This is not surprising, if its first down then the offense is 10 yards to go. We can do more analysis here, how many yards to go are there on 2nd down or 3rd down usually? 
# 

# %% [markdown]
# ###  Feature 3 Play type (with missing values) analysis and distributions

# %%
df_copy1['playType']

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("playType").size().plot(kind='bar')
# Label the axes
plt.xlabel("Play Type")
plt.ylabel("Frequency")
plt.title("Offensive Play type Distribution Chart")
# Show the figure
plt.show()

# %% [markdown]
# ### Play Type Analysis
# This feature will be removed from the research, all the plays in the data that we have are passing plays, even if they end in a sack. We determined unknown plays were passing plays earlier and the distribution indicates that there is no variation. Sacks are a result of a failed passing play. 

# %% [markdown]
# ### Feature 4: Offensive formation (with missing values) analysis and distributions

# %%
df_copy1["offenseFormation"]

# %%
df_copy1["offenseFormation"].value_counts()


# %% [markdown]
# Reviewing the value counts we can already see the variation is higher than most of the other features 

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("offenseFormation").size().plot(kind='bar')
# Label the axes
plt.xlabel("PlayBook Formation")
plt.ylabel("Frequency")
plt.title("Offensive Playbook Formation Distribution Chart")
# Show the figure
plt.show()

# %% [markdown]
# ### Offensive Formation feature Analysis
# This is no surprise as we can see in our earlier methods we determined that shotgun is the most common pass play formation and a shotgun formation consist of 1 rb 1 TE and 3 WR, which is the most common value in personnelO feature. These two features are identical and we will be using only one feature to represent the offensive formation. 
# 
# 

# %% [markdown]
# ### Feature 5 Defenders In The Box (with missing values) analysis and distributions

# %%
df_copy1['defendersInTheBox']

# %%
df_copy1["defendersInTheBox"].value_counts()

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("defendersInTheBox").size().plot(kind='bar')
# Label the axes
plt.xlabel("Number of Defenders in the Box")
plt.ylabel("Frequency")
plt.title("Defenders near the Line of Scrimmage")
# Show the figure
plt.show()

# %% [markdown]
# ### Defenders in the box Analysis
# There are usually between 5-7 defenders near the line of scrimmage, with 6 being the most common. What would be interesting is to see if the number of defenders in the box correlates/has a relationship with sacks. 

# %% [markdown]
# ### Defenders in the box == more pass rushers? what about sacks? 
# In the next analysis we are looking to look at what defensive personnel data points have an impact and potential relationship with the result of the play.

# %% [markdown]
# ### Feature 6: number Of Pass Rushers (with missing values) analysis and distributions

# %%
df_copy1['numberOfPassRushers']

# %%
df_copy1['numberOfPassRushers'].value_counts()

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("numberOfPassRushers").size().plot(kind='bar')
# Label the axes
plt.xlabel("Number of Defenders Rushing")
plt.ylabel("Frequency")
plt.title("The Number of Defenders Rushing the Quarterback on Passing Plays")
# Show the figure
plt.show()

# %% [markdown]
# ### Feature Number of Pass Rushers Analysis
# We thought it would be interesting to see if there are any relationships between number of rushers, number of defenders in the box and a sack play result. Most NFL coaches either have 3 or 4 linemen on the defensive line. I would consider a 5 rushers to be a blitz. 

# %% [markdown]
# ### Feature 7: Defensive Formation (with missing values) analysis and distributions

# %%
df_copy1['personnelD']

# %%
df_copy1['personnelD'].value_counts()

# %%
# visual of distribution
plt.figure(figsize=(12,10))
df_copy1.groupby("personnelD").size().plot(kind='bar')
# Label the axes
plt.xlabel("Defensive Formation")
plt.ylabel("Frequency")
plt.title("The Formation of the Defense on Passing Downs")
# Show the figure
plt.show()

# %% [markdown]
# ### Feature Defensive Formation Analysis
# In the bar chart we can conclude that there are in fact more defensive formations than offensive formation (it's not even close). Although the variation is more than the offensive formation we can see that 4DL 2 LB and 5 DB is the most common formation. I would say that this is a cover 3 with an extra corner in the slot. Knowing this data consists of passing plays it would make sense that that the defense would have an extra corner out. A quick glance analysis I would say that defenses play 5 DBs when the offense is in shotgun. 

# %% [markdown]
# ### Question: Do shotgun plays == 5 or more DBs defensive formation? Defenders in the box == more pass rushers? what about sacks? 
# 
# See if there is a relationship via the analysis below. 

# %%
df_copy1

# %%
sack = df_copy1.loc[(df_copy['playType'] == "play_type_sack")]
sack

# %%
sack.numberOfPassRushers.value_counts()

# %%
plt.rcParams['figure.figsize'] = (5, 5)
fig, axes = plt.subplots()

sack.groupby("numberOfPassRushers").size().plot(kind='bar', color="teal")
plt.xlabel("Number of Pass Rushers", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.xticks(rotation = 0, fontsize=10)
plt.title("The Number of Pass Rushers vs. Play Result Ending in a Sack", fontsize=10)

plt.show()

# %%
sack.passResult.value_counts() #Done to make sure there were no mistakes in the data

# %%
rush = df_copy1.loc[(df_copy['numberOfPassRushers'] == 6)]
rush

# %%
#print(rush.passResult.value_counts())
#com = rush.loc[(rush['passResult'] == "C")]
#print("When rushing 6 defenders the QB is %.2f%%", (com/rush.shape[0]*100),
#     "more likily to complete a pass")

# %%
rush1 = df_copy1.loc[(df_copy['numberOfPassRushers'] == 5)]
rush1

# %% [markdown]
# ### Feature 7: Play Result (with missing values) analysis and distributions
# Our playresult variable that present the number of net yards gained in each play

# %%
print("Describe Play Result")
print(df['playResult'].describe())

# %%
plt.rcParams['figure.figsize'] = (15, 10)
fig, axes = plt.subplots()


plt.hist(df['playResult'], bins=20, color="green", alpha=0.8)
plt.xlabel("Net Yards Gained", fontsize=20)
plt.ylabel("count", fontsize=20)
plt.suptitle("Play Result Histogram", fontsize=30)

plt.show()

# %%
plt.boxplot(df['playResult'])

plt.show()
    

# %%
print("Describe Pass Result")
print(df['passResult'].describe())

# %%
df.groupby("passResult").size().plot(kind='bar', color="teal")
plt.xlabel("Pass Result", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(rotation = 0, fontsize=15)
plt.title("Pass Result", fontsize=30)

plt.show()

# %%
explode = (0.1, 0, 0, 0, 0.1)
plt.pie(df.passResult.value_counts(), autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 15}, pctdistance=0.6, explode=explode)
plt.legend(df.passResult.value_counts().index, labels= df.passResult.value_counts().index, title="Completions",loc="center left", bbox_to_anchor=(1.0, 0, 0.5, 1), fontsize=15)
plt.axis('equal') 
plt.suptitle('Pass Result', fontsize = 20)
plt.show()

# %%
ax = sns.catplot(x='passResult', y='playResult', kind = 'box', data = df, height=12, aspect=1)
ax.fig.suptitle('Pass Result and Play result')

# %% [markdown]
# To begin our analysis, we will use an l apply to add an outcome column to the data set. This is the column we will use as our dependent variable. If the play results in any sort of positive yardage we counted that as a success 

# %%
df['outcome'] = df['playResult'].apply(lambda c: 1 if c > 0 else 0)

df.head()

# %%
sns.countplot(x='outcome', data=df, palette = 'hls').set_title('Outcome Counts')

# %%
ax = sns.catplot(x='outcome', y='playResult', kind = 'box', data = df, height=12, aspect=1)
ax.fig.suptitle('Pass Result and Play result')

# %% [markdown]
# As expected Incomplete passes have a negative or 0 play result. We can also see that Complete passes have a Play Result between 1 and aproximately 15 yards

# %% [markdown]
# # Feature Engineering
# Converting values of personnelO and personnelD

# %%
df.columns

# %%
df['personnelO'].value_counts()

# %% [markdown]
# # Removing special teams plays 
# (plays that occur less than 10 times have a defensive player in the formation meaning it is special teams play) 

# %%
# Removing all the values that only show up 10 or less times
value_counts = df['personnelO'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
to_remove = value_counts[value_counts <= 10].index

# Keep rows where the city column is not in to_remove
df = df[~df.personnelO.isin(to_remove)]
df.personnelO.value_counts()

# %% [markdown]
# ### The personnelO values and their new representation (dummy value)
# 1 RB, 1 TE, 3 WR:       a
# 
# 1 RB, 2 TE, 2 WR:       b
# 
# 2 RB, 1 TE, 2 WR:       c
# 
# 1 RB, 3 TE, 1 WR:       d
# 
# 1 RB, 0 TE, 4 WR:       e
# 
# 0 RB, 1 TE, 4 WR:       f 
# 
# 2 RB, 2 TE, 1 WR:       g
# 
# 2 RB, 0 TE, 3 WR:       h
# 
# 6 OL, 1 RB, 1 TE, 2 WR: i
# 
# 2 QB, 1 RB, 1 TE, 2 WR: j
# 
# 0 RB, 2 TE, 3 WR:       k
# 
# 6 OL, 1 RB, 2 TE, 1 WR: l     
# 
# 0 RB, 0 TE, 5 WR:       m          
# 
# 6 OL, 1 RB, 0 TE, 3 WR: n      
# 
# 6 OL, 2 RB, 2 TE, 0 WR: o     
# 
# 3 RB, 1 TE, 1 WR:       p           
# 
# 3 RB, 0 TE, 2 WR:       q           
# 
# 2 RB, 3 TE, 0 WR:       r           
# 
# 6 OL, 2 RB, 1 TE, 1 WR: s   

# %%
df['personnelO'] = df['personnelO'].replace(['1 RB, 1 TE, 3 WR','1 RB, 2 TE, 2 WR','2 RB, 1 TE, 2 WR','1 RB, 3 TE, 1 WR',
                                             '1 RB, 0 TE, 4 WR','0 RB, 1 TE, 4 WR','2 RB, 2 TE, 1 WR','2 RB, 0 TE, 3 WR',
                                             '6 OL, 1 RB, 1 TE, 2 WR','2 QB, 1 RB, 1 TE, 2 WR','0 RB, 2 TE, 3 WR',
                                             '6 OL, 1 RB, 2 TE, 1 WR','0 RB, 0 TE, 5 WR','6 OL, 1 RB, 0 TE, 3 WR',
                                             '6 OL, 2 RB, 2 TE, 0 WR','3 RB, 1 TE, 1 WR','3 RB, 0 TE, 2 WR','2 RB, 3 TE, 0 WR',
                                             '6 OL, 2 RB, 1 TE, 1 WR'],['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                                                                        'o','p','q','r','s'])


# %%
df["personnelO"].value_counts()

# %%
df["personnelD"].value_counts()

# %%
# Removing all the values that only show up 10 or less times
value_d = df['personnelD'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
removed = value_d[value_d <= 10].index

# Keep rows where the city column is not in to_remove
df = df[~df.personnelD.isin(removed)]
df.personnelD.value_counts()

# %% [markdown]
# ### The personnelD values and their new representation (dummy value)
# 
# 4 DL, 2 LB, 5 DB: ab
# 3 DL, 3 LB, 5 DB: ac
# 4 DL, 3 LB, 4 DB: ad
# 2 DL, 4 LB, 5 DB: ae
# 4 DL, 1 LB, 6 DB: af
# 3 DL, 2 LB, 6 DB: ag
# 2 DL, 3 LB, 6 DB: ah
# 3 DL, 4 LB, 4 DB: ai    
# 1 DL, 4 LB, 6 DB: aj
# 1 DL, 5 LB, 5 DB: ak
# 1 DL, 3 LB, 7 DB: al    
# 5 DL, 2 LB, 4 DB: am      
# 3 DL, 1 LB, 7 DB: an    
# 2 DL, 2 LB, 7 DB: ao     
# 0 DL, 4 LB, 7 DB: ap     
# 4 DL, 0 LB, 7 DB: aq     
# 4 DL, 4 LB, 3 DB: ar     
# 0 DL, 5 LB, 6 DB: as      
# 5 DL, 1 LB, 5 DB: at      
# 5 DL, 3 LB, 3 DB: au      

# %%
df['personnelD'] = df['personnelD'].replace(['4 DL, 2 LB, 5 DB','3 DL, 3 LB, 5 DB','4 DL, 3 LB, 4 DB','2 DL, 4 LB, 5 DB',
                                             '4 DL, 1 LB, 6 DB','3 DL, 2 LB, 6 DB','2 DL, 3 LB, 6 DB','3 DL, 4 LB, 4 DB',
                                             '1 DL, 4 LB, 6 DB','1 DL, 5 LB, 5 DB','1 DL, 3 LB, 7 DB','5 DL, 2 LB, 4 DB',
                                             '3 DL, 1 LB, 7 DB','2 DL, 2 LB, 7 DB','0 DL, 4 LB, 7 DB','4 DL, 0 LB, 7 DB',
                                             '4 DL, 4 LB, 3 DB','0 DL, 5 LB, 6 DB','5 DL, 1 LB, 5 DB','5 DL, 3 LB, 3 DB'],
                                            ['ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap',
                                             'aq','ar','as','at','au'])



# %%
df["personnelD"].value_counts()

# %%
df.head()

# %% [markdown]
# ### OHE Categorical Variables
# 
# In order to run our models and complete out analysis we need to use Pandas Get Dummies to One Hot Encode our categorical variables.

# %%
processed = df[['gameId', 'playId', 'quarter', 'down', 'yardsToGo','yardlineNumber', 'defendersInTheBox',
       'numberOfPassRushers', 'preSnapVisitorScore', 'preSnapHomeScore',
       'absoluteYardlineNumber', 'offensePlayResult', 'playResult', 'epa', 'outcome']].copy()

# %%
processed.head()

# %%
cat_columns = ['offenseFormation', 'personnelD', 'personnelO']
cat_dummies = [col for col in processed 
               if "__" in col 
               and col.split("__")[0] in cat_columns]
processed = pd.get_dummies(df, prefix_sep="__",
                              columns=cat_columns)


# %%
processed.head()

# %%
for col in processed.columns: 
    print(col) 

# %% [markdown]
# ##### Dropping variables that we are not are interested in analyzing or are not relevant to the problem

# %%
processed = processed.drop(['playDescription', 'possessionTeam','playType','yardlineSide','typeDropback','gameClock',
                            'penaltyCodes','penaltyJerseyNumbers','offensePlayResult','playResult','isDefensivePI',
                            'playId','gameId'], axis=1)




# %%
processed.head()

# %%
for col in processed.columns: 
    print(col) 

# %% [markdown]
# ### Missing values
# 
# 
# The criteria used to handling missing values is to drop the rows that has them, this criteria was decided after analysis and conclude that plays with missing data is not a play the we can analyze and does not aport information to the data set

# %%
processed = processed.dropna(axis=1)

# %%
processed.reset_index()

# %%
#Checking if we have any NaN value left
processed.isnull().sum().sum()

# %% [markdown]
# ## Correlation
# 
# We are running a correlation to look at the potential relationships between our variables. We are doing this as a prelude to our feature selection. 

# %%
corrdf = processed.corr()

# %%
positivec = corrdf[corrdf["outcome"]>=0.1]

# %%
positivec

# %%
negativec = corrdf[corrdf["outcome"]<= -0.1]

# %%
negativec

# %%
import seaborn as sns

corr = processed.corr(method="pearson")
fig, ax = plt.subplots(figsize=(20,20))
sns.set(font_scale= 1.1)
sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, cmap=sns.diverging_palette(20, 220, n=50), square=True, cbar_kws={"shrink": 0.5})
plt.title('Correlation Heat Map', fontsize = 5)
plt.show()

# %% [markdown]
# ### Our correlation results were largely inconclusive - this may have to do with the high level of cardinality in our data set. 
# ## Some Conclusions from the heatmap
# 1. There seems to be a strong-negative relationship between PersonnelO_a and PersonnelD_ad
# 2. Shotgun and outcome features have a surprisingly strong-negative relationship aswell.
# 3. epa and outcome features have a strong postive relationship, I believe this will be the most significant aspect.
# 4. Offensive formation and offensive Personnels have relationship that do not matter, they represent the same thing. 

# %% [markdown]
# # Modeling

# %% [markdown]
# ### Split into train and test sets

# %%
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
data_model_y = processed.replace({'outcome': {1: 'Yard gained', 0: 'No Yard gained'}})['outcome']

# %%
X_train, X_val, y_train, y_val = train_test_split(processed.loc[:, processed.columns != 'outcome'],
                                                    data_model_y, test_size =0.7, random_state=22, stratify=processed['outcome'])

# %% [markdown]
# ##### Based on this, we will move forward with our dataset as is. As our analysis continues to evolve we may choose to drop some columns from our dataset but this is a very preliminary result. 

# %% [markdown]
# ### Model #1: Logistic Regression
# 
# We are running several models, starting with a logistic regression. 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#!pip install yellowbrick
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC

# %%
model = LogisticRegression()

# %%
X_train.columns

# %%
#The ConfusionMatrix visualizer taxes a model
classes = ['Yard ganied','No Yard gained']
cm = ConfusionMatrix(model, classes=classes, percent=False)

cm.fit(X_train, y_train)

cm.score(X_val, y_val)

for label in cm.ax.texts:
    label.set_size(30)

cm.poof()
plt.show()

# %%
# Precision, Recall, and F1 Score

plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20


visualizer = ClassificationReport(model, classes=classes)

visualizer.score(X_val, y_val) 
g = visualizer.poof()

# %%
y_pred = model.predict(X_val)
print("Accuracy of Logistic Regression Model is:\n", accuracy_score(y_val, y_pred))

# %% [markdown]
# # Using XGBoost library to run randomized search optimizing technique

# %%
import warnings

# %%
y_true = y_val.map({'No Yard gained': 1, 'Yard gained': 0}).astype(int)
y_trainn = y_train.map({'No Yard gained': 1, 'Yard gained': 0}).astype(int)
#y_val
#y_train

# %%
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from os import path
import pickle
import time

n_folds = 30
xgb_fit_dict = {
    'eval_metric': 'auc',
    "early_stopping_rounds": 15,
    "eval_set": [(X_val, y_true)],
    'verbose': 100
}

xgb_param_dict = {
    'n_estimators': np.arange(10, 100, 10),
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'learning_rate': [0.05, 0.1, 0.3],
    'subsample': stats.uniform(loc=0.2, scale=0.8),
    'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'gamma': [0.0, 0.1, 0.2],
    'max_depth': [5, 7, 10],
    'min_child_samples': stats.randint(100, 500), 
    "objective": ["binary:logistic"],
    'alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}


xgb_model = XGBClassifier(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(xgb_model, random_state = 42,  
                            param_distributions = xgb_param_dict, 
                            n_iter = 50, 
                            cv = n_folds, 
                            scoring = 'f1', 
                            verbose = False) 


save1 = 'xgb_saved1.pickle.dat'

if path.exists(save1):
    print('I already have a saved model.')
    
    # Load in saved model
    placed_best = pickle.load(open(save1, 'rb'))
    
    print('Saved Model Parameters')
    print(placed_best.get_xgb_params())
    
    # Compute saved model's MSE for test set
    best_xgb_preds = placed_best.predict(X_val)
    start = time.time()
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("F1 Score on Test Set: %.4f" % f1_score(y_true, best_xgb_preds))
    
else:
    print('Starting to train...')
    
    # Fit via RandomizedSearch
    start = time.time()
    rs_clf.fit(X_train, y_trainn, **xgb_fit_dict)
    print("RandomizedSearch took %.2f seconds to complete." % (time.time()-start))
    
    # Get best params
    xgb_best_params = rs_clf.best_params_
    
    # Train using best params
    placed_best = XGBClassifier(**xgb_best_params, seed = 42)
    start = time.time()
    placed_best.fit(X_train, y_trainn)
    
    # Get MSE
    best_xgb_preds = placed_best.predict(X_val)
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("F1 Score on Test Set: %.4f" % f1_score(y_true, best_xgb_preds))
    
    # Save best xgb model
    pickle.dump(placed_best, open(save1, 'wb'))





# %%
# Compute ROC curve and AUC 
fpr, tpr, thresholds = roc_curve(y_true, 
                                 placed_best.predict(X_val))
calc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
         lw = 2, label='ROC curve (area = %0.2f)' % calc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc="lower right")
plt.show()

# %%
f, ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(placed_best, ax = ax)
plt.show()

# %% [markdown]
# ### K-Nearest Neighbors (KNN)

# %% [markdown]
# K-nearest neighbor classifier is going to be used in this problem, this algorithm will predict the observation to be in one class or the other one depening on the closer class of other obvservation.

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
nearest_n = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)

# %%
nearest_n.predict(X_val)

# %%
#The ConfusionMatrix visualizer taxes a model
classes = ['Yard ganied','No Yard gained']
cm = ConfusionMatrix(nearest_n, classes=classes, percent=False)

cm.score(X_val, y_val)

for label in cm.ax.texts:
    label.set_size(20)

cm.poof()
plt.show()

# %%
# Precision, Recall, and F1 Score

plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20


visualizer = ClassificationReport(nearest_n, classes=classes)

visualizer.score(X_val, y_val) 
g = visualizer.poof()


# %%
y_pred = model.predict(X_val)

# %%
print("Accuracy of K-Nearest Neightbord Model is:\n", accuracy_score(y_val, y_pred))

# %% [markdown]
# ### Random Forest Classifier
# 
# 
# Third option a Random Forest Classifier, this model will be used because is another option for classification problem.

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
random_f = RandomForestClassifier()
model = random_f.fit(X_train, y_train)

# %%
#The ConfusionMatrix visualizer taxes a model
classes = ['Yard ganied','No Yard gained']
cm = ConfusionMatrix(model, classes=classes, percent=False)

cm.score(X_val, y_val)

for label in cm.ax.texts:

    cm.poof()

# %%
# Precision, Recall, and F1 Score

plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20


visualizer = ClassificationReport(model, classes=classes)

visualizer.score(X_val, y_val) 
g = visualizer.poof()

# %%
y_pred = model.predict(X_val)
print("Accuracy of Random Forest Model is:\n", accuracy_score(y_val, y_pred))

# %% [markdown]
# # Conclusion
# 
# Our preliminary conclusions indicate that the best model for the research is the Logistic Regression model. The RandomizedSearch method increased the F1 score significantly making it the most effective model for implementation. One of the biggest takeaways from the RandomizedSearch is the impact that EPA had on the predictions.  
# 
# Logistic Regression model Performance metrics:
# 
# F1: 90.013
# 
# AUC: 0.974
# 
# Run time: .80 seconds
# 
# 
# 

# %% [markdown]
# # References 
# 
# [1] Statista. (2020). Total revenue of all National Football League teams from 2001 to 2019 https://www.statista.com/statistics/193457/total-league-revenue-of-the-nfl-since-2005/
# 
# [2] Aubrey, J. (2020, June 9). The Future of NFL Data Analytics https://www.samford.edu/sports-analytics/fans/2020/The-Future-of-NFL-Data-Analytics)
# 
# [3] Loyola, K. (2020, September 16). The true story behind Tom Brady and the Deflategate scandal. Bolavip US. https://us.bolavip.com/nfl/the-true-story-behind-tom-brady-and-the-deflategate-scandal-20200915-0014.html
# 
# [4] NFL Big Data Bowl 2021. (n.d.). Retrieved December 12, 2020, from https://www.kaggle.com/c/nfl-big-data-bowl-2021/rules
# 
# [5] Brownlee, J. (2020, August 20). SMOTE for Imbalanced Classification with Python. Retrieved December 12, 2020, from https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# 
# [6]Brownlee, J. (2020b, August 21). How to Implement Bayesian Optimization from Scratch in Python. Machine Learning Mastery. https://machinelearningmastery.com/what-is-bayesian-optimization/
# 
# [7] Seif, G. (2021, February 14). A Beginner’s guide to XGBoost - Towards Data Science. Medium. https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
# 
# 

# %%



