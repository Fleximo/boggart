
# coding: utf-8

# In[1]:

#IMPORTS
import re

import pandas as pd
import numpy as np
import scipy
import matplotlib
#matplotlib.use('Qt4Agg')
# get_ipython().magic('matplotlib inline')
from matplotlib import pylab as plt
plt.rcParams['figure.figsize'] = (18, 8)

import statsmodels.api as sm
import seaborn as sns


# In[99]:

#CONSTANTS
USER_FEATURES='Userid,followers,friends,num_tweets,account_created_at,fraction_of_tweets_that_are_mentions_and_not_retweets,fraction_of_tweets_that_are_retweets,fraction_of_tweets_containing_pictures,fraction_of_tweets_containing_youtube_videos,fraction_of_tweets_containing_other_urls,fraction_of_tweets_containing_a_hashtag,number_of_distinct_users_mentioned,number_of_distinct_users_retweeted,average_summed_retweet_count_for_this_user_averaged_over_the_total_number_of_tweets_by_this_user,fraction_of_the_users_tweets_that_were_retweeted,average_summed_retweet_count_per_week,average_summed_retweet_count_for_things_that_have_non_zero_retweets,average_summed_retweet_count_per_week_by_num_followers,average_summed_retweet_count_for_things_that_have_non_zero_retweets_by_num_followers,avg_no_of_friends_of_friends,avg_no_of_followers_of_friends,BE,FR,DE,JP,JM,BR,FI,CO,VE,PR,RU,NL,PT,RS,TT,TR,LV,NZ,TH,PH,RO,CA,PL,AE,GR,CL,BH,EG,ZA,IT,AR,AU,GB,IN,IE,ID,ES,KE,SG,NO,US,KR,KW,SA,MY,MX,SE,is_male,is_female,is_father,is_mother,is_student,leaning_democrat,leaning_republican,is_from_city,is_from_rural,categories_count,gaming,comedy,animals,finance,film_animation,science_tech,travel,people_blogs,howto,entertainment,sports,autos,music,scifi,news_politics,nonprofit,education,movies,shows,num_youtube_videos_shared,min_lag_between_time_uploaded_and_time,mean_lag,median_lag,max_lag,std_dev_of_lag,min_num_of_views_for_videos_shared_by_this_user,mean_num_of_views,median_num_of_views,max_num_of_views,std_dev_of_num_of_views,min_num_of_comments,mean_num_of_comments,median_num_of_comments,max_num_of_comments,std_dev_of_num_of_comments,min_polarization_score_of_users_videos,mean_polarization,median_polarization,max_polarization,std_dev_of_polarization_score,vgaming,vcomedy,vanimals,vfinance,vfilm_animation,vscience_tech,vtravel,vpeople_blogs,vhowto,ventertainment,vsports,vtrailers,vautos,vmusic,vscifi,vnews_politics,vnonprofit,veducation,vmovies,vshows,vis_non_promotional'.split(',')
TWEET_INTERESTS='gaming,comedy,animals,finance,film_animation,science_tech,travel,people_blogs,howto,entertainment,sports,autos,music,scifi,news_politics,nonprofit,education,movies,shows'.split(',')
VIDEO_INTERESTS='vgaming,vcomedy,vanimals,vfinance,vfilm_animation,vscience_tech,vtravel,vpeople_blogs,vhowto,ventertainment,vsports,vautos,vmusic,vscifi,vnews_politics,vnonprofit,veducation,vmovies,vshows'.split(',')#trailers omitted to keep same vectors
save = True
load = True

#plot parameters for enhanced plots
plt.rcParams.update({'axes.titlesize': 22})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'legend.fontsize': 20})
plt.rcParams.update({'legend.fancybox':True})
plt.rcParams.update({'xtick.labelsize':14})
plt.rcParams.update({'ytick.labelsize':14})

# In[100]:

#FUNCTIONS

def is_retweet(text):
    regex = r'(RT|via)((?:\b\W*@\w+)+)'
    match = re.search(regex, text)
    if match:
        return True
    return False

def plot_predictions(y_true, y_pred, title=""):
    x = np.arange(y_true.shape[0])
    plt.plot(x, y_true, "bo")
    plt.plot(x, y_true, "b-", label="y_true")

    plt.plot(x, y_pred, "ro")
    plt.plot(x, y_pred, "r-", label="y_predicted")
    plt.legend()
    plt.title(title)
    if(save):
        plt.savefig('../figures/'+title+'.png', format='png', dpi=300)
    else:
        plt.show()
    

def cosine_dist(u, v):
    return scipy.spatial.distance.cosine(u, v)


def get_content_dist_df(df):
    '''
    @param df: A dataframe containing TWEET_INTERESTS and VIDEO_INTERESTS columns
    for each user
    '''
    content_dist_df = pd.DataFrame(np.zeros(df.shape[0]), columns=['content_dist'])
    
    for i in range(df.shape[0]):
        content_dist_df.ix[i, 'content_dist'] = cosine_dist(df.ix[i,TWEET_INTERESTS].values, 
                                                            df.ix[i, VIDEO_INTERESTS].values)
    
    return content_dist_df


def add_content_dist(data, ufeats):
    '''
    This methods calculates a distance measure between a user's interests and 
    his/her video interests. 
    @param data: A dataframe containing a column 'userid'
    @return: adds "content_dist"  column to the given dataframe 'data'.
    '''
    
    newdat = data.merge(ufeats[['Userid'] + TWEET_INTERESTS + VIDEO_INTERESTS],
                   on='Userid', how='left')
    content_dist_df = get_content_dist_df(newdat[TWEET_INTERESTS+VIDEO_INTERESTS])
    
    return pd.tools.merge.concat([data, content_dist_df], axis=1)
    

def get_Au2v(data, ufeats):
    '''
    This method builds a measure of called Au2v which means the number of times a user's
    tweet is re-tweeted by 
    @param data: a data frame containing "userid" column
    @param ufeats: user features dataframe 
    '''
#     #read followership data
#     followingdat = pd.read_csv('../data/user-followership.txt', delimiter='\t')
#     
#     #read user_tweet_video.txt
#     utweetdat = pd.read_csv('../data/user_tweet_video.txt', delimiter='\t')
#     utweetdat.columns = ['Userid', 'Tweetid', 'Videoid']
#     utweetdat = utweetdat.merge(data, on='Userid', how='right')
# 
#     #read tweet_context.txt
#     tweets = pd.read_csv('../data/tweet-content-final.txt', delimiter='\t', encoding="ISO-8859-1")
#     tweets.columns = ["Tweetid", 'Text']
#     #add is_retweet column to this dataframe
#     tweets['Is_retweet'] = tweets['Text'].apply(lambda tweet: is_retweet(tweet))
    
    newdat = data.merge(ufeats[['Userid', 'fraction_of_the_users_tweets_that_were_retweeted']],
                   on='Userid', how='left')
        
    return newdat['fraction_of_the_users_tweets_that_were_retweeted']
    
    

def get_data():
    '''
    @return: X, y ; where X is a dataframe (Independent Variables:
    score, dscore, content_dist), and
    y is a series, dependent variable, which is the number of times a user's tweet 
    is retweeted by his/her followers. 
    '''
    #read user_all_features.txt for number of retweets for each user
    ufeats = pd.read_csv('../data/user_all_features.txt', delimiter='\t', 
                         header=None)
    ufeats.columns = USER_FEATURES
    
    if(load):
        data = pd.read_csv('../data/data.csv', index_col='index')
    else:
        #read score files
        scoredat = pd.read_csv('../data/userid_score.txt', delimiter='\t')
        dscoredat = pd.read_csv('../data/userid_dscore.txt', delimiter='\t')
        
        #merge both on common userids 
        data = scoredat.merge(dscoredat, on='Twitter_UID', how='inner')
        data = data.rename(columns={"Twitter_UID":"Userid", "D_Score":'Dscore'})
        
        #build content-distance column for each userid
        data = add_content_dist(data, ufeats)
        
        #handle nans
        data = data.dropna()
        data.index = np.arange(data.shape[0]) #re-index
        
        #save data
        data.to_csv('../data/data.csv', index_label='index')
    
    #get dependent variable
    y = get_Au2v(data, ufeats)
    
    #set independent variables
    X = data.drop('Userid', axis=1)

    return X, y 


# ## Regression Analysis

# In[101]:

#get data
X, y = get_data()


# In[102]:

y.name = 'Au2v'


# Shape i.e. size of the data: 

# In[103]:

X.shape


# X contains three columns, as : 

# In[104]:

X.head()


# ####Linear regression on data. 

# In[105]:

#Fit a linear model
mod = sm.OLS(endog=y, exog=X).fit()


# In[106]:

#print summary
mod.summary()


# The above regression summary shows that coeffiecients of all predictors are significant (P>|t| is greater less than 0.05). Therefore, Score, Dscore and content_dist can be considered as good predictors. 

# ####Regression plots for each independent variable:

# In[107]:

#plot regression plots for each independent variable
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
if(save):
    plt.savefig('../figures/PartialRegressionPlot.png', format='png', dpi=300)


# The above regression plots for each predictor shows the effect of outliers on the estimated regression coefficient. Regression line is pulled out of its optimal tracjectory due these outliers. 

# ####Regression Plots for individual Predictors

# ##### 1. Regression  plots for "Score" : 

# In[108]:

#all regression plots for "Score"
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(mod, "Score", fig=fig)
if(save):
    plt.savefig('../figures/RegressionPlots_Score.png', format='png', dpi=300)


# ##### 2. Regression  plots for "D_Score" : 

# In[109]:

#all regression plots for DScore
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(mod, "Dscore", fig=fig)
if(save):
    plt.savefig('../figures/RegressionPlots_Dscore.png', format='png', dpi=300)


# ##### 3. Regression plots for cosine distance i.e. "content_dist" : 

# In[110]:

#all regression plots for cosine content_distance
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(mod, "content_dist", fig=fig)
if(save):
    plt.savefig('../figures/RegressionPlots_content_dist.png', format='png', dpi=300)


# #### Plot for each Predictors,  depicting  fitted values with confidence interval of predictions: 

# The following plots show that fitted values of Au2v and its prediction confidence for each independent variable. These plots show that fitted values are quite close the true values of Au2v except for the outlier points. This suggests that removal of outliers would yield a better estimate. Removal of outliers is explored in later sections. 

# ##### 1. Fitted values of Au2v Vs Score

# In[111]:

#fitted values with prediction confidence interval, for Score
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod, "Score", ax=ax)
if(save):
    plt.savefig('../figures/FittedPlot_Score.png', format='png', dpi=300)


# ##### 2. Fitted values of Au2v Vs Dscore

# In[112]:

#fitted values with prediction confidence interval, for Dscore
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod, "Dscore", ax=ax)
if(save):
    plt.savefig('../figures/fittedPlot_Dscore.png', format='png', dpi=300)


# ##### 3. Fitted values of Au2v Vs content_dist

# In[113]:

#fitted values with prediction confidence interval, for content_distance
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod, "content_dist", ax=ax)
if(save):
    plt.savefig('../figures/fittedPlot_content_dist.png', format='png', dpi=300)


# From all above plots it is clear that scale is skewed due to presence of outliers. We should handle these outliers manually or use automated robust linear regression methods for handling outliers. 
# In the next section, we experiment on the following: 
# 
# 1. Robust linear regression for handling outliers
# 2. Building and including 3rd independent variable "content_distance" into analysis.
# 

# ### Linear Regression (after removing outliers)

# A rough estimate of detecting outliers can be done by using the quantile distributions of each independent variable, as follows:

# In[114]:

#distributions for detecting outlier thresholds
X.describe()


# From the above table, as a guess we could take values of score and dscore only upto 10 and 5  respectively. 

# In[115]:

X1 = X[(X['Score'] < 10) & (X['Dscore'] < 5)]
y1 = y[(X['Score'] < 10) & (X['Dscore'] < 5)]

X1.shape


# ####Fit ordinary least squares regression model on data obtained after removing outlier data points:

# In[116]:

#Fit a robust linear model
mod = sm.OLS(endog=y1, exog=X1).fit()
mod.summary()


# The above results show considerable improvement w.r.t. R-squared. Also, Durbin-watson statistic close to 2 confirms normality assumption of residuals. Also, all predictor's coefficients are significant as P>|t| is less than 0.05 and hence are good predictors. 

# ##### Test normality of residuals using qqplot: 

# In[117]:

#see qqplot for normality test
from statsmodels.graphics.api import qqplot

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(mod.resid, line='q', ax=ax, fit=True)
if(save):
    plt.savefig('../figures/qqplot.png', format='png', dpi=300)


# Above qqplot suggests deviation from normality at higher quantiles. 

# ####Regression plots for each independent variable:

# In[118]:

#plot regression plots for each independent variable
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
if(save):
    plt.savefig('../figures/PartialRegressionPlot(Outliers Removed).png', format='png', dpi=300)


# From the above plots it is clear that after removing outliers we get a better fit of regression line on each independent variable. That is, path of regression line is now more aligned to the the optimal path. 

# ###Predictive Modeling 

# In[156]:

#Model
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

rnd = np.random.RandomState(seed=13)
shuffle = False
nfolds = 10

def do_cv(X, y):
    kfcv = KFold(n=X.shape[0], n_folds=nfolds, shuffle=shuffle, random_state=rnd)
    reg_lr = linear_model.LinearRegression(fit_intercept=False, 
                                           normalize=False, 
                                           copy_X=True, 
                                           n_jobs=-1)
    cv_preds = np.zeros(y.shape[0])
    for k, (cv_train, cv_test) in enumerate(kfcv):
        reg_lr.fit(X.iloc[cv_train,:], y.iloc[cv_train])
        ypred = reg_lr.predict(X.iloc[cv_test,:])
        cv_preds[cv_test] = ypred
        rmse_fold = round(np.sqrt(mean_squared_error(y.iloc[cv_test].values, ypred)), 4)
        print("=============================================")
        print("CV Fold "+ str(k) +" RMSE= "+ str(rmse_fold))
    print("#######Total RMSE %s"%(round(np.sqrt(mean_squared_error(y.values, cv_preds)), 4)))
    return cv_preds


# In[120]:

#model using linear regression on score, dscore and content-dist features 
cv_preds = do_cv(X, y)

#plot predictions
plot_predictions(y[:200], cv_preds[:200], title='Linear Regression Performance')


# Above, we did a 10 fold cross-validation for predicting Au2v dependent variable from Score, Dscore and content-dist independent variables. The results of the predictive modeling using linear regression above show that we achieve a root mean squared error of 0.2728 (across all folds), which means that our prediction varies by 0.2728 amount from the true value of A2uv. 

# In[121]:

#model using linear regression using score, dscore and content-dist features on outlier filtered data
cv_preds_or = do_cv(X1, y1)

#plot predictions
plot_predictions(y1[:200], cv_preds_or[:200], title="Linear Regression Performance (Outliers removed)")


# Above, we did a 10 fold cross-validation for predicting Au2v dependent variable from Score, Dscore and content-dist independent variables. This time we did predictive modeling after removing outliers from the data. The results of the predictive modeling using linear regression above show that we achieve a root mean squared error of 0.20 (across all folds), which means that our prediction varies by 0.20 amount from the true value of A2uv. This shows a considerable improvement in prediction error than modeling with original data (with outliers). 

# ###Classification : Predicting popularity of a user 

# If Au2v crosses a threshold, say 0.3, i.e. if more than 30% tweets of user 'u' are retweeted by others users then user 'u' can be considered as a popular user. This can also be interpreted as "if a user u's tweets will be popular or not given user u's DScore, score and content-distance(measure of his tweet's similiarity with his video interests) 

# In[122]:

#plot both predictions to compare 
N = 150 #number of points to plot 
title = "Linear Regression Performance(with & without outliers)"
x = np.arange(N)
plt.plot(x, y[:N], "bo")
plt.plot(x, y[:N], "b-", label="y_true")

plt.plot(x, cv_preds[:N], "r^")
plt.plot(x, cv_preds[:N], "r:", label="y_predicted(with outliers)")

plt.plot(x, cv_preds_or[:N], "gs")
plt.plot(x, cv_preds_or[:N], "g--", label="y_predicted(without outliers)")

plt.legend()
plt.title(title)
if(save):
    plt.savefig('../figures/'+title+'.png', format='png', dpi=300)


#create binary class target, popular or not popular
thres = 0.25
ybinary_preds = pd.Series(cv_preds)
ybinary_preds[ybinary_preds >= thres] = 1
ybinary_preds[ybinary_preds < thres] = 0

ybinary_true = y1.copy(True)
ybinary_true[y1 >= thres] = 1
ybinary_true[y1 < thres] = 0

#random prediction
yrand = np.random.randint(0, 2, size=y1.shape[0])


# In[123]:

#calculate precision and recall metrics on our predictions
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
precision = precision_score(ybinary_true, ybinary_preds)
recall = recall_score(ybinary_true, ybinary_preds)
prand = precision_score(ybinary_true, yrand)
rrand = recall_score(ybinary_true, yrand)

print("Model  ## Precision: %f, Recall: %f"%(precision, recall))
print("Random ## Precision: %f, Recall: %f"%(prand, rrand))


# The above precision and recall of our model shows that precision is much better than random predictions, i.e. our model does learn from the predictors. However, recall is low which suggests the need of additional features/predictors to improve model. 

# #### Using Naive Bayes Classifier

# In[157]:

from sklearn.naive_bayes import GaussianNB

#create binary classification target
yb = y1.copy(True)
yb[y1 >= thres] = 1
yb[y1 < thres] = 0

#do 10 fold cross validation with naive bayes classifier
def do_cv_clf(X, y, clf):
    kfcv = KFold(n=X.shape[0], n_folds=nfolds, shuffle=shuffle, random_state=rnd)    
    cv_preds = np.zeros(y.shape[0])
    for k, (cv_train, cv_test) in enumerate(kfcv):
        clf.fit(X.iloc[cv_train,:], y.iloc[cv_train])
        ypred = clf.predict(X.iloc[cv_test,:])
        cv_preds[cv_test] = ypred
        precisioncv = precision_score(y.iloc[cv_test].values, ypred)
        recallcv = recall_score(y.iloc[cv_test].values, ypred)
        print("=============================================")
        print("CV Fold %d: precision= %f, recall= %f"%(k, precisioncv, recallcv))
        
    print("\n###### Overall Precision %s"%(precision_score(y, cv_preds)))
    print("###### Overall Recall %s"%(recall_score(y, cv_preds)))
    return cv_preds


# In[125]:

#create naive bayes classifier and do 10 fold crossvalidation
gnb = GaussianNB()
cv_preds = do_cv_clf(X1, yb, gnb)


# #### Using Random Forest Classifier

# In[134]:

#do random forest classification
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, 
                            n_jobs=-1,
                            max_features=None,
                            min_samples_split=2
                           )
cv_preds = do_cv_clf(X1, yb, rf)


# From above two classifiers, we see that for :
# 1. Naive Bayes classifier, precision is 0.58 (better than linear regression) and recall is 0.34(worse than linear regression)
# 2. Random Forest Classifier, precision is 0.53(better than baseline but not better than linear regression and naive bayes). While, recall is 0.47 (better than both naive bayes, linear regression and slightly closer to baseline). 

# ### Experimenting with log transformations

# In[140]:

X1.describe()


# In[172]:

#do log transformation on indpendent variables
X2 = pd.DataFrame()
X2['Score'] = np.log1p(X1['Score'].values)
X2['Dscore'] = X1['Dscore'].values
X2['content_dist'] = np.log1p(X1['content_dist'].values)


# In[173]:

X2.describe()


# #####Training linear regression model on log(1+x) transformed data

# In[243]:

#model using linear regression using score, dscore and content-dist features on outlier filtered data
cv_preds_lr = do_cv(X2, y1)

cv_preds_lr_binary = pd.Series(cv_preds_lr)
cv_preds_lr_binary[cv_preds_lr >= thres] = 1
cv_preds_lr_binary[cv_preds_lr < thres] = 0

precision = precision_score(ybinary_true, cv_preds_lr_binary)
recall = recall_score(ybinary_true, cv_preds_lr_binary)
prand = precision_score(ybinary_true, yrand)
rrand = recall_score(ybinary_true, yrand)

print("Model  ## Precision: %f, Recall: %f"%(precision, recall))
print("Random ## Precision: %f, Recall: %f"%(prand, rrand))


# The above results show that log(1+x) transformation improves the model's recall. Both Precision and recall are better than the baseline. 
# 

# #####Training Random Forest Classifier on log(1+x) transformed data

# In[238]:

#train random forest classifier on transformed data
rf = RandomForestClassifier(n_estimators=200, 
                            n_jobs=-1,
                            max_features=None,
                            min_samples_split=2
                           )
cv_preds_rf = do_cv_clf(X2, yb, rf)


# #### Training Naive Bayes classifier on log(1+x) transformed data

# In[240]:

gnb = GaussianNB()
cv_preds_nb = do_cv_clf(X2, yb, gnb)


# The above results on naive bayes classifier show that log(1+x) transformation has improved recall from 0.34 to 0.40, precision is slightly decreased. However, recall still remains below the baseline. 

# ### Comparing precision and recall of various models

# In[258]:

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precisionlr, recalllr, _ = precision_recall_curve(ybinary_true, cv_preds_lr_binary)
auc_lr = average_precision_score(ybinary_true, cv_preds_lr_binary)  

precisionrf, recallrf, _ = precision_recall_curve(ybinary_true, cv_preds_rf)
auc_rf = average_precision_score(ybinary_true, cv_preds_rf)  

precisionnb, recallnb, _ = precision_recall_curve(ybinary_true, cv_preds_nb)
auc_nb = average_precision_score(ybinary_true, cv_preds_nb)  

plt.plot(precisionlr, recalllr, '-b', label='Linear Regression (AUC:%0.3f)'%auc_lr)
plt.plot(precisionnb, recallnb, '-r', label='Naive Bayes (AUC:%0.3f)'%auc_nb)
plt.plot(precisionrf, recallrf, '-g', label='Random Forest (AUC:%0.3f)'%auc_rf)

plt.title("Precision Recall Curves")
plt.legend()
plt.savefig('../figures/precision_recall_curves.png', format='png', dpi=1200)
# plt.show()


# The above precision-recall curves shows a comparision of performances for Linear Regression, Naive Bayes and Random Forest methods. Higher Area under this curve (AUC) represents better performance, with AUC=1 being the highest achievable performance by a classifier. Above plot shows that Linear Regression has the highest AUC of 0.625. 

# ### Linear Regression and Plots with log(1+x) transformations

# In[174]:

#Fit a robust linear model
y2 = y1.copy(True); y2.index = np.arange(y2.shape[0])
mod = sm.OLS(endog=y2, exog=X2).fit()
mod.summary()


# #### High Resolution plots

# In[175]:

#plot regression plots for each independent variable
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
if(save):
    plt.savefig('../figures/PartialRegressionPlot(Outliers Removed, log1p).png', format='png', dpi=300)


# In[226]:

#fitted values with prediction confidence interval, for Score
indx = np.random.randint(0, y2.shape[0], 100)
mod1 = sm.OLS(endog=y2[indx], exog=X2.ix[indx,:]).fit()


# In[231]:

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod1, "Score", ax=ax)
fig.savefig('../figures/log1p(Score)_fitted.png', format='png', dpi=1200)


# In[232]:

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod1, "Dscore", ax=ax)
fig.savefig('../figures/Dscore_fitted.png', format='png', dpi=1200)


# In[233]:

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(mod1, "content_dist", ax=ax)
fig.savefig('../figures/log1p(content_dist)_fitted.png', format='png', dpi=1200)

