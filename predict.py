import numpy as np  # Linear algebra
import pandas as pd  # Data processing
import matplotlib.pyplot as plt
from scipy.stats import norm

col_types = {'overall': np.int32, 'age': np.int32}

# read only the columns that we will use (name and photo are loaded only to visualize the results at the end)
df = pd.read_csv("players_20.csv", usecols=['short_name', 'value_eur', 'overall', 'age', 'attacking_finishing'],
                 dtype=col_types)

# nprint(df.isnull().sum())

# Nobody can have a value lower or equal than zero, so those values are bad entries and we need to remove them
df = df.loc[df.value_eur > 0]


def between_1_and_99(s):
    try:
        n = int(s)
        return (1 <= n and n <= 99)
    except ValueError:
        return False


# remove not valid entries for Finishing
df = df.loc[df['attacking_finishing'].apply(lambda x: between_1_and_99(x))]

# now we can define Finishing as integers
df['attacking_finishing'] = df['attacking_finishing'].astype('int')

############### DESCRIBE DATA #####################
# print(df.overall.describe())
# print(df.nlargest(5, columns='overall'))
#
# # plot the histogram
# plt.hist(df.overall, bins=16, normed=True, alpha=0.6, color='g')
# plt.title("#Players per Overall")
# plt.xlabel("Overall")
# plt.ylabel("Count")
#
# overall_mean = df.overall.mean()
# overall_std = df.overall.std()
#
# # Plot the probability density function for norm
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, overall_mean, overall_std)
# plt.plot(x, p, 'k', linewidth=2, color='r')
# title = "#Players per Overall, Fit results: mean = %.2f,  std = %.2f" % (overall_mean, overall_std)
# plt.title(title)
#
# plt.show()


############### ML #####################
# Algorithm fall in the supervised learning category (because we can train a model using observations where the
# expected result is well known and we can “teach” it to our algorithm).

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.20, random_state=99)

xtrain = train[['value_eur']]
ytrain = train[['overall']]

xtest = test[['value_eur']]
ytest = test[['overall']]

# Create linear regression object
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# Make predictions using the testing set
y_pred = regr.predict(xtest)

plt.scatter(xtest, ytest, color='black')
plt.plot(xtest, y_pred, color='blue', linewidth=3)
plt.xlabel("Value")
plt.ylabel("Overall")
plt.show()

from sklearn.metrics import mean_squared_error, r2_score  # common metris to evaluate regression models

print('Linear Regression Model')
# The mean squared error. LOW
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred))
# Explained variance score: 1 is perfect prediction. HIGH
print('Variance score: %.2f' % r2_score(ytest, y_pred))
print()

# In this case, we’ll use polynomials as our basis functions and the Ridge model to solve the regression problem.
# Using Ridge regression (instead of the standard linear regression model) can help minimize overfitting, which is a
# possible colateral issue of adding basis functions to our regression model.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pol = make_pipeline(PolynomialFeatures(6), linear_model.Ridge())
pol.fit(xtrain, ytrain)

y_pol = pol.predict(xtest)
plt.scatter(xtest, ytest, color='black')
plt.scatter(xtest, y_pol, color='blue')
plt.xlabel("Value")
plt.ylabel("Overall")
plt.show()
print('Ridge Regression Model')

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pol))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(ytest, y_pol))
print()

# The last model that we will try is called Support Vector Regression which can be seen as an extension of the
# Support Vector Machine method used for classification problems. In particular, we will
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', gamma=1e-3, C=100, epsilon=0.1)
svr_rbf.fit(xtrain, ytrain.values.ravel())

y_rbf = svr_rbf.predict(xtest)
plt.scatter(xtest, ytest, color='black')
plt.scatter(xtest, y_rbf, color='blue')
plt.xlabel("Value")
plt.ylabel("Overall")
plt.show()
print('Support Vector Regression Model')
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_rbf))
print('Variance score: %.2f' % r2_score(ytest, y_rbf))
print()

# Improve
xtrain = train[['value_eur', 'age', 'attacking_finishing']]
xtest = test[['value_eur', 'age', 'attacking_finishing']]

regr_more_features = linear_model.LinearRegression()
regr_more_features.fit(xtrain, ytrain)
y_pred_more_features = regr_more_features.predict(xtest)
print('New Linear Regression Model')
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred_more_features))
print('Variance score: %.2f' % r2_score(ytest, y_pred_more_features))
print()

pol_more_features = make_pipeline(PolynomialFeatures(4), linear_model.Ridge())
pol_more_features.fit(xtrain, ytrain)
y_pol_more_features = pol_more_features.predict(xtest)
print('New Ridge Regression Model')
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pol_more_features))
print('Variance score: %.2f' % r2_score(ytest, y_pol_more_features))
print()

svr_rbf_more_features = SVR(kernel='rbf', gamma=1e-3, C=100, epsilon=0.1)
svr_rbf_more_features.fit(xtrain, ytrain.values.ravel())
y_rbf_more_features = svr_rbf_more_features.predict(xtest)
print('New Support Vector Regression Model')
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_rbf_more_features))
print('Variance score: %.2f' % r2_score(ytest, y_rbf_more_features))
print()

# Use New SVR model as the best canidate model
pd.options.mode.chained_assignment = None
test['Overall_Prediction_RBF'] = y_rbf_more_features
test['Error_Percentage'] = np.abs((test.overall - y_rbf_more_features) / test.overall * 100)

plt.hist(test.Error_Percentage, bins=16)
plt.title("#Players per %error")
plt.xlabel("%error")
plt.ylabel("Count")
plt.show()

print("Use trained model to predict for all players")
y_rbf_all = svr_rbf_more_features.predict(df[['value_eur', 'age', 'attacking_finishing']])
print("Mean squared error: %.2f" % mean_squared_error(df[['overall']], y_rbf_all))
print('Variance score: %.2f' % r2_score(df[['overall']], y_rbf_all))

import ipywidgets as widgets

pd.options.mode.chained_assignment = None
df['Overall_Prediction_RBF'] = y_rbf_all
df['Error_Percentage'] = np.abs((df.overall - y_rbf_all) / df.overall * 100)
jsonDf = df.to_json(orient='records')
widgets.HTML(value=''' players = ''' + jsonDf)

f = open("fifa20players.js", "w")
f.write('players = ''' + jsonDf)
f.close()
