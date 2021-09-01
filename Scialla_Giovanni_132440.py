import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt
import time

np.random.seed(0)
#leggo il dataset

dataset = "day.csv"
df = pd.read_csv(dataset)


#-------------------EDA-------------------------
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.value_counts())

df = pd.read_csv(dataset, usecols=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15],
                 names=['season', 'month', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'ftemp', 'humidity', 'windspeed', 'rent-count'],
                 skiprows=1)


df.drop_duplicates(inplace=True)
print(df.isnull().values.sum())
#d_season = {1 : "spring", 2 : "summer", 3 : "autumn", 4 : "winter"}
#df['season'] = df['season'].map(d_season)
#d_month = {1 : "january", 2 : "february", 3 : "march", 4 : "april", 5 : "may", 6 : "june", 7 : "july", 8 : "august",
#        9 : "september", 10 : "october", 11 : "november", 12 : "dicember"}
#df["month"] = df["month"].map(d_month)
#d_weather = {1: "clear", 2: "cloudy", 3: "rainy"}
#df['weather'] = df['weather'].map(d_weather)

#numerical features
numerical = ['temp', 'ftemp', 'humidity', 'windspeed', 'rent-count']

#categorical features
categorical = ['season', 'month', 'holiday', 'weekday', 'workingday', 'weather', 'rent-count']


fig, ax = plt.subplots(figsize=(10, 8))
sns.set_style(style='whitegrid')

#matrice di correlazione per le features numeriche
df_corr = df[numerical].corr()
sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='Blues', vmin=-1, vmax=1)
plt.show()

#rimuovo la multicollinearità
df.drop('ftemp', axis=1, inplace=True)

numerical = ['temp', 'humidity', 'windspeed', 'rent-count']
sns.pairplot(df[numerical])
plt.show()

print(df['season'].unique())

for i in categorical:
    sns.countplot(data=df, x=i)
    plt.show()

for i in categorical:
    sns.boxplot(data=df, x=i, y='rent-count', hue='workingday')
    plt.show()

sns.scatterplot(data=df, x="rent-count", y="temp", hue="workingday", palette="deep")
plt.show()
sns.scatterplot(data=df, x="rent-count", y="temp", hue="season", palette="deep")
plt.show()
sns.scatterplot(data=df, x="rent-count", y="temp", hue="weather", palette="deep")
plt.show()
sns.scatterplot(data=df, x="rent-count", y="temp", hue="weekday", palette="deep")
plt.show()

sns.displot(data=df, x="rent-count", hue="month")
plt.show()


sns.barplot(x="month", y="rent-count", hue="weather", data=df)
plt.show()
sns.barplot(x="month", y="rent-count", hue="workingday", data=df)
plt.show()
month_count = df['month'].value_counts()
sns.barplot(month_count.index, month_count.values, alpha=0.9)
plt.show()

#FEATURES SELECTION CON WRAPPER METHODS (STEP FORWARD SELECTION)

#training set and testing set split
features = list(set(df.columns) - set(['rent-count']))
X = df[features].values
y = df['rent-count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

best_n = 1
best_score = 0.0
for i in range(1,9,1):
    sfs = SequentialFeatureSelector(estimator=Ridge(), n_features_to_select=i, direction='forward', scoring='r2', cv=5, n_jobs=-1)
    sfs.fit(X_train, y_train)
    print(sfs.get_params())
    print(sfs.get_support(indices=True))
    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)
    rr = Ridge()
    rr.fit(X_train_sfs, y_train)
    y_pred_rr = rr.predict(X_test_sfs)
    r2score = metrics.r2_score(y_test, y_pred_rr)
    if(best_score < r2score):
        best_n = i
        best_score = r2score


sfs = SequentialFeatureSelector(estimator=Ridge(), n_features_to_select=best_n, direction='forward', scoring='r2', cv=5, n_jobs=-1)
sfs.fit(X_train, y_train)
X_train = sfs.transform(X_train)
X_test = sfs.transform(X_test)
print("il miglior numero di features è: ", best_n)
print("dimensione del vettore di features: ",X_train.shape, X_test.shape)
print("features selezionate: ", df.columns[sfs.get_support(indices=True)])

#----------------PREPROCESSING---------------

#validation set
X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print(X_train_2.shape, y_train_2.shape)
print(X_val.shape, y_val.shape)
#min max scaler
mms = MinMaxScaler()
mms.fit(X_train)
X_train_mms = mms.transform(X_train)
X_test_mms = mms.transform(X_test)

#standard scaler
ss = StandardScaler()
ss.fit(X_train)
X_train_ss = ss.transform(X_train)
X_test_ss = ss.transform(X_test)
#-----------TRAINING AND MODEL SELECTION-------------

#ridge regression
rr = Ridge()
rr.fit(X_train_2, y_train_2)

#grid search cv for model selection
parameters = {"alpha": (0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0), "solver": ("svd", "lsqr", "sag")}
model = rr
clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='r2', cv=5)
clf.fit(X_val, y_val)
print('ridge regression')
print('Overall, the best parameters for alpha is:', clf.best_params_.get("alpha"),
      'the best solver is: ', clf.best_params_.get("solver"),
      ' since it leads to r2-score = ', clf.best_score_)

#ridge regression con dati originali
rr = Ridge(alpha=0.1, solver='svd')
rr.fit(X_train, y_train)
y_pred_rr = rr.predict(X_test)
print('RIDGE REGRESSION')
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_rr))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rr)))
print("R2 Score:", metrics.r2_score(y_test, y_pred_rr))

#ridge regression with min max scaler
rr = Ridge(alpha=0.1, solver='svd')
rr.fit(X_train_mms, y_train)
y_pred_rr = rr.predict(X_test_mms)
print('RIDGE REGRESSION with min max scaler')
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_rr))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rr)))
print("R2 Score:", metrics.r2_score(y_test, y_pred_rr))

#ridge regression with standard scaler
rr = Ridge(alpha=0.1, solver='svd')
rr.fit(X_train_ss, y_train)
y_pred_rr = rr.predict(X_test_ss)
print('RIDGE REGRESSION with standard scaler')
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_rr))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rr)))
print("R2 Score:", metrics.r2_score(y_test, y_pred_rr))

#neural network
nn = MLPRegressor(max_iter=5000)
nn.fit(X_train_2, y_train_2)

#cross validation
hidden_layer_sizes = [(50, 20), (60, 20), (60, 30)]
hyper_parameter = [{'solver': ['adam','sgd','lbfgs'],
                'activation': ['tanh','identity', 'relu'],
                'hidden_layer_sizes': hidden_layer_sizes,
                'alpha': [0.1, 0.001, 0.0001],
                'early_stopping': [True, False]}]
clf_nn = GridSearchCV(estimator=nn, param_grid=hyper_parameter, refit=True, cv=5)
clf_nn.fit(X_val, y_val)
print(clf_nn.best_params_, clf_nn.best_score_)

#neural network original features
nn = MLPRegressor(activation="identity", solver="lbfgs", max_iter=5000, hidden_layer_sizes=(60, 30), alpha=0.1, random_state=42, early_stopping=False)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print('NEURAL NETWORK with original features')
print("Error on test (MSE) = ", metrics.mean_squared_error(y_test, y_pred_nn))
print("Error on test (MAE) = ", metrics.mean_absolute_error(y_test, y_pred_nn))
print('Error on test (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_nn)))
print("Performance on test (R2) = ", metrics.r2_score(y_test, y_pred_nn))

#neural network with min max scaler
nn = MLPRegressor(activation="identity", solver="lbfgs", max_iter=5000, hidden_layer_sizes=(60, 30), alpha=0.1, random_state=42, early_stopping=False)
nn.fit(X_train_mms, y_train)
y_pred_nn = nn.predict(X_test_mms)
print('NEURAL NETWORK with min max scaler')
print("Error on test (MSE) = ", metrics.mean_squared_error(y_test, y_pred_nn))
print("Error on test (MAE) = ", metrics.mean_absolute_error(y_test, y_pred_nn))
print('Error on test (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_nn)))
print("Performance on test (R2) = ", metrics.r2_score(y_test, y_pred_nn))


#neural network with standard scaler
nn = MLPRegressor(activation="identity", solver="lbfgs", max_iter=5000, hidden_layer_sizes=(60, 30), alpha=0.1, random_state=42, early_stopping=False)
nn.fit(X_train_ss, y_train)
y_pred_nn = nn.predict(X_test_ss)
print('NEURAL NETWORK with standard scaler')
print("Error on test (MSE) = ", metrics.mean_squared_error(y_test, y_pred_nn))
print("Error on test (MAE) = ", metrics.mean_absolute_error(y_test, y_pred_nn))
print('Error on test (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_nn)))
print("Performance on test (R2) = ", metrics.r2_score(y_test, y_pred_nn))

#------------TESTING BEST VERSION----------------
#ridge regression best version original data
rr = Ridge(alpha=clf.best_params_.get("alpha"), solver=clf.best_params_.get("solver"))
start_time = time.time_ns()
rr.fit(X_train, y_train)
end_time = time.time_ns()
elapsed_time = (end_time - start_time)
print("tempo di addestramento %s seconds" % elapsed_time)
y_pred_rr = rr.predict(X_test)
print('RIDGE REGRESSION best version')
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rr))
print("R2 Score:", metrics.r2_score(y_test, y_pred_rr))

print("R2 Score su training set:", metrics.r2_score(y_train, rr.predict(X_train)))
print("Mean Absolute Error su training set:", metrics.mean_absolute_error(y_train, rr.predict(X_train)))

x_plot = plt.scatter(y_pred_rr, (y_pred_rr - y_test), c='b')
plt.hlines(y=0, xmin=0, xmax=8000)
plt.title('Residual plot (ridge regression best version)')
plt.show()

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rr})
df1 = df.head(20)
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('bar plot (ridge regression best version)')
plt.xlabel("instance")
plt.ylabel("rent-count")
plt.show()

#neural network best version standardized data
nn = MLPRegressor(activation=clf_nn.best_params_.get("activation"),
                  solver=clf_nn.best_params_.get("solver"),
                  max_iter=5000,
                  hidden_layer_sizes=clf_nn.best_params_.get("hidden_layer_sizes"),
                  alpha=clf_nn.best_params_.get("alpha"),
                  random_state=clf_nn.best_params_.get("random_state"),
                  early_stopping=True)
start_time = time.time_ns()
nn.fit(X_train_ss, y_train)
end_time = time.time_ns()
elapsed_time = (end_time - start_time) / 1000000
print("tempo di addestramento in milliseconds", elapsed_time)
y_pred_nn = nn.predict(X_test_ss)
print('NEURAL NETWORK best version')
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_nn))
print("R2 Score:", metrics.r2_score(y_test, y_pred_nn))

x_plot = plt.scatter(y_pred_nn, (y_pred_nn - y_test), c='b')
plt.hlines(y=0, xmin=0, xmax=8000)
plt.title('Residual plot (neural network best version)')
plt.show()

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_nn})
df1 = df.head(20)
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('bar plot (neural network best version)')
plt.xlabel("instance")
plt.ylabel("rent-count")
plt.show()

#linear regression
lr = LinearRegression()
start_time = time.time_ns()
lr.fit(X_train, y_train)
end_time = time.time_ns()
elapsed_time = (end_time - start_time) / 1000000
print("tempo di addestramento in milliseconds", elapsed_time)
y_pred_lr = lr.predict(X_test)
print('LINEAR REGRESSION')
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_lr))
print("R2 Score:", metrics.r2_score(y_test, y_pred_lr))

x_plot = plt.scatter(y_pred_lr, (y_pred_lr - y_test), c='b')
plt.hlines(y=0, xmin=0, xmax=8000)
plt.title('Residual plot (linear regression)')
plt.show()

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lr})
df1 = df.head(20)
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.title('bar plot (linear regression)')
plt.xlabel("instance")
plt.ylabel("rent-count")
plt.show()