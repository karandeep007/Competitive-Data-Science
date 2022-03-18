import pandas as pd
import os
import numpy as np
import itertools
import plots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# split data 80-20 and trainx validatex
# then run model on testxs
def missing_summary(df):
    cols = df.columns
    for col in cols:
        print(col, " ", df[col].isnull().sum())

def data_prep(df,catcols,numcols):
    # simple imputation by Mean for Numeric
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(df[numcols])
    df[numcols] = imputer.transform(df[numcols])
    return df

def split_data(X, y, pct_split, cat_features):
    # split into Training and validation set
    trainX, valX, trainy, valy = train_test_split(X,y,test_size=pct_split,random_state=5)
    # encode categorical
    trainX = pd.get_dummies(trainX, columns=[i for i in cat_features],
                              prefix=[i for i in cat_features])
    # encode categorical
    valX = pd.get_dummies(valX, columns=[i for i in cat_features],
                              prefix=[i for i in cat_features])
    return trainX, valX, trainy, valy

def lr_classifer(trainX, trainY, testX):
    lr_clf = LogisticRegression(solver='liblinear')
    # train classifier
    lr_clf = lr_clf.fit(X= trainX, y = trainY)
    # make predicitions
    predY = lr_clf.predict(testX)
    return predY, lr_clf

def make_predictions(testX, clf):
    predY = clf.predict(testX)
    return predY

if __name__ == '__main__':

    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # Not sure of type = ['Cabin','Ticket']
    cat_features = ['Pclass','Sex','Embarked']
    # Fare Numeric removed as it was least imp, no change in model metrics
    num_features = ['Age','SibSp']
    target = 'Survived'
    print(train_df.columns)
    print(train_df.head())

    # plots for Numeric
    plots.scatterplot(train_df,num_features,"plots/",target)
    # view missing value proportion

    # Some data prep
    train_df = data_prep(train_df,cat_features, num_features)
    test_df = data_prep(test_df, cat_features, num_features)
    # missing_summary(train_df)

    X = train_df.loc[ : ,cat_features+num_features].copy()
    y = train_df.loc[:,target]
    testX = test_df.loc[:, cat_features + num_features].copy()

    # split into training and validation & encode features
    trainX, valX, trainy, valy = split_data(X, y, 0.20, cat_features)
    print('Training dataset :' ,trainX.shape)
    print('Holdout dataset :',valX.shape)

    predY , model = lr_classifer(trainX, trainy, valX)
    valy_asarray = valy.to_numpy()

    # c_matrix = confusion_matrix(valy_asarray, predY)
    # print(c_matrix)
    print(classification_report(valy_asarray,predY))

    # prep test data
    testX = pd.get_dummies(testX, columns=[i for i in cat_features],
                           prefix=[i for i in cat_features])
    final_pred = make_predictions(testX, model)
    testY = pd.DataFrame(test_df['PassengerId'],index=None,columns=['PassengerId'],copy=True)
    testY['Survived'] = final_pred.tolist()
    testY.to_csv(os.path.join(os.getcwd()+'/FinalPredictions.csv'),index=False)

    # ToDo: Calculate prediction accuracy

    # Feature Importance
    feature_names = model.feature_names_in_
    feature_imp = model.coef_
    re_1 =  np.reshape(model.feature_names_in_, (-1, 1))
    re_2 = np.reshape(model.coef_, (-1, 1))
    htack_fimp = np.hstack((re_1, re_2))
    feature_imp_df = pd.DataFrame(htack_fimp, columns=['Feature', 'Imp'])

    # Top Features
    feature_imp_df['Imp_Abs'] = feature_imp_df['Imp'].abs()
    feature_imp_df = feature_imp_df.sort_values(by=['Imp_Abs'], ascending=False)

    print('END')