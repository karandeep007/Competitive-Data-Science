'''
https://towardsdatascience.com/introduction-to-decision-tree-classifiers-from-scikit-learn-32cd5d23f4d
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_text #export the decision rules

train_df = pd.read_csv('Titanic/data/train.csv')

cat = ['Pclass', 'Sex', 'Embarked']
num = ['Age', 'Fare']
target = 'Survived'

X_num = train_df[num]
X_cat = train_df[cat].astype(str)
# enc = OneHotEncoder()
# # ToDo: Difference between get dummies and OneHotEncoder
# X_cat = pd.get_dummies(X_cat)

# ToDo: How does IterativeImputer work?
imp = IterativeImputer(max_iter=75, random_state=0)
imp.fit(X_num)
X_num = pd.DataFrame(imp.transform(X_num), columns=list(X_num.columns))

df_X = pd.concat([X_num,X_cat], axis=1)
feature_names = df_X.columns
df_y = train_df[target]
X_train, test_x, y_train, test_lab = train_test_split(df_X, df_y ,test_size = 0.4, random_state = 42)

clf = DecisionTreeClassifier(max_depth =3, random_state = 42)
clf.fit(X_train, y_train)

tree_rules = export_text(clf, feature_names = list(feature_names))
print(tree_rules)
