# penguins prediction 
import seaborn as sns
df = sns.load_dataset('penguins')

print (df.shape)
print ("is nulls are:",df.isnull().sum())
print ("species are:",df['species'].value_counts())
print ("columns ex:",df.head())

df = df.dropna(subset =['bill_length_mm'] )
print (df.shape)
print ("NEW SHAPE IS : ",df.isnull().sum())

from sklearn.model_selection import train_test_split
X = df.drop(columns ='species')
y = df['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =42,stratify= y,test_size= 0.2)

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include= 'object').columns.tolist()

print (num_cols)
print (cat_cols)
print (X_train.shape,X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

prep = ColumnTransformer([
    ('num',Pipeline([
        ('imputer',SimpleImputer(strategy = 'median')),
        ('sc',StandardScaler())
    ]),num_cols),

    ('cat',Pipeline([
        ('imputer',SimpleImputer(strategy = 'most_frequent')),
        ('encoding',OneHotEncoder(handle_unknown ='ignore'))
    ]),cat_cols)
])


models = {
    'Logistic':LogisticRegression(max_iter= 1000),
    'RandomForest':RandomForestClassifier(n_estimators=200,random_state=42)
}
for name , m in models.items():
  pipe = Pipeline([('prep',prep), ('model',m)])
  scores = cross_val_score(pipe,X_train,y_train,cv=5,scoring ='accuracy')
  print (name, scores.mean())

  from sklearn.model_selection import GridSearchCV
  pipe = Pipeline([('prep',prep), ('model',RandomForestClassifier(random_state=42))])
  param_grid = {
      'model__n_estimators':[100,200],
      'model__max_depth':[3,5,None]
  }
  grid = GridSearchCV(pipe, param_grid ,cv=5, scoring= 'accuracy')
  grid.fit(X_train,y_train)
  print ("Best params:",grid.best_params_)
  print ("best cv score:",grid.best_score_)
  print ("test accurazcy:", grid.score(X_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
best = grid.best_estimator_
y_pred = best.predict(X_test)

print (confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))
