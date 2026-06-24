import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('data.csv')
data = data.drop(['customerID', 'TotalCharges', 'gender','PhoneService'], axis=1)


cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data[cols] = data[cols].fillna(data[cols].mode().iloc[0])
x= data.drop('Churn', axis=1)
y= data['Churn']
category = x.select_dtypes(include=['object']).columns
numeric = x.select_dtypes(exclude=['object']).columns


transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), category)
])
model = Pipeline(steps=[
    ('preprocessor', transformer),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy*100 ,"%")
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)