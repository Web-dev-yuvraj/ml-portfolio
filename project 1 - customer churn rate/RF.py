import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('data.csv')
data = data.drop(['customerID','Contract', 'TotalCharges'], axis=1)
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['Partner'] = data['Partner'].map({'No': 0, 'Yes': 1})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'No': 0, 'Yes': 1})
data['Dependents'] = data['Dependents'].map({'No': 0, 'Yes': 1})
data['PhoneService'] = data['PhoneService'].map({'No': 0, 'Yes': 1})
data['OnlineSecurity'] = data['OnlineSecurity'].map({'No': 0, 'Yes': 1})
data['OnlineBackup'] = data['OnlineBackup'].map({'No': 0, 'Yes': 1})
data['DeviceProtection'] = data['DeviceProtection'].map({'No': 0, 'Yes': 1})
data['TechSupport'] = data['TechSupport'].map({'No': 0, 'Yes': 1})
data['StreamingTV'] = data['StreamingTV'].map({'No': 0, 'Yes': 1})
data['StreamingMovies'] = data['StreamingMovies'].map({'No': 0, 'Yes': 1})
data['InternetService'] = data['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
data['MultipleLines'] = data['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
data['PaymentMethod'] = data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], [0, 1, 2, 3])
cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data[cols] = data[cols].fillna(0)

x= data.drop('Churn', axis=1)
y= data['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(x_train)
xtest_scaled = scaler.transform(x_test)

model.fit(xtrain_scaled, y_train)
y_pred = model.predict(xtest_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy*100 ,"%")
print(
    classification_report(
        y_test,
        y_pred
    )
)