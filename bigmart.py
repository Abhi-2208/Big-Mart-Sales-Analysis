import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
training_data=pd.read_csv('bigmart_train.csv')
training_data['Item_Weight']=training_data['Item_Weight'].apply(lambda x:training_data['Item_Weight'].mean() if np.isnan(x) else x)
training_data.drop('Item_Identifier',axis=1,inplace=True)
#print(training_data.info())
#print(set(training_data['Outlet_Size']))
le=LabelEncoder()
training_data['Outlet_Size']=le.fit_transform(training_data['Outlet_Size'])
#print(set(training_data['Outlet_Size']))
#print(training_data.info())
training_data['Item_Fat_Content']=le.fit_transform(training_data['Item_Fat_Content'])
#print(set(training_data['Item_Fat_Content']))
training_data['Item_Type']=le.fit_transform(training_data['Item_Type'])
#print(set(training_data['Item_Type']))
training_data['Outlet_Identifier']=le.fit_transform(training_data['Outlet_Identifier'])
#print(set(training_data['Outlet_Identifier']))
#print(set(training_data['Outlet_Location_Type']))
training_data['Outlet_Location_Type']=le.fit_transform(training_data['Outlet_Location_Type'])
#print(set(training_data['Outlet_Location_Type']))
training_data['Outlet_Type']=le.fit_transform(training_data['Outlet_Type'])
#print(set(training_data['Outlet_Type']))
y_train=training_data['Item_Outlet_Sales']
x_train=training_data.drop('Item_Outlet_Sales',axis=1)


testing_data=pd.read_csv('bigmart_test.csv')
#print(testing_data.info())
testing_data['Item_Weight']=testing_data['Item_Weight'].apply(lambda x:testing_data['Item_Weight'].mean() if np.isnan(x) else x)
testing_data['Outlet_Size']=le.fit_transform(testing_data['Outlet_Size'])
#print(set(testing_data['Outlet_Size']))
testing_data.drop('Item_Identifier',axis=1,inplace=True)
testing_data['Item_Fat_Content']=le.fit_transform(testing_data['Item_Fat_Content'])
testing_data['Item_Type']=le.fit_transform(testing_data['Item_Type'])
testing_data['Outlet_Identifier']=le.fit_transform(testing_data['Outlet_Identifier'])
testing_data['Outlet_Location_Type']=le.fit_transform(testing_data['Outlet_Location_Type'])
testing_data['Outlet_Type']=le.fit_transform(testing_data['Outlet_Type'])
#print(testing_data.info())
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
print("The accuracy score of training_data is : {}".format(rf.score(x_train,y_train)))
random_forest=rf.score(x_train,y_train)
print("Random forest accuracy",random_forest)
pred=rf.predict(testing_data)
#print(pred)

knn=KNeighborsRegressor()
knn.fit(x_train,y_train)
knn_acc=knn.score(x_train,y_train)
print("Knn accuracy",knn_acc)

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_acc=dt.score(x_train,y_train)
print("Decision tree accuracy",dt_acc)

lr=LinearRegression()
lr.fit(x_train,y_train)
lr_acc=lr.score(x_train,y_train)
print("linear regression accuracy:",lr_acc)


accuracies=[knn_acc,random_forest,dt_acc,lr_acc]
algorithms=["KNN","Random Forest","Decision tree","Linear regression"]
fig=plt.figure(figsize=(10,5))
plt.bar(algorithms,accuracies,color="maroon",width=0.4)
plt.xlabel("Algorithms")
plt.ylabel("Accuracies")
plt.title("accuracy for different algorithms")
plt.show()
