# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns
# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
#importing the dataset
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)

#visualizing the dataset
print(f"\n Here are the first 5 rows of the dataset {data.head()}")

#separated the data in features and target
x = data["YearsExperience"]
y = data["Salary"]

#using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(data["YearsExperience"], data["Salary"], color="red") # actual plot
sns.regplot(data = data, x = "YearsExperience", y = "Salary") #regression line
plt.show() # renderize the plot to show it

#splitting the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02, random_state = 101)

#checking the train and test to proove they are 80% and 20% respectively
print(f"\n the total x size is: {x.shape[0]}")
print(f"\n the total x_train size is: {x_train.shape[0]}, and is the {x_train.shape[0]/x.shape[0] * 100} % of the total X")
print(f"\n the total x_test size is: {x_test.shape[0]}, and is the {x_test.shape[0]/x.shape[0] * 100} % of the total X")



# --- END OF MAIN CODE ---


