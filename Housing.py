# Download and load a real dataset (California housing data).
# Split it into training and testing data in a smart way.
# Ensure that the test set represents the whole dataset well (avoid sampling bias).
# Prepare the data so we can start building a machine learning model later







import os  #Helps us work with folders and files
import tarfile  #Helps us open zipped (.tgz) files
import urllib.request  #Lets us download files from the internet
#  The base link where our dataset is found
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
#  This is the local folder where we will save the data
HOUSING_PATH = os.path.join("datasets", "housing")
#  The full web link to the zipped dataset file
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#Function to download and extract the housing data



def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    #If the folder doesn't exist, create it!
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    #This is the path where we’ll save the downloaded zip file
    tgz_path = os.path.join(housing_path, "housing.tgz")
    #Download the dataset from the internet and save it in our folder
    urllib.request.urlretrieve(housing_url, tgz_path)
    #Open the downloaded zip file
    housing_tgz = tarfile.open(tgz_path)
    #Unzip (extract) all files from the zip into our folder
    housing_tgz.extractall(path=housing_path)
    #Done with the zip file, so close it
    housing_tgz.close()
#Actually call the function to download and unzip the data
fetch_housing_data()



#Now let's use pandas (a tool for working with tables)
import pandas as pd
# Function to load the housing data from the CSV file
def load_housing_data(housing_path=HOUSING_PATH):
    # Find the path to the CSV file that was extracted
    csv_path = os.path.join(housing_path, "housing.csv")
    #Use pandas to read the CSV and return it as a table (DataFrame)
    return pd.read_csv(csv_path)
# Now we actually load the data and save it in a variable called "housing"
housing = load_housing_data()
# Show the first 5 rows of the table
print(housing.head())
# Print out info **about** the table (like column names and data types)
print(housing.info)  # NOTE: this should be housing.info() with parentheses!
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()



import numpy as np
#creating a func to seprate data(data for training) and for testing(20per)
def split_train_test(data,test_ratio):
    #writing program so my rows get shuffled i mean just like shuffling cards randomly
    shuffled_indices=np.random.permutation(len(data))
    #calculating how many rows to use for testing
    test_set_size=int(len(data) * test_ratio)
    #taking the first 20 random this become test set
    test_indices=shuffled_indices[:test_set_size]
    #taking other rows for training
    train_indices=shuffled_indices[test_set_size]
    #This line below return rows for training and rows for testing
    return data.iloc[train_indices],data.iloc[test_indices]
train_set, test_set=split_train_test(housing,0.2)


#Now i am setting unique id for each row 
#I am hashing the id and that number wil decide if a row wiil go in training set


from zlib import crc32
def test_set_check(identifier, test_ratio):# I am turning identifier into a number
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * test_ratio*2**32 #I am also checking if this number is in lowest 20%
#Now i am spliting the data based on id
def split_train_test_by_id(data, test_ratio ,id_column):
    ids=data[id_column] #pick your id column
    in_test_set=ids.apply(lambda id :test_set_check(id ,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set] 

#What if no column 
housing_with_id=housing.reset_index()#This gives each row an index like 0,1,2,3,4,...
#Now splitting 


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#Now I am doing stratified sampling in code so i make sure data is split based on category
#First I am creating income categories
housing["income_cat"] = pd.cut(#pd.cut cuts up continuous column in categories 
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

from sklearn.model_selection import StratifiedShuffleSplit

split= StratifiedShuffleSplit(n_splits=1 , test_size=0.2 ,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    split_train_test=housing.loc[train_index]
    split_test_index=housing.loc[test_index]
    strat_test_index=housing.loc[test_index]

#The following shows the percen of each category in test set
strat_test_index["income_cat"].value_counts() /len(strat_test_index)

#Now i don't need income cat for my training so i will remove it
for set_ in(split_train_test,strat_test_index):
    set_.drop("income_cat" ,axis=1 ,inplace=True)























