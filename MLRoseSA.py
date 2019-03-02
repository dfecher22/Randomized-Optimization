
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlrose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("heart.csv", header=None)


dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[1:303,0:12].astype(float)
Y = dataset[1:303,13].astype(int)




X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state = 3)
# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# One hot encode target values


lr=[0.001,0.01,0.1]
max_iters=[1000,5000,10000,25000,50000]
decays=[mlrose.ArithDecay(),mlrose.GeomDecay(),mlrose.ExpDecay()]
res=[]


nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2,7,1], activation = 'relu',algorithm = 'simulated_annealing',
                                 schedule=decays[0],max_iters = 70000,
                                 bias = True, is_classifier = True, learning_rate = 0.01,
                                 early_stopping = False, clip_max = 10, max_attempts =30)



# Predict labels for train set and assess accuracy
nn_model1.fit(X_train_scaled, y_train)
y_train_pred = nn_model1.predict(X_train_scaled)
y_test_pred = nn_model1.predict(X_test_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
res.append((y_train_accuracy, y_test_accuracy))

res=numpy.asarray(res)
print(res)
'''numpy.savetxt("SAdecayES.csv", res, delimiter=",")'''



