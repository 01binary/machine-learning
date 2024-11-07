from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
  # Create an Adam optimizer with the given learning rate
  opt = Adam(lr = learning_rate)
  
  # Create your binary classification model  
  model = Sequential()
  model.add(Dense(128, input_shape = (30,), activation = activation))
  model.add(Dense(256, activation = activation))
  model.add(Dense(1, activation = 'sigmoid'))
  
  # Compile your model with your optimizer, loss, and metrics
  model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return model

# Tuning part 1

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 128, 256], 
    'epochs': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long! 
show_results()

# Tuning part 2

# Create a KerasClassifier
model = KerasClassifier(
  build_fn = create_model(
    learning_rate = 0.001,
    activation = 'relu'
  ),
  epochs = 50,
  batch_size = 128,
  verbose = 0
)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv = 3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())
