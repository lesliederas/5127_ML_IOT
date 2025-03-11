import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from person import Person
rng = np.random.default_rng(2022)

# Create list comprehensions for lengths of names
list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]

name_lengths = [len(name) for name in list_of_names]
print(f"Name lengths: {name_lengths}")

# Create a dictionary of person objects
people = {name: Person(name=name, age=age, height=height)
          for name, age, height in zip(list_of_names, list_of_ages, list_of_heights_cm)}

# Print person information
for name, person_obj in people.items():
    print(person_obj)

# Convert lists to numpy arrays
ages_array = np.array(list_of_ages)
heights_array = np.array(list_of_heights_cm)

# Compute average age
average_age = np.mean(ages_array)
print(f"Average age: {average_age:.2f}")

# Scatter plot of ages vs heights
plt.figure(figsize=(8, 6))
plt.scatter(ages_array, heights_array, color='purple')
plt.grid(True)
plt.xlabel("Age (years)")
plt.ylabel("Height (cm)")
plt.title("Age vs Height Scatter Plot")
plt.savefig("age_vs_height.png")

# Load the iris dataset
iris_db = load_iris(as_frame=True)
x_data = iris_db['data'].to_numpy()
y_labels = iris_db['target'].to_numpy()
target_names = iris_db['target_names']

# Plot iris data
plt.figure(figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')
l_colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(target_names):
    plt.scatter(x_data[y_labels == n, 0], x_data[y_labels == n, 1],
                c=l_colors[n], label=species)
plt.xlabel(iris_db['feature_names'][0])
plt.ylabel(iris_db['feature_names'][1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('iris_data.png')


# Define the classify_iris function
def classify_iris(features):
    weights = np.array([
        [5, 8, 11, 0.1],   # Weights for class 0
        [12, 0.4, 8, 0.6],  # Weights for class 1
        [10.1, 0.3, 0.4, 6.7]  # Weights for class 2
    ])
    biases = np.array([100.9, 120, 4])
    scores = (np.matmul(weights, features) + biases)
    predicted = np.argmax(scores)
    return predicted
def classify_rand(x):
      return rng.integers(0,2, endpoint=True)

def evaluate_classifier(cls_func, x_data, labels, print_confusion_matrix=True):
    n_correct = 0
    n_total = x_data.shape[0]
    cm = np.zeros((3,3))
    for i in range(n_total):
        x = x_data[i,:]
        y = cls_func(x)
        y_true = labels[i]
        cm[y_true, y] += 1
        if y == y_true:
            n_correct += 1    
        acc = n_correct / n_total
    print(f"Accuracy = {n_correct} correct / {n_total} total = {100.0*acc:3.2f}%")
    if print_confusion_matrix:
        print(f"{12*' '}Estimated Labels")
        print(f"              {0:3.0f}  {1.0:3.0f}  {2.0:3.0f}")
        print(f"{12*' '} {15*'-'}")
        print(f"True    0 |   {cm[0,0]:3.0f}  {cm[0,1]:3.0f}  {cm[0,2]:3.0f} ")
        print(f"Labels: 1 |   {cm[1,0]:3.0f}  {cm[1,1]:3.0f}  {cm[1,2]:3.0f} ")
        print(f"        2 |   {cm[2,0]:3.0f}  {cm[2,1]:3.0f}  {cm[2,2]:3.0f} ")
        print(f"{40*'-'}")
  ## done printing confusion matrix  

    return acc, cm

## Now evaluate the classifier we've built.  This will evaluate the
# random classifier, which should have accuracy around 33%.
acc, cm = evaluate_classifier(classify_iris, x_data.to_numpy(), y_labels.to_numpy())
