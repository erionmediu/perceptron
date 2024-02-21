# Perceptron Implementation from Scratch

This project demonstrates a simple implementation of the perceptron algorithm, one of the foundational algorithms in machine learning. The perceptron is a type of linear classifier that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

## Steps:
1. **Data Reading**: Loading and preprocessing data.
2. **Vector Operations**: Implementing basic vector operations needed for the perceptron algorithm.
3. **Algorithm Implementation**: The core logic of the perceptron learning algorithm.

## Data Format
Each line in the dataset represents a single data point. The expected format is `"target feature_index:feature_value ..."`. For example, a line `"1 0:2.0 3:4.0 123:1.0"` means that the target label is `1`, feature `0` has a value of `2.0`, feature `3` is `4.0`, feature `123` is `1.0`, and all other features are `0.0` by default, indicating a sparse data representation.

## Implementation

Below is the Python code implementing the perceptron algorithm from scratch, including data preparation, vector operations, and the perceptron model itself.

```python
# Import necessary libraries
import random

# Initialize random seed for reproducibility
random.seed(123)

# Function to binarize ratings into binary class labels
def binarize(ratings):
    """Convert numeric ratings into binary class labels."""
    return [1 if rating > 5 else -1 for rating in ratings]

# Function to parse a data line into features and target label
def parse_line(line):
    """Extracts features and target label from a line."""
    elements = line.split()
    target = int(elements[0])
    features = {int(k): float(v) for k, v in (element.split(":") for element in elements[1:])}
    return features, target

# Function to prepare data from a file
def prepare_data(filepath, sample_rate=0.9):
    """Reads and prepares data from a file."""
    data = []
    for line in open(filepath):
        if random.random() < sample_rate:
            data.append(parse_line(line))
    random.shuffle(data)
    return data

# Dot product function for vector operations
def dot_product(vec1, vec2):
    """Computes the dot product between two feature vectors."""
    return sum(vec1.get(key, 0) * value for key, value in vec2.items())

# Function to increment one vector by another
def increment(vec, increment_vec, scale=1.0):
    """Updates vector 'vec' by adding 'increment_vec' scaled by 'scale'."""
    for key, value in increment_vec.items():
        vec[key] = vec.get(key, 0.0) + value * scale

# Function to initialize the perceptron model
def initialize_model():
    """Initializes the Perceptron model with empty weights and zero bias."""
    return {'weights': {}, 'bias': 0.0}

# Function for making predictions with the perceptron model
def predict(model, features):
    """Predicts the class label for given features."""
    weighted_sum = dot_product(model['weights'], features) + model['bias']
    return 1 if weighted_sum >= 0 else -1

# Function to update the model based on prediction error
def update_model(model, features, target, learning_rate=1.0):
    """Updates the Perceptron model weights and bias."""
    prediction = predict(model, features)
    if prediction != target:
        increment(model['weights'], features, scale=target * learning_rate)
        model['bias'] += target * learning_rate

# Example usage
data = [
    ({0: 2, 1: 1}, 1),
    ({0: 1, 1: 3}, -1),
    ({0: 4, 1: 2}, 1),
    ({0: 1, 1: 1}, -1),
    ({0: 3, 1: 4}, -1),
]

model = initialize_model()
for features, target in data:
    update_model(model, features, target, learning_rate=1.0)

# Evaluate the model
for features, target in data:
    prediction = predict(model, features)
    print(f"True Label: {target}, Predicted Label: {prediction}")
