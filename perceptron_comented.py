# Perceptron Implementation from Scratch

# Steps:
# 1. Data Reading
# 2. Vector Operations
# 3. Algorithm Implementation

# Data Format: Each line represents a data point. The format is "target feature_index:feature_value ...".
# Example: "1 0:2.0 3:4.0 123:1.0" indicates the target label is 1, feature 0 has value 2.0, and so on.
# Unmentioned features are considered to have a value of 0.0, following a sparse representation.

# Print the first three data examples, demonstrating use of enumerate for index-value pairs.
for index, line in enumerate(open('sentiment.feat')):
    if index < 3:
        print(line)

# Example of enumerate on a list for demonstration purposes.
for index, value in enumerate([4, 44, 444, 4444, 44444, 444]):
    print(index, value)

# Binarization of integer ratings into binary labels (positive/negative) using a threshold.
def binarize(ratings):
    """Convert numeric ratings into binary class labels."""
    return [1 if rating > 5 else -1 for rating in ratings]

# Alternative binarization function with explicit loop for clarity.
def binarize_explicit(ratings):
    binary_labels = []
    for rating in ratings:
        binary_labels.append(1 if rating > 5 else -1)
    return binary_labels

# Demonstration of binarization functions.
ratings_example = list(range(11))
print(binarize(ratings_example))
print(binarize_explicit(ratings_example))

# Function to parse a data line into feature values and a target label.
def parse_line(line):
    """Extracts features and target label from a line."""
    elements = line.split()
    target = int(elements[0])
    features = {int(k): float(v) for k, v in (element.split(":") for element in elements[1:])}
    return features, target

# Example usage of parse_line.
example_line = '1 0:2.0 3:4.0 123:1.0'
print(parse_line(example_line))

# Data Preparation: Reads, shuffles, and formats training data.
import random
random.seed(123)  # Ensure reproducibility

def prepare_data(filepath, sample_rate=0.9):
    """Reads and prepares data from a file, sampling and shuffling the entries."""
    data = []
    for line in open(filepath):
        if random.random() < sample_rate:
            data.append(parse_line(line))
    random.shuffle(data)
    return data

# Vector operations for the Perceptron algorithm.

# Dot product between two feature vectors.
def dot_product(vec1, vec2):
    """Computes the dot product between two feature vectors."""
    return sum(vec1.get(key, 0) * value for key, value in vec2.items())

# Increment one vector by another, used for weight updates.
def increment(vec, increment_vec, scale=1.0):
    """Updates vector 'vec' by adding 'increment_vec' scaled by 'scale'."""
    for key, value in increment_vec.items():
        vec[key] = vec.get(key, 0.0) + value * scale

# Scale a vector by a constant factor.
def scale_vector(vector, factor):
    """Scales a vector by a given factor."""
    return {key: value * factor for key, value in vector.items()}

# Perceptron model initialization.
def initialize_model():
    """Initializes the Perceptron model with empty weights and zero bias."""
    return {'weights': {}, 'bias': 0.0}

# Prediction function for the Perceptron.
def predict(model, features):
    """Predicts the class label for given features using the Perceptron model."""
    weighted_sum = dot_product(model['weights'], features) + model['bias']
    return 1 if weighted_sum >= 0 else -1

# Model update rule based on prediction error.
def update_model(model, features, target, learning_rate=1.0):
    """Updates the Perceptron model weights and bias based on the prediction error."""
    prediction = predict(model, features)
    if prediction != target:
        increment(model['weights'], features, scale=target * learning_rate)
        model['bias'] += target * learning_rate

# Example data preparation and model usage.
# Note: Replace 'your_data_path' with the actual path to your data file.
data = prepare_data('your_data_path')
model = initialize_model()
for features, target in data:
    update_model(model, features, target)

# This code structure provides a clear and modular approach to implementing a Perceptron.
# Each function is designed for a specific part of the process, facilitating easy modification and extension.

# Testing the model :

# Create a simple dataset of feature vectors and binary labels (-1, 1).
# Features: x1, x2 (simple 2D points for easy visualization)
# Label: 1 if x1 > x2, else -1 (a simple linearly separable problem)
data = [
    ({0: 2, 1: 1}, 1),  # Point (2, 1) -> Label: 1
    ({0: 1, 1: 3}, -1), # Point (1, 3) -> Label: -1
    ({0: 4, 1: 2}, 1),  # Point (4, 2) -> Label: 1
    ({0: 1, 1: 1}, -1), # Point (1, 1) -> Label: -1
    ({0: 3, 1: 4}, -1), # Point (3, 4) -> Label: -1
]

# Step 2: Train the Perceptron
model = initialize_model()  # Initialize the perceptron model.
for features, target in data:
    update_model(model, features, target, learning_rate=1.0)

# Step 3: Evaluate the Model
# For each example in our dataset, predict the label and compare it with the true label.
for features, target in data:
    prediction = predict(model, features)
    print(f"True Label: {target}, Predicted Label: {prediction}")

# Note: this is an over-simplified scenario
