class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, num_epochs=100, threshold=0.5):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.weights = [0] * (num_features + 1)  # Add 1 for bias term
 
    def predict(self, features):
        activation = self.weights[0]  # Initialize with bias term
        for i in range(self.num_features):
            activation += self.weights[i+1] * features[i]
        return 1 if activation >= self.threshold else 0
 
    def train(self, training_data, labels):
        # ^w = a . (yi - ^yi) . xi
        # ^b = a . (yi - ^yi)
 
        for _ in range(self.num_epochs):
            for features, label in zip(training_data, labels):
                prediction = self.predict(features)
                update = self.learning_rate * (label - prediction)
                self.weights[0] += update  # Update bias term
                for i in range(self.num_features):
                    self.weights[i+1] += update * features[i]
 
    def evaluate(self, test_data, labels):
        correct = 0
        for features, label in zip(test_data, labels):
            prediction = self.predict(features)
            if prediction == label:
                correct += 1
        return correct/(len(test_data))
 
if __name__=="__main__":
    # Sample data
    training_data = [
        [2, 3],
        [4, 1],
        [1, 6],
        [3, 4],
    ]
    labels = [0, 0, 1, 1]
 
    # Create and train the perceptron
    num_features = len(training_data[0])
    perceptron = Perceptron(num_features=num_features)
    perceptron.train(training_data, labels)
 
    # Test the perceptron
    test_data = [
        [5, 2],
        [2, 1],
        [3, 5],
    ]
    
    labels_test = [0, 0, 0]
    
    # Predict
    for features in test_data:
        prediction = perceptron.predict(features)
        print(f"Prediction for {features}: {prediction}")
 
    # Evaluate
    print(f"Success rate: {perceptron.evaluate(test_data, labels_test):.2}%")
