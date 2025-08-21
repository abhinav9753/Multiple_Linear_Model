import numpy as np
from flask import Flask, request, render_template
import pickle
import sklearn
from sklearn.linear_model import LinearRegression

# Create a Flask web application instance
app = Flask(__name__)

# Load the trained model. Make sure 'model.pkl' is in the same directory as this script.
# You need to replace 'model.pkl' with the actual name of your saved model file.
# The 'rb' stands for 'read binary'.
try:
    with open('MLR_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: model.pkl not found. Please make sure your trained model file is in the same directory.")
    model = None # Set model to None if the file is not found

@app.route('/')
def home():
    """Renders the HTML form page."""
    # This will look for 'index.html' in a 'templates' folder.
    # Make sure to save the HTML code provided above as 'templates/index.html'.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, uses the model to predict the profit,
    and returns the result to the user.
    """
    if model is None:
        return "Error: Model not loaded. Please check the server logs."

    # Get the data from the HTML form
    rnd_spend = float(request.form['rnd_spend'])
    admin_spend = float(request.form['admin_spend'])
    marketing_spend = float(request.form['marketing_spend'])
    state = int(request.form['state'])

    # Create a NumPy array from the input data. The model expects data in this format.
    # The double brackets are important to maintain the correct shape (1 row, 4 columns).
    features = np.array([[rnd_spend, admin_spend, marketing_spend, state]])

    # Make a prediction using the loaded model
    prediction = model.predict(features)
    predicted_profit = round(prediction[0], 2)

    # Return the predicted profit as a formatted string to the user
    # Note: In a real-world app, you might want to render a new HTML page
    # or use JavaScript to display the result on the same page.
    return f"The predicted profit is: ${predicted_profit:,.2f}"

if __name__ == "__main__":
    # Run the Flask application
    app.run(debug=True) # debug=True allows for automatic reloading on code changes
