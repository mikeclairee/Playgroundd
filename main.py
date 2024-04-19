from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = pd.read_csv("FinalPreference.csv")

# Initialize the model
X = data.drop(columns=["PreparationMethod", "Ingredients", "ServingTemperature", "Intensity", "CoffeeName", "ImageLabel"])
y = data["ImageLabel"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        # Get user inputs from the form
        age = request.form.get("age")
        gender = request.form.get("gender")
        preparation = request.form.get("preparation")
        ingredients = request.form.get("ingredients") 
        temperature = request.form.get("temperature")
        intensity = request.form.get("intensity")

        
        # Validate inputs
        try:
            age = int(age)
            gender = int(gender)
            preparation = str(preparation)
            ingredients = str(ingredients)
            temperature = str(temperature)
            intensity = str(intensity)
            # Validate other inputs if necessary
        except ValueError:
            return render_template("error.html", message="Invalid input! Please enter valid values.")

        # Make prediction based on user inputs
        predicted_value = model.predict([[age, gender]])
        predicted_image_label = data[data["ImageLabel"] == predicted_value[0]]["CoffeeName"].values[0]

        return render_template("index.html", predicted_coffee=predicted_value[0], image_label=predicted_image_label)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
