import pickle
import streamlit as st
import urllib.request
from sklearn.linear_model import LogisticRegression


# Raw URL to the model
url = "https://raw.githubusercontent.com/umeshpardeshi9545/titanic-survival-streamlit/main/Titanic_model.pkl"

# Load the model
try:
    with urllib.request.urlopen(url) as response:
        model_data = response.read()  # Read bytes
        model = pickle.loads(model_data)  # Load model from bytes
except Exception as e:
    model = None
    st.error(f"Error loading the model: {e}")

def main():
    st.title("Titanic Survival Prediction ")
    
    # Input variables
    Pclass = st.text_input("Passenger class -(1,2,3) : ")
    Age = st.text_input("Age")
    SibSp = st.text_input(" Number of Siblings/Spouses of Passenger (0-5) ")
    Parch = st.text_input("Number of Parents/Children Aboard (0-6)")
    Fare = st.text_input("Fare")
    Sex_male = st.text_input("Gender (Male-1, Female-0)")
    Embarked_Q = st.text_input("Boarded at Queenstons (Yes-1,No-0)")
    Embarked_S = st.text_input("Boarded at Southampton (Yes-1,No-0)")

    if st.button("Predict"):
        if model is None:
            st.error("Model could not be loaded. Please check the file URL or format.")
            return

        try:
            # Convert inputs to floats
            inputs = [float(Pclass), float(Age), float(SibSp), float(Parch), float(Fare), 
                      float(Sex_male), float(Embarked_Q), float(Embarked_S)]
            
            # Make prediction
            prediction = model.predict([inputs])
            st.success(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
        except ValueError:
            st.error("Please enter valid numeric inputs for all fields.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
