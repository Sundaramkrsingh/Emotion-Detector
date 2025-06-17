import streamlit as st 
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Emotion Detector")
st.write("Enter a sentence and emotion will be detected")

user_input = st.text_area("Your Sentence", "")

if st.button("Predict"):
    if user_input.strip():
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        st.subheader("Predicted Emotion:")
        st.success(f"{predicted_label}")

        st.subheader("Confidence Scores:")
        for idx, score in enumerate(proba):
            st.write(f"{le.inverse_transform([idx])[0]}: {score:.2f}")
    else:
        st.warning("Please enter some text.")