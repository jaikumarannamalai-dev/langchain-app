import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="Healthy Breakfast Ideas",
    page_icon="🍳",
    layout="centered",
)

st.title("🍳 Healthy Breakfast Ideas")
st.write("Choose your favorite cuisine and discover five healthy breakfast ideas!")

# Select breakfast cuisine
cuisine = st.selectbox(
    "Choose your breakfast cuisine:",
    [
        "Indian Breakfast",
        "North Indian Breakfast",
        "South Indian Breakfast",
        "American Breakfast",
    ],

)

# Read the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is missing. "
        "Please add it to your .env file."
    )
    st.stop()

# Initialize the LangChain ChatOpenAI model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)

# Generate ideas when the button is clicked

if st.button("Generate Breakfast Ideas", type="primary"):

    if cuisine == "Indian Breakfast":
        prompt = """
        Suggest exactly 5 healthy Indian breakfast ideas.
        Include dishes from different Indian regions, such as idli,
        dosa, poha, upma, pesarattu, dhokla, and vegetable paratha.
        Prefer vegetables, lentils, whole grains, and protein-rich foods.
        """

    elif cuisine == "North Indian Breakfast":
        prompt = """
        Suggest exactly 5 healthy North Indian breakfast ideas.
        Consider besan chilla, vegetable poha, dal cheela,
        vegetable daliya, and paratha served with yogurt.
        Prefer whole grains, vegetables, lentils, and protein-rich foods.
        """

    elif cuisine == "South Indian Breakfast":
        prompt = """
        Suggest exactly 5 healthy South Indian breakfast ideas.
        Consider idli with sambar, pesarattu, ragi dosa,
        vegetable uttapam, and vegetable upma.
        Prefer lentils, whole grains, vegetables, and protein-rich foods.
        """

    else:
        prompt = """
        Suggest exactly 5 healthy American breakfast ideas.
        Consider oatmeal with berries, vegetable omelets,
        Greek yogurt bowls, whole-grain pancakes, and avocado toast.
        Prefer whole grains, fruits, vegetables, protein, and healthy fats.
        """

       

    prompt += """
    NUTRITION REQUIREMENTS:
    - Generate exactly 5 distinct breakfast ideas.
    - Include the dish name and a short description.
    - Include a typical serving size.
    - Estimate calories per serving in kcal.
    - Estimate protein per serving in grams.
    - Estimate carbohydrates per serving in grams.
    - Estimate fat per serving in grams.
    - Label all nutritional values as estimates.

    OUTPUT FORMAT:
    1. Dish Name — Description | Serving: 1 plate | Calories: 300 kcal | Protein: 12 g | Carbs: 40 g | Fat: 8 g

    The values above are examples of formatting only.
    Estimate values for each actual dish.

    - Number the ideas from 1 to 5.
    - Write each idea on one line.
    - Do not include an introduction or conclusion.
"""
   

    try:
        with st.spinner("Preparing your breakfast ideas..."):
            response = model.invoke(prompt)

        # Display the generated breakfast ideas
        st.subheader(f"🍳 Your 5 Healthy {cuisine} Ideas")
        st.markdown(response.content)

        # Initialize nutrition totals
        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0

        # Extract nutrition values from the model response
        import re

        for line in response.content.splitlines():
            calories_match = re.search(
                r"Calories:\s*([\d.]+)\s*kcal", line, re.IGNORECASE
            )
            protein_match = re.search(
                r"Protein:\s*([\d.]+)\s*g", line, re.IGNORECASE
            )
            carbs_match = re.search(
                r"Carbs:\s*([\d.]+)\s*g", line, re.IGNORECASE
            )
            fat_match = re.search(
                r"Fat:\s*([\d.]+)\s*g", line, re.IGNORECASE
            )

            if calories_match:
                total_calories += float(calories_match.group(1))

            if protein_match:
                total_protein += float(protein_match.group(1))

            if carbs_match:
                total_carbs += float(carbs_match.group(1))

            if fat_match:
                total_fat += float(fat_match.group(1))

        # Nutrition dashboard
        st.subheader("📊 Nutrition Dashboard")
        st.caption(
            "Estimated totals for all five breakfast ideas, "
            "assuming one serving of each."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Calories", f"{total_calories:.0f} kcal")

        with col2:
            st.metric("Total Protein", f"{total_protein:.1f} g")

        with col3:
            st.metric("Total Carbs", f"{total_carbs:.1f} g")

        with col4:
            st.metric("Total Fat", f"{total_fat:.1f} g")

        if total_calories == 0 and total_protein == 0:
            st.warning(
                "Nutrition totals could not be extracted. "
                "Check that the AI response follows the requested format."
            )

    except Exception as e:
        st.error("Unable to generate breakfast ideas.")
        st.exception(e)