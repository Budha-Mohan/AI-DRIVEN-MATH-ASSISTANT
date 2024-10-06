import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import openai
import streamlit as st
from PIL import Image
import pytesseract

#OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#Reading functions
def generate_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems.Provide detailed solutions and direct numerical answers when possible.Do not respond to non-mathematical queries."},
                {"role": "user", "content": query}
            ],
            max_tokens=250
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"

# To check for math related query
def is_math_query(query):
    math_keywords = re.compile(
        r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation|solve|evaluate|simplify|equate|factor|expand|differentiate|integrate|limit|' \
'function|graph|plot|expression|variable|constant|pi|e|ln|polynomial|quadratic|linear|exponential|logarithmic|trigonometric|complex|number|math|mathematics|' \
'multiply|times|what is|find|calculate|add|sum|subtract|divide|result|total|differences|product)\b', 
        re.IGNORECASE)
    math_symbols = re.compile(r'[+\-*/^=()]')
    if math_keywords.search(query) or math_symbols.search(query):
        return True
    return False

# Function to handle math queries
def handle_math_query(query):
    if not is_math_query(query):
        return "Please ask a valid mathematical question."

    # To check if the query is ARITHMETIC PROBLEM to SOLVE
    simple_arithmetic = re.match(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query)
    if simple_arithmetic:
        return generate_response(query)
    
    # To check if the query is an EQUATIONS to SOLVE
    equation_match = re.match(r'^\s*(.+?)\s*=\s*(.+?)\s*$', query)
    if equation_match:
        return generate_response(query)

    # For general maths use the OpenAI API
    return generate_response(query)

# Functions for plotting: polynomials, general equations, trig functions
def plot_expression(expression, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 1000)

    if isinstance(expression, list):                  # Polynomial case
        y_vals = np.polyval(expression, x_vals)
        poly_expr = np.poly1d(expression)
        label = f'Polynomial: {poly_expr}'
    elif isinstance(expression, (str, sp.Expr)):      # General equation or trig function case
        x = sp.symbols('x')
        if isinstance(expression, str):
            expression = sp.sympify(expression)        # Convert string to sympy expression
        func_lambdified = sp.lambdify(x, expression, 'numpy')
        y_vals = func_lambdified(x_vals)
        label = f'y = {expression}'
    else:
        st.error("Invalid input for expression.")
        return

    plt.plot(x_vals, y_vals, label=label)
    plt.title('Plot of the Expression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Function to extract text using OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Streamlit UI setup
st.title("AI Driven Math Assistant", anchor='middle')

# Create some space between the title and the columns
st.write("")
st.write("")

# Create two columns with more spacing
col1, col2 = st.columns([1, 1], gap="large")

# First Column: General Solution Section
with col1:
    st.subheader("General Math Solution")
    user_input_solution = st.text_input("Enter your mathematical query for a solution", "")

    if user_input_solution:
        if is_math_query(user_input_solution):
            response = generate_response(user_input_solution)
            st.write("Solution:")
            st.write(response)
        else:
            st.write("Please ask a valid mathematical question.")

# Second Column: Plot Graph Section
with col2:
    st.subheader("Plot a Graph")
    plot_type = st.radio("Select what to plot:", ["Polynomial", "General Function"])

    if plot_type == "Polynomial":
        st.write("**Example input:** `2, -3, 1` for the polynomial `2x² - 3x + 1`")
        coeffs = st.text_input("Enter polynomial coefficients (comma-separated)", "")
        x_range = st.slider("Select x range", -10, 10, (-10, 10))
        if coeffs:
            coeff_list = list(map(float, coeffs.split(',')))
            plot_expression(coeff_list, x_range)

    elif plot_type == "General Function":
        st.write("**Example inputs:** `sin(x)`, `x**2 + 4*x + 4`, `cos(x)`")
        expression = st.text_input("Enter the function/expression to plot (e.g., sin(x) or x**2 + 4*x + 4):", "")
        x_range = st.slider("Select x range", -10, 10, (-10, 10))
        if expression:
            plot_expression(expression, x_range)

# OCR Section remains below
st.write("  ")  # Add some space
st.header("Upload Your Problems")
uploaded_file = st.file_uploader("Upload an image containing a math problem (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    extracted_text = extract_text_from_image(image)
    st.write("Extracted Text:")
    st.write(extracted_text)
    
    if is_math_query(extracted_text):
        st.write("Detected a math-related query. Solving...")
        response = generate_response(extracted_text)
        st.write("Solution:")
        st.write(response)
    else:
        st.write("No math-related content detected. Please upload a math-related image.")

