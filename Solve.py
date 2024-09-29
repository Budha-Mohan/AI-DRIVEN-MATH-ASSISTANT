import streamlit as st
import openai
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import re
import os

import os  # Import the os module

openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to preprocess and correct natural language queries
def preprocess_query(query):
    response = openai.ChatCompletion.create( # Use openai.ChatCompletion instead of openai.Completion
        model="gpt-4",
        messages=[{"role": "user", "content": f"Correct and convert the following natural language query into a mathematical expression: {query}"}], # Provide prompt as a message
        max_tokens=50
    )
    corrected_query = response['choices'][0]['message']['content'].strip() # Extract content from response
    return corrected_query

def generate_response(query):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",  # You can also use "gpt-3.5-turbo" or other available models
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems."},
                {"role": "user", "content": query},
                {"role": "system", "content": "You are an assistant that converts natural language math queries into symbolic mathematical expressions."},
                {"role": "user", "content": f"Convert this into a mathematical expression: {query}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Function to check if the expression is valid
def is_valid_expression(expr):
    try:
        # Define the variable
        x = sp.symbols('x')
        # Try to convert the expression to a SymPy object
        sp.sympify(expr)
        return True
    except (sp.SympifyError, TypeError, ValueError):
        return False

# Function to plot the mathematical function based on user input
def plot_function(expr):
    # Generate x values
    x_vals = np.linspace(-10, 10, 400)

    # Check if the expression is valid
    if not is_valid_expression(expr):
        return "Cannot draw graph: The expression is invalid. Please enter a valid mathematical expression."

    # Define the variable and the expression
    x = sp.symbols('x')
    expr_sympy = sp.sympify(expr)  # Convert to SymPy expression

    # Generate y values
    y_vals = []
    for val in x_vals:
        try:
            y_val = expr_sympy.subs(x, val).evalf()
            y_vals.append(float(y_val))
        except (TypeError, ValueError):
            y_vals.append(np.nan)  # Use NaN for invalid values

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, label=f'y = {expr}', color='blue')
    plt.title(f'Graph of the Function: y = {expr}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    return f"Here is the graph of the function: y = {expr}"

# Get all known functions from SymPy's library
sympy_functions = {name for name in dir(sp.functions) if callable(getattr(sp.functions, name))}

# Update the chatbot function to handle graph requests
def chatbot(query):
    # Keywords to identify graph plotting requests
    graph_keywords = ["draw graph", "plot", "show graph", "visualize", "graph", "draw"]

    # Check if any of the graph keywords are in the query
    if any(keyword in query.lower() for keyword in graph_keywords):
        query_lower = query.lower()

        # Check for the presence of "for" or "of", otherwise use the last part of the query
        if "for" in query_lower:
            expr = query.split("for")[-1].strip()
        elif "of" in query_lower:
            expr = query.split("of")[-1].strip()
        else:
            expr = query.split()[-1].strip()  # Extract the last word if no "for" or "of" is found

        # Clean and format the expression (like "sinx" to "sin(x)")
        expr = expr.replace(" ", "")

        # Automatically detect any known SymPy functions in the expression and add parentheses if needed
        for func in sympy_functions:
            # Look for functions followed by a variable with no parentheses, e.g., "sinx"
            pattern = rf'(?<!\w)({func})(?!\()'  # Match the function if not followed by "("
            expr = re.sub(pattern, rf'\1(', expr)  # Add the opening parenthesis after the function
            # Use a non-lookbehind approach to add closing parenthesis
            if '(' in expr and ')' not in expr:
                expr += ')'
        # Insert '*' where necessary between numbers and variables (like in "2x" -> "2*x")
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)  # Add '*' between number and variable


        try:
            sp.sympify(expr)  # Validate the expression
        except sp.SympifyError:
            return f"Cannot understand the expression: {expr}. Please enter a valid one."


        return plot_function(expr)  # Call the function to plot the given mathematical expression

    # For non-graph related queries, generate response using OpenAI API
    response = generate_response(query)  # Use OpenAI for math responses
    return response

# Assuming your OpenAI API response function is already defined
def is_math_query(text):
    math_keywords = re.compile(r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation)\b', re.IGNORECASE)
    math_symbols = re.compile(r'[+\-*/^=()]')

    if math_keywords.search(text) or math_symbols.search(text):
        return True
    return False

# Function to process the image and extract text
def process_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    if is_math_query(extracted_text):
        return extracted_text, True
    else:
        return extracted_text, False

# Function to solve the problem if it is math-related
def solve_math_problem(image_path):
    extracted_text, is_math = process_image(image_path)

    if is_math:
        print("Math problem detected:", extracted_text)
        # Call the OpenAI API to solve the problem
        response = generate_response(extracted_text)  # You already have this function in your chatbot
        return response
    else:
        return "This doesn't seem to be a math question. Please ask math-related questions."

# Streamlit UI
st.title("Mathematics Chatbot")
st.sidebar.header("Options")

# User input for math query
query = st.text_input("Enter your math query:")
if st.button("Submit"):
    if query:
        response = generate_response(query)
        st.write("Response:", response)
    else:
        st.write("Please enter a valid query.")

# Image upload for math problem
uploaded_file = st.file_uploader("Upload an image of a math problem", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    solution = solve_math_problem(image_path)
    st.write("Solution:", solution)

# Test Queries Section
st.sidebar.header("Test Queries")
test_query = st.sidebar.text_input("Enter a test query:")
if st.sidebar.button("Run Test"):
    if test_query:
        test_response = generate_response(test_query)
        st.sidebar.write("Test Response:", test_response)
    else:
        st.sidebar.write("Please enter a valid test query.")

