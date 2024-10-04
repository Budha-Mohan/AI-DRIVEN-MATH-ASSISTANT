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

def generate_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems. Provide direct numerical answers when possible."},
                {"role": "user", "content": query}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Function to solve mathematical expressions directly
def solve_expression(expression):
    try:
        result = sp.sympify(expression).evalf()
        return result
    except (sp.SympifyError, TypeError, ValueError):
        return "Invalid mathematical expression."

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
    if not is_valid_expression(expr):
        return "Invalid mathematical expression for plotting."

    # Generate x values
    x_vals = np.linspace(-10, 10, 400)

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
    st.pyplot(plt)  # Use Streamlit's pyplot function to display the plot
    plt.clf()  # Clear the current plot for the next one

    return f"Here is the graph of the function: y = {expr}"

# Update the chatbot function to handle graph requests
def chatbot(query):
    # Keywords to identify graph plotting requests
    graph_keywords = ["draw graph", "plot", "show graph", "visualize", "graph", "draw"]

    # Check if any of the graph keywords are in the query
    if any(keyword in query.lower() for keyword in graph_keywords):
        # Extract expression from the query
        expr = re.sub(r"[^0-9a-zA-Z\+\-\*\/\^\(\)\s]", "", query)  # Keep valid math symbols

        return plot_function(expr)  # Call the function to plot the given mathematical expression

    # For non-graph related queries, generate response using OpenAI API
    response = generate_response(query)  # Use OpenAI for math responses
    return response
# Streamlit UI
st.title("Mathematics Chatbot")
st.sidebar.header("Options")

# User input for math query
query = st.text_input("Enter your math query:")
if st.button("Submit"):
    if query:
        response = chatbot(query)  # Call the chatbot function
        st.write("Response:", response)
    else:
        st.write("Please enter a valid query.")
