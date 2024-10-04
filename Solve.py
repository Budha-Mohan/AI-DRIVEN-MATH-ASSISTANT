import streamlit as st
import openai
import sympy as sp
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Set up your OpenAI API key
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

# Function to handle math queries
def handle_math_query(query):
    # Check if the query is a simple arithmetic problem
    simple_arithmetic = re.match(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query)
    if simple_arithmetic:
        return solve_expression(query)
    
    # Check if the query is an equation to solve
    equation_match = re.match(r'^\s*(.+?)\s*=\s*(.+?)\s*$', query)
    if equation_match:
        lhs, rhs = equation_match.groups()
        try:
            x = sp.symbols('x')
            solution = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
            return solution
        except (sp.SympifyError, TypeError, ValueError):
            return "Invalid equation."

    # For other types of math queries, use the OpenAI API
    return generate_response(query)

# Function to check if the expression is valid
def is_valid_expression(expr):
    try:
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
    st.pyplot(plt)  # Use Streamlit to display the plot

    return f"Here is the graph of the function: y = {expr}"

# Streamlit UI
st.title("Mathematics Chatbot")
st.sidebar.header("Options")

# User input for math query
query = st.text_input("Enter your math query:")
if st.button("Submit"):
    if query:
        if "plot" in query.lower():
            expr = query.split()[-1].strip()  # Extract the last word as the expression
            response = plot_function(expr)
            st.write(response)
        else:
            response = handle_math_query(query)
            st.write("Response:", response)
    else:
        st.write("Please enter a valid query.")
  
