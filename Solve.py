import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import openai
import streamlit as st

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems. Provide detailed solutions and direct numerical answers when possible. Do not respond to non-mathematical queries."},
                {"role": "user", "content": query}
            ],
            max_tokens=250
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Function to check if the query is math-related
def is_math_query(query):
    math_keywords = re.compile(
        r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation|solve|evaluate|simplify|factor|expand|differentiate|integrate|limit|function|graph|plot|expression|variable|constant|polynomial|quadratic|linear|exponential|logarithmic|trigonometric|complex|number|math|mathematics|multiply|times|what is|find|calculate|add|subtract|divide|result|total|sum|product|difference)\b', 
        re.IGNORECASE
    )
    math_symbols = re.compile(r'[+\-*/^=()]')

    if math_keywords.search(query) or math_symbols.search(query):
        return True
    return False

# Function to handle math queries
def handle_math_query(query):
    if not is_math_query(query):
        return "Please ask a valid mathematical question."

    # Check if the query is a simple arithmetic problem
    simple_arithmetic = re.match(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query)
    if simple_arithmetic:
        return generate_response(query)
    
    # Check if the query is an equation to solve
    equation_match = re.match(r'^\s*(.+?)\s*=\s*(.+?)\s*$', query)
    if equation_match:
        return generate_response(query)

    # For other types of math queries, use the OpenAI API
    return generate_response(query)


# Check if the query is math-related
def is_math_query(query):
    math_keywords = re.compile(
        r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation|solve|evaluate|simplify|factor|expand|differentiate|integrate|limit|function|graph|plot|expression|variable|constant|polynomial|quadratic|linear|exponential|logarithmic|trigonometric|complex|number|math|mathematics|multiply|times|what is|find|calculate|add|subtract|divide|result|total|sum|product|difference)\b', 
        re.IGNORECASE
    )
    math_symbols = re.compile(r'[+\-*/^=()]')

    return bool(math_keywords.search(query) or math_symbols.search(query))

# Unified plotting function for expressions (polynomials, general equations, trig functions)
def plot_expression(expression, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 1000)

    if isinstance(expression, list):  # Polynomial case
        y_vals = np.polyval(expression, x_vals)
        poly_expr = np.poly1d(expression)
        label = f'Polynomial: {poly_expr}'
    elif isinstance(expression, (str, sp.Expr)):  # General equation or trig function case
        x = sp.symbols('x')
        if isinstance(expression, str):
            expression = sp.sympify(expression)  # Convert string to sympy expression
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
# Streamlit UI setup
st.title("AI-Assisted Math Chatbot")

# First search bar: General Solution Section
st.header("General Math Solution")
user_input_solution = st.text_input("Enter your mathematical query for a solution", "")

if user_input_solution:
    if is_math_query(user_input_solution):
        response = generate_response(user_input_solution)
        st.write("### GPT-4 Response:")
        st.write(response)
    else:
        st.write("Please ask a valid mathematical question.")

# Second search bar: Plot Graph Section
st.header("Plot a Graph")
plot_type = st.radio("Select what to plot:", ["Polynomial", "General Function/Expression"])

if plot_type == "Polynomial":
    st.write("**Example input:** `2, -3, 1` for the polynomial `2x² - 3x + 1`")
    coeffs = st.text_input("Enter polynomial coefficients (comma-separated)", "")
    x_range = st.slider("Select x range", -10, 10, (-10, 10))
    if coeffs:
        coeff_list = list(map(float, coeffs.split(',')))
        plot_expression(coeff_list, x_range)

elif plot_type == "General Function/Expression":
    st.write("**Example inputs:** `sin(x)`, `x**2 + 4*x + 4`, `cos(x)`")
    expression = st.text_input("Enter the function/expression to plot (e.g., sin(x) or x**2 + 4*x + 4):", "")
    x_range = st.slider("Select x range", -10, 10, (-10, 10))
    if expression:
        plot_expression(expression, x_range)
