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
    math_keywords = re.compile(r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation|solve|evaluate|simplify|factor|expand|differentiate|integrate|limit|function|graph|plot|expression|variable|constant|polynomial|quadratic|linear|exponential|logarithmic|trigonometric|complex|number|math|mathematics)\b', re.IGNORECASE)
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
    math_keywords = re.compile(r'\b(algebra|calculus|geometry|integral|derivative|matrix|equation|solve|evaluate|simplify|factor|expand|differentiate|integrate|limit|function|graph|plot|expression|variable|constant|polynomial|quadratic|linear|exponential|logarithmic|trigonometric|complex|number|math|mathematics)\b', re.IGNORECASE)
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
st.write("Enter a mathematical query or ask to plot a function or expression.")

# Input form for query
user_input = st.text_input("Your Query", "")

# Main response logic
if user_input:
    if is_math_query(user_input):
        response = generate_response(user_input)
        st.write("### GPT-4 Response:")
        st.write(response)
        
        # Check for plot requests in user query
        if "plot" in user_input.lower() or "graph" in user_input.lower():
            plot_type = st.selectbox("Select what to plot:", ["Polynomial", "General Function/Expression"])
            
            if plot_type == "Polynomial":
                coeffs = st.text_input("Enter coefficients (comma-separated)", "")
                x_range = st.slider("Select x range", -10, 10, (-10, 10))
                if coeffs:
                    coeff_list = list(map(float, coeffs.split(',')))
                    plot_expression(coeff_list, x_range)
            elif plot_type == "General Function/Expression":
                expression = st.text_input("Enter the function/expression to plot (e.g., sin(x) or x**2 + 4*x + 4):", "")
                x_range = st.slider("Select x range", -10, 10, (-10, 10))
                if expression:
                    plot_expression(expression, x_range)

    else:
        st.write("Please ask a valid mathematical question.")



# # Streamlit UI
# st.title("Mathematics Chatbot")
# st.sidebar.header("Options")

# # User input for math query
# query = st.text_input("Enter your math query:")
# if st.button("Submit"):
#     if query:
#         if "plot" in query.lower():
#             expr = query.split()[-1].strip()  # Extract the last word as the expression
#             response = plot_function(expr)
#             st.write(response)
#         else:
#             response = handle_math_query(query)
#             st.write("Response:", response)
#     else:
#         st.write("Please enter a valid query.")
