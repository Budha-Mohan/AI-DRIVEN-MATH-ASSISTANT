import streamlit as st
import openai
import sympy as sp
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_response(query):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that solves mathematical problems. Provide direct numerical answers when possible."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=150
#         )
#         return response.choices[0].message['content'].strip()
#     except Exception as e:
#         return f"An error occurred: {e}"

# Function to generate response using OpenAI API
def generate_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems. Provide detailed solutions and direct numerical answers when possible. Do not respond to non-mathematical queries."},
                {"role": "user", "content": query}
            ],
            max_tokens=150
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

# # Function to solve mathematical expressions directly
# def solve_expression(expression):
#     try:
#         result = sp.sympify(expression).evalf()
#         return result
#     except (sp.SympifyError, TypeError, ValueError):
#         return "Invalid mathematical expression."

# # Function to handle math queries
# def handle_math_query(query):
#     # Check if the query is a simple arithmetic problem
#     simple_arithmetic = re.match(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query)
#     if simple_arithmetic:
#         return solve_expression(query)
    
#     # Check if the query is an equation to solve
#     equation_match = re.match(r'^\s*(.+?)\s*=\s*(.+?)\s*$', query)
#     if equation_match:
#         lhs, rhs = equation_match.groups()
#         try:
#             x = sp.symbols('x')
#             solution = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
#             return solution
#         except (sp.SympifyError, TypeError, ValueError):
#             return "Invalid equation."

#     # For other types of math queries, use the OpenAI API
#     return generate_response(query)
    
# # Function to check if the expression is valid
# def is_valid_expression(expr):
#     try:
#         x, y = sp.symbols('x y')
#         sp.sympify(expr)
#         return True
#     except (sp.SympifyError, TypeError, ValueError):
#         return False

# # Function to plot the mathematical function based on user input
# def plot_function(expr):
#     # Generate x values
#     x_vals = np.linspace(-10, 10, 400)

#     # Check if the expression is valid
#     if not is_valid_expression(expr):
#         return "Cannot draw graph: The expression is invalid. Please enter a valid mathematical expression."

#     # Define the variable and the expression
#     x, y = sp.symbols('x y')
#     expr_sympy = sp.sympify(expr)  # Convert to SymPy expression

#     # Check if the expression is an implicit function (e.g., a circle)
#     if isinstance(expr_sympy, sp.Equality):  # Handle equations in the form of x^2 + y^2 = 25
#         lhs, rhs = expr_sympy.lhs, expr_sympy.rhs
#         if lhs.has(y):
#             # Parametric plotting for implicit functions
#             y_vals = []
#             for val in x_vals:
#                 try:
#                     # Solve for y in terms of x
#                     solutions = sp.solve(lhs.subs(x, val) - rhs, y)
#                     for sol in solutions:
#                         y_vals.append(float(sol.evalf()))  # Add all valid y values
#                 except Exception:
#                     y_vals.append(np.nan)  # Use NaN for invalid values
#             y_vals = np.array(y_vals)

#             # Plotting
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.plot(x_vals, y_vals, label=f'Implicit curve: {expr}', color='blue')
#             ax.set_title(f'Graph of the Implicit Function: {expr}')
#             ax.set_xlabel('x')
#             ax.set_ylabel('y')
#             ax.axhline(0, color='black', linewidth=0.5, ls='--')
#             ax.axvline(0, color='black', linewidth=0.5, ls='--')
#             ax.grid(color='gray', linestyle='--', linewidth=0.5)
#             ax.legend()
#             st.pyplot(fig)  # Use Streamlit to display the plot
#             return f"Here is the graph of the implicit function: {expr}"

#     # Generate y values for explicit functions
#     y_vals = []
#     for val in x_vals:
#         try:
#             y_val = expr_sympy.subs(x, val).evalf()
#             y_vals.append(float(y_val))
#         except (TypeError, ValueError):
#             y_vals.append(np.nan)  # Use NaN for invalid values

#     # Plotting explicit functions
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(x_vals, y_vals, label=f'y = {expr}', color='blue')
#     ax.set_title(f'Graph of the Function: y = {expr}')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.axhline(0, color='black', linewidth=0.5, ls='--')
#     ax.axvline(0, color='black', linewidth=0.5, ls='--')
#     ax.grid(color='gray', linestyle='--', linewidth=0.5)
#     ax.legend()
#     st.pyplot(fig)  # Use Streamlit to display the plot

#     return f"Here is the graph of the function: y = {expr}"

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
  
# Streamlit UI
st.title("Mathematics Chatbot")
st.sidebar.header("Options")

# User input for math query
query = st.text_input("Enter your math query:")
if st.button("Submit"):
    if query:
        response = handle_math_query(query)
        st.write("Response:", response)
    else:
        st.write("Please enter a valid query.")
