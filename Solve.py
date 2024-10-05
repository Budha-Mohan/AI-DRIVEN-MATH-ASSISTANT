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
                {"role": "system", "content": "You are a helpful assistant that solves mathematical problems. Provide detailed solutions and direct numerical answers when possible. Do not respond to non-mathematical queries."},
                {"role": "user", "content": query}
            ],
            max_tokens=300
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

def handle_math_query(query):
    if not is_math_query(query):
        return "Please ask a valid mathematical question."

    # Check if the query is about a constant
    constant_keywords = re.compile(r'\b(value of|what is|find|give me|calculate|determine)\b.*\b(pi|π|e|Euler\'s number|golden ratio|phi|sqrt\(2\)|sqrt\(3\))\b', re.IGNORECASE)

    # Log the query for debugging
    print(f"Received query: {query}")  # Debugging line

    if constant_keywords.search(query):
        # Let the OpenAI API handle the response for constants
        response = generate_response(query)
        return response
    
    # Check for simple arithmetic problem
    simple_arithmetic = re.match(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query)
    if simple_arithmetic:
        return generate_response(query)
    
    # Check if the query is an equation to solve
    equation_match = re.match(r'^\s*(.+?)\s*=\s*(.+?)\s*$', query)
    if equation_match:
        return generate_response(query)
        
    # For other types of math queries, use Sympy
    try:
        expr = sp.sympify(query)
        result = expr.evalf()
        return f"The result is: {result}"
    except Exception as e:
        return f"An error occurred while processing the expression: {e}"
    # For other types of math queries, use the OpenAI API
    return generate_response(query)
   

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
  
