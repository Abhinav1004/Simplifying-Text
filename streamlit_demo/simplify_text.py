import streamlit as st

st.title("Text Simplification Tool")


# Function to simplify text (you can replace this with a more complex model or logic)
def simplify_text(input_text):
    # Example logic: Convert text to lowercase and replace complex words (dummy simplification)
    simplified_text = input_text.lower().replace("difficult", "easy").replace("complex", "simple")
    return simplified_text


def main():
    # Text input area
    user_input = st.text_area("Enter Text", placeholder="Type your text here...")

    # Button to process text
    simplify_btn = st.button("Simplify Text")

    if simplify_btn:
        if not user_input.strip():
            st.error("Invalid input, please enter some text.")
        else:
            with st.spinner('Simplifying text...'):
                # Simplify the input text
                simplified_output = simplify_text(user_input)

                # Display the output
                st.success("Simplified Text:")
                st.write(simplified_output)


if __name__ == '__main__':
    main()
