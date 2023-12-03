import itertools
import os
import shutil
import zipfile

import chardet
import docx2txt
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
import streamlit as st
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from GoogleChecker import (calculate_cosine_similarity, check_plagiarism,
                           highlight_plagiarized, plot_plagiarism_graph,
                           preprocess_text)

st.set_page_config(page_title="EthicalCheck", page_icon="search.png")

# Custom HTML and CSS code for highlighting plagiarized parts
highlight_style = """
<style>
    .plagiarized {
        background-color: #000080; /* Light red for highlighting */
    }
</style>
"""

# Streamlit app
st.title('EthicalCheck - Plagiarism Checker')

instructions = st.markdown("## Instructions:")
instructions_text = """
- Choose an option from the sidebar to perform different plagiarism checking tasks.
- For 'Text Input', enter two pieces of text and click 'Calculate Similarity'.
- For 'Upload Files', upload two files and click 'Compare'.
- For 'Upload Multiple Files', upload multiple files and click 'Compare Files'.
- For 'Check Plagiarism Online', enter text and click 'Check Plagiarism Online'.
"""

instructions.markdown(instructions_text)

empty_space = st.empty()

st.sidebar.title("Plagiarism Checker Options")
st.sidebar.markdown(
    "Choose an option from the radio button below to perform different plagiarism checking tasks."
)

option = st.sidebar.radio("Choose an option:", ("Text Input", "Upload Files", "Upload Multiple Files","Check Plagiarism Online"))

if option == "Text Input":
    text1 = st.text_area('Enter the first text or code:')
    text2 = st.text_area('Enter the second text or code:')
    
    if st.button('Calculate Similarity'):

        instructions.markdown("")

        if text1 and text2:
            similarity_score = calculate_cosine_similarity(text1, text2)
            similarity_percentage = similarity_score * 100

            fuzz_score = fuzz.token_set_ratio(text1, text2)  # Calculate FuzzyWuzzy score

            if similarity_score >= 0.3 or fuzz_score >= 60:  # Adjust the thresholds accordingly
                st.write(f'Similarity score: {similarity_percentage:.2f}%')
                st.write(f"Similarity score (paraphrased): {fuzz_score}% - Flagged as plagiarised.")
            else:
                st.write(f'Similarity score: {similarity_percentage:.2f}%')
                st.write(f"Similarity score (paraphrased): {fuzz_score}% - Not plagiarised.")

            # Highlight plagiarized parts in the texts
            highlighted_text1 = highlight_plagiarized(text1, text2)
            highlighted_text2 = highlight_plagiarized(text2, text1)

            st.markdown('<div class="plagiarized">' + highlighted_text1 + '</div>', unsafe_allow_html=True)
            st.markdown('<div class="plagiarized">' + highlighted_text2 + '</div>', unsafe_allow_html=True)
                
        else:
            st.warning('Please enter both texts or code to calculate similarity.')

elif option == "Upload Files":

    file1 = st.file_uploader("Upload File 1", type=["txt","pdf","docx","py","java","php","css"])
    file2 = st.file_uploader("Upload File 2", type=["txt","pdf","docx","py","java","php","css"])

    if file1 and file2:
        clicked = st.button('Compare')

        if clicked:
            text1 = file1.read().decode("utf-8") if file1.type in ["txt", "pdf", "docx"] else file1.read().decode("latin1")
            text2 = file2.read().decode("utf-8") if file2.type in ["txt", "pdf", "docx"] else file2.read().decode("latin1")

            score = calculate_cosine_similarity(text1, text2)
            fuzz_score = fuzz.token_set_ratio(text1, text2)  # Calculate fuzzywuzzy score

            threshold = 0.3
            if score >= threshold or fuzz_score >= 60:  # Set a threshold for fuzzywuzzy score
                st.write(f"Similarity score: {score:.2%}")
                st.write(f"Similarity score (paraphrased): {fuzz_score}% - Flagged as plagiarised.")
            else:
                st.write(f"Similarity score: {score:.2%}")
                st.write(f"Similarity score (paraphrased): {fuzz_score}% - Not plagiarised.")
            
        

elif option == "Upload Multiple Files":
    uploaded_files = st.file_uploader("Upload multiple files (.docx)", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Number of uploaded files: {len(uploaded_files)}")
        file_contents = []
        file_names = []

        for uploaded_file in uploaded_files:
            # Read file as bytes
            content = uploaded_file.getvalue()
            # Detect encoding using chardet
            detected_encoding = chardet.detect(content)['encoding']
            # If chardet couldn't detect an encoding, set a default encoding
            encoding = detected_encoding if detected_encoding else 'utf-8'
            # Decode using the detected encoding

            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                try:
                    # Read the docx content as text
                    docx_text = docx2txt.process(uploaded_file)
                    file_contents.append(docx_text)
                    file_names.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
            
        if st.button('Compare Files'):
            instructions.markdown("")
            if file_contents:  # Ensure at least two files are uploaded for comparison
                similarity_scores = []
                similarity_table = []

                for i in range(len(file_contents)):
                    for j in range(i + 1, len(file_contents)):
                        similarity_score = calculate_cosine_similarity(file_contents[i], file_contents[j])
                        
                        # Use fuzzywuzzy for improved similarity check
                        fuzz_score = fuzz.token_set_ratio(file_contents[i], file_contents[j])
                        
                        similarity_scores.append(similarity_score)
                        similarity_table.append([file_names[i], file_names[j], similarity_score, fuzz_score])

                if similarity_table:
                    # Adjust the DataFrame to only display percentages for the plagiarism column
                    similarity_table_percentages = [
                        [doc1, doc2, f"{score * 100:.2f}%" if isinstance(score, float) else None, f"{fuzz_score}%" if isinstance(fuzz_score, int) else None]
                        for doc1, doc2, score, fuzz_score in similarity_table
                    ]

                    # Display similarity table
                    df = pd.DataFrame(similarity_table_percentages, columns=['Document 1', 'Document 2', 'Similarity score', 'Similarity score (paraphrased)'])
                    st.write('Similarity scores among the documents:')
                    st.write(df)

                    # Plotting plagiarism graph
                    plot_plagiarism_graph(similarity_scores)

                    # Download button for CSV report
                    csv_file = df.to_csv(index=False).encode()
                    st.download_button(
                        label="Download CSV report",
                        data=csv_file,
                        file_name='similarity_report.csv',
                        mime='text/csv'
                    )
                else:
                    st.write('No similarity scores calculated.')

elif  option == "Check Plagiarism Online":
    text_to_check = st.text_area("Enter the text to check for plagiarism", "")

    if st.button("Check Plagiarism Online"):
        instructions.markdown("")
        if text_to_check:
            original_text = preprocess_text(text_to_check)

            results = check_plagiarism(text_to_check, original_text)

            # Display the search results with similarity percentages
            if results:
                result_df = pd.DataFrame(results)
                
                # Format the 'similarity' and 'semantic_similarity' columns to show percentages with two decimal places
                result_df['similarity'] = result_df['similarity'].map('{:.2f}%'.format)
                result_df['semantic_similarity'] = result_df['semantic_similarity'].map('{:.2f}%'.format)
                
                # Display the results in a table
                st.dataframe(result_df, height=len(result_df)*30)

            else:
                st.warning("No results found.")

        else:
            st.warning("Please enter text to check for plagiarism.")


