# Group7_PlagiarismChecker
# EthicalCheck - Plagiarism Checker

## Overview

EthicalCheck is a plagiarism-checking tool developed in Python. It provides local and online options for users to check for plagiarism in text or code. The tool employs various techniques, including cosine similarity and FuzzyWuzzy, to assess similarity between documents.

## Features

- **Text Input Comparison:**
  - Input two pieces of text or code.
  - Calculate similarity scores using cosine similarity and FuzzyWuzzy.
  - Highlight plagiarized sections in the displayed texts.

- **File Upload Comparison:**
  - Upload two files for comparison.
  - Calculate similarity scores using cosine similarity and FuzzyWuzzy.
  - Display whether the content is flagged as plagiarized or not.

- **Multiple File Upload Comparison:**
  - Upload multiple files for batch comparison.
  - Calculate similarity scores among all pairs of documents.
  - Visualize results in a table and histogram.

- **Check Plagiarism Online:**
  - Enter text to check for plagiarism online.
  - Fetch search results from Google Custom Search API.
  - Display similarity percentages.

## Prerequisites

- Python 3.x
- Install required dependencies using `requirements.txt`.

  ```bash
  pip install -r requirements.txt

# Clone the repository
git clone https://github.com/cliffnkansah/Group7_PlagiarismChecker.git
cd Group7_PlagiarismChecker

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run Checker.py

## Online Deployment

### 1. Choose a Cloud Service
Choose a cloud service provider for deployment, such as Heroku, AWS, or Google Cloud.

### 2. Configure API Keys and Permissions
Ensure that you have the necessary configurations for API keys and access permissions.

### 3. Update Streamlit Configuration
Update the Streamlit app's configuration settings as needed for deployment.

### 4. Deploy the Application
Use the deployment mechanisms of your chosen cloud service to deploy the application.

### 5. Access the Application
Once deployed, access the provided URL to use the application.

## Acknowledgments

The project uses the following technologies:

- **Streamlit**: Used for the web application interface.
- **NLTK, BeautifulSoup, and other Python libraries**: Employed for text processing and similarity calculations.

## How to Run Locally

Follow these steps to run the application locally for development or testing purposes:
