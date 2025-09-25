End-to-End Customer Feedback Analysis System
This project is a complete, production-ready web application for analyzing, categorizing, and generating insights from customer feedback. It leverages a modern stack of open-source NLP and GenAI tools to provide a comprehensive dashboard for product analysis, comparison, and AI-powered recommendations.

🔹 Core Features
Sentiment Analysis: Automatically classifies reviews into Positive or Negative categories.

AI-Powered Summarization: Uses a Retrieval-Augmented Generation (RAG) pipeline with a T5 model to generate concise, accurate summaries of product feedback.

Interactive Dashboard: A sleek, modern UI built with HTML, CSS, and Chart.js to visualize sentiment distribution and other key metrics.

Product Comparison: A dedicated view to compare two products side-by-side, complete with an AI-generated comparative analysis.

Suggestion Bot: An intelligent chat assistant that uses the RAG pipeline to answer user queries (e.g., "Which product is more durable?") based on actual review data.

Persistent Storage: Uses SQLite to store all processed reviews and analysis results, ensuring fast load times and data persistence.

🔹 Technical Stack
Backend: Python, Flask

Frontend: HTML, CSS, JavaScript (with Chart.js)

Database: SQLite

AI & NLP:

sentence-transformers: For generating high-quality text embeddings.

faiss-cpu: For efficient similarity search on embeddings.

transformers: For sentiment analysis and generative AI (T5).

torch: The deep learning framework.

nltk: For text preprocessing.

🔹 Setup and Installation
Prerequisites: Python 3.9+ and pip.

Set Up a Virtual Environment (Highly Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
Create a requirements.txt file with the contents from this project and run:

pip install -r requirements.txt

Download NLTK Data:
Run the following command in your terminal. The preprocessing.py script will handle the download if needed.

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

Get the Dataset:

Download the Amazon Review Dataset from Kaggle: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

From the downloaded archive, take either train.ft.txt.bz2 or test.ft.txt.bz2.

Rename it to amazon_reviews.csv and place it inside the data/ folder.

🔹 Running the Application
First-Time Setup (Process Data):
This is a crucial one-time step. Run the embeddings.py script from the main project folder. It will read the raw data, perform all cleaning and AI analysis, and create the feedback.db database and review_index.faiss search file.

python backend/embeddings.py

Note: This can take several minutes depending on your computer.

Start the Backend Server:
Run the Flask API server from the main project directory.

flask --app backend/app run

The server will start, typically on http://127.0.0.1:5000.

Launch the Frontend:
Open the frontend/index.html file in your web browser. The application will automatically connect to the running backend.