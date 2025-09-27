Universal Product Intelligence Engine
This is a complete, end-to-end system for analyzing customer feedback and product reputation from the general internet. Instead of relying on a single, fragile e-commerce site scraper, this application performs a general web search for any given product, aggregates reviews and opinions from multiple sources, and uses a powerful AI pipeline to deliver a comprehensive analysis.

This final version features a dynamic, NLP-based topic extraction engine that can understand and analyze reviews for any product category, from electronics to carrots.

🚀 Final Setup and Running Instructions
Follow these steps to get the application running.

Step 1: Install All Dependencies
First, you need to install all the required Python libraries, including the new NLP toolkit.

Open your terminal in the main project folder.

Run this command:

pip install -r requirements.txt

Step 2: Download NLP Data (New, One-Time Step)
Our new, smarter AI needs some data from the NLTK library to understand grammar. You only need to do this once.

Open your terminal and start the Python interpreter by typing python.

Run the following two commands inside the Python interpreter:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

A small window might pop up. Once the downloads are complete, you can close it and type exit() to leave the Python interpreter.

Step 3: Run the Local Data Setup (Optional but Recommended)
This step analyzes your local amazon_reviews.csv file to power the local dataset features and the suggestion bot.

In your terminal, from the main project folder, run:

python backend/embeddings.py

Step 4: Start the Backend Server
This starts the main "brain" of your application.

In the same terminal, run:

flask --app backend/app run
