# AI Assistant Web Application

A Flask-based web application that provides an AI-powered chat interface with various capabilities including text analysis, data visualization, and more.

## Features

- User authentication (login/signup)
- Real-time chat interface
- Conversation history
- Code block formatting
- Special commands for various functionalities:
  - `/news` - Get latest news
  - `/stock` - Get stock data
  - `/analyze` - Analyze text
  - `/wolfram` - Query Wolfram Alpha

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

5. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

6. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

1. Create a `.env` file in the project root directory with the following variables:
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
OPENWEATHERMAP_API_KEY=your-api-key
WOLFRAM_ALPHA_APP_ID=your-app-id
```

## Running the Application

1. Make sure your virtual environment is activated

2. Run the Flask application:
```bash
flask run
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Create a new account or login with existing credentials
2. Start chatting with the AI assistant
3. Use special commands for additional functionality:
   - `/news <topic>` - Get latest news about a topic
   - `/stock <symbol>` - Get stock data for a symbol
   - `/analyze <text>` - Analyze text sentiment and entities
   - `/wolfram <query>` - Get computational knowledge results

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── templates/         # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   └── chat.html
└── static/           # Static files (CSS, JS, images)
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 