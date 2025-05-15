from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import json
import datetime
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import wikipedia
import nltk
from nltk.tokenize import sent_tokenize
import re
import hashlib
import secrets
from pathlib import Path
import markdown
import pygments
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import tiktoken
from typing import List, Dict, Any
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import yfinance as yf
import wolframalpha
import wikipediaapi
import spacy
from textblob import TextBlob
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.figure_factory as ff

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    salt = db.Column(db.String(32), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    preferences = db.relationship('UserPreference', backref='user', uselist=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    theme = db.Column(db.String(20), default='light')
    language = db.Column(db.String(10), default='en')
    notification_settings = db.Column(db.String(200), default='{}')

# Helper Functions
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return key.hex(), salt

def verify_password(stored_password, stored_salt, provided_password):
    key, _ = hash_password(provided_password, stored_salt)
    return key == stored_password

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and verify_password(user.password_hash, user.salt, password):
            login_user(user)
            return redirect(url_for('chat'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        password_hash, salt = hash_password(password)
        user = User(username=username, password_hash=password_hash, salt=salt, email=email)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/chat')
@login_required
def chat():
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.created_at.desc()).all()
    return render_template('chat.html', conversations=conversations)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    data = request.json
    user_input = data.get('message')
    
    # Generate response using the AI model
    response = generate_response_with_context(user_input, [], model, tokenizer)
    
    # Save conversation
    conversation = Conversation.query.filter_by(id=data.get('conversation_id')).first()
    if not conversation:
        conversation = Conversation(user_id=current_user.id, title=user_input[:50])
        db.session.add(conversation)
        db.session.commit()
    
    # Save messages
    user_message = Message(conversation_id=conversation.id, role='user', content=user_input)
    assistant_message = Message(conversation_id=conversation.id, role='assistant', content=response)
    db.session.add(user_message)
    db.session.add(assistant_message)
    db.session.commit()
    
    return jsonify({
        'response': response,
        'conversation_id': conversation.id
    })

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Initialize database and load models
with app.app_context():
    db.create_all()
    model, tokenizer, sentence_model, qa_pipeline = load_models()

if __name__ == '__main__':
    app.run(debug=True)
