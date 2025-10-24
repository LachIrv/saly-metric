
from flask import Flask, request, render_template_string
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
    return "Upload interface coming soon."
