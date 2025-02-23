from flask import Flask, render_template, request, jsonify
from medical_ethics_assistant import MedicalEthicsAssistant
import asyncio

app = Flask(__name__)
assistant = None

@app.before_first_request
async def initialize():
    global assistant
    assistant = await MedicalEthicsAssistant.create()

@app.route('/')
def home():
    return render_template('query_form.html')

@app.route('/submit_query', methods=['POST'])
async def submit_query():
    query = request.form['query']
    response = await assistant.get_ethical_guidance(query)
    return jsonify(response)

@app.route('/follow_up', methods=['POST'])
async def follow_up():
    question = request.form['question']
    context = request.form['context']
    response = await assistant.get_follow_up_guidance(question, context)
    return jsonify(response) 