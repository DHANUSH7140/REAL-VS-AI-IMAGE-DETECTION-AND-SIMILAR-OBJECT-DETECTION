"""
Unified Router App (Flask)
Serves as the main entry point, handles authentication,
and hosts the two detection modules in iframes.
"""

import os
import sys
import subprocess
import socket
import time
import atexit
from flask import Flask, render_template, request, redirect, session, url_for

from shared.auth import register_user, authenticate_user

app = Flask(__name__, static_folder="modules/similar_object_detection/static")
app.secret_key = "unified_router_secret_key"

# Ensure absolute paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_APP_DIR = os.path.join(ROOT_DIR, "real-vs-ai-detector")
SIMILAR_APP_DIR = os.path.join(ROOT_DIR, "modules", "similar_object_detection")

AI_PORT = 5003
SIMILAR_PORT = 5004

processes = []

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_app(cwd, port, script="app.py"):
    if not check_port(port):
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["PYTHONPATH"] = cwd
        print(f"Starting {cwd} on port {port}...")
        proc = subprocess.Popen([sys.executable, script], cwd=cwd, env=env)
        processes.append(proc)
        
        # Wait for port to be available
        start_time = time.time()
        while not check_port(port):
            if time.time() - start_time > 10:
                print(f"Timeout waiting for port {port}")
                break
            time.sleep(0.5)

def cleanup_processes():
    print("Cleaning up subprocesses...")
    for p in processes:
        p.terminate()

atexit.register(cleanup_processes)

def auth():
    return 'username' in session

@app.route('/')
def index():
    if not auth():
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        success, msg = authenticate_user(username, password)
        if success:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = msg
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        if password != password_confirm:
            error = "Passwords do not match."
        else:
            success, msg = register_user(username, password)
            if success:
                return render_template('login.html', message=msg + " Please log in.")
            else:
                error = msg
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("Starting background modules...")
    start_app(AI_APP_DIR, AI_PORT)
    start_app(SIMILAR_APP_DIR, SIMILAR_PORT)
    
    print("Starting Main Router on port 8500...")
    # Run the main router on port 8500
    app.run(host='0.0.0.0', port=8500, debug=False, use_reloader=False)
