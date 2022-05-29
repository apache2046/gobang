from flask import Flask, request, g, abort, jsonify, render_template, redirect
import os
__all__ = [
    "serv"
]

def serv(port=8080):
    static_dir = os.path.dirname(__file__) + "/../front_end/dist"
    print(static_dir)
    app = Flask('srv', static_folder = static_dir, static_url_path="/")

    @app.route('/')
    def index():
        return redirect('/index.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        print('login')
        return "abc"

    app.run(host="0.0.0.0", port=port, debug=False, threaded=False, processes=2)

if __name__ == "__main__":
    serv()