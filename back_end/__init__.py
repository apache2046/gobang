from flask import Flask, request, g, abort, jsonify, render_template, redirect
import os
__all__ = [
    "serv"
]

def serv(port=8080, boardsize_cb=None, clearboard_cb=None, play_cb=None, genmove_cb=None):
    static_dir = os.path.dirname(__file__) + "/../front_end/dist"
    print(static_dir)
    app = Flask('go_srv', static_folder = static_dir, static_url_path="/")

    @app.route('/')
    def index():
        return redirect('/index.html')

    @app.route('/boardsize', methods=['GET', 'POST'])
    def boardsize():
        print('boardsize_cb')
        params = request.json
        if boardsize_cb is not None:
            boardsize_cb(params["size"])
        return jsonify({'status':'ok'})
    
    @app.route('/clearboard', methods=['GET', 'POST'])
    def clearboard():
        print('clearboard')
        # params = request.json
        if clearboard_cb is not None:
            clearboard_cb()
        return jsonify({'status':'ok'})

    @app.route('/play', methods=['POST'])
    def play():
        print('play')
        params = request.json
        ret = True
        if play_cb is not None:
            ret = play_cb(params["actor"], params["pos"])
    
        if ret:
            return jsonify({'status':'ok'})
        else:
            return jsonify({'status':'failed'})
    
    @app.route('/genmove', methods=['POST'])
    def genmove():
        print('genmove')
        params = request.json
        if genmove_cb is not None:
            ret = genmove_cb(params["actor"])
            print('genmove2 ', ret, type(ret), type(ret[1]))
            return jsonify({'status':'ok', 'pos': ret})
        return jsonify({'status':'failed'})
    
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True, processes=1)

if __name__ == "__main__":
    serv()