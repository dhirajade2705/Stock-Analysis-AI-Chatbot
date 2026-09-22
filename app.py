import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from config.config import Config
from api.stock_routes import stock_bp
from api.prediction_routes import prediction_bp
from api.chatbot_routes import chatbot_bp

load_dotenv()


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    CORS(app)

    app.register_blueprint(stock_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(chatbot_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Resource not found."}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=Config.DEBUG)
