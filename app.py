from flask import Flask, request, jsonify, send_file
import os
import uuid

from download_video import download_video
import main
import config

app = Flask(__name__)

@app.route("/")
def home():
    return "Sports Analytics API Running 🚀"


@app.route("/process", methods=["POST"])
def process_video():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    job_id = str(uuid.uuid4())

    input_path = os.path.join(config.DATA_DIR, f"{job_id}.mp4")
    output_path = os.path.join(config.OUTPUT_DIR, f"{job_id}_out.mp4")

    # 1. Download video
    success = download_video(url, input_path)
    if not success:
        return jsonify({"error": "Download failed"}), 500

    # 2. Run pipeline
    args = main.parse_args()
    args.video = input_path
    args.output = output_path

    main.run(args)

    return jsonify({
        "message": "Processing complete",
        "download_url": f"/download/{job_id}"
    })


@app.route("/download/<job_id>")
def download(job_id):
    path = os.path.join(config.OUTPUT_DIR, f"{job_id}_out.mp4")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)