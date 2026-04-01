import streamlit as st
import os
import uuid

import config
from download_video import download_video
import main

st.set_page_config(page_title="Sports Analyzer", layout="wide")

st.title("🏏 Sports Video Analyzer")

# Fixed URL (as you said)
DEFAULT_URL = "https://www.youtube.com/watch?v=0pYlJ7hA5Zo"

url = st.text_input("Enter Video URL", value=DEFAULT_URL)

if st.button("🚀 Process Video"):

    job_id = str(uuid.uuid4())

    input_path = os.path.join(config.DATA_DIR, f"{job_id}.mp4")
    output_path = os.path.join(config.OUTPUT_DIR, f"{job_id}_out.mp4")

    st.info("📥 Downloading video...")
    success = download_video(url, input_path)

    if not success:
        st.error("Download failed ❌")
        st.stop()

    st.info("⚙️ Processing video... (this may take time)")

    args = main.parse_args()
    args.video = input_path
    args.output = output_path

    main.run(args)

    st.success("✅ Done!")

    # Show video directly
    web_output = output_path.replace(".mp4", "_web.mp4")

    convert_to_web_format(output_path, web_output)

    st.video(web_output)

    # Download button
    with open(output_path, "rb") as f:
        st.download_button(
            "⬇️ Download Video",
            f,
            file_name="output.mp4"
        )