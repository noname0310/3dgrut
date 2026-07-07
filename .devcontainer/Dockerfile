FROM 3dgrut:cuda12

SHELL ["/bin/bash", "-lc"]

RUN source /workspace/.venv/bin/activate && \
    uv pip install -U huggingface_hub hf_xet
