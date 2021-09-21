docker run -d --name deepfm_serving -p 8500:8500 -p 8501:8501 -v $PWD/deep_ctr/deep_ctr_fm:/models/deepfm -e MODEL_NAME=deepfm tensorflow/serving:1.15.0

curl http://127.0.0.1:8501/v1/models/deepfm/metadata
