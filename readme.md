1. `feats.py` to compute features from images
2. `search.py` to execute a search

Note: `lib` points to `https://github.com/vizlab-fsc/canary/tree/master/lib`

## Tensorflow on AWS

Build from source to take advantage of AVX2 and FMA (for CPU, not using GPU):

```bash
# Install bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh
chmod +x bazel-0.16.0-installer-linux-x86_64.sh
./bazel-0.16.0-installer-linux-x86_64.sh --user

# Clone tensorflow repo
git clone https://github.com/tensorflow/tensorflow
cd tensorflow

# Build
# If using GPU, can add `--config=cuda`
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both -k //tensorflow/tools/pip_package:build_pip_package
```

## References

- <https://adamspannbauer.github.io/2018/03/04/using-keras-to-build-an-image-search-engine/>
- <https://github.com/yinleon/pydata2017/blob/master/pydata-yin-reverse_img_search.ipynb>