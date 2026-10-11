#!/usr/bin/env bash
set -euo pipefail

uv run pytest --timeout 10 -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

# 设置输出 tar.gz 文件的名称。
output_file="cs336-spring2025-assignment-1-submission.zip"
rm "$output_file" || true

# 将当前目录中的文件压缩为一个 zip 文件。
zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '.*' \
    -x 'tests/fixtures' \
    -x 'tests/_snapshots' \
    -x '*.pt' \
    -x '*.pth' \
    -x '*.npy' \
    -x '*.npz'

echo "All files have been compressed into $output_file"