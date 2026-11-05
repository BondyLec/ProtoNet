#!/bin/bash

MINI="bondy@Bondys-Mac-Mini.local"
REMOTE_DIR="/Users/bondy/Research/projects/ProtoNet"

echo "Push code to Github..."

git push

echo "Update Mac mini and start training..."

ssh "$MINI" "
    cd $REMOTE_DIR &&
    git pull &&
    tmux kill-session -t protonet 2>/dev/null || true
    tmux new-session -d -s protonet \
    'uv run python train.py 2>&1 | tee results/train_1shot.log'
"

echo "ProtoNet training started on Mac mini."