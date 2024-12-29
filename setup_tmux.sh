#!/bin/bash

# Create a new tmux session
tmux new-session -d -s andrei_initial_test_training_session

# Terminal 1: In /bigdata/andrei_thesis/Underdiagnosis_NatMed/ALLData directory
tmux rename-window -t andrei_initial_test_training_session:0 'Training'
tmux send-keys -t andrei_initial_test_training_session:0 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:0 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:0 'conda activate cxr_underdiag' C-m
tmux send-keys -t andrei_initial_test_training_session:0 'cd /bigdata/andrei_thesis/Underdiagnosis_NatMed/ALLData' C-m
tmux send-keys -t andrei_initial_test_training_session:0 'echo "Ready for training!"' C-m

# Terminal 2: NVIDIA-SMI monitoring
tmux new-window -t andrei_initial_test_training_session:1 -n 'Monitor'
tmux send-keys -t andrei_initial_test_training_session:1 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:1 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:1 'conda activate cxr_underdiag' C-m
tmux send-keys -t andrei_initial_test_training_session:1 'cd /bigdata/andrei_thesis/Underdiagnosis_NatMed/ALLData' C-m
tmux send-keys -t andrei_initial_test_training_session:1 'watch -n 1 nvidia-smi' C-m

# Terminal 3: Log monitoring
tmux new-window -t andrei_initial_test_training_session:2 -n 'Logs'
tmux send-keys -t andrei_initial_test_training_session:2 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:2 'conda deactivate' C-m
tmux send-keys -t andrei_initial_test_training_session:2 'conda activate cxr_underdiag' C-m
tmux send-keys -t andrei_initial_test_training_session:2 'cd /bigdata/andrei_thesis/Underdiagnosis_NatMed/ALLData' C-m
tmux send-keys -t andrei_initial_test_training_session:2 'tail -f train.log' C-m

# Attach to the tmux session
tmux attach -t andrei_initial_test_training_session
