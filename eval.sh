#!/bin/sh
# Base model configuration
modelname=DeepSeekPPOv1
BASE_MODEL="checkpoints_data/OpenRS-PPO_v1"

# modelname=TinyLlamaGRPO-accuracy
# BASE_MODEL="checkpoints_data/OpenRS-TinyLlamaGRPO-accuracy"
NUM_GPUS=4
BASE_MODEL_ARGS="dtype=bfloat16,data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# Define evaluation tasks
TASKS="custom_aime24 custom_aime25 custom_math_500 custom_gpqa_diamond custom_minerva custom_amc23 custom_olympiadbench"
# TASKS="custom_aime24 custom_math_500 amc23"

# Function to get checkpoint path for experiment and step
get_checkpoint_path() {
    exp=$1
    step=$2
    echo "${BASE_MODEL}/checkpoint-${step}"
}

# Function to get steps for a given experiment
get_steps() {
    exp=$1
    
    case $exp in
        1) echo "50 100 150 200 250" ;;
        2) echo "50 100 150 200 250 300 350 400" ;;
        3) echo "50 100 150 200 250 300 350 400" ;;
        *) echo "" ;;
    esac
}

# Function to run evaluations for a given step and checkpoint
run_evaluation() {
    experiment=$1
    step=$2
    checkpoint_path=$(get_checkpoint_path "$experiment" "$step")
    output_dir="logs/evals_Model${modelname}/Exp${experiment}_${step}"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "Error: Checkpoint not found at $checkpoint_path"
        return 1
    fi
    
    # Set model args with the checkpoint path
    model_args="model_name=$checkpoint_path,$BASE_MODEL_ARGS"
    
    echo "----------------------------------------"
    echo "Running evaluations for experiment $experiment, step $step"
    echo "Checkpoint: $checkpoint_path"
    echo "Output directory: $output_dir"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Run evaluations for each task
    for task in $TASKS; do
        echo "Evaluating task: $task"
        VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm "$model_args" "$task" \
            --custom-tasks src/open_r1/evaluate.py \
            --output-dir "$output_dir"
    done
    echo "----------------------------------------"
}

# Function to run an experiment
run_experiment() {
    exp_num=$1
    steps=$(get_steps "$exp_num")
    
    # Check if experiment exists
    if [ -z "$steps" ]; then
        echo "Error: Experiment $exp_num not defined"
        return 1
    fi
    
    echo "========================================"
    echo "Running Experiment $exp_num"
    echo "Steps: $steps"
    echo "========================================"
    
    # Run evaluation for each step in the experiment
    for step in $steps; do
        run_evaluation "$exp_num" "$step"
    done
}

# Function to list all available experiments and checkpoints
list_configurations() {
    echo "Available Experiments:"
    
    for exp_num in 1 2 3; do
        steps=$(get_steps "$exp_num")
        echo "  Experiment $exp_num: Steps = $steps"
        
        # List checkpoints for this experiment
        echo "  Checkpoints:"
        for step in $steps; do
            checkpoint_path=$(get_checkpoint_path "$exp_num" "$step")
            if [ -d "$checkpoint_path" ]; then
                echo "    Step $step: $checkpoint_path [EXISTS]"
            else
                echo "    Step $step: $checkpoint_path [NOT FOUND]"
            fi
        done
        echo ""
    done
    
    echo "Available checkpoints in ${BASE_MODEL}:"
    if [ -d "$BASE_MODEL" ]; then
        ls -d ${BASE_MODEL}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-/checkpoint-/' || echo "  No checkpoints found"
    else
        echo "  Base model directory not found"
    fi
}

# Main function to run experiments
main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [options] <experiment_number> [experiment_number2 ...]"
        echo "Options:"
        echo "  --list, -l    List all available experiments and checkpoints"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 1           # Run experiment 1"
        echo "  $0 1 2 3       # Run experiments 1, 2, and 3"
        echo "  $0 --list      # List all configurations"
        echo ""
        list_configurations
        exit 0
    fi
    
    if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
        list_configurations
        exit 0
    fi
    
    for exp_num in "$@"; do
        if [ "$exp_num" = "1" ] || [ "$exp_num" = "2" ] || [ "$exp_num" = "3" ]; then
            run_experiment "$exp_num"
        else
            echo "Error: Experiment $exp_num not defined"
        fi
    done
}

# Execute main function with command line arguments
main "$@"