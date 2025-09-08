"""Shared configuration for evaluation scripts."""

# Shared Evaluation Configuration
EVALUATION_CONFIG = {
    # Models
    "MODEL_AI": "siliconflow:Qwen/Qwen3-8B",  # AI agent model
    "MODEL_PERSONA": "siliconflow:THUDM/GLM-4-9B-0414",  # Simulated user model
    "MODEL_EVALUATOR": "siliconflow:THUDM/GLM-Z1-9B-0414",  # Evaluator model
    # Dataset naming
    "DATASET_PREFIX": "react-agent-eval",  # Consistent prefix for all evaluation datasets
    # Common settings
    "LANGGRAPH_URL": "http://localhost:2024",
    "MAX_TURNS": 3,
}
