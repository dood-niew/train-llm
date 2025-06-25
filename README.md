# Fine-Tuning on LANTA Supercomputer
## Prerequisites
1. LANTA Supercomputer Access  
   Ensure you have permission to run multi-node GPU jobs on the LANTA Supercomputer.
2. Modules and Environment Setup  
   The following system modules should be loaded before running the script:
   - Mamba
   - CUDA Toolkit >= 11.8
   - GCC >= 10.3.0
   - PrgEnv-gnu
   - cpe-cuda
   
   A Conda environment is also required, with all necessary dependencies installed.
3. Python Dependencies  
   Inside the activated Conda environment, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Model and Dataset
   - Base Model: `/project/lt200258-aithai/llm/base-model/google/gemma-2-27b`
   - Dataset:<br>
     The training dataset is in ShareGPT format, compatible with Hugging Face's `datasets` library.
     Each sample contains a JSON string with a `messages` field.
        In the codebase (`/src/lora_finetune/data.py`), data is loaded as follows:
         ```python
          # Line 71 in data.py
          messages = eval(dataset["text"])["messages"]
         ```
     Make sure the `text` column in your dataset contains valid JSON-encoded strings structured like:
    ```
    {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi, how can I help you?"}
    ]
    }
    ```
