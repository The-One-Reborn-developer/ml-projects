# Gemma 3 Fine-Tuning for Visual Recognition Task

## Setup with 1 GPU

### Fine-Tuning

Python 3.11.7

1. ```python -m venv .venv```

2. ```source .venv/bin/activate```

3. ```pip install -r requirements.txt```

4. Download dataset.

5. ```python create_image_urls.py```

6. ```python create_jsonl.py```

7. ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine_tune.py```

8. ```python merge.py```

### Inference

1. ```python inference.py path/to/file```

## Setup with multiple GPUs

### Accelerate

Python 3.11.7

1. ```python -m venv .venv```

2. ```source .venv/bin/activate```

3. ```pip install -r requirements.txt```

4. ```accelerate config```

   Answers to the following questions:
   Q: In which compute environment are you running?
   A: This machine

   Q: Which type of machine are you using?
   A: multi-GPU

   Q: How many different machines will you use (use more than 1 for multi-node training)? [1]:
   A: Hit Enter

   Q: Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
   A: NO

   Q: Do you wish to optimize your script with torch dynamo?[yes/NO]:
   A: NO

   Q: Do you want to use DeepSpeed? [yes/NO]:
   A: NO

   Q: Do you want to use FullyShardedDataParallel? [yes/NO]:
   A: NO

   Q: Do you want to use TensorParallel? [yes/NO]:
   A: NO

   Q: Do you want to use Megatron-LM ? [yes/NO]:
   A: NO

   Q: How many GPU(s) should be used for distributed training? [1]:
   A: Number of GPUs as integer (e.g. 4)

   Q: What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
   A: List the number of GPUs starting from 0 (e.g. 0,1,2,3)

   Q: Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
   A: NO

   Q: Do you wish to use mixed precision?
   A: bf16

5. ```accelerate launch fine_tune.py```
