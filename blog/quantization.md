+++
date = "2025-02-12T11:07:50Z"
draft = false
title = "Using Quantization to speed up and slim down your LLM"
categories = ["llm", "python", "ai", "inference optimization", "quantization", "bitsandbytes", "autogptq", "autoround", "deep learning", "deployment", "llama-factory"]
tags = ["llm", "large language models", "python ai", "inference speed", "quantization techniques", "bitsandbytes", "autogptq", "autoround", "int8", "int4", "fp16", "model optimization", "deep learning deployment", "llama-factory quantization"]
+++

## Summary

Large Language Models (LLMs) are powerful, but their size can lead to slow inference speeds and high memory consumption, hindering real-world deployment. Quantization, a technique that reduces the precision of model weights, offers a powerful solution. 
This post will explore how to use quantization techniques like `bitsandbytes`, `AutoGPTQ`, and `AutoRound` to dramatically improve LLM inference performance.

## What is Quantization?

Quantization reduces the computational and storage demands of a model by representing its weights with lower-precision data types. 
Lets imagine data is water and we hold that water in buckets, most of the time we don't need massive floating point buckets to hold data that can be represented by integers. 
`Quantization` is using smaller buckets to hold the same amount of water – you save space and can move the containers more quickly. `Quantization` trades a tiny amount of precision for significant gains in speed and memory efficiency.

**Benefits of Quantization:**

*   **⚡️ Blazing-Fast Inference:** Reduced computation per token leads to faster response times.
*   **💾 Lower Memory Footprint:** Smaller model size allows deployment on less powerful hardware, including consumer GPUs.
*   **🔋 Energy Efficiency:** Ideal for cloud and edge deployments where power consumption is a concern.

**Types of Quantization:**

| Type        | Precision    | Speedup      | Memory Reduction | Accuracy Impact | Use Cases                                     |
|-------------|--------------|--------------|------------------|-----------------|----------------------------------------------|
| FP16        | 16-bit float | 1.5x         | 50%              | Minimal         | General purpose, good starting point           |
| BF16        | 16-bit float | Similar to FP16 | Similar to FP16 | Minimal         | Training, less prone to overflow              |
| INT8        | 8-bit integer | 2x           | 75%              | Moderate        | General purpose, balanced performance         |
| INT4        | 4-bit integer | 3-4x         | 90%              | Significant     | Memory-constrained environments, edge devices |

## Quantization Techniques for LLMs
There are several approaches to quantization, each with its own strengths. Below, we’ll explore three popular methods: **bitsandbytes**, **AutoGPTQ**, and **AutoRound**.

### Method 1: 4-bit Quantization with `bitsandbytes` (Recommended for Fine-tuning)

`bitsandbytes` excels at quantizing models *during* fine-tuning, especially when using Low-Rank Adaptation (LoRA). It's particularly well-suited for NVIDIA GPUs (Ampere and newer).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations (improves performance)
    bnb_4bit_use_double_quant=True,       # Double quantization for better accuracy
    # llm_int8_threshold=6.0, # Optional: Adjust for better performance on some models
    # bnb_4bit_quant_type="nf4", # Optional: Use NF4 quantization type
)

# Load the quantized model
model_name = "./fine-tuned-mistral"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

print("✅ Model loaded with 4-bit quantization!")

# Inference example
input_text = "Will the S&P 500 exceed 6,000 before June 2025?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")  # Move input to the same device as the model
output = model.generate(input_ids, max_length=50)
print("Forecast:", tokenizer.decode(output[0], skip_special_tokens=True))

```
### Method 2: GPTQ (8-bit/4-bit) with `auto-gptq` (For Maximum Speed)

`auto-gptq` performs *post-training* quantization, making it ideal for deploying pre-trained models where maximizing inference speed is the top priority.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "./fine-tuned-mistral"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

quant_config = BaseQuantizeConfig(
    bits=4,         # Use 4 or 8 for INT4 or INT8 quantization
    group_size=128,  # Adjust group size if needed
    desc_act=False,  # Disable activation quantization (often helps with accuracy)
)

quantized_model = AutoGPTQForCausalLM.from_pretrained(model, quant_config)
quantized_model.save_quantized("./fine-tuned-mistral-gptq")

print("✅ Model quantized with GPTQ and saved!")
```

### Method 3: **AutoRound Quantization**  

While `bitsandbytes` and `AutoGPTQ` are excellent for GPU-based quantization, AutoRound offers a unique approach optimized for Intel CPUs. This is how it works and how to use it.

#### **What is AutoRound?**  

AutoRound is a method for **weight quantization** that optimally rounds model parameters to minimize accuracy degradation. Unlike traditional quantization approaches that often introduce **rounding errors**, AutoRound leverages optimization techniques to find a **better set of quantized weights**.  

**Key Features of AutoRound:**  
- **Loss-aware quantization**: Optimizes rounding to minimize accuracy drop.  
- **Post-training**: No need for retraining or fine-tuning.  
- **Compatible with CPUs**: Designed for Intel architectures but also works on other hardware.  

#### **Why Use AutoRound?**  

Traditional quantization methods (e.g., uniform rounding, KL-divergence-based approaches) can cause a noticeable drop in accuracy, especially for large models. AutoRound helps by:  

✅ **Reducing accuracy loss** compared to naive quantization.  
✅ **Optimizing for Intel CPUs**, ensuring efficient execution.  
✅ **Working post-training**, so you don’t need access to training data.  

This makes it ideal for **deploying AI models in production** where performance and efficiency are critical.  

---

#### **How to Use AutoRound for Model Quantization**  


```bash

#pip install auto-round[cpu]
pip install auto-round[gpu] 

```

Running auto round

```bash
auto-round \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --format "auto_gptq,auto_awq,auto_round" \
    --disable_eval \
    --output_dir ./tmp_autoround
```

Using the api

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits =4
group_size = 128
sym = True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format='auto_round', inplace=True) 
```

* **model**: The PyTorch model to be quantized.
* **tokenizer**: An optional tokenizer for processing input data. If none, a dataset must be provided.
* **bits** (int): Number of bits for quantization (default is 4).
* **group_size** (int): Size of the quantization group (default is 128).
* **sym** (bool): Whether to use symmetric quantization (default is True).
* **enable_quanted_input** (bool): Whether to use the output of the previous quantized block as the input for the current block for tuning (default is True).
* **enable_minmax_tuning** (bool): Whether to enable weight min-max tuning (default is True).
* **iters** (int): Number of tuning iterations (default is 200).
* **lr** (float): The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).
* **minmax_lr** (float): The learning rate for min-max tuning (default is None, it will be set to lr automatically).
* **nsamples** (int): Number of samples for tuning (default is 128).
* **seqlen** (int): Data length of the sequence for tuning (default is 2048).
* **batch_size** (int): Batch size for training (default is 8).
* **scale_dtype** (str): The data type of quantization scale to be used (default is "float16"), different kernels have different choices.
* **amp** (bool): Whether to use automatic mixed precision (default is True).
* **nblocks** (int): Packing several blocks as one for tuning together (default is 1).
* **gradient_accumulate_steps** (int): Number of gradient accumulation steps (default is 1).
* **low_gpu_mem_usage** (bool): Whether to save GPU memory at the cost of ~20% more tuning time (default is False).
* **dataset** Union[str, list, tuple, torch.utils.data.DataLoader]: The dataset name for tuning (default is " NeelNanda/pile-10k"). Local json file and combination of datasets have been supported, e.g. " ./tmp.json,NeelNanda/pile-10k:train, mbpp:train+validation+test"
* **layer_config** (dict): Configuration for weight quantization (default is None), mainly for mixed bits or mixed precision.
* **device**: The device to be used for tuning. The default is set to 'auto', allowing for automatic detection.

---

## Hyperparameter tuning

Hyperparameter tuning aims to balance `efficiency gains` and `model accuracy` for the specific task and hardware.


#### 1. **Controlling Quantization Precision**
   - **What it means**: Quantization involves converting high-precision weights (e.g., FP32 or FP16) into lower-precision formats (e.g., INT8, INT4). The choice of precision directly impacts both the speed and accuracy of the model.
   - **Why tune**: Different tasks and models may require different levels of precision. For example:
     - **INT8** might work well for some applications but could lead to noticeable accuracy drops for others.
     - **INT4** offers even greater memory savings and speedups but at the cost of increased quantization error.
   - **How to tune**: Experiment with various quantization levels (e.g., FP16, INT8, INT4) and evaluate their impact on inference speed, memory usage, and accuracy.


#### 2. **Adjusting Quantization Ranges**
   - **What it means**: During quantization, the range of values that weights or activations can take is constrained. This range must be carefully chosen to minimize information loss.
   - **Why tune**: If the range is too narrow, important details in the data may be lost. If it's too wide, the benefits of quantization (e.g., reduced memory usage) diminish.
   - **How to tune**: Use techniques like **min-max scaling** or **histogram-based quantization** to determine optimal ranges for each layer or tensor. Tools like TensorFlow Model Optimization or PyTorch’s quantization utilities provide options for customizing these ranges.

#### 3. **Calibrating Quantization Parameters**
   - **What it means**: Calibration involves determining the best quantization parameters (e.g., scale factors and zero points) for each layer based on representative datasets.
   - **Why tune**: Poorly calibrated parameters can lead to significant degradation in model performance. Proper calibration ensures that the quantized model closely approximates the behavior of the full-precision model.
   - **How to tune**: Use a subset of your training or validation dataset to calibrate the model before deploying it. Techniques like **post-training quantization** rely heavily on accurate calibration.

#### 4. **Optimizing Quantization Methods**
   - **What it means**: There are multiple quantization methods, such as symmetric vs. asymmetric quantization, per-tensor vs. per-channel quantization, and dynamic vs. static quantization. Each method has trade-offs in terms of performance and complexity.
   - **Why tune**: The optimal method depends on the architecture of the model, the target hardware, and the specific application.
   - **How to tune**: Test different quantization strategies and compare their effects on inference time, memory usage, and accuracy. For example:
     - **Per-channel quantization** often works better for convolutional layers in CNNs.
     - **Dynamic quantization** is useful for recurrent neural networks (RNNs).

#### 5. **Balancing Speed and Accuracy**
   - **What it means**: Quantization improves inference speed and reduces memory consumption but can also reduce model accuracy. Hyperparameter tuning helps find the sweet spot where the gains in efficiency outweigh the losses in performance.
   - **Why tune**: The acceptable level of accuracy loss varies depending on the use case. For example:
     - In real-time applications like chatbots, faster response times might be prioritized over minor accuracy drops.
     - In medical imaging, maintaining high accuracy might be more important than achieving maximum speed.
   - **How to tune**: Measure key metrics (e.g., FLOPs, latency, accuracy) under different quantization settings and select the configuration that meets your project’s requirements.

#### 6. **Hardware-Specific Optimization**
   - **What it means**: Different hardware platforms (e.g., CPUs, GPUs, TPUs, edge devices) have varying support for quantized operations. Some accelerators perform better with INT8, while others excel with FP16 or BF16.
   - **Why tune**: To maximize performance on a specific piece of hardware, you need to tailor the quantization settings accordingly.
   - **How to tune**: Profile the model on the target hardware and adjust hyperparameters like batch size, quantization precision, and parallelism to achieve the best results.

#### 7. **Handling Layer-Specific Behavior**
   - **What it means**: Not all layers in a model respond equally well to quantization. Some layers may tolerate low precision without much loss in accuracy, while others may require higher precision.
   - **Why tune**: Applying uniform quantization across all layers can lead to suboptimal results. Fine-grained control allows you to allocate resources more effectively.
   - **How to tune**: Identify critical layers (e.g., early layers in a transformer) and apply higher precision to them while using lower precision for less sensitive layers.

#### 8. **Mitigating Quantization Noise**
   - **What it means**: Quantization introduces noise due to the loss of precision. This noise can accumulate during inference, especially in deep models with many layers.
   - **Why tune**: Reducing quantization noise is crucial for maintaining model accuracy.
   - **How to tune**: Techniques like **double quantization**, **mixed-precision quantization**, and **quantization-aware training** can help mitigate noise. Hyperparameters controlling these techniques must be carefully tuned.

#### Example: Hyperparameter Tuning in Practice

Suppose you’re quantizing a large language model for deployment on an edge device. Here’s how you might approach hyperparameter tuning:

1. **Start with FP16 quantization** to establish a baseline for performance and accuracy.
2. Gradually move to INT8 and then INT4, evaluating the impact on inference speed and memory usage.
3. Use calibration data to fine-tune quantization ranges for each layer.
4. Experiment with per-channel quantization for convolutional layers and per-tensor quantization for fully connected layers.
5. Profile the model on the target hardware to identify bottlenecks and adjust batch sizes or parallelism accordingly.
6. Compare the final quantized model against the full-precision version to ensure the accuracy drop is within acceptable limits.

---

## Refrences
[bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)

[AutoGPTQ Documentation](https://github.com/PanQiWei/AutoGPTQ)

[AutoRound](https://github.com/intel/auto-round)

[A Comprehensive Study on Quantization Techniques for Large Language Models](https://arxiv.org/abs/2411.02530)

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)