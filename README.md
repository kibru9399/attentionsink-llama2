 **# Llama2 Streaming LLM for Official Weights**

This repository provides a compatible implementation of the streaming LLM paper specifically designed for the official Llama2 weights from Meta.

## Key Features

- **Compatibility with Official Weights:** Addresses compatibility issues arising when using the official weights with the original paper's implementation or HuggingFace.
- **Robust Handling of Meta-Specific Implementations:** Accommodates differences in RoPE embedding, KV cache, and model parameter naming conventions.
- **Streamlined Inference:** Offers the `inference.py` file for immediate execution of streaming tasks.
- **Adaptable for Diverse Use Cases:** The `inference.py` file can be tailored for various scenarios requiring long-range context dependency.

## Quick Start

1. **Obtain Official Llama2 Weights:** Download the weights from Meta's designated resource.
2. **Clone This Repository:** 
   ```bash
   git clone https://github.com/kibru9399/attentionsink-llama2.git
   ```
3. **Run Streaming Inference:** 
   ```bash
   python inference.py
   ```
4. **Adapt for Custom Use Cases:** Modify the `inference.py` file to align with your specific requirements.

## Additional Information

- **Paper:** For details on the streaming LLM approach, refer to the original paper ([Paper](https://arxiv.org/pdf/2309.17453.pdf).

 

