# Deploy Pretrained, Optimized MLFF Model with NVIDIA ALCHEMI NIM Models

<details>
<summary>What is NVIDIA NIM?</summary>
NVIDIA NIM (NVIDIA Inference Microservices) is a container-based AI models that were already pretrained and optimized, and ready to be used for high-performance inference at scale. It provides a standardized way to deploy and serve AI models, including LLMs, vision models, and also scientific ML models like machine learning force field (MLFF) for quantum chemistry purpose. 
</details>

<details>
<summary>What is NVIDIA ALCHEMI NIM?</summary>
NVIDIA ALCHEMI NIM is a NIM that is specialized for MLFF models. That is, it is a high-level API wrapper for running MLFF models for quantum chemistry purpose.
</details>

<details>
<summary>Limitation of NVIDIA NIM</summary>
The model must use an architecture supported by the underlying engines (TensorRT-LLM or vLLM) optimized by NVIDIA team. If you have designed a completely proprietary, non-standard neural network architecture from scratch that isn't supported by these frameworks, you won't be able to run it inside the standard NIM LLM container without deep custom development. But for 99% of custom, fine-tuned, or open-weight models, NIM can run them smoothly.  
</details>

<details>
<summary>I build my own MLFF from scratch, can I deplot it with NVIDIA NIM?</summary>
The answer is NO. you cannot deploy a completely custom, from-scratch MLFF (Machine Learning Force Field) / MLIP (Machine Learning Interatomic Potential) with a completely new architecture directly inside the standard NVIDIA NIM container. Nonetheless, you could do so but you need a deep, low-level customization.
</details>


### Deploying NVIDIA NIM models

### Deploying model-free NVIDIA NIM models

### Self-Hosting and deploying NVIDIA ALCHEMI NIM models on Local HPC Cluster
