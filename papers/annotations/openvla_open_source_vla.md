# Paper Annotation

## Paper Info
- **Title:** OpenVLA: An Open-Source Vision-Language-Action Model
- **Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. (Stanford, UC Berkeley, Google DeepMind, Toyota Research Institute)
- **Year:** 2024
- **Venue:** Conference on Robot Learning (CoRL 2024)
- **arXiv:** https://arxiv.org/abs/2406.09246
- **Project:** https://openvla.github.io/
- **Code:** https://github.com/openvla/openvla
- **Model:** https://huggingface.co/openvla/openvla-7b

## Summary
OpenVLA is the open-source equivalent of RT-2. It's a 7B-parameter VLA trained on 970k real-world robot demonstrations from the Open X-Embodiment dataset. It outperforms the closed-source RT-2-X (55B) by 16.5% on generalist manipulation tasks while using 7x fewer parameters. Everything is MIT-licensed: training code, model weights, data pipeline.

## Key Ideas
- Fused visual encoder: combines SigLIP (semantic vision) + DINOv2 (spatial/geometric features) → better visual grounding
- Prismatic VLM backbone: projector maps fused visual embeddings into Llama 2's input space
- Actions discretized into 256 bins per dimension, predicted autoregressively as language tokens
- Trained on Open X-Embodiment (OXE): 970k trajectories, 22 robot types, diverse tasks
- Full open-source release (MIT): weights, training code, evaluation suite

## Methods
- Visual encoder: SigLIP (400M) + DINOv2 (ViT-L/14) fused via learned MLP projector
- LLM backbone: Llama 2 7B
- Action head: 7-DoF actions tokenized as 6 continuous + 1 discrete (gripper) dimensions
- Training: 64 A100 GPUs, 15 days
- Fine-tuning: LoRA adapters allow efficient adaptation to new robots with <1k demos

## Results
- +16.5% absolute success rate over RT-2-X (55B) across 29 tasks, multiple embodiments
- Out-of-the-box generalization to 9+ robot types not seen during training
- Fine-tuning with LoRA reaches strong performance on new tasks in <1 day of GPU training
- Competitive with closed RT-2-X at 7x fewer parameters

## Relevance to My Research
- THIS is the starting point for building your own VLA — fully open weights, clean codebase
- The SigLIP+DINOv2 fusion is a practical encoder choice: semantic + spatial features
- LoRA fine-tuning recipe is what you'll use to adapt to your target robot/task
- Explainability angle: two distinct visual streams (semantic vs. spatial) are separately interrogatable

## Questions/Thoughts
- Can we add attention visualization on the SigLIP vs DINOv2 pathways to understand what drives actions?
- The 7B backbone is large for real-time control — see OpenVLA-OFT for efficient inference variants
- Action quantization still introduces error on fine-grained manipulation; pi0's flow matching solves this
- What's the minimum dataset size to fine-tune to a new task well?

## Code/Resources
- Training code: https://github.com/openvla/openvla
- Weights: https://huggingface.co/openvla/openvla-7b
- Dataset: Open X-Embodiment https://robotics-transformer-x.github.io/
