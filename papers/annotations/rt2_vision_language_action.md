# Paper Annotation

## Paper Info
- **Title:** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **Authors:** Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. (Google DeepMind)
- **Year:** 2023
- **Venue:** Conference on Robot Learning (CoRL 2023)
- **arXiv:** https://arxiv.org/abs/2307.15818
- **Project:** https://robotics-transformer2.github.io/

## Summary
RT-2 is the foundational paper establishing the VLA paradigm. It co-fine-tunes large vision-language models (VLMs) — PaLI-X and PaLM-E — on both robotic trajectory data and internet-scale vision-language tasks. The key trick: robot actions are represented as text tokens, fed into the same tokenizer as natural language. This lets a single model do QA, captioning, AND control.

## Key Ideas
- Represent continuous robot actions as discretized text tokens (e.g., "128 64 200 ...")
- Co-fine-tune on VLT tasks (visual QA) + robotic demonstrations simultaneously
- Emergent capabilities arise from internet pretraining: semantic reasoning, novel object generalization, multi-step chains of thought
- Larger models generalize better; 55B PaLI-X significantly outperforms 5B version
- Chain-of-thought prompting enables multi-stage reasoning (e.g., pick the "improvised hammer" → picks rock)

## Methods
- Base models: PaLI-X (5B, 55B) and PaLM-E (12B)
- Actions tokenized into 256 bins per dimension, represented as text tokens appended to vocabulary
- Co-fine-tuning: standard VLT tasks + robot trajectory data in same training loop
- Input: image + language instruction → Output: tokenized action sequence
- Evaluated on 6,000 real-world robot trials across seen/unseen environments and tasks

## Results
- RT-2 (55B) achieves 62% success on emergent tasks vs 32% for RT-1 baseline
- Significantly better generalization to novel objects, backgrounds, and semantic concepts
- Chain-of-thought RT-2 enables compositional reasoning with intermediate text steps
- Inference at ~1-3 Hz (slow — important limitation for real-time control)

## Relevance to My Research
- Core architecture reference for building any VLA — this is where the paradigm started
- The action tokenization scheme (discretized bins as text tokens) is the simplest VLA output head
- Emergent reasoning from web pretraining is relevant to my explainability work: can we trace which web-learned concepts drive specific robotic decisions?
- The gap between semantic understanding and physical control is a central XAI question here

## Questions/Thoughts
- How do we explain WHICH web knowledge drove a specific action? SHAP/attention on the VLM backbone?
- Action tokenization introduces quantization error — flow matching (see pi0) addresses this
- RT-2 is closed-source; OpenVLA replicates and open-sources the approach

## Code/Resources
- No official open-source code (proprietary Google model)
- Community replication: https://github.com/kyegomez/RT-2
- OpenVLA (arXiv:2406.09246) is the open-source successor to study
