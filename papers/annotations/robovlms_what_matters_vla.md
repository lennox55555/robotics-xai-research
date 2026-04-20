# Paper Annotation

## Paper Info
- **Title:** Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models
- **Authors:** Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, Huaping Liu (Tsinghua, NUS, ByteDance)
- **Year:** 2024
- **Venue:** Nature Machine Intelligence (2025); arXiv preprint December 2024
- **arXiv:** https://arxiv.org/abs/2412.14058
- **Project:** https://robovlms.github.io/
- **Code:** https://github.com/Robot-VLAs/RoboVLMs

## Summary
This is the ablation study / design guide for VLAs. The authors ran 600+ controlled experiments across 8 VLM backbones, 4 policy architectures, and multiple training configurations to answer: what actually matters when building a VLA? The result is RoboVLMs — a framework that lets you plug any VLM into a VLA with minimal code, and a detailed recipe for design choices. Published in Nature Machine Intelligence.

## Key Ideas
- Continuous action spaces outperform discrete tokenization (confirms pi0's direction)
- Policy head architecture matters more than VLM backbone size — get the head right first
- VLMs with stronger vision-language pre-training generalize better (KosMos, PaliGemma >> others)
- Cross-embodiment data during post-training/fine-tuning significantly helps few-shot generalization
- The RoboVLMs framework is a plug-and-play VLA builder: swap backbone, swap policy head, evaluate

## Methods
- 8 VLM backbones tested: LLaVA, InstructBLIP, Flamingo, KosMos-2, PaliGemma, Qwen-VL, mPLUG-Owl, mPLUG-Owl2
- 4 policy heads: autoregressive token prediction, MLP regression, diffusion head, hybrid
- Evaluation: CALVIN benchmark (simulation) + real-world tabletop manipulation
- Cross-embodiment data: Open X-Embodiment mixed in at varying ratios
- 600+ distinct experiments — the most systematic VLA ablation to date

## Results
- Best RoboVLM achieves 4.25/5 tasks on CALVIN zero-shot (previous SOTA: 3.16)
- Continuous action head (MLP/diffusion) consistently outperforms discrete tokenization
- PaliGemma and KosMos backbones significantly outperform LLaVA-based models
- Cross-embodiment data boosts few-shot performance by 30-50% in low-data regimes
- Results published in Nature Machine Intelligence, lending strong credibility

## Relevance to My Research
- THIS is the design guide to read before building your VLA — tells you what choices matter
- Use PaliGemma as VLM backbone (best tradeoff of performance + open weights + reasonable size)
- Use continuous action head (MLP regressor or diffusion) over discrete tokenization
- Cross-embodiment pretraining with OXE is worth the effort even for task-specific deployment
- The RoboVLMs codebase is the cleanest open-source VLA framework available — build on top of it

## Questions/Thoughts
- Why does PaliGemma dominate? Hypothesis: better visual token alignment from SigLIP training
- The diffusion head beats MLP on contact-rich tasks — validates pi0's direction
- Can we use the RoboVLMs framework as an XAI testbed? Swap backbones and measure explainability
- The 600-experiment ablation is a goldmine for understanding what VLM properties transfer to robotics

## Code/Resources
- RoboVLMs code: https://github.com/Robot-VLAs/RoboVLMs
- CALVIN benchmark: https://github.com/mees/calvin
- Open X-Embodiment: https://robotics-transformer-x.github.io/
