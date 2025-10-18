# How large language models encode theory-of-mind: a study on sparse parameter patterns

**Authors:** Yuheng Wu¹, Wentao Guo², Zirui Liu³, Heng Ji⁴, Zhaozhuo Xu⁵ & Denghui Zhang⁶

¹Department of Electrical Engineering, Stanford University, Stanford, CA, USA
²Department of Computer Science, Princeton University, Princeton, NJ, USA
³Department of Computer Science & Engineering, University of Minnesota Twin Cities, Minneapolis, MN, USA
⁴Department of Computer Science, University of Illinois Urbana-Champaign, Champaign, IL, USA
⁵Department of Computer Science, Stevens Institute of Technology, Hoboken, NJ, USA
⁶School of Business, Stevens Institute of Technology, Hoboken, NJ, USA

## Abstract

This paper investigates the emergence of Theory-of-Mind (ToM) capabilities in large language models (LLMs) from a mechanistic perspective, focusing on the role of extremely sparse parameter patterns. We introduce a novel method to identify ToM-sensitive parameters and reveal that perturbing as little as 0.001% of these parameters significantly degrades ToM performance while also impairing contextual localization and language understanding. To understand this effect, we analyze their interactions with core architectural components of LLMs. Our findings demonstrate that these sensitive parameters are closely linked to the positional encoding module, particularly in models using Rotary Position Embedding (RoPE), where perturbations disrupt dominant frequency activations critical for contextual processing. Furthermore, we show that perturbing ToM-sensitive parameters affects LLMs' attention mechanism by modulating the angle between queries and keys under positional encoding. These insights provide a deeper understanding of how LLMs acquire social reasoning abilities, bridging AI interpretability with cognitive science.

## Introduction

Theory-of-Mind (ToM) refers to the ability to infer and reason about the mental states of others, which is a fundamental aspect of human cognition. ToM evaluation tasks have been widely used in cognitive and developmental psychology to assess social reasoning abilities, particularly in early childhood and neurodevelopmental studies.

A typical ToM task involves reasoning about the discrepancy between reality and an agent's beliefs. For example, in Fig. 1, Sam (protagonist) encounters a bag labeled "chocolate," but the bag contains popcorn. LLMs (the ToM task taker) should be able to infer from the story that: (a) the bag contains popcorn, and (b) the protagonist believes the bag contains chocolate.

Understanding how ToM-like reasoning emerges in Large Language Models (LLMs) is a critical area of research, with significant implications for the cognitive modeling of artificial intelligence (AI). By exploring how LLMs develop the ability to infer mental states, we can better align LLM systems with human social cognition, fostering more trustworthy and interpretable interactions. Recent studies have found that to some extent, ToM capabilities already emerge in LLMs. However, existing research on ToM in LLMs primarily treats LLMs as black boxes, either evaluating their ToM performance across different scenarios or leveraging ToM for prompt engineering. To date, few works have explored the emergence of ToM capabilities at the parameter level; the underlying mechanisms in LLM architecture that give rise to ToM capabilities remain unclear. This gap raises two key questions:

- Which parameters in LLMs are sensitive to ToM capabilities?
- How do these parameters influence ToM reasoning performance?

In this paper, we investigate the internal structures of LLMs that encode ToM capabilities, moving beyond task-based evaluation to analyze the specific parameters sensitive to ToM-related behavior. We introduce a novel framework to identify extremely sparse and low-rank ToM-sensitive parameter patterns, uncovering a strong connection between ToM-related performance and the LLM's positional encoding mechanisms. In particular, we demonstrate that these sensitive parameters influence ToM capabilities by modulating the positional encoding process, which alters the attention mechanism's internal dynamics. Our key contributions include:

- **Sparse parameter sensitivity:** We propose a method to identify an extremely sparse, low-rank, ToM-sensitive parameter pattern in LLMs. Perturbing as little as 0.001% of model parameters leads to significant changes in ToM capabilities.

- **Connection to positional encoding:** We demonstrate that the functionality of the observed ToM-sensitive parameter pattern is tightly linked to Rotary Position Embedding (RoPE)-based positional encoding in LLMs. Specifically, perturbing these parameters disrupts dominant frequency activations critical for contextual reasoning. In contrast, models without this frequency-dependent activation structure exhibit distinct sensitivity patterns.

- **Impact on attention mechanisms:** We show that perturbing the ToM-sensitive parameter pattern alters the geometric relationship between queries and keys under positional encoding, leading to shifts in attention sinks. These shifts degrade the model's ability to form coherent representations, impairing its language understanding capabilities.

Our findings contribute to advancing the interpretability of LLM systems and deepen the understanding of how ToM-like reasoning emerges in LLMs. By identifying sparse, low-rank parameter patterns sensitive to ToM capabilities and uncovering their connection to positional encoding and attention mechanisms, we provide new insights into the functionality of LLM architectures in supporting social reasoning behavior. These discoveries have significant implications for LLM alignment and the development of society-aware AI systems. Our work not only bridges the gap between task-based evaluation and mechanistic understanding of ToM in LLMs but also paves the way for future research on controllable and interpretable social reasoning in AI.

## Results

### ToM tasks for LLMs

ToM tasks assess an agent's ability to infer and reason about others' mental states. In evaluating LLMs, a variety of ToM tasks have been employed, each targeting different aspects of social reasoning. Among these, false-belief tasks (FB) are the most widely used. FB tasks assess whether an LLM can understand that an agent may hold a belief that differs from the actual state of the world. Two classic forms of FB tasks are unexpected contents and unexpected transfer tasks.

**Unexpected contents task:** This task involves an agent encountering an object with misleading packaging (e.g., a chocolate box containing popcorn). Participants must infer both the true content and the agent's false belief. We illustrate this task in Fig. 1 of the introduction.

**Unexpected transfer task:** This task evaluates whether an LLM can infer that an agent will act based on their outdated belief about an object's location.

Here is an unexpected transfer task sample:

**Context:** James puts his car keys in the drawer before heading out to exercise. While James is out, his wife Linda decides to clean the house. She finds the car keys in the drawer and thinks they would be safer in the key cabinet. She moves them there and continues cleaning. Later, James returns from his run and wants to get his car keys.

- Prompt 1: The keys will be taken out of the key cabinet.
- Prompt 2: James will look for the keys in the drawer.

During testing, the context is concatenated with each prompt separately to form two distinct inputs, and the LLM generates responses auto-regressively for each. We evaluate the model's response by checking the first generated token using an exact match. To pass the test, the LLM must correctly understand both that (a) the keys were moved to the key cabinet and (b) James is unaware of this change.

Additionally, true-control tasks are used to verify that LLMs are not simply responding based on surface-level word associations. For instance, if James had witnessed Linda moving the keys before leaving, the correct response to both prompts should be key cabinet instead of drawer. Further examples and variations of this test can be found in Section B of supplementary information.

Beyond false-belief tasks, ToM reasoning extends to more complex social scenarios, such as:

- **Faux pas tasks:** Can the LLM detect when someone has made an inappropriate or socially awkward remark?
- **Irony tasks:** Can the LLM distinguish between a literal statement and a sarcastic or ironic remark?

In this study, we focus on false belief tasks (unexpected contents and unexpected transfer tasks) because they have clear, objective answers. In contrast, tasks such as faux pas and irony detection involve some degree of subjectivity, making automatic evaluation more challenging.

### Methods and findings overview

To investigate the mechanistic basis of ToM capabilities in LLMs, we developed an analysis framework that connects LLMs' behavior when answering ToM-related questions to their internal computational workflow involving model parameters. Using a Hessian-based sensitivity analysis, we identified an extremely sparse subset of LLM parameters (at the 0.001% level of all parameters) in linear transformation parameter matrices in LLMs, including W_Q, W_K, W_V, W_O, W_Gate, W_Up, and W_Down. We denote these selected LLM parameters as ToM-sensitive parameters. Next, to isolate the functionality of these ToM-sensitive parameters, we perturbed them by replacing each identified parameter with the average value of other non-sensitive parameters in the same matrix. This method (illustrated in Fig. 2) allows us to pinpoint LLM behaviors specific to the ToM-sensitive parameters. We applied this approach to four LLM families: Llama, Qwen, DeepSeek, and Jamba.

Under the proposed perturbation, we observe the resulting changes in both LLMs' behaviors and their internal states given the same input. Our results show that even when only a tiny fraction of ToM-sensitive parameters are altered, the LLMs suffer a significant drop in ToM-related performance, an effect not seen when we randomly select the same amount of parameters and perturb as a control. This stark contrast indicates that the identified parameters play a critical role in the LLMs' capacity for ToM. Further analysis suggests that the performance impairment arises because the LLM loses its ability to localize context and maintain proper language understanding once those weights are perturbed. To understand why, we then examine the underlying mechanism driving this effect.

Next, we examined how these sparse ToM-sensitive parameters interact with the core architectural components of the LLMs. Our results show that the ToM-sensitive parameters predominantly influence the positional encoding mechanism. In LLMs using RoPE, the positional encoding naturally produces activation patterns concentrated at specific dominant frequencies. We found that the ToM-sensitive parameters align precisely with these frequency patterns, perturbing the sensitive parameters selectively disrupted the dominant frequency activations (Fig. 3). This finding explains the earlier noted loss of contextual localization: by breaking the frequency-based structure that normally underpins positional relationships in the sequence, the perturbation prevents the model from accurately anchoring tokens to their positions in context.

Importantly, this phenomenon is architecture-dependent. LLMs that do not use RoPE-based positional encoding do not exhibit the same concentrated frequency pattern and do not show such extreme sensitivity to perturbations in a tiny subset of parameters. This contrast confirms that the observed ToM-sensitive parameter effect is tightly linked to the RoPE positional encoding scheme.

Finally, our framework reveals how perturbations in positional encoding propagate into the model's attention mechanism, altering the geometry of query-key interactions. The ToM-sensitive parameters regulate the relationship between certain query and key vectors: they affect the angle between the current token's query vector (q) and the beginning-of-sequence key vector (k_BOS). Under normal conditions, RoPE ensures that q and k_BOS are non-orthogonal, creating a stable "attention sink" at the BOS token. However, when we perturb the ToM-sensitive parameters, we observe that k_BOS rotates toward orthogonality relative to q. This rotation destabilizes the previously stable attention sink, causing the model's attention weights to shift and spread toward irrelevant positions in the sequence (see Figs. 4 and 5). Such a geometric disruption in attention directly degrades the model's language understanding: without a stable attention sink, the model struggles to maintain coherent relationships between tokens, leading to a breakdown in its ability to form consistent and accurate interpretations of the input.

### Sensitivity to perturbations and its impact on ToM and language processing

Firstly, we show that even at an extreme sparsity level κ (10⁻⁵), perturbing ToM-sensitive parameters causes a significant decline in ToM performance across all RoPE-based models while having minimal impact on perplexity (Table 1). This contrast underscores the specialized role of these parameters in ToM reasoning. In comparison, random perturbations produce no measurable effect, further highlighting the structured nature of ToM-related computations. Details on the search process for the optimal κ and results on random perturbations can be found in Section B of supplementary information. For more results on additional ToM Benchmark, please refer to Section B of supplementary information. For a detailed analysis of how varying perturbation strength affects ToM performance and language perplexity, please also refer to Section B of supplementary information.

Secondly, we show that perturbing ToM-sensitive parameters not only affects ToM tasks but also degrades contextual localization and language understanding. As shown in Fig. 6, RoPE-based models struggle to maintain positional accuracy, especially in longer token sequences. The impact extends to language understanding, as seen in the decline in MMLU benchmark performance (Fig. 7), with ToM-relevant categories such as business ethics experiencing the sharpest drop (Fig. 8).

Thirdly, unlike RoPE-based architectures, non-RoPE models do not exhibit clear ToM-sensitive parameter patterns. Notably, perturbations in models such as Jamba-1.5-Mini resulted in improved ToM task performance alongside reduced perplexity, suggesting an alternative strategy for encoding ToM reasoning. The absence of RoPE prevents dominant frequency activations, rendering the perturbation approach ineffective in disrupting positional encoding. This distinction underscores fundamental differences in how these architectures internalize and process ToM-related intelligence. For more information about RoPE and dominant frequency activations, please refer to Section "Rotary positional encoding". More results are provided in Section B in supplementary information.

**Finding 1:** An extremely sparse ToM-sensitive parameter pattern exists, whose perturbation significantly affects RoPE-based models' ToM capabilities, while random perturbations do not. Our experiments further demonstrate that this degradation is linked to a reduction in contextual localization and language understanding.

### Characteristics of ToM-sensitive parameters and their impact on positional encoding

Firstly, we show that ToM-sensitive parameter pattern exhibits strong sparsity and low-rank structure, with significant perturbations concentrated in the W_Q and W_K matrices. In Llama3-8B, the average rank of the masked parameters in these matrices is 21.69 and 10.5, respectively, highlighting a structured low-rank nature. Moreover, perturbed weights in W_Q and W_K are significantly larger than those in other matrices, indicating a link between ToM-related computations and the attention mechanism. For detailed results, please refer to Section B in supplementary information.

Secondly, as shown in Fig. 9, the ToM-sensitive parameter pattern primarily perturbs dominant frequency activations, which closely align with the frequencies exhibiting the highest activation norm. This suggests that these parameters modulate positional encoding by selectively targeting key frequency components. However, this alignment is absent in Jamba, which does not employ RoPE and lacks a clear dominant frequency structure. Consequently, perturbing the ToM-sensitive parameter pattern in Jamba might not affect contextual localization through positional encoding. Visualizations are provided in Section B in supplementary information.

**Finding 2:** The functionality of the ToM-sensitive parameter pattern relates to the positional encoding module in LLM architectures. Perturbing the proposed ToM-sensitive parameter pattern in LLMs with RoPE disrupts dominant frequency activations induced by positional encoding, thereby impairing contextual localization. In contrast, LLMs without RoPE lack this frequency-dependent activation structure and exhibit different sensitivity patterns.

### From positional encoding to attention map

We next investigate how these effects propagate from positional encoding to the attention map. Recent studies have identified a phenomenon known as attention sinks in LLMs, where attention maps across layers and heads predominantly focus on the relationship between the query token and k_BOS. This appears as a pronounced vertical stripe in the first column of the attention map. Despite its smaller norm compared to other tokens, k_BOS occupies a distinct manifold, allowing it to act as a bias that absorbs excess attention scores, thereby stabilizing attention dynamics.

Perturbing the ToM-sensitive parameter pattern leads to significant shifts in attention sinks. Using a threshold of 0.01 to define a shift, we find that over 30% of attention sinks in layer 10 are displaced (Fig. 10), severely disrupting the attention structure. This perturbation causes the model to incorrectly select irrelevant features in W_V, impairing language understanding by selecting irrelevant features.

We analyze the q tokens at positions where attention sink shifts occur, computing their angles with k_BOS and k_others. As shown in Table 2, we find that the magnitudes of the vectors remain largely unchanged before and after perturbation, and q remains nearly orthogonal to k_others, with little change in their inner product. However, for the angle between q and k_BOS, we observe that the change introduced by RoPE is minimal, whereas the ToM perturbation causes a significant angular shift. This perturbation completely overwhelms the positional information encoded by RoPE, explaining the decline in the model's contextual localization ability. Additionally, it leads to a smaller inner product between q and k_BOS, destabilizing the attention sink and causing shifts that further degrade language understanding.

As shown in Fig. 11, perturbing ToM-sensitive parameters introduces two key distortions. First, incorrect attention relationships emerge: an attention head originally attending to function words such as "the" (article), "of" (preposition), and "-lest" (subordinating conjunction) begins misallocating attention to punctuation marks like commas. Second, existing attention relationships are distorted: the attention scores assigned to certain tokens are altered, which undermine the model's ability to maintain stable feature representations, impairing its overall language understanding capabilities.

**Finding 3:** Perturbing ToM-sensitive parameter patterns affects the attention mechanism, thereby influencing language understanding. Perturbing the ToM-sensitive parameter pattern alters the angle between q and k_BOS under positional encoding. This disruption breaks the RoPE encoding, causing q and k_BOS to become more orthogonal. As a result, the attention sink is destabilized, distorting the attention matrix and impairing the model's ability to capture correct feature relationships, ultimately diminishing its ToM capabilities.

## Discussion

Our study uncovers a fundamental link between sparse parameter structures and ToM capabilities in LLMs, demonstrating that social reasoning behaviors are governed by a highly localized and low-rank subset of model weights. A key insight from our findings is the pivotal role of positional encoding, particularly RoPE, in shaping ToM-related inferences. We observe that ToM-sensitive parameters modulate dominant frequency activations, influencing geometric relationships in the attention mechanism and ultimately shifting attention sinks. This mechanistic perspective suggests that LLMs leverage structured positional and relational representations to model implicit beliefs and perform ToM-related reasoning.

While our findings offer new insights into the structural basis of ToM reasoning in LLMs, our study does not exhaustively evaluate the full spectrum of ToM abilities. We primarily focus on unexpected transfer and unexpected contents tasks, which are among the most rigorous ToM benchmarks. However, future work is needed to assess whether similar parameter structures support a broader range of social reasoning skills. At the same time, our analysis of ToM-sensitive parameters provides a broader perspective, revealing their role beyond ToM tasks in contextual localization and language understanding. This suggests that ToM reasoning may not be an isolated cognitive faculty but rather an emergent property of general mechanisms underlying token positioning and meaning construction.

Beyond theoretical implications, our results raise important considerations for AI interpretability and controllability. The ability to identify and manipulate ToM-sensitive parameters opens avenues for designing models that can adaptively regulate their social reasoning behaviors—an essential feature for AI systems deployed in high-stakes domains such as healthcare, legal analysis, and human-AI collaboration. However, this structural localization also presents risks: if ToM capabilities are concentrated in a sparse parameter subset, adversarial interventions could be used to either suppress or exaggerate social reasoning, potentially leading to deceptive or manipulative AI behaviors.

Our findings also open several promising avenues for future investigation. Firstly, for targeted model alignment, can ToM-sensitive parameters be leveraged to ensure AI systems align with human ethical norms while mitigating unintended social biases? Secondly, for comparative cognitive modeling, how do the identified sparse parameter structures compare to neural representations of ToM in the human brain? Could similar mechanisms underlie social reasoning across biological and artificial systems? Thirdly, for robustness and adversarial testing, if ToM capabilities depend on sparse and structured parameter subsets, could targeted attacks degrade LLM reasoning abilities? Understanding these vulnerabilities is critical for developing more resilient AI architectures.

While our work centers on ToM reasoning in LLMs, we acknowledge the potential to extend our framework to multimodal settings such as visual question answering (VQA). In VQA, some studies have explored the impact of language priors (e.g., ESC-Net) and robustness under perturbations (e.g., R-VQA), which align with our broader interest in how small parameter changes affect reasoning behavior. These connections suggest possible directions for future work beyond the language-only setting.

By illuminating the structural underpinnings of social intelligence in AI, our study bridges the gap between deep learning, cognitive science, and AI ethics. As LLMs continue to evolve, understanding how they acquire, encode, and manipulate social reasoning will be essential for ensuring their transparency, reliability, and alignment with human values.

## Methods

### Sparse ToM-sensitive parameter patterns

In this subsection, we identify sparse parameter patterns critical for ToM capabilities. Using the Fisher information matrix, we derive a binary mask m_κ to isolate ToM-sensitive parameters. We further combine this with a pre-training task performance mask m'_κ to ensure perturbations specifically impair ToM capabilities without degrading overall language performance.

We start with an introduction of Fisher information matrix. Let D_ToMTrain = {(x_i, y_i)}_{i=1}^n be a dataset with loss

L(θ, D_ToMTrain) = (1/n) Σ_{i=1}^n ℓ(θ, x_i, y_i)

In the later stage of training, the first-order gradient term of the loss L is nearly zero, so the second-order term, governed by the Hessian matrix, primarily determines how the loss increases under small parameter perturbations. We denote the Hessian of the loss L at parameters θ by H(θ). In practice, this Hessian is often approximated by the Fisher information matrix F, which can be estimated via the empirical Fisher F̂. Concretely, let g_i = ∇_θ ℓ(θ; x_i, y_i), then in the late-training regime, we approximate the overall gradient and Hessian of L by

∇_θ L(θ) ≈ (1/n) Σ_{i=1}^n g_i,    H ≈ F ≈ F̂ = (1/n) Σ_{i=1}^n g_i g_i^T    (1)

In practical scenarios, we further simplify F̂ by ignoring its off-diagonal elements, focusing only on the diagonal entries as a per-parameter sensitivity estimates. Under this approximation, larger diagonal values indicate that the corresponding parameters have a greater impact on the model's performance.

Next, we showcase how to identify ToM-sensitive parameter patterns. Let d be the number of parameters in the current layer or matrix being analyzed. We seek a sparse binary mask m_κ ∈ {0,1}^d with exactly κd non-zero entries (κ ∈ [0,1] is the proportion) such that it maximizes the total sensitivity.

**Definition 1 (ToM-sensitive Parameters).** Using the Hessian H from Equation (1), a sensitive parameter mask m_κ ∈ {0,1}^d with κd non-zero entries is defined by

m_κ = arg max_{m_κ ∈ {0,1}^d} Σ_{i=1}^d m_κ(i) H_ii

We applied m_κ directly to the model and observed that while the model's ToM capabilities diminished, the model's perplexity also increased significantly. We hypothesize that this occurs because m_κ includes not only parameters relevant to ToM-related tasks but also those essential for maintaining the model's language processing capabilities.

Inspired by previous work, to prevent degradation of these language capabilities, we employ another dataset D_pretraining to derive m'_κ, identifying parameters critical for overall language modeling performance. The final ToM-sensitive pattern is then defined as:

m''_κ = m_κ ⊙ m̄'_κ

Here, m̄'_κ represents the complement of m'_κ, and ⊙ denotes element-wise product. This formulation isolates parameters specifically sensitive to ToM tasks while preserving those vital for language processing, ensuring that applying m''_κ impairs ToM capabilities without substantially affecting the model's overall linguistic performance.

### Rotary positional encoding

For Transformer decoder-based models, a widely used positional encoding method is RoPE.

We start by introducing RoPE and feature frequencies. RoPE applies token position-dependent rotations to feature pairs in activations Q and K. Formally, RoPE defines a rotational encoding angle as:

θ(p, m) = p · (1/50000)^(2m/d_h)

where p is the token position, m is the feature index within an attention head, d_h denotes the per-head feature dimension. The encoding applies a rotation matrix M(p, m) to each feature pair x_p^m ∈ ℝ²:

Enc(x_p^m, p, m) = [cos(θ(p,m))  -sin(θ(p,m))]  · x_p^m
                    [sin(θ(p,m))   cos(θ(p,m))]

                  = M(p, m) · x_p^m

Given two token activations q_i, k_j ∈ ℝ^{d_h}, their RoPE-encoded activation interaction is:

RoPE(q_i, k_j) = Σ_{m=0}^{d_h/2-1} Enc(q_i^m, i, m)^T · Enc(k_j^m, j, m)
                = Σ_{m=0}^{d_h/2-1} (q_i^m)^T · M(j-i, m) · k_j^m

This formulation shows that RoPE assigns smaller encoding angles to later feature dimensions in Q and K, meaning that these dimensions rotate more slowly across token positions. As a result, lower-indexed dimensions correspond to higher frequencies, while higher-indexed dimensions correspond to lower frequencies in the positional encoding.

Recent studies have shown that activations tend to concentrate at certain frequencies, with low-frequency components of Q = XW_Q and K = XW_K exhibiting higher magnitudes. One possible explanation is that low-frequency dimensions rotate more slowly, which may allow them to encode information more stably over longer token dependencies. We observe that this phenomenon occurs specifically in models using RoPE, while it is absent in models without RoPE.