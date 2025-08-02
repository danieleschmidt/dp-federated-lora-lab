# ADR-0001: Use LoRA for Parameter-Efficient Fine-tuning

## Status

Accepted

## Context

Large Language Models (LLMs) have billions of parameters, making full fine-tuning computationally expensive and memory-intensive. In federated learning scenarios, clients often have limited computational resources and cannot afford to train full models. Additionally, differential privacy mechanisms become more challenging to apply effectively when the parameter space is large, as the noise required scales with dimensionality.

We need a parameter-efficient fine-tuning method that:
- Reduces computational and memory requirements for clients
- Maintains model quality comparable to full fine-tuning
- Works well with differential privacy mechanisms
- Supports efficient aggregation in federated settings
- Is compatible with various transformer architectures

## Decision

We will use Low-Rank Adaptation (LoRA) as the primary parameter-efficient fine-tuning method for our federated learning system.

LoRA works by:
1. Freezing the original pre-trained model weights
2. Adding trainable low-rank decomposition matrices to attention layers
3. Training only these low-rank adapters (typically <1% of original parameters)
4. Combining adapter outputs with frozen weights during inference

Our implementation will:
- Support adaptive rank selection (4-64) based on task complexity and data characteristics
- Target key attention modules (q_proj, v_proj, k_proj, o_proj)
- Use the HuggingFace PEFT library as the foundation
- Implement noise-aware adaptation strategies for differential privacy

## Consequences

### Positive
- **Reduced Resource Requirements**: Clients need 10-100x less memory and computation
- **Faster Training**: Fewer parameters to train and communicate
- **Better Privacy**: Smaller parameter space means better differential privacy utility
- **Efficient Communication**: Only adapter weights need to be shared (few MB vs GB)
- **Model Compatibility**: Works with most transformer architectures out-of-the-box
- **Quality Preservation**: Research shows LoRA achieves 95-99% of full fine-tuning performance

### Negative
- **Limited Expressiveness**: May not capture all task-specific adaptations compared to full fine-tuning
- **Hyperparameter Sensitivity**: Rank selection requires careful tuning
- **Architecture Dependence**: Optimal rank and target modules vary by model architecture
- **Research Maturity**: Less established than full fine-tuning in federated settings

### Neutral
- **Implementation Complexity**: Moderate complexity to implement adaptive rank selection
- **Debugging**: Requires understanding both base model and adapter behavior

## Alternatives Considered

### Full Fine-tuning
- **Pros**: Maximum expressiveness, well-established
- **Cons**: Prohibitively expensive for federated clients, poor DP utility
- **Verdict**: Rejected due to resource constraints

### Prefix Tuning
- **Pros**: Very parameter-efficient, good for some tasks
- **Cons**: Task-specific, less general than LoRA, unstable training
- **Verdict**: Rejected for lack of generality

### Adapter Layers
- **Pros**: Simple to implement, good performance
- **Cons**: Adds inference latency, more parameters than LoRA
- **Verdict**: Rejected due to efficiency concerns

### P-Tuning v2
- **Pros**: Strong performance on understanding tasks
- **Cons**: Task-specific, complex implementation
- **Verdict**: Rejected for complexity and specificity

### BitFit
- **Pros**: Extremely parameter-efficient (only bias terms)
- **Cons**: Limited expressiveness, poor performance on complex tasks
- **Verdict**: Rejected due to performance limitations

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [Noise-Aware LoRA for Differential Privacy](https://arxiv.org/abs/2507.09990)

---

**Date**: 2025-01-15
**Author(s)**: Daniel Schmidt
**Reviewers**: Terragon Labs Team