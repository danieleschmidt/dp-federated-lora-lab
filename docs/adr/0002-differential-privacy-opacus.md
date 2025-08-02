# ADR-0002: Adopt Differential Privacy via Opacus

## Status

Accepted

## Context

Privacy is a critical requirement for federated learning systems, especially when dealing with sensitive data like healthcare records, financial information, or personal communications. Without formal privacy guarantees, federated learning can still leak information about individual data points through model updates, gradients, or inference attacks.

We need a differential privacy solution that:
- Provides formal mathematical privacy guarantees
- Integrates seamlessly with PyTorch training loops
- Supports efficient per-sample gradient computation
- Offers flexible privacy accounting mechanisms
- Works well with parameter-efficient fine-tuning (LoRA)
- Has active community support and maintenance

The challenge is balancing privacy (lower ε) with model utility, especially in federated settings where communication rounds are limited.

## Decision

We will use PyTorch Opacus as our primary differential privacy framework.

Our implementation will:
- Use Gaussian noise mechanism for gradient perturbation
- Implement Rényi Differential Privacy (RDP) for tight privacy accounting
- Support per-client privacy budgets with heterogeneous requirements
- Apply noise-aware LoRA adaptation strategies
- Provide comprehensive privacy budget tracking and reporting

Key configuration:
```python
privacy_engine = PrivacyEngine(
    module=lora_model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,        # σ = 1.1 * sensitivity
    max_grad_norm=1.0,           # Gradient clipping bound
    batch_size=32,               # Logical batch size
    sample_size=len(dataset),    # For sampling probability
    epochs=num_epochs,
    target_epsilon=8.0,          # Privacy budget
    target_delta=1e-5,           # Failure probability
    accounting_mode="rdp"        # Rényi DP accounting
)
```

## Consequences

### Positive
- **Formal Guarantees**: Mathematical proof of (ε, δ)-differential privacy
- **PyTorch Integration**: Native support for PyTorch models and optimizers
- **Performance**: Optimized per-sample gradient computation using functorch
- **Flexibility**: Configurable noise mechanisms and accounting methods
- **Community**: Active development by Meta AI Research team
- **LoRA Compatibility**: Works well with parameter-efficient fine-tuning
- **Advanced Accounting**: RDP provides tighter privacy bounds than basic composition

### Negative
- **Utility Trade-off**: Adding noise reduces model performance (typically 2-10% accuracy drop)
- **Computational Overhead**: Per-sample gradient computation is 2-5x slower
- **Memory Usage**: Increased memory footprint for gradient storage
- **Hyperparameter Sensitivity**: Requires careful tuning of noise_multiplier and clipping
- **Batch Size Constraints**: Logical batch size affects privacy-utility trade-off

### Neutral
- **Learning Curve**: Requires understanding of differential privacy concepts
- **Integration Complexity**: Moderate complexity to integrate with federated learning
- **Debugging**: DP training can be harder to debug due to noise injection

## Alternatives Considered

### TensorFlow Privacy
- **Pros**: Well-established, good documentation, TF ecosystem integration
- **Cons**: Requires TensorFlow, not compatible with our PyTorch stack
- **Verdict**: Rejected due to framework mismatch

### JAX-Privacy
- **Pros**: Functional programming paradigm, good for research
- **Cons**: JAX ecosystem, less mature, smaller community
- **Verdict**: Rejected for ecosystem reasons

### Custom DP Implementation
- **Pros**: Full control, tailored to our specific needs
- **Cons**: High development cost, error-prone, hard to validate
- **Verdict**: Rejected due to risk and maintenance burden

### PySyft DP Module
- **Pros**: Designed for federated learning, privacy-focused
- **Cons**: Less mature than Opacus, smaller community, complex setup
- **Verdict**: Rejected for maturity concerns

### Autodp Library
- **Pros**: Pure privacy accounting, flexible mechanisms
- **Cons**: Low-level, requires manual integration, no gradient computation
- **Verdict**: Rejected for lack of ML framework integration

## Implementation Strategy

### Phase 1: Basic Integration
- Integrate Opacus with LoRA training loops
- Implement basic Gaussian mechanism
- Add privacy budget tracking

### Phase 2: Advanced Features
- Implement adaptive clipping strategies
- Add support for heterogeneous privacy requirements
- Optimize for federated learning communication patterns

### Phase 3: Production Readiness
- Add comprehensive monitoring and alerting
- Implement privacy-aware hyperparameter tuning
- Create detailed privacy reports and auditing

## Privacy Parameters

### Recommended Settings
- **Target Epsilon (ε)**: 1.0-10.0 depending on sensitivity
  - Healthcare: ε ≤ 1.0
  - Financial: ε ≤ 4.0
  - General: ε ≤ 10.0
- **Target Delta (δ)**: 1/n² where n is dataset size, max 1e-5
- **Noise Multiplier**: Start with 1.1, tune based on utility requirements
- **Max Grad Norm**: 0.1-2.0, typically 1.0
- **Batch Size**: Larger batches improve privacy amplification

### Monitoring Metrics
- Current privacy expenditure (ε, δ)
- Privacy budget remaining
- Utility metrics (accuracy, perplexity)
- Noise-to-signal ratio
- Gradient clipping frequency

## References

- [Opacus: User-Friendly Differential Privacy Library in PyTorch](https://arxiv.org/abs/2109.12298)
- [Rényi Differential Privacy](https://arxiv.org/abs/1702.07476)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Opacus Documentation](https://opacus.ai/)
- [PyTorch Opacus GitHub](https://github.com/pytorch/opacus)
- [Privacy Amplification by Subsampling](https://arxiv.org/abs/1405.7085)

---

**Date**: 2025-01-15
**Author(s)**: Daniel Schmidt
**Reviewers**: Terragon Labs Team