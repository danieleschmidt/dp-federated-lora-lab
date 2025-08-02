# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the dp-federated-lora-lab project.

## About ADRs

Architecture Decision Records (ADRs) are documents that capture important architectural decisions made during the project development, along with their context and consequences.

## ADR Format

We use the following format for our ADRs:

```markdown
# ADR-XXXX: [Title]

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

[Describe the forces at play, including technological, political, social, and project local factors]

## Decision

[Describe our response to these forces]

## Consequences

[Describe the resulting context, after applying the decision]

## Alternatives Considered

[List other options that were considered]

## References

[Links to relevant documents, discussions, or external resources]
```

## Current ADRs

- [ADR-0001: Use LoRA for Parameter-Efficient Fine-tuning](0001-lora-parameter-efficient-finetuning.md)
- [ADR-0002: Adopt Differential Privacy via Opacus](0002-differential-privacy-opacus.md)
- [ADR-0003: Implement Secure Aggregation Protocol](0003-secure-aggregation-protocol.md)
- [ADR-0004: Choose gRPC for Client-Server Communication](0004-grpc-communication-protocol.md)
- [ADR-0005: Use PyTorch as Primary ML Framework](0005-pytorch-ml-framework.md)

## Creating New ADRs

1. Copy the template from `template.md`
2. Number the ADR sequentially (XXXX format with zero padding)
3. Use a descriptive title
4. Fill out all sections thoroughly
5. Submit for review via pull request
6. Update this README with the new ADR

## ADR Workflow

1. **Proposed**: Initial draft, under discussion
2. **Accepted**: Decision has been made and is being implemented
3. **Deprecated**: No longer recommended, but not actively harmful
4. **Superseded**: Replaced by a newer decision (reference the superseding ADR)