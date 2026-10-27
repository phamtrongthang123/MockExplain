"""Small explicit literature rules. These recommendations are never experiments."""
RULE_VERSION = "mockexplain-rules-0.1.1"
SOURCES_CHECKED = "2026-10-27"


def method(key, title, purpose, reason, prerequisites, limitations, references):
    return {"id": key, "title": title, "purpose": purpose, "reason": reason,
            "prerequisites": prerequisites, "limitations": limitations,
            "references": [{"title": title, "url": url} for title, url in references],
            "status": "Literature recommendation; not run", "sources_checked": SOURCES_CHECKED}


def recommend(structures):
    kinds = {s['kind'] for s in structures}
    convolution = 'convolution' in kinds
    attention = 'attention' in kinds
    language = attention and 'embedding' in kinds and 'vocabulary_head' in kinds
    if convolution and attention:
        family = "Hybrid convolution / transformer candidate"
    elif convolution:
        family = "CNN / convolutional candidate"
    elif language:
        family = "Language transformer candidate"
    elif attention:
        family = "Transformer / attention candidate"
    else:
        family = "Unknown / not covered"
    suggestions = []
    if convolution:
        suggestions.append(method("filter-inspection", "Filter inspection",
            "Look at convolution kernels.",
            "Convolution-like layers or weights were found.",
            ["Convolution weights and their layout. No input or training is needed."],
            ["Weights alone rarely explain deeper features. Confirm that the weights are convolutional."],
            [("Visualizing Weights", "https://distill.pub/2020/circuits/visualizing-weights/")]))
        suggestions.append(method("grad-cam", "Grad-CAM",
            "Highlight regions associated with an output.",
            "Convolutional layers may provide suitable feature maps.",
            ["A runnable model, input, target output, feature maps and gradients. No retraining is needed."],
            ["Maps are coarse and do not prove causation. Check that the model has suitable spatial features."],
            [("Grad-CAM paper", "https://arxiv.org/abs/1610.02391")]))
    if language:
        suggestions.append(method("jacobian-lens", "J-lens",
            "Read language-model activations in vocabulary space.",
            "Attention, embeddings and a possible vocabulary head were found.",
            ["A compatible decoder, tokenizer, input and residual activations.",
             "A compatible pre-fitted J-lens, or fitting with corpus Jacobians."],
            ["Check model compatibility; the averaged readout is approximate."],
            [("J-lens paper", "https://arxiv.org/abs/2607.15495"),
             ("J-lens code", "https://github.com/anthropics/jacobian-lens")]))
        suggestions.append(method("logit-lens", "Logit lens",
            "Read intermediate activations as vocabulary predictions.",
            "Attention, embeddings and a possible vocabulary head were found.",
            ["A runnable language model, tokenizer, input, activations and compatible unembedding.",
             "The plain logit lens needs no separately trained lens; a tuned lens does."],
            ["Intermediate representations may differ from the final-layer basis, making readouts misleading."],
            [("Tuned Lens paper", "https://arxiv.org/abs/2303.08112"),
             ("LogitLens and TunedLens code", "https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py")]))
    if attention:
        suggestions.append(method("sae", "Sparse autoencoders (SAEs)",
            "Find sparse features in model activations.",
            "Attention-related structure was found.",
            ["A runnable model and representative activations.",
             "A trained SAE matched to the model and layer, or new training."],
            ["Features depend on the SAE and data. Results from language models may not transfer to other modalities."],
            [("SAE paper", "https://arxiv.org/abs/2309.08600")]))
        if not language:
            suggestions.append(method("activation-analysis", "Activation inspection",
                "Inspect activations at layers or heads.",
                "Attention-related structure was found.",
                ["A runnable model, representative inputs and access to its activations."],
                ["Attention and activation patterns alone do not establish causal influence. Tool support varies by model."],
                [("TransformerLens", "https://github.com/TransformerLensOrg/TransformerLens")]))
    if convolution and attention:
        suggestions = [s for s in suggestions if s['id'] in ('filter-inspection', 'grad-cam', 'jacobian-lens', 'sae')]
    for suggestion in suggestions:
        relevant = {'convolution'} if suggestion['id'] in ('filter-inspection', 'grad-cam') else {'attention', 'embedding', 'vocabulary_head'}
        evidence = next((s for s in structures if s['kind'] in relevant), None)
        if evidence:
            suggestion['reason'] += f" Evidence: {evidence['evidence']} ({evidence['source']})."
    inference = {"family": family, "confidence": "Architecture and task unverified",
                 "evidence": structures,
                 "note": "Names, shapes and source declarations are clues, not proof of architecture or behavior."}
    return inference, suggestions
