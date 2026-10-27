"""AST inspection only. Uploaded source is never compiled, imported, or executed."""
import ast
import json
import math

from .limits import InspectionError, MAX_AST_NODES, MAX_CONFIG, MAX_SOURCE


MODULES = {
    "Conv1d": "convolution", "Conv2d": "convolution", "Conv3d": "convolution",
    "ConvTranspose1d": "convolution", "ConvTranspose2d": "convolution",
    "ConvTranspose3d": "convolution", "Embedding": "embedding",
    "MultiheadAttention": "attention", "Transformer": "attention",
    "TransformerEncoder": "attention", "TransformerEncoderLayer": "attention",
    "TransformerDecoder": "attention", "TransformerDecoderLayer": "attention",
    "Linear": "linear", "LayerNorm": "normalization", "GroupNorm": "normalization",
    "BatchNorm1d": "normalization", "BatchNorm2d": "normalization",
    "BatchNorm3d": "normalization", "RMSNorm": "normalization",
}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def finite_literal(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [finite_literal(v) for v in value]
    if type(value) in (str, int, float, bool, type(None)):
        return value
    return None


def inspect_source(data, name):
    if len(data) > MAX_SOURCE:
        raise InspectionError("Python source exceeds the 512 KiB static-inspection limit.")
    try:
        tree = ast.parse(data.decode("utf-8-sig"), filename="uploaded-architecture.py")
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        raise InspectionError(f"Cannot parse Python source{f' at line {line}' if line else ''}. Supply valid UTF-8 Python.") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise InspectionError("Python source is too complex for bounded AST inspection (50,000 nodes).")
    aliases = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    declarations, structures = [], []
    for node in sorted((n for n in nodes if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
        call = dotted(node.func)
        head, *tail = call.split('.')
        resolved = '.'.join([aliases.get(head, head), *tail])
        constructor = resolved.rsplit('.', 1)[-1]
        if not resolved.startswith("torch.nn.") or constructor not in MODULES:
            continue
        owner = parents.get(node)
        target = dotted(owner.targets[0]) if isinstance(owner, ast.Assign) else ""
        if isinstance(owner, ast.AnnAssign):
            target = dotted(owner.target)
        args = []
        for arg in node.args[:8]:
            try:
                value = ast.literal_eval(arg)
                args.append(finite_literal(value))
            except (ValueError, TypeError, RecursionError):
                args.append(None)
        keywords = {}
        for kw in node.keywords:
            if kw.arg:
                try:
                    value = ast.literal_eval(kw.value)
                    if isinstance(value, (str, int, float, bool, tuple, list, type(None))):
                        keywords[kw.arg] = finite_literal(value)
                except (ValueError, TypeError, RecursionError):
                    pass
        declaration = {"name": target or constructor, "constructor": resolved, "line": node.lineno,
                       "args": args, "keywords": keywords, "source": name}
        declarations.append(declaration)
        kind = MODULES[constructor]
        structures.append({"kind": kind, "evidence": f"{target or constructor}: {resolved} call at line {node.lineno}", "source": name})
        if kind == "linear" and any(token in target.lower() for token in ("lm_head", "unembed", "vocab_head")):
            structures.append({"kind": "vocabulary_head", "evidence": f"{target}: named vocabulary-head candidate at line {node.lineno}", "source": name})
    classes = [n.name for n in nodes if isinstance(n, ast.ClassDef)]
    return {"classes": classes, "declarations": declarations, "structures": structures,
            "notes": ["Call sites may be unused or conditional; runtime layers and order are unknown.",
                      "Source alone does not reveal weights, dtype, runtime shapes or imported modules."]}


def json_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite JSON value")
    return value


def inspect_config(data, name):
    if len(data) > MAX_CONFIG:
        raise InspectionError("Configuration exceeds the 128 KiB JSON limit.")
    try:
        obj = json.loads(data, parse_float=json_float, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise InspectionError("Configuration must be valid finite JSON data.") from None
    if not isinstance(obj, dict):
        raise InspectionError("Configuration must be a JSON object.")
    # Record only useful scalar declarations; config claims are never measured facts.
    fields = {key: obj[key] for key in ("model_type", "architectures", "hidden_size", "num_hidden_layers",
              "num_attention_heads", "vocab_size", "is_decoder", "is_encoder_decoder", "torch_dtype")
              if key in obj and isinstance(obj[key], (str, int, bool, list))}
    fields = json.loads(json.dumps(fields, allow_nan=False))
    return {"source": name, "fields": fields, "note": "Configuration values and their match to the checkpoint are unverified."}
