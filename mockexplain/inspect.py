"""Input identity, reader orchestration, and report schema."""
import hashlib
import json
from pathlib import PurePath

from .checkpoint import inspect_checkpoint
from .limits import (InspectionError, MAX_UPLOAD, MAX_SAMPLE_TENSORS,
                     SAMPLES_PER_TENSOR, MAX_SAMPLE_BYTES)
from .rules import RULE_VERSION, recommend
from .source import inspect_source, inspect_config

READER_VERSION = "mockexplain-readers-0.1.0"
SCHEMA_VERSION = 1


def inspect_files(files):
    """files is a list of (display filename, bytes); no path is ever opened."""
    if not files or len(files) > 3:
        raise InspectionError("Upload one .pt checkpoint and/or one .py architecture, with one optional .json configuration (up to three files).")
    if sum(len(data) for _, data in files) > MAX_UPLOAD:
        raise InspectionError("Combined upload exceeds the 128 MiB limit.")
    inputs, seen, parsed = [], set(), {}
    for name, data in files:
        name = name.replace('\\', '/').rsplit('/', 1)[-1]
        if not name or len(name) > 180 or any(ord(c) < 32 for c in name):
            raise InspectionError("Use a short filename without control characters.")
        kind = PurePath(name).suffix.lower()
        if kind not in ('.pt', '.py', '.json'):
            raise InspectionError("Supported inputs are .pt tensor state_dict ZIPs, .py architecture source, and optional .json configuration.")
        if kind in seen:
            raise InspectionError(f"Only one {kind} file can belong to a model profile. Upload other models as separate profiles.")
        if not data:
            raise InspectionError(f"{name} is empty. Select a nonempty model file.")
        seen.add(kind)
        inputs.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind})
        parsed[kind] = (name, data)
    if not seen.intersection({'.pt', '.py'}):
        raise InspectionError("Configuration alone cannot describe inspected structure. Attach it with a .pt checkpoint or .py architecture.")
    checkpoint, source, config = None, None, None
    if '.pt' in parsed:
        name, data = parsed['.pt']
        checkpoint = inspect_checkpoint(data, name)
    if '.py' in parsed:
        name, data = parsed['.py']
        source = inspect_source(data, name)
    if '.json' in parsed:
        name, data = parsed['.json']
        config = inspect_config(data, name)
    structures = (checkpoint['structures'] if checkpoint else []) + (source['structures'] if source else [])
    inference, methods = recommend(structures)
    missing = ["Architecture, connectivity, task and training data are unverified.",
               "Methods have not been run."]
    if checkpoint:
        missing += ["The checkpoint may be incomplete or contain buffers and shared weights; trainable parameters are unknown."]
        if not source:
            missing += ["Architecture source is missing; attach .py or JSON for context."]
    else:
        missing += ["No weights: tensor shapes, counts, dtypes and statistics are unavailable."]
    if source:
        missing += source['notes']
    if config:
        missing += [config['note']]
    sampled = checkpoint['sampled_values'] if checkpoint else 0
    total = checkpoint['tensor_element_count'] if checkpoint else None
    inputs.sort(key=lambda i: (i['kind'], i['name']))
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    profile = {"schema_version": SCHEMA_VERSION, "id": digest,
        "reader_version": READER_VERSION, "rule_version": RULE_VERSION, "inputs": inputs,
        "facts": {"tensors": checkpoint['tensors'] if checkpoint else [],
                  "tensor_element_count": total, "parameter_count": None,
                  "parameter_count_note": "Tensor counts may include buffers and shared weights.",
                  "storage_bytes": checkpoint['storage_bytes'] if checkpoint else None,
                  "dtypes": sorted({t['dtype'] for t in checkpoint['tensors']}) if checkpoint else [],
                  "classes": source['classes'] if source else [],
                  "declarations": source['declarations'] if source else [],
                  "config": config, "checkpoint_wrapper": checkpoint['wrapper'] if checkpoint else None},
        "inferences": inference, "recommendations": methods, "missing": missing,
        "coverage": {"metadata": "All supported tensor entries" if checkpoint else "Static AST call sites only",
                     "partial": True, "partial_reason": "Architecture is inferred; weight statistics are sampled.",
                     "tensors_inspected": len(checkpoint['tensors']) if checkpoint else 0,
                     "tensors_sampled": checkpoint['sampled_tensors'] if checkpoint else 0,
                     "sampled_values": sampled, "total_tensor_values": total,
                     "sample_bytes": checkpoint['sample_bytes'] if checkpoint else 0,
                     "sample_limit": {"tensors": MAX_SAMPLE_TENSORS, "values_per_tensor": SAMPLES_PER_TENSOR, "bytes": MAX_SAMPLE_BYTES},
                     "notes": (checkpoint['notes'] if checkpoint else []) +
                         (["Unsampled storage has no CRC check."] if checkpoint else []) +
                         (["Attached files are not checked against each other."] if len(inputs) > 1 else []) +
                         ["Uploaded code and pickle callables are not executed."]}}
    return profile
