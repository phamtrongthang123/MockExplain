"""One set of public inspection bounds; all readers enforce these limits."""
MAX_UPLOAD = 128 * 1024 * 1024
MAX_SOURCE = 512 * 1024
MAX_CONFIG = 128 * 1024
MAX_METADATA = 2 * 1024 * 1024
MAX_ENTRIES = 12000
MAX_TENSORS = 10000
MAX_OPS = 200000
MAX_AST_NODES = 50000
MAX_SAMPLE_TENSORS = 32
SAMPLES_PER_TENSOR = 256
MAX_SAMPLE_BYTES = MAX_SAMPLE_TENSORS * SAMPLES_PER_TENSOR * 8


class InspectionError(ValueError):
    """A bounded, user-actionable rejection of an input."""
