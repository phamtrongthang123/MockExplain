"""Read a bounded subset of torch.save ZIPs as data, without pickle execution.

The pickle opcode interpreter below constructs only builtin containers and our
metadata records. GLOBAL names are inert symbols; REDUCE never calls a callable.
There is intentionally no pickle.load, torch.load, dynamic import, or fallback.
"""
from dataclasses import dataclass
import io
import math
import pickletools
import struct
import zipfile

from .limits import (InspectionError, MAX_ENTRIES, MAX_METADATA, MAX_OPS,
                     MAX_SAMPLE_TENSORS, MAX_TENSORS, MAX_UPLOAD, SAMPLES_PER_TENSOR)

DTYPES = {
    "FloatStorage": ("float32", "f", 4), "DoubleStorage": ("float64", "d", 8),
    "HalfStorage": ("float16", "e", 2), "BFloat16Storage": ("bfloat16", None, 2),
    "LongStorage": ("int64", "q", 8), "IntStorage": ("int32", "i", 4),
    "ShortStorage": ("int16", "h", 2), "CharStorage": ("int8", "b", 1),
    "ByteStorage": ("uint8", "B", 1), "BoolStorage": ("bool", "?", 1),
}
DTYPE_NAMES = {value[0]: key for key, value in DTYPES.items()}
BUILDERS = {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_parameter"}


@dataclass(frozen=True)
class Symbol:
    module: str
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    dtype: str
    size: int
    location: str
    untyped: bool = False


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def fail(message="Unsupported or unsafe checkpoint metadata. Export a plain tensor state_dict with torch.save(model.state_dict(), path)."):
    raise InspectionError(message)


def symbol(module, name):
    allowed = ((module == "collections" and name == "OrderedDict") or
               (module == "torch._utils" and name in BUILDERS) or
               (module == "torch" and name in {*DTYPES, *DTYPE_NAMES, "UntypedStorage"}) or
               (module == "torch.storage" and name == "UntypedStorage"))
    if not allowed:
        fail(f"Unsupported or unsafe checkpoint object: {module}.{name}. Use a tensor state_dict.")
    return Symbol(module, name)


def integer(value, maximum=2**48):
    return type(value) is int and 0 <= value <= maximum


class MetadataParser:
    def __init__(self):
        self.stack, self.memo = [], {}
        self.mark = object()
        self.storages = {}
        self.tensor_count = 0

    def pop(self):
        if not self.stack:
            fail()
        return self.stack.pop()

    def marked(self):
        items = []
        while self.stack:
            value = self.pop()
            if value is self.mark:
                return items[::-1]
            items.append(value)
        fail()

    @staticmethod
    def setitems(obj, items):
        if type(obj) is not dict or len(items) % 2:
            fail()
        for key, value in zip(items[::2], items[1::2]):
            if not isinstance(key, (str, int)) or (isinstance(key, str) and len(key) > 4096):
                fail()
            if key in obj:
                fail("Duplicate mapping key in checkpoint metadata; export an unambiguous state_dict.")
            obj[key] = value

    def storage(self, value):
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage":
            fail()
        _, kind, key, location, size = value
        if not isinstance(kind, Symbol) or kind.name not in {*DTYPES, "UntypedStorage"}:
            fail()
        if not isinstance(key, str) or not key.isdecimal() or len(key) > 32:
            fail()
        if not isinstance(location, str) or len(location) > 128 or not integer(size):
            fail()
        result = Storage(key, kind.name, size, location)
        if key in self.storages and self.storages[key] != result:
            fail("Conflicting storage descriptions in checkpoint.")
        self.storages[key] = result
        return result

    def reduce(self, function, args):
        if not isinstance(function, Symbol) or type(args) is not tuple:
            fail()
        if function == Symbol("collections", "OrderedDict"):
            if args:
                fail()
            return {}
        if function == Symbol("torch._utils", "_rebuild_parameter"):
            if len(args) != 3 or not isinstance(args[0], Tensor) or type(args[1]) is not bool or args[2] != {}:
                fail()
            return args[0]
        if function.module != "torch._utils" or function.name not in BUILDERS:
            fail()
        expected = {"_rebuild_tensor": (4,), "_rebuild_tensor_v2": (6, 7), "_rebuild_tensor_v3": (7, 8)}
        if len(args) not in expected[function.name]:
            fail()
        storage, offset, shape, stride = args[:4]
        if not isinstance(storage, Storage) or not integer(offset):
            fail()
        if type(shape) is not tuple or type(stride) is not tuple or len(shape) != len(stride) or len(shape) > 16:
            fail()
        if not all(integer(v) for v in (*shape, *stride)) or math.prod(shape) > 2**48:
            fail("Tensor dimensions exceed the bounded reader's limits.")
        if len(args) >= 6 and (type(args[4]) is not bool or args[5] not in ({}, None)):
            fail()
        if function.name == "_rebuild_tensor_v3":
            dtype = args[6]
            if not isinstance(dtype, Symbol) or dtype.module != "torch" or dtype.name not in DTYPE_NAMES:
                fail("Unsupported tensor dtype; use dense real or integer tensors.")
            if storage.dtype != "UntypedStorage":
                fail()
            storage = Storage(storage.key, DTYPE_NAMES[dtype.name], storage.size, storage.location, True)
        elif storage.dtype not in DTYPES:
            fail()
        if len(args) == expected[function.name][-1] and function.name != "_rebuild_tensor" and len(args) > (7 if function.name.endswith("v3") else 6):
            if args[-1] not in ({}, None):
                fail("Tensor metadata flags are unsupported; export plain dense tensors.")
        self.tensor_count += 1
        if self.tensor_count > MAX_TENSORS:
            fail("Checkpoint exceeds the 10,000-tensor metadata limit.")
        return Tensor(storage, offset, shape, stride)

    def parse(self, data):
        try:
            for count, (op, arg, pos) in enumerate(pickletools.genops(data), 1):
                if count > MAX_OPS or len(self.stack) > 50000 or len(self.memo) > 50000:
                    fail("Checkpoint metadata exceeds bounded parser limits.")
                code = op.name
                if code in {"PROTO", "FRAME"}:
                    if code == "PROTO" and arg > 5:
                        fail("Unsupported checkpoint pickle protocol.")
                elif code == "MARK":
                    self.stack.append(self.mark)
                elif code in {"NONE", "NEWTRUE", "NEWFALSE"}:
                    self.stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[code])
                elif code in {"BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "BINFLOAT", "FLOAT",
                              "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"}:
                    if isinstance(arg, float) and not math.isfinite(arg):
                        fail("Non-finite scalar checkpoint metadata is unsupported.")
                    self.stack.append(arg)
                elif code in {"EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"}:
                    self.stack.append({} if code == "EMPTY_DICT" else [] if code == "EMPTY_LIST" else ())
                elif code in {"TUPLE", "LIST", "DICT"}:
                    items = self.marked()
                    if code == "DICT":
                        value = {}
                        self.setitems(value, items)
                    else:
                        value = tuple(items) if code == "TUPLE" else items
                    self.stack.append(value)
                elif code in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                    size = int(code[-1])
                    if len(self.stack) < size:
                        fail()
                    value = tuple(self.stack[-size:])
                    del self.stack[-size:]
                    self.stack.append(value)
                elif code in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}:
                    index = len(self.memo) if code == "MEMOIZE" else int(arg)
                    if not integer(index, 50000) or not self.stack:
                        fail()
                    self.memo[index] = self.stack[-1]
                elif code in {"BINGET", "LONG_BINGET", "GET"}:
                    self.stack.append(self.memo[int(arg)])
                elif code in {"SETITEM", "SETITEMS"}:
                    if code == "SETITEM":
                        value, key = self.pop(), self.pop()
                        items = [key, value]
                    else:
                        items = self.marked()
                    self.setitems(self.stack[-1], items)
                elif code in {"APPEND", "APPENDS"}:
                    items = [self.pop()] if code == "APPEND" else self.marked()
                    if type(self.stack[-1]) is not list:
                        fail()
                    self.stack[-1].extend(items)
                elif code == "GLOBAL":
                    self.stack.append(symbol(*arg.split(' ', 1)))
                elif code == "STACK_GLOBAL":
                    name, module = self.pop(), self.pop()
                    self.stack.append(symbol(module, name))
                elif code == "BINPERSID":
                    self.stack.append(self.storage(self.pop()))
                elif code == "REDUCE":
                    args, function = self.pop(), self.pop()
                    self.stack.append(self.reduce(function, args))
                elif code == "BUILD":
                    state = self.pop()
                    # state_dict's OrderedDict may carry inert version metadata.
                    if type(self.stack[-1]) is not dict or type(state) is not dict or set(state) != {"_metadata"}:
                        fail()
                elif code == "STOP":
                    if len(self.stack) != 1 or pos != len(data) - 1:
                        fail()
                    return self.stack[0]
                else:
                    fail(f"Unsupported checkpoint opcode {code}; use a tensor state_dict ZIP.")
        except InspectionError:
            raise
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, UnicodeError):
            fail("Malformed checkpoint metadata. Supply a valid torch.save tensor state_dict ZIP.")
        fail("Incomplete checkpoint metadata.")


def sample_tensor(raw_storage, tensor, endian):
    dtype, code, size = DTYPES[tensor.storage.dtype]
    total = math.prod(tensor.shape)
    count = min(total, SAMPLES_PER_TENSOR)
    if not count:
        return None
    # Spread a deterministic sample across the logical tensor, including endpoints.
    indices = [i * (total - 1) // max(1, count - 1) for i in range(count)]
    values = []
    for index in indices:
        offset = tensor.offset
        for dim, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
            offset += (index % dim) * stride
            index //= dim
        raw = raw_storage[offset * size:(offset + 1) * size]
        if len(raw) != size:
            fail("Truncated tensor storage.")
        if dtype == "bfloat16":
            bits = int.from_bytes(raw, "little" if endian == '<' else "big") << 16
            value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
        else:
            value = struct.unpack(endian + code, raw)[0]
        values.append(value)
    finite = [v for v in values if math.isfinite(v)]
    return {"sampled_values": count, "total_values": total, "sample_bytes": count * size,
            "strategy": "evenly spaced logical indices; deterministic, not random or a full distribution estimate",
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "mean": math.fsum(v / len(finite) for v in finite) if finite else None,
            "zero_fraction": sum(v == 0 for v in values) / count, "nonfinite_values": count - len(finite)}


def inspect_checkpoint(data, name):
    try:
        return _inspect_checkpoint(data, name)
    except InspectionError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError, EOFError, struct.error):
        fail("Malformed or truncated PyTorch ZIP checkpoint. Export a plain tensor state_dict using torch.save.")


def _inspect_checkpoint(data, name):
    if len(data) > MAX_UPLOAD:
        fail("Checkpoint exceeds the 128 MiB upload limit.")
    if not zipfile.is_zipfile(io.BytesIO(data)):
        fail("Unsupported checkpoint. Legacy raw pickles and full models are not supported; export trusted weights as a state_dict ZIP.")
    # Check directory size/count before ZipFile allocates one object per entry.
    tail = data[-65557:]
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0 or len(tail) - eocd < 22:
        fail("Invalid ZIP directory.")
    disk, directory_disk, disk_count, count, directory_size, _, comment_len = struct.unpack_from('<4H2LH', tail, eocd + 4)
    if disk or directory_disk or disk_count != count or count > MAX_ENTRIES or directory_size > 4 * 1024 * 1024 or eocd + 22 + comment_len != len(tail):
        fail("ZIP directory exceeds reader limits or uses an unsupported multi-disk layout.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ENTRIES:
            fail("Checkpoint exceeds the 12,000 ZIP-entry limit.")
        names = [e.filename for e in entries]
        if len(set(names)) != len(names) or any('..' in n.split('/') or n.startswith('/') or '\\' in n for n in names):
            fail("Ambiguous or unsafe ZIP paths in checkpoint.")
        metadata = [e for e in entries if e.filename.endswith('/data.pkl') or e.filename == 'data.pkl']
        if len(metadata) != 1:
            fail("Expected one tensor state_dict data.pkl entry; TorchScript and packaged models are unsupported.")
        meta = metadata[0]
        if meta.file_size > MAX_METADATA:
            fail("Checkpoint metadata exceeds the 2 MiB inspection limit.")
        if any(e.compress_type != zipfile.ZIP_STORED or e.flag_bits & 1 for e in entries):
            fail("Compressed or encrypted checkpoint entries are unsupported. Use the ordinary uncompressed torch.save ZIP format.")
        prefix = meta.filename[:-len('data.pkl')]
        if any(n.startswith(prefix + 'code/') or n == prefix + 'constants.pkl' for n in names):
            fail("TorchScript archives are unsupported; export a tensor state_dict.")
        obj = MetadataParser().parse(archive.read(meta))
        wrapper = None
        if type(obj) is not dict:
            fail("Expected a mapping of tensor names to dense tensors.")
        for key in ("state_dict", "model_state_dict"):
            if key in obj and type(obj[key]) is dict:
                wrapper, obj = key, obj[key]
                break
        if not obj or any(type(k) is not str or not isinstance(v, Tensor) for k, v in obj.items()):
            fail("No plain tensor state_dict found. Only string tensor keys and dense real/integer tensors are supported.")
        if len(obj) > MAX_TENSORS:
            fail("Checkpoint exceeds the 10,000-tensor limit.")
        by_name = {e.filename: e for e in entries}
        endian = '<'
        order_name = prefix + 'byteorder'
        if order_name in by_name:
            if by_name[order_name].file_size > 16:
                fail("Invalid checkpoint byte order.")
            order = archive.read(order_name).decode('ascii')
            if order not in ('little', 'big'):
                fail("Invalid checkpoint byte order.")
            endian = '<' if order == 'little' else '>'
        rows, storages, structures = [], {}, []
        # Validate all metadata and storage extents before reading any weight bytes.
        for key, tensor in sorted(obj.items()):
            storage = tensor.storage
            dtype, _, size = DTYPES[storage.dtype]
            path = prefix + 'data/' + storage.key
            if path not in by_name:
                fail(f"Missing raw tensor storage for {key}.")
            entry = by_name[path]
            expected_bytes = storage.size if storage.untyped else storage.size * size
            if entry.file_size != expected_bytes:
                fail(f"Tensor storage size disagrees with metadata for {key}.")
            storage_view(data, entry)  # Validate local headers/extents for unsampled tensors too.
            total = math.prod(tensor.shape)
            last = tensor.offset + sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride)) if total else tensor.offset
            if (total and (last + 1) * size > entry.file_size) or (not total and tensor.offset * size > entry.file_size):
                fail(f"Tensor view exceeds storage bounds for {key}.")
            storages[storage.key] = {"bytes": entry.file_size}
            row = {"name": key, "shape": list(tensor.shape), "stride": list(tensor.stride), "dtype": dtype,
                   "numel": total, "storage_key": storage.key, "device_recorded": storage.location,
                   "statistics": None, "source": name}
            rows.append(row)
            lower = key.lower()
            if len(tensor.shape) in (3, 4, 5) and key.endswith('weight'):
                structures.append({"kind": "convolution", "evidence": f"{key}: rank-{len(tensor.shape)} weight {list(tensor.shape)}; convolution candidate, not confirmed", "source": name})
            if any(t in lower for t in ('attn.', 'attention.', 'q_proj.', 'k_proj.', 'v_proj.', 'in_proj_weight')):
                structures.append({"kind": "attention", "evidence": f"{key}: attention-related key naming", "source": name})
            if len(tensor.shape) == 2 and any(t in lower for t in ('embed', 'wte.', 'tok_embeddings.')):
                structures.append({"kind": "embedding", "evidence": f"{key}: embedding-related matrix naming", "source": name})
            if len(tensor.shape) == 2 and any(t in lower for t in ('lm_head.', 'unembed.', 'vocab_head.')):
                structures.append({"kind": "vocabulary_head", "evidence": f"{key}: vocabulary-head matrix naming", "source": name})
            if any(t in lower for t in ('norm.', 'ln_', 'ln.', 'bn.', 'running_mean', 'running_var')):
                structures.append({"kind": "normalization", "evidence": f"{key}: normalization-related key naming", "source": name})
        sampled_bytes = 0
        for row, (_, tensor) in zip(rows[:MAX_SAMPLE_TENSORS], sorted(obj.items())):
            row['statistics'] = sample_tensor(storage_view(data, by_name[prefix + 'data/' + tensor.storage.key]), tensor, endian)
            if row['statistics']:
                sampled_bytes += row['statistics']['sample_bytes']
        return {"tensors": rows, "structures": structures, "tensor_element_count": sum(r['numel'] for r in rows),
                "storage_bytes": sum(s['bytes'] for s in storages.values()), "sample_bytes": sampled_bytes,
                "sampled_values": sum(r['statistics']['sampled_values'] for r in rows if r['statistics']),
                "sampled_tensors": sum(r['statistics'] is not None for r in rows), "wrapper": wrapper,
                "byteorder": "little" if endian == '<' else "big",
                "notes": (["No byteorder entry: assumes little-endian storage (older torch.save convention)."] if order_name not in by_name else [])}


def storage_view(data, entry):
    """Direct slices avoid ZIP stream seeks that could silently scan whole weights."""
    start = entry.header_offset
    if data[start:start + 4] != b'PK\x03\x04' or start + 30 > len(data):
        fail("Invalid raw-storage ZIP header.")
    flags, compression = struct.unpack_from('<HH', data, start + 6)
    name_len, extra_len = struct.unpack_from('<HH', data, start + 26)
    offset = start + 30 + name_len + extra_len
    if compression != zipfile.ZIP_STORED or flags & 1 or offset + entry.file_size > len(data):
        fail("Invalid raw-storage ZIP extent.")
    try:
        local_name = data[start + 30:start + 30 + name_len].decode('utf-8' if flags & 0x800 else 'cp437')
    except UnicodeError:
        fail('Invalid raw-storage filename encoding.')
    if local_name != entry.filename:
        fail('Local and directory storage names disagree.')
    return memoryview(data)[offset:offset + entry.file_size]
