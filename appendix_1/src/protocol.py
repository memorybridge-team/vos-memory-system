"""Cold versus prefilled-memory SAM2 protocol and passive memory auditing."""
from collections import UserDict

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODELS = {
    "tiny": ("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "small": ("configs/sam2/sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "plus": ("configs/sam2/sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "large": ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}
MODEL_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
CONDITIONS = ("cold", "full_memory")
PROTOCOL_NAME = "prefilled_native_memory_v1"


class MemoryAuditDict(UserDict):
    """Behavior-preserving dict that passively records native SAM2 memory use.

    ``initial`` contains the ordinary-frame records copied from the completed
    Cold pass. Lookups return exactly the requested key, and writes proceed
    normally. The wrapper records whether each successful lookup used an
    injected first-pass record or a record written during the second pass.
    """

    def __init__(self, initial, first_frame):
        super().__init__()
        self.data.update(initial)
        self.provenance = {key: "first_pass" for key in initial}
        self.initial_keys = set(initial)
        self.first_frame = first_frame
        self.events = []
        self.active = False
        self.query_frame = None
        self.writes = 0
        self.updated_keys = set()
        self.overwritten_first_pass_keys = set()
        self.removed_first_pass_keys = set()

    def _query_frame(self):
        return self.query_frame

    def get(self, key, default=None):
        value = self.data.get(key, default)
        if self.active and key in self.data:
            self.events.append({
                "query_frame": self._query_frame(),
                "requested_frame": key,
                "memory_frame": key,
                "source_pass": self.provenance[key],
                "is_future": (self.query_frame is not None and key > self.query_frame),
            })
        return value

    def __setitem__(self, key, value):
        self.writes += 1
        if key in self.data and self.provenance.get(key) == "first_pass":
            self.overwritten_first_pass_keys.add(key)
        self.data[key] = value
        self.provenance[key] = "second_pass"
        self.updated_keys.add(key)

    def __delitem__(self, key):
        if self.provenance.get(key) == "first_pass":
            self.removed_first_pass_keys.add(key)
        del self.data[key]
        self.provenance.pop(key, None)

    def remaining_first_pass_records(self):
        return sum(self.provenance.get(key) == "first_pass" for key in self.data)

    def unique_events(self):
        seen, result = set(), []
        for event in self.events:
            identity = tuple(event.values())
            if identity not in seen:
                seen.add(identity)
                result.append(event)
        return result


def instrument_memory_state(inference_state, first_frame):
    """Wrap prefilled ordinary-frame banks without changing their behavior."""
    audits = {}
    for obj_index, outputs in inference_state["output_dict_per_obj"].items():
        audit = MemoryAuditDict(outputs["non_cond_frame_outputs"], first_frame)
        outputs["non_cond_frame_outputs"] = audit
        audits[obj_index] = audit
    return audits


def set_audit_active(audits, active):
    for audit in audits.values():
        audit.active = active


def audit_summary(audits):
    events = []
    for obj_index, audit in audits.items():
        events.extend({"object_index": obj_index, **event}
                      for event in audit.unique_events())
    counts = {"first_pass": 0, "second_pass": 0}
    for event in events:
        counts[event["source_pass"]] += 1
    return {"events": events, "unique_memory_hits": counts}


def memory_record_count(inference_state):
    return sum(
        len(records)
        for outputs in inference_state["output_dict_per_obj"].values()
        for records in outputs.values()
    )
