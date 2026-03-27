import json
import re

import numpy as np


class _SchemaFSMV2:
    def __init__(self, schema: dict):
        self.schema = schema
        defs = schema["$defs"]
        self.schema_version = schema["properties"]["schema_version"]["const"]
        self.base_iri_re = re.compile(defs["iriString"]["pattern"])
        self.local_name_re = re.compile(defs["localName"]["pattern"])
        self.datatype_values = set(defs["datatypeId"]["enum"])
        self.ontology_item_specs = {
            "DeclareClass": [("type", "enum"), ("id", "local_name")],
            "DeclareObjectProperty": [("type", "enum"), ("id", "local_name")],
            "DeclareDataProperty": [("type", "enum"), ("id", "local_name")],
            "SubClassOf": [("type", "enum"), ("subclass", "local_name"), ("superclass", "local_name")],
            "ObjectPropertyDomain": [("type", "enum"), ("property", "local_name"), ("class", "local_name")],
            "ObjectPropertyRange": [("type", "enum"), ("property", "local_name"), ("class", "local_name")],
        }
        self.instance_item_specs = {
            "DeclareIndividual": [("type", "enum"), ("id", "local_name")],
            "ClassAssertion": [("type", "enum"), ("individual", "local_name"), ("class", "local_name")],
            "ObjectPropertyAssertion": [
                ("type", "enum"),
                ("subject", "local_name"),
                ("property", "local_name"),
                ("object", "local_name"),
            ],
            "DataPropertyAssertion": [
                ("type", "enum"),
                ("subject", "local_name"),
                ("property", "local_name"),
                ("value", "text"),
                ("datatype", "datatype"),
            ],
        }
        # Flatten all possible enum values for constraint checking
        self.all_enum_values = set(self.ontology_item_specs.keys()) | set(self.instance_item_specs.keys())
        self.reset()

    @staticmethod
    def _is_ws(ch: str) -> bool:
        return ch in " \t\r\n"

    def reset(self):
        self.mode = "top_object_open"
        self.pos = 0
        self.string_buf = ""
        self.string_escape = False
        self.finished = False

        self.array_kind = None
        self.array_after_open = False
        self.array_waiting_next = False

        self.item_type = None
        self.item_specs = None
        self.item_field_index = 0
        self.item_key_buf = ""
        self.item_value_kind = None
        self.item_expected_value = None

    def clone(self):
        other = _SchemaFSMV2.__new__(_SchemaFSMV2)
        other.schema = self.schema
        other.schema_version = self.schema_version
        other.base_iri_re = self.base_iri_re
        other.local_name_re = self.local_name_re
        other.datatype_values = self.datatype_values
        other.ontology_item_specs = self.ontology_item_specs
        other.instance_item_specs = self.instance_item_specs

        other.mode = self.mode
        other.pos = self.pos
        other.string_buf = self.string_buf
        other.string_escape = self.string_escape
        other.finished = self.finished
        other.array_kind = self.array_kind
        other.array_after_open = self.array_after_open
        other.array_waiting_next = self.array_waiting_next
        other.item_type = self.item_type
        other.item_specs = self.item_specs
        other.item_field_index = self.item_field_index
        other.item_key_buf = self.item_key_buf
        other.item_value_kind = self.item_value_kind
        other.item_expected_value = self.item_expected_value
        return other

    def can_consume(self, text: str) -> bool:
        test = self.clone()
        return test.consume(text)

    def consume(self, text: str) -> bool:
        for ch in text:
            if ord(ch) < 0x20 and ch not in ("\n", "\r", "\t"):
                return False
            if not self._consume_char(ch):
                return False
        return True

    def _consume_exact_key(self, ch: str, target: str, next_mode: str) -> bool:
        if self.pos == 0:
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self.pos = 1
            return True
        idx = self.pos - 1
        if idx < len(target):
            if ch != target[idx]:
                return False
            self.pos += 1
            return True
        if idx == len(target):
            if ch != '"':
                return False
            self.mode = next_mode
            self.pos = 0
            return True
        return False

    def _start_string_value(self, kind: str, expected_value: str | None = None):
        self.mode = "string_value"
        self.string_buf = ""
        self.string_escape = False
        self.item_value_kind = kind
        self.item_expected_value = expected_value

    def _current_item_spec(self):
        return self.item_specs[self.item_field_index]

    def _can_extend_enum_value(self, prefix: str) -> bool:
        """Check if any valid enum value starts with the prefix."""
        for allowed_value in self.all_enum_values:
            if allowed_value.startswith(prefix):
                return True
        return False

    def _complete_string_value(self) -> bool:
        value = self.string_buf
        kind = self.item_value_kind
        if kind == "schema_version":
            if value != self.schema_version:
                return False
            self.mode = "top_after_schema_value"
            return True
        if kind == "base_iri":
            if not self.base_iri_re.fullmatch(value):
                return False
            self.mode = "top_after_base_iri"
            return True
        if kind == "enum":
            allowed = self.ontology_item_specs if self.array_kind == "ontology" else self.instance_item_specs
            if value not in allowed:
                return False
            self.item_type = value
            self.item_specs = allowed[value]
            self.item_field_index = 1
            self.mode = "item_after_field"
            return True
        if kind == "local_name":
            if not self.local_name_re.fullmatch(value):
                return False
            self.mode = "item_after_field"
            return True
        if kind == "datatype":
            if value not in self.datatype_values:
                return False
            self.mode = "item_after_field"
            return True
        if kind == "text":
            self.mode = "item_after_field"
            return True
        return False

    def _consume_char(self, ch: str) -> bool:
        if self.mode == "done":
            return False

        if self.mode == "top_object_open":
            if self._is_ws(ch):
                return True
            if ch != '{':
                return False
            self.mode = "top_key_schema_version"
            self.pos = 0
            return True

        if self.mode == "top_key_schema_version":
            return self._consume_exact_key(ch, "schema_version", "top_colon_schema_version")

        if self.mode == "top_colon_schema_version":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "top_schema_value_open"
            return True

        if self.mode == "top_schema_value_open":
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self._start_string_value("schema_version")
            return True

        if self.mode == "top_after_schema_value":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_base_iri"
            self.pos = 0
            return True

        if self.mode == "top_key_base_iri":
            return self._consume_exact_key(ch, "base_iri", "top_colon_base_iri")

        if self.mode == "top_colon_base_iri":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "top_base_iri_open"
            return True

        if self.mode == "top_base_iri_open":
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self._start_string_value("base_iri")
            return True

        if self.mode == "top_after_base_iri":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_prefixes"
            self.pos = 0
            return True

        if self.mode == "top_key_prefixes":
            return self._consume_exact_key(ch, "prefixes", "top_colon_prefixes")

        if self.mode == "top_colon_prefixes":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "top_prefixes_open"
            return True

        if self.mode == "top_prefixes_open":
            if self._is_ws(ch):
                return True
            if ch != '[':
                return False
            self.mode = "top_prefixes_close"
            return True

        if self.mode == "top_prefixes_close":
            if self._is_ws(ch):
                return True
            if ch != ']':
                return False
            self.mode = "top_after_prefixes"
            return True

        if self.mode == "top_after_prefixes":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_ontology"
            self.pos = 0
            return True

        if self.mode == "top_key_ontology":
            return self._consume_exact_key(ch, "ontology", "top_colon_ontology")

        if self.mode == "top_colon_ontology":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "top_ontology_open"
            return True

        if self.mode == "top_ontology_open":
            if self._is_ws(ch):
                return True
            if ch != '[':
                return False
            self.array_kind = "ontology"
            self.array_after_open = True
            self.array_waiting_next = False
            self.mode = "item_array"
            return True

        if self.mode == "top_after_ontology":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_instances"
            self.pos = 0
            return True

        if self.mode == "top_key_instances":
            return self._consume_exact_key(ch, "instances", "top_colon_instances")

        if self.mode == "top_colon_instances":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "top_instances_open"
            return True

        if self.mode == "top_instances_open":
            if self._is_ws(ch):
                return True
            if ch != '[':
                return False
            self.array_kind = "instances"
            self.array_after_open = True
            self.array_waiting_next = False
            self.mode = "item_array"
            return True

        if self.mode == "top_object_close":
            if self._is_ws(ch):
                return True
            if ch != '}':
                return False
            self.finished = True
            self.mode = "done"
            return True

        if self.mode == "item_array":
            if self.array_after_open:
                if self._is_ws(ch):
                    return True
                if ch == ']':
                    self.array_after_open = False
                    self.mode = "top_after_ontology" if self.array_kind == "ontology" else "top_object_close"
                    return True
                if ch != '{':
                    return False
                self.array_after_open = False
                self.item_type = None
                self.item_specs = None
                self.item_field_index = 0
                self.item_key_buf = ""
                self.mode = "item_key"
                return True

            if self.array_waiting_next:
                if self._is_ws(ch):
                    return True
                if ch == ',':
                    self.array_waiting_next = False
                    self.mode = "item_open_after_comma"
                    return True
                if ch == ']':
                    self.array_waiting_next = False
                    self.mode = "top_after_ontology" if self.array_kind == "ontology" else "top_object_close"
                    return True
                return False
            return False

        if self.mode == "item_open_after_comma":
            if self._is_ws(ch):
                return True
            if ch != '{':
                return False
            self.item_type = None
            self.item_specs = None
            self.item_field_index = 0
            self.item_key_buf = ""
            self.mode = "item_key"
            return True

        if self.mode == "item_key":
            expected_key = "type" if self.item_field_index == 0 else self._current_item_spec()[0]
            return self._consume_exact_key(ch, expected_key, "item_colon")

        if self.mode == "item_colon":
            if self._is_ws(ch):
                return True
            if ch != ':':
                return False
            self.mode = "item_value_open"
            return True

        if self.mode == "item_value_open":
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            if self.item_field_index == 0:
                self._start_string_value("enum")
            else:
                _, value_kind = self._current_item_spec()
                self._start_string_value(value_kind)
            return True

        if self.mode == "string_value":
            if self.string_escape:
                self.string_buf += ch
                self.string_escape = False
                return True
            if ch == "\\":
                self.string_escape = True
                return True
            if ch == '"':
                return self._complete_string_value()
            # For enum values, only accept characters that could extend a valid enum
            if self.item_value_kind == "enum":
                test_buf = self.string_buf + ch
                if not self._can_extend_enum_value(test_buf):
                    return False
            self.string_buf += ch
            return True

        if self.mode == "item_after_field":
            if self._is_ws(ch):
                return True
            if self.item_field_index + 1 < len(self.item_specs):
                if ch != ',':
                    return False
                self.item_field_index += 1
                self.mode = "item_key"
                self.pos = 0
                return True
            if ch != '}':
                return False
            self.mode = "item_array"
            self.array_waiting_next = True
            return True

        return False


class SchemaConstrainedDecoderV2:
    def __init__(self, tokenizer, schema_path: str):
        self.tokenizer = tokenizer
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.fsm = _SchemaFSMV2(self.schema)
        self.token_text_cache = {}

    def reset(self):
        self.fsm.reset()

    def is_finished(self) -> bool:
        return self.fsm.finished

    def _token_to_text(self, token_id: int) -> str:
        cached = self.token_text_cache.get(token_id)
        if cached is not None:
            return cached
        text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.token_text_cache[token_id] = text
        return text

    def can_accept_token(self, token_id: int) -> bool:
        text = self._token_to_text(token_id)
        if text == "":
            return False
        return self.fsm.can_consume(text)

    def apply_token(self, token_id: int) -> bool:
        text = self._token_to_text(token_id)
        if text == "":
            return False
        return self.fsm.consume(text)

    def select_next_token(self, logits: np.ndarray, top_k: int = 4096) -> int | None:
        if logits.ndim != 1:
            raise ValueError("Expected 1D logits for constrained token selection")

        vocab_size = logits.shape[0]
        k = min(max(top_k, 1), vocab_size)
        candidate_ids = np.argpartition(logits, -k)[-k:]
        ranked_ids = candidate_ids[np.argsort(logits[candidate_ids])[::-1]]

        for token_id in ranked_ids.tolist():
            if self.can_accept_token(token_id):
                return int(token_id)

        if k < vocab_size:
            k2 = min(vocab_size, max(k * 2, 16384))
            candidate_ids = np.argpartition(logits, -k2)[-k2:]
            ranked_ids = candidate_ids[np.argsort(logits[candidate_ids])[::-1]]
            for token_id in ranked_ids.tolist():
                if self.can_accept_token(token_id):
                    return int(token_id)

        ranked_ids = np.argsort(logits)[::-1]
        for token_id in ranked_ids.tolist():
            if self.can_accept_token(token_id):
                return int(token_id)

        return None