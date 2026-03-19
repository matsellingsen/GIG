import json
import re

import numpy as np


class _SchemaFSM:
    """Incremental FSM for a canonical minified JSON format driven by schema rules."""

    def __init__(self, schema: dict):
        defs = schema.get("$defs", {})
        required = schema.get("required", [])
        expected_order = ["schema_version", "base_iri", "prefixes", "ontology", "instances"]

        if required != expected_order:
            raise ValueError(
                "Schema required field order must be exactly: "
                "schema_version, base_iri, prefixes, ontology, instances"
            )

        self.schema_version = schema["properties"]["schema_version"]["const"]
        self.base_iri_re = re.compile(defs["iriString"]["pattern"])

        # Prefix object fields
        self.prefix_re = re.compile(defs["prefixDecl"]["properties"]["prefix"]["pattern"])
        self.prefix_iri_re = re.compile(defs["iriString"]["pattern"])

        # Axiom rules
        self.tbox_rules = [re.compile(item["pattern"]) for item in defs["tboxAxiom"]["oneOf"]]
        self.abox_rules = [re.compile(item["pattern"]) for item in defs["aboxAxiom"]["oneOf"]]

        self.reset()

    def reset(self):
        self.mode = "top_object_open"
        self.literal = ""
        self.pos = 0

        self.string_buf = ""
        self.string_escape = False
        self.string_validator = None

        self.array_item_buf = ""
        self.array_item_escape = False
        self.array_rules = None
        self.array_after_open = False
        self.array_waiting_next = False

        self.prefix_obj_field = ""
        self.prefix_value_buf = ""
        self.prefix_escape = False
        self.prefix_prefix = None
        self.prefix_iri = None

        self.finished = False

    @staticmethod
    def _is_ws(ch: str) -> bool:
        return ch in " \t\r\n"

    def clone(self):
        other = _SchemaFSM.__new__(_SchemaFSM)
        other.schema_version = self.schema_version
        other.base_iri_re = self.base_iri_re
        other.prefix_re = self.prefix_re
        other.prefix_iri_re = self.prefix_iri_re
        other.tbox_rules = self.tbox_rules
        other.abox_rules = self.abox_rules

        other.mode = self.mode
        other.literal = self.literal
        other.pos = self.pos

        other.string_buf = self.string_buf
        other.string_escape = self.string_escape
        other.string_validator = self.string_validator

        other.array_item_buf = self.array_item_buf
        other.array_item_escape = self.array_item_escape
        other.array_rules = self.array_rules
        other.array_after_open = self.array_after_open
        other.array_waiting_next = self.array_waiting_next

        other.prefix_obj_field = self.prefix_obj_field
        other.prefix_value_buf = self.prefix_value_buf
        other.prefix_escape = self.prefix_escape
        other.prefix_prefix = self.prefix_prefix
        other.prefix_iri = self.prefix_iri

        other.finished = self.finished
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

    def _switch_literal(self, literal: str):
        self.mode = "literal"
        self.literal = literal
        self.pos = 0

    def _consume_exact_key(self, ch: str, target: str, next_mode: str) -> bool:
        if self.pos == 0:
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self.pos = 1
            return True

        key_index = self.pos - 1
        if key_index < len(target):
            if ch != target[key_index]:
                return False
            self.pos += 1
            return True

        if key_index == len(target):
            if ch != '"':
                return False
            self.mode = next_mode
            self.pos = 0
            return True

        return False

    def _finish_literal_or_advance(self):
        if self.mode != "literal" or self.pos != len(self.literal):
            return True

        lit = self.literal
        if lit == '}':
            self.finished = True
            self.mode = "done"
        else:
            return False
        return True

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
            self.mode = "fixed_string"
            self.string_buf = ""
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
            self.mode = "validated_string"
            self.string_validator = self.base_iri_re
            self.string_buf = ""
            self.string_escape = False
            self.literal = ',"prefixes":['
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
            self.mode = "prefix_array"
            self.array_after_open = True
            self.array_waiting_next = False
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
            self.mode = "axiom_array"
            self.array_rules = self.tbox_rules
            self.array_after_open = True
            self.array_waiting_next = False
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
            self.mode = "axiom_array"
            self.array_rules = self.abox_rules
            self.array_after_open = True
            self.array_waiting_next = False
            return True

        if self.mode == "top_object_close":
            if self._is_ws(ch):
                return True
            if ch != '}':
                return False
            self.finished = True
            self.mode = "done"
            return True

        if self.mode == "literal":
            if self.pos >= len(self.literal) or ch != self.literal[self.pos]:
                return False
            self.pos += 1
            return self._finish_literal_or_advance()

        if self.mode == "fixed_string":
            target = self.schema_version
            idx = len(self.string_buf)
            if idx == 0 and self._is_ws(ch):
                return True
            if idx >= len(target) and self._is_ws(ch):
                return True
            if idx >= len(target):
                return False
            if ch != target[idx]:
                return False
            self.string_buf += ch
            if self.string_buf == target:
                self.mode = "top_after_schema_value"
                self.string_buf = ""
            return True

        if self.mode == "top_after_schema_value":
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self.mode = "top_after_schema_comma"
            return True

        if self.mode == "top_after_schema_comma":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_base_iri"
            self.pos = 0
            return True

        if self.mode == "validated_string":
            if self.string_escape:
                self.string_buf += ch
                self.string_escape = False
                return True
            if ch == "\\":
                self.string_escape = True
                return True
            if ch == '"':
                if self.string_validator is None or not self.string_validator.fullmatch(self.string_buf):
                    return False
                self.string_buf = ""
                self.string_validator = None
                if self.literal == ',"prefixes":[':
                    self.mode = "top_after_base_iri"
                else:
                    self.mode = "literal"
                return True
            self.string_buf += ch
            return True

        if self.mode == "top_after_base_iri":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_prefixes"
            self.pos = 0
            return True

        if self.mode == "prefix_array":
            if self.array_after_open:
                if self._is_ws(ch):
                    return True
                if ch == ']':
                    self.array_after_open = False
                    self.mode = "top_after_prefixes"
                    return True
                if ch == '{':
                    self.array_after_open = False
                    self.mode = "prefix_obj_key"
                    self.prefix_obj_field = ""
                    return True
                return False
            if self.array_waiting_next:
                if self._is_ws(ch):
                    return True
                if ch == ',':
                    self.array_waiting_next = False
                    self.mode = "prefix_obj_key_wait"
                    return True
                if ch == ']':
                    self.array_waiting_next = False
                    self.mode = "top_after_prefixes"
                    return True
                return False
            return False

        if self.mode == "top_after_prefixes":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_ontology"
            self.pos = 0
            return True

        if self.mode == "prefix_obj_key_wait":
            if self._is_ws(ch):
                return True
            if ch != '{':
                return False
            self.mode = "prefix_obj_key"
            self.prefix_obj_field = ""
            return True

        if self.mode == "prefix_obj_key":
            if self.prefix_obj_field == "":
                if self._is_ws(ch):
                    return True
                if ch != '"':
                    return False
                self.prefix_obj_field = "collect"
                self.string_buf = ""
                return True
            if self.prefix_obj_field == "collect":
                if ch == '"':
                    if self.string_buf == "prefix":
                        self.prefix_obj_field = "colon_prefix"
                        return True
                    if self.string_buf == "iri":
                        self.prefix_obj_field = "colon_iri"
                        return True
                    return False
                self.string_buf += ch
                return True
            if self.prefix_obj_field == "colon_prefix":
                if self._is_ws(ch):
                    return True
                if ch != ':':
                    return False
                self.prefix_obj_field = "value_prefix_open"
                return True
            if self.prefix_obj_field == "value_prefix_open":
                if self._is_ws(ch):
                    return True
                if ch != '"':
                    return False
                self.prefix_obj_field = "value_prefix"
                self.prefix_value_buf = ""
                self.prefix_escape = False
                return True
            if self.prefix_obj_field == "value_prefix":
                if self.prefix_escape:
                    self.prefix_value_buf += ch
                    self.prefix_escape = False
                    return True
                if ch == "\\":
                    self.prefix_escape = True
                    return True
                if ch == '"':
                    if not self.prefix_re.fullmatch(self.prefix_value_buf):
                        return False
                    self.prefix_prefix = self.prefix_value_buf
                    self.prefix_obj_field = "comma"
                    return True
                self.prefix_value_buf += ch
                return True
            if self.prefix_obj_field == "comma":
                if self._is_ws(ch):
                    return True
                if ch != ',':
                    return False
                self.prefix_obj_field = "iri_key_open"
                return True
            if self.prefix_obj_field == "iri_key_open":
                if self._is_ws(ch):
                    return True
                if ch != '"':
                    return False
                self.prefix_obj_field = "iri_key_collect"
                self.string_buf = ""
                return True
            if self.prefix_obj_field == "iri_key_collect":
                if ch == '"':
                    if self.string_buf != "iri":
                        return False
                    self.prefix_obj_field = "colon_iri"
                    return True
                self.string_buf += ch
                return True
            if self.prefix_obj_field == "colon_iri":
                if self._is_ws(ch):
                    return True
                if ch != ':':
                    return False
                self.prefix_obj_field = "value_iri_open"
                return True
            if self.prefix_obj_field == "value_iri_open":
                if self._is_ws(ch):
                    return True
                if ch != '"':
                    return False
                self.prefix_obj_field = "value_iri"
                self.prefix_value_buf = ""
                self.prefix_escape = False
                return True
            if self.prefix_obj_field == "value_iri":
                if self.prefix_escape:
                    self.prefix_value_buf += ch
                    self.prefix_escape = False
                    return True
                if ch == "\\":
                    self.prefix_escape = True
                    return True
                if ch == '"':
                    if not self.prefix_iri_re.fullmatch(self.prefix_value_buf):
                        return False
                    self.prefix_iri = self.prefix_value_buf
                    self.prefix_obj_field = "end"
                    return True
                self.prefix_value_buf += ch
                return True
            if self.prefix_obj_field == "end":
                if self._is_ws(ch):
                    return True
                if ch != '}':
                    return False
                self.mode = "prefix_array"
                self.array_waiting_next = True
                self.prefix_obj_field = ""
                return True
            return False

        if self.mode == "axiom_array":
            if self.array_after_open:
                if self._is_ws(ch):
                    return True
                if ch == ']':
                    self.array_after_open = False
                    if self.array_rules is self.tbox_rules:
                        self.mode = "top_after_ontology"
                    else:
                        self.mode = "top_object_close"
                    return True
                if ch == '"':
                    self.array_after_open = False
                    self.mode = "axiom_item"
                    self.array_item_buf = ""
                    self.array_item_escape = False
                    return True
                return False
            if self.array_waiting_next:
                if self._is_ws(ch):
                    return True
                if ch == ',':
                    self.array_waiting_next = False
                    self.mode = "axiom_item_open"
                    return True
                if ch == ']':
                    self.array_waiting_next = False
                    if self.array_rules is self.tbox_rules:
                        self.mode = "top_after_ontology"
                    else:
                        self.mode = "top_object_close"
                    return True
                return False
            return False

        if self.mode == "top_after_ontology":
            if self._is_ws(ch):
                return True
            if ch != ',':
                return False
            self.mode = "top_key_instances"
            self.pos = 0
            return True

        if self.mode == "axiom_item_open":
            if self._is_ws(ch):
                return True
            if ch != '"':
                return False
            self.mode = "axiom_item"
            self.array_item_buf = ""
            self.array_item_escape = False
            return True

        if self.mode == "axiom_item":
            if self.array_item_escape:
                self.array_item_buf += ch
                self.array_item_escape = False
                return True
            if ch == "\\":
                self.array_item_escape = True
                return True
            if ch == '"':
                if not any(rx.fullmatch(self.array_item_buf) for rx in self.array_rules):
                    return False
                self.mode = "axiom_array"
                self.array_waiting_next = True
                return True
            self.array_item_buf += ch
            return True

        return False


class SchemaConstrainedDecoder:
    def __init__(self, tokenizer, schema_path: str):
        self.tokenizer = tokenizer
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.fsm = _SchemaFSM(self.schema)
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

        # Fallback: expand search once if top_k was too narrow.
        if k < vocab_size:
            k2 = min(vocab_size, max(k * 2, 16384))
            candidate_ids = np.argpartition(logits, -k2)[-k2:]
            ranked_ids = candidate_ids[np.argsort(logits[candidate_ids])[::-1]]
            for token_id in ranked_ids.tolist():
                if self.can_accept_token(token_id):
                    return int(token_id)

        # Final fallback: scan full vocabulary in descending logit order.
        ranked_ids = np.argsort(logits)[::-1]
        for token_id in ranked_ids.tolist():
            if self.can_accept_token(token_id):
                return int(token_id)

        return None
