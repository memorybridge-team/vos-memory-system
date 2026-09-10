import unittest

from src.protocol import (MemoryAuditDict, audit_summary, instrument_memory_state,
                          memory_record_count, set_audit_active, MODELS)


class MemoryTests(unittest.TestCase):
    def test_prefilled_bank_preserves_native_lookup_and_allows_writes(self):
        bank = MemoryAuditDict({1: "cold-one", 2: "cold-two"}, 0)
        bank.active = True
        bank.query_frame = 1
        self.assertEqual(bank.get(0, "missing"), "missing")
        self.assertEqual(bank.unique_events(), [])
        self.assertEqual(bank.get(2), "cold-two")
        bank[1] = "new-one"
        self.assertEqual(bank.get(1), "new-one")
        self.assertEqual(bank.writes, 1)
        self.assertEqual(bank.overwritten_first_pass_keys, {1})
        self.assertEqual(bank.remaining_first_pass_records(), 1)
        self.assertEqual(bank.unique_events()[0]["query_frame"], 1)
        self.assertEqual(bank.unique_events()[0]["memory_frame"], 2)
        self.assertTrue(bank.unique_events()[0]["is_future"])
        self.assertEqual(bank.unique_events()[1]["source_pass"], "second_pass")
        del bank[2]
        self.assertEqual(bank.removed_first_pass_keys, {2})
        self.assertEqual(bank.remaining_first_pass_records(), 0)

    def test_original_four_model_mapping(self):
        self.assertEqual(set(MODELS), {"tiny", "small", "plus", "large"})
        self.assertTrue(MODELS["plus"][0].endswith("sam2_hiera_b+.yaml"))
        self.assertTrue(all("sam2.1" not in item for pair in MODELS.values() for item in pair))

    def test_first_pass_entries_are_distinguished_from_overwrites(self):
        memory = MemoryAuditDict({1: "old-1", 2: "old-2"}, first_frame=0)
        memory.active = True
        self.assertEqual(memory.get(1), "old-1")
        memory[1] = "new-1"
        self.assertEqual(memory.get(1), "new-1")
        self.assertEqual(memory.get(2), "old-2")
        self.assertEqual(
            [event["source_pass"] for event in memory.unique_events()],
            ["first_pass", "second_pass", "first_pass"],
        )

    def test_state_instrumentation_does_not_remove_memory(self):
        state = {"output_dict_per_obj": {0: {
            "cond_frame_outputs": {0: "prompt"},
            "non_cond_frame_outputs": {1: "one", 2: "two"},
        }}}
        audits = instrument_memory_state(state, 0)
        self.assertEqual(memory_record_count(state), 3)
        set_audit_active(audits, True)
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"].get(2)
        summary = audit_summary(audits)
        self.assertEqual(summary["unique_memory_hits"]["first_pass"], 1)
        self.assertEqual(summary["events"][0]["memory_frame"], 2)
        self.assertEqual(summary["events"][0]["requested_frame"], 2)


if __name__ == "__main__":
    unittest.main()
