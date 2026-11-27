"""Minimal stand-in for torch: the worker only seeds and asks whether CUDA is available."""


def manual_seed(seed):
    return None


class cuda:  # noqa: N801 — mirrors torch's attribute shape
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def max_memory_allocated() -> int:
        return 0
