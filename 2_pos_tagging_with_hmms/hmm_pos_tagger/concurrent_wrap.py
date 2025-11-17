from typing import Iterable, Self, override
from multiprocessing.synchronize import Barrier
import multiprocessing
import threading

from .base import HmmPosTaggerInterface, HmmPosTaggerBase


class HmmPosTaggerMultithreadingWrap(HmmPosTaggerInterface):
    """
    Multithreading support of HmmPosTagger classes

    Only support multithreading in tagging. It is single-threading for fitting.
    """

    def __init__(self, base_model: HmmPosTaggerBase, num_threads: int) -> None:
        assert num_threads > 0
        self._base_model = base_model
        self._num_threads = num_threads
        self._running = True

        self._signal_list_input_ready = list[threading.Semaphore]()
        self._signal_output_ready = threading.Barrier(self._num_threads + 1)
        self._shared = {"input": None, "output": None, "model": self._base_model}

        self._threads_list = list[threading.Thread]()
        for idx in range(self._num_threads):
            self._signal_list_input_ready.append(threading.Semaphore(0))
            thread = threading.Thread(
                target=self._proxy_thread,
                args=(
                    idx,
                    self._num_threads,
                    self._signal_list_input_ready[-1],
                    self._signal_output_ready,
                    self._shared,
                ),
            )
            thread.start()
            self._threads_list.append(thread)

    def __del__(self) -> None:
        self._shared.clear()
        self._signal_input_ready()
        for thread in self._threads_list:
            thread.join()

    @override
    def export(self) -> dict[str, object]:
        return self._base_model.export()

    @override
    def load(self, data: dict[str, object]) -> Self:
        self._base_model.load(data)
        return self

    @override
    def fit(
        self,
        sentences: Iterable[Iterable[str]],
        pos_tags_list: Iterable[Iterable[str]] | None = None,
    ) -> Self:
        self._base_model.fit(sentences, pos_tags_list)
        return self

    @override
    def tag(self, sentences: Iterable[Iterable[str]]) -> list[list[tuple[str, float]]]:
        self._shared["input"] = list(sentences)
        self._shared["output"] = [None] * len(sentences)
        self._signal_input_ready()
        self._signal_output_ready.wait()
        return self._shared["output"]

    def _signal_input_ready(self):
        for signal in self._signal_list_input_ready:
            signal.release()

    @staticmethod
    def _proxy_thread(
        thread_index: int,
        num_threads: int,
        signal_input_ready: threading.Semaphore,
        signal_output_ready: threading.Barrier,
        shared: dict[str, object],
    ) -> None:
        while True:
            signal_input_ready.acquire()
            if len(shared) == 0:
                break

            len_input = len(shared["input"])
            per_thread_len = len_input // num_threads
            mod = len_input % num_threads
            beg = per_thread_len * thread_index + min(mod, thread_index)
            end = per_thread_len * (thread_index + 1) + min(mod, thread_index + 1)

            out = shared["model"].tag(shared["input"][beg:end])
            #assert shared["output"][beg:end] == [None] * (end - beg)
            shared["output"][beg:end] = out
            signal_output_ready.wait()
