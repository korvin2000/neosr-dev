import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest
import torch

from neosr.utils.stop import StopFlag, sigint_ignored

ROOT = Path(__file__).resolve().parents[1]

# Runs in its own session, so killpg(SIGINT) reaches the trainer and every loader
# worker like a terminal Ctrl-C would. Workers re-import this file as __mp_main__,
# hence neosr is imported only under the main guard. argv[1] = "0" drops the bracket.
SCENARIO = """
import os, signal, sys
import torch
from torch.utils.data import DataLoader, Dataset

N, EPOCHS = 20, 2

class DS(Dataset):
    def __len__(self):
        return N
    def __getitem__(self, i):
        return torch.tensor(signal.getsignal(signal.SIGINT) is signal.SIG_IGN)

def init(_):
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN, "worker SIGINT not ignored"

if __name__ == "__main__":
    import contextlib
    bracket = sys.argv[1] == "1"
    sys.argv = [sys.argv[0], "-opt", os.environ["NEOSR_OPT"]]
    from neosr.utils.stop import StopFlag, sigint_ignored
    for persistent in (True, False):
        loader = DataLoader(DS(), num_workers=3, multiprocessing_context="spawn",
                            persistent_workers=persistent, worker_init_fn=init)
        ignored = 0
        for epoch in range(EPOCHS):
            stop = StopFlag()
            stop.install()
            with sigint_ignored() if bracket else contextlib.nullcontext():
                it = iter(loader)
            if epoch == 0:
                os.killpg(os.getpgrp(), signal.SIGINT)  # bootstrap: workers still starting
            for b, x in enumerate(it):
                ignored += int(x.item())
                if epoch == 1 and b == N // 4:
                    os.killpg(os.getpgrp(), signal.SIGINT)  # steady state
            assert stop.requested(torch.device("cpu")), (persistent, epoch)
        assert ignored == N * EPOCHS, (persistent, ignored)
    print("OK")
"""


@pytest.fixture
def restore_sigint():
    prev = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, prev)


def run_scenario(tmp_path: Path, bracket: bool) -> subprocess.CompletedProcess:
    script = tmp_path / "sigint_scenario.py"
    script.write_text(SCENARIO)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "NEOSR_OPT": str(ROOT / "tests" / "loss_test_options.toml"),
    }
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, str(script), str(int(bracket))],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        start_new_session=True,
        check=False,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX process groups")
def test_workers_survive_ctrl_c(tmp_path):
    r = run_scenario(tmp_path, bracket=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "OK"


@pytest.mark.usefixtures("restore_sigint")
def test_second_sigint_raises():
    stop = StopFlag()
    stop.install()
    assert not stop.requested(torch.device("cpu"))
    signal.raise_signal(signal.SIGINT)
    assert stop.requested(torch.device("cpu"))
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


@pytest.mark.usefixtures("restore_sigint")
def test_sigint_ignored_restores():
    stop = StopFlag()
    stop.install()
    with sigint_ignored():
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) == stop._on_sigint

    class BodyError(Exception):
        pass

    with pytest.raises(BodyError), sigint_ignored():
        raise BodyError
    assert signal.getsignal(signal.SIGINT) == stop._on_sigint
    assert not stop.requested(torch.device("cpu"))
