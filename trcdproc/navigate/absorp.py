from typing import Generator
from trcdproc.core import DeltaAChannel, H5File, subgroups


def paths_for_signal(src_file: H5File, chan: DeltaAChannel) -> Generator[str, None, None]:
    """Yields the paths to each dataset corresponding to the specified signal

    Args:
        src_file (H5File): The file containing the experiment data
        chan (DeltaAChannel): The channel whose paths will be produced

    Yields:
        Paths to each of the datasets of the specified signal
    """
    rounds_root = src_file['rounds']
    rounds = sorted(subgroups(rounds_root))
    wavelengths = sorted(subgroups(rounds_root[rounds[0]]))
    for rnd in rounds:
        for wav in wavelengths:
            path = f'/rounds/{rnd}/{wav}/{chan.name}'
            try:
                src_file[path]
            except KeyError:
                continue
            yield path
