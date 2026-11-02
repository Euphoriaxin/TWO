"""Install METEOR's English paraphrases from a checksum-pinned release wheel."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

WHEEL_SHA256 = '083ed7910f1aec000b0a237ef6665f74edf19954204d0b1cbdb8399ed132228d'
DATA_SHA256 = 'c147ac7d2c91f2fbb3ad31e4b352235061eb83145e0434daf217ee9ca5975f48'
MEMBER = 'pycocoevalcap/meteor/data/paraphrase-en.gz'
DESTINATION = Path(__file__).resolve().parents[1] / MEMBER


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def install(wheel=None, destination=DESTINATION):
    if destination.exists():
        if sha256(destination) != DATA_SHA256:
            raise RuntimeError(f'Checksum mismatch: {destination}. Preserve/remove this file before retrying.')
        print(f'METEOR data verified: {destination}')
        return
    with tempfile.TemporaryDirectory(prefix='two-meteor-') as temporary:
        temporary = Path(temporary)
        if wheel is None:
            requirements = temporary / 'requirements.txt'
            requirements.write_text(f'pycocoevalcap==1.2 --hash=sha256:{WHEEL_SHA256}\n')
            subprocess.run([
                sys.executable, '-m', 'pip', 'download', '--no-deps',
                '--only-binary=:all:', '--require-hashes', '-r', str(requirements),
                '--dest', str(temporary),
            ], check=True)
            wheel = temporary / 'pycocoevalcap-1.2-py3-none-any.whl'
        wheel = Path(wheel)
        if sha256(wheel) != WHEEL_SHA256:
            raise RuntimeError('Unexpected pycocoevalcap wheel checksum; refusing extraction.')
        with zipfile.ZipFile(wheel) as archive:
            data = archive.read(MEMBER)
        if hashlib.sha256(data).hexdigest() != DATA_SHA256:
            raise RuntimeError('METEOR data checksum mismatch; refusing installation.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write to an adjacent temporary file so an interrupted write is not used.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            pending = Path(output.name)
            try:
                output.write(data)
                output.close()
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
        print(f'METEOR data installed and verified: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, help='Use an already downloaded release wheel (offline).')
    args = parser.parse_args()
    install(args.wheel)
