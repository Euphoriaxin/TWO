#!/usr/bin/env python
# Python wrapper for METEOR, originally by Xinlei Chen; Python 3 by Sabarish Sivanath.
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

METEOR_JAR = 'meteor-1.5.jar'


class Meteor:
    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_p = None
        self._stderr = None
        directory = Path(__file__).resolve().parent
        if not (directory / METEOR_JAR).is_file():
            raise FileNotFoundError(f'METEOR JAR missing: {directory / METEOR_JAR}')
        if not (directory / 'data' / 'paraphrase-en.gz').is_file():
            raise FileNotFoundError(
                'METEOR English data missing. From the project root run: python scripts/setup_meteor.py')
        if shutil.which('java') is None:
            raise RuntimeError('METEOR requires Java. Install a Java runtime and verify with: java -version')
        self.meteor_cmd = ['java', '-Xmx2G', '-jar', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        # A file avoids filling a stderr pipe and preserves Java's actual error.
        self._stderr = tempfile.TemporaryFile(mode='w+t', encoding='utf-8')
        try:
            self.meteor_p = subprocess.Popen(
                self.meteor_cmd, cwd=os.fspath(directory), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._stderr, text=True,
                encoding='utf-8', bufsize=1)
        except Exception:
            self.close()
            raise

    def _failure(self):
        try:
            self.meteor_p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        self._stderr.seek(0)
        error = self._stderr.read(8192).strip()
        return RuntimeError(
            f'METEOR Java process failed (exit={self.meteor_p.poll()}). '
            f'{error or "No Java stderr was produced."}')

    def _write(self, line):
        try:
            self.meteor_p.stdin.write(line + '\n')
            self.meteor_p.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._failure() from exc

    def _read(self):
        line = self.meteor_p.stdout.readline()
        if not line:
            raise self._failure()
        return line.strip()

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        if not gts:
            raise ValueError('METEOR needs at least one reference/hypothesis pair')
        with self.lock:
            statistics = []
            for key in gts:
                assert len(res[key]) == 1
                statistics.append(self._stat(res[key][0], gts[key]))
            self._write('EVAL ||| ' + ' ||| '.join(statistics))
            scores = [float(self._read()) for _ in gts]
            return float(self._read()), scores

    def method(self):
        return 'METEOR'

    @staticmethod
    def _clean(text):
        return ' '.join(text.replace('|||', '').split())

    def _stat(self, hypothesis_str, reference_list):
        fields = ['SCORE'] + [self._clean(ref) for ref in reference_list]
        fields.append(self._clean(hypothesis_str))
        self._write(' ||| '.join(fields))
        return self._read()

    def _score(self, hypothesis_str, reference_list):
        return self.compute_score({0: reference_list}, {0: [hypothesis_str]})[0]

    def close(self):
        with self.lock:
            process = self.meteor_p
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout):
                    try:
                        stream.close()
                    except OSError:
                        pass
                self.meteor_p = None
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
