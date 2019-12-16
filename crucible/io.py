import os
import sys
import json
import csv
from pprint import pprint

"""
method for read/load config files, and logging.
"""

def load_json(filename):
    with open(filename, "r") as f:
        res = json.load(f)
    return res

def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


class logger:
    def __init__(self, workspace=None, flush=True, mute=False):
        """
        logger class.
        param:
            workspace: path to save log file, if None, only print to stdout.
            flush: force flushing when printing.
        """
        self.workspace = workspace
        self.flush = flush
        self.mute = mute
        if workspace is not None:
            os.makedirs(workspace, exist_ok=True)
            self.log_file = os.path.join(workspace, "log.txt")
            self.fp = open(self.log_file, "a+")
        else:
            self.fp = None

    def __del__(self):
        if self.fp: 
            self.fp.close()

    def _print(self, text, use_pprint=False):
        if not self.mute:
            print(text) if not use_pprint else pprint(text)
        if self.fp:
            print(text, file=self.fp)
        if self.flush:
            sys.stdout.flush()


    def log(self, text, level=0):
        text = "\t"*level + text
        text.replace("\n", "\n"+"\t"*level)
        self._print(text)

    def log1(self, text):
        self.log(text, level=1)

    def info(self, text):
        text = "[INFO] " + text
        text.replace("\n", "\n"+"[INFO] ")
        self._print(text)

    def error(self, text):
        text = "[ERROR] " + text
        text.replace("\n", "\n"+"[ERROR] ")
        self._print(text)

    def logblock(self, text):
        self._print("#####################")
        self._print(text, use_pprint=True)
        self._print("#####################")
