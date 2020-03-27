from os import devnull
from subprocess import check_call, STDOUT


class Wrapper:
    def __init__(self, binary, quiet=False, logfile=None):
        self.binary = binary
        self.quiet = quiet
        self.logfile = logfile

    def tofile(self, command, file):
        with open(file, 'a') as f:
            check_call(command, stdout=f, stderr=STDOUT)

    def __call__(self, options):
        command = [self.binary, *options]
        if not self.quiet:
            print("Calling command")
            print(" ".join(command))
        if self.logfile is not None:
            self.tofile(command, self.logfile)
        elif self.quiet:
            self.tofile(command, devnull)
        else:
            check_call(command)
