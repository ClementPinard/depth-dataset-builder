from subprocess import check_call, PIPE


class Wrapper:
    def __init__(self, binary, quiet=False):
        self.binary = binary
        self.quiet = quiet
        self.pipe = PIPE if self.quiet else None

    def __call__(self, options):
        command = [self.binary, *options]
        if not self.quiet:
            print("Calling command")
            print(" ".join(command))
        check_call(command, stderr=self.pipe, stdout=self.pipe)
