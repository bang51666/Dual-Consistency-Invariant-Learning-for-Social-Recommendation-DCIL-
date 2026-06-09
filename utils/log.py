from pathlib import Path


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "log.txt").open("a", encoding="utf-8")

    def write(self, message):
        print(message, end="")
        self.file.write(message)
        self.file.flush()

    def close(self):
        self.file.close()
