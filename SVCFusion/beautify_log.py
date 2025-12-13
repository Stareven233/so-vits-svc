import os


class HightLightedLog:
    def __enter__(self):
        width = self._get_width()
        print()
        print("\033[91m" + "=" * width + "\033[0m")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        width = self._get_width()
        print("\033[91m" + "=" * width + "\033[0m")
        print()

    def _get_width(self):
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 50  # fallback
