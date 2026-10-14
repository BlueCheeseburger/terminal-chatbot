#!/usr/bin/env python3
"""Show the raw bytes your terminal sends for each key, to find out what Shift+Enter produces.

    python3 tools/show_keys.py

Press Enter and Shift+Enter one at a time. Press q to quit.
"""

import os
import select
import sys
import termios
import tty


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Run this in a terminal.")
    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    print("Press keys (q quits). Each line shows the bytes received.\r")
    try:
        tty.setraw(descriptor)
        while True:
            data = os.read(descriptor, 32)
            # Gather the rest of a multi-byte sequence that arrives together.
            while select.select([descriptor], [], [], 0.05)[0]:
                data += os.read(descriptor, 32)
            print("{!r}   hex: {}\r".format(data, data.hex(" ")))
            if data == b"q":
                break
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


if __name__ == "__main__":
    main()
