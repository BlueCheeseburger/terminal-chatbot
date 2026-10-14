# macOS Command-Key Shortcuts

Terminal applications do not send the Command modifier to programs running inside them. To make the TUI's Command-key equivalents work in macOS Terminal, open **Terminal > Settings > Profiles > Keyboard**, press **+**, choose **Send Hex Code**, then create these mappings:

| Shortcut | Send Hex Code | TUI action |
| --- | --- | --- |
| `Command-K` | `0x0B` | Actions menu |
| `Command-L` | `0x0C` | Model picker |
| `Shift-Return` | `1B 5B 31 33 3B 32 75` | New line in the prompt (Enter still sends) |

Those hex values are the same control characters the TUI already handles for `Ctrl-K` and `Ctrl-L`. Shift-Return sends the code `Esc [ 1 3 ; 2 u` (type those seven bytes into the hex field separated by spaces), which the prompt treats as a new line. Ctrl-J is intentionally not a new-line shortcut. This approach keeps the terminal's normal Command-key handling under your control and requires no background utility.
