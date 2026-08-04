# macOS Command-Key Shortcuts

Terminal applications do not send the Command modifier to programs running inside them. To make the TUI's Command-key equivalents work in macOS Terminal, open **Terminal > Settings > Profiles > Keyboard**, press **+**, choose **Send Hex Code**, then create these mappings:

| Shortcut | Send Hex Code | TUI action |
| --- | --- | --- |
| `Command-K` | `0x0B` | Actions menu |
| `Command-L` | `0x0C` | Model picker |
| `Command-N` | `0x0E` | Clear context |

Those hex values are the same control characters the TUI already handles for `Ctrl-K`, `Ctrl-L`, and `Ctrl-N`. This approach keeps the terminal's normal Command-key handling under your control and requires no background utility.
