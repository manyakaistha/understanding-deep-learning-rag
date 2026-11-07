# Notices and attribution

Original application code: Copyright © 2026 Manya Kaistha. Released under GNU AGPL v3 or later; see [LICENSE](LICENSE). This choice is compatible with the open-source distribution path of the PyMuPDF dependency. It does not relicense third-party learning sources or vendor assets.

## Learning sources

*Understanding Deep Learning* — Simon J.D. Prince, MIT Press. Official project: https://udlbook.github.io/udlbook/. The book is separately licensed CC BY-NC-ND 4.0. Its PDF, figures, and extracted corpus are not distributed here. Original license text: [UDL-BOOK-LICENSE.txt](docs/third-party/UDL-BOOK-LICENSE.txt).

Official notebook code: Copyright 2023 Simon Prince, MIT License. The README screenshots show excerpts of these notebook exercises. Full notice: [UDL-NOTEBOOKS-LICENSE.txt](docs/third-party/UDL-NOTEBOOKS-LICENSE.txt). The runtime downloads the pinned upstream notebook revision only on an explicit build.

The benchmark contains short source quotations to identify evidence locations. These retain attribution and original source terms; a reference in the benchmark does not grant rights to the book as a whole.

## Bundled frontend components

| Component | Pinned version | Upstream terms / notice |
| --- | --- | --- |
| DOMPurify | 3.2.4 | Apache-2.0 or MPL-2.0; [full notice](docs/third-party/DOMPurify-LICENSE.txt) |
| Marked | 15.0.7 | MIT; [full notice](docs/third-party/Marked-LICENSE.md) |
| KaTeX | 0.16.21 | MIT; [full notice](docs/third-party/KaTeX-LICENSE.txt) |
| Prism | 1.29.0 | MIT; [full notice](docs/third-party/Prism-LICENSE.txt) |

KaTeX fonts use SIL OFL 1.1; their embedded copyright and Reserved Font Name notices plus the full license are in [KaTeX-FONTS-NOTICES.txt](docs/third-party/KaTeX-FONTS-NOTICES.txt). See also [the original vendor inventory](src/static/vendor/VENDOR_LICENSES.md).

Python dependencies and downloaded neural-model weights retain their own licenses. In particular, PyMuPDF is available under GNU AGPL or a commercial license. This repository does not distribute the embedding checkpoint.
