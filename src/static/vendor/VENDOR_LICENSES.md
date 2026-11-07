# Third-Party Vendor Assets and Licenses

The UDL Study Interface utilizes pinned local vendor assets to ensure offline capability, zero runtime CDN or external network dependencies, and security compliance.

## 1. DOMPurify
- **Version**: 3.2.4
- **Source**: https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js
- **License**: Apache-2.0 / Mozilla Public License 2.0 (MPL-2.0)
- **Copyright**: (c) 2024 Mario Heiderich, Cure53
- **Usage**: Mandatory sanitization of all dynamic HTML renderings, Markdown outputs, and equation transcribers.

## 2. Marked.js
- **Version**: 15.0.7
- **Source**: https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js
- **License**: MIT License
- **Copyright**: (c) 2018+, Christopher Jeffrey
- **Usage**: Client-side Markdown parsing for assistant replies and explanations.

## 3. KaTeX
- **Version**: 0.16.21
- **Source**: https://github.com/KaTeX/KaTeX/releases/download/v0.16.21/katex.zip
- **License**: MIT License (Code) & SIL Open Font License 1.1 (Fonts)
- **Copyright**: (c) 2013-2024 Khan Academy and other contributors
- **Usage**: Fast, accessible mathematical typesetting with local fonts (`fonts/`). Always invoked with `trust: false`.

## 4. PrismJS
- **Version**: 1.29.0
- **Source**: https://cdn.jsdelivr.net/npm/prismjs@1.29.0/
- **License**: MIT License
- **Copyright**: (c) 2012 Lea Verou
- **Usage**: Code syntax highlighting for Python notebooks, bash, and JSON snippets.
