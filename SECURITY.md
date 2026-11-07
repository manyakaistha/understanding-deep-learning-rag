# Security

This is a local, single-user study application. It binds to localhost and has no multi-user authentication, tenant isolation, or internet-facing deployment model. Do not expose it directly as a public server.

Provider keys and browser conversations stay in page memory. A selected remote provider receives the question, bounded history, and retrieved excerpts. Do not put confidential material into remote requests unless you intend to send it there. Source and crop routes use allowlisted local snapshots; notebook cells and generated code are not executed.

Never include API keys, raw data, SQLite databases, or model weights in issues. For a security concern, use GitHub's private vulnerability reporting where available, or contact the repository owner privately through an established channel. Include affected version, reproduction steps, and expected impact without posting credentials publicly.
