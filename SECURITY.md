# Security

Report suspected vulnerabilities privately to the maintainer through an existing
trusted contact channel. If this repository offers GitHub's **Security → Report
a vulnerability** option, use it. Do not post tokens, private prompts or sensitive
experiment traces in an issue.

If a credential was committed, revoke it at its provider immediately; deleting a
file or adding an ignore rule does not remove a secret from Git history. Hugging
Face and other service credentials belong in local environment variables or the
provider's credential store, never in configuration checked into this repository.

Treat external checkpoints, probe files and model-generated content as untrusted.
Prefer safe tensor formats, review any required remote model code before enabling
it, and load pickle or joblib artifacts only from trusted sources. Model outputs
are experimental data and must not be executed as commands.

Pull-request CI uses no model or service credentials and does not download gated
checkpoints. Keep real-model workflows separate from routine offline validation.
Review raw traces for private content and check dataset permissions before sharing
them. Action dependencies are pinned to commit SHAs; review dependency updates
before merging them.
