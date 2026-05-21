# Security Policy

## Scope
This project runs locally and does not use external AI APIs by default. Security still matters for code, data, and model artifacts.

## Security Guidelines
- Do not commit sensitive/private datasets or model files.
- Do not commit local machine paths, personal tokens, or credentials.
- If file-upload or file-input features are added later, validate user-provided files before processing.
- Keep dependencies updated and monitor for known vulnerabilities.
- Review model and data files before sharing public snapshots.

## Responsible Usage
- Treat training data as potentially sensitive.
- Avoid publishing logs that contain personal information.
- Keep the repository and runtime environment patched.

## Reporting Issues
If you find a security issue:
1. Do not open a public exploit-style issue with sensitive details.
2. Share a private report with clear reproduction steps and impact summary.
3. Provide affected files/paths and a suggested mitigation if possible.
