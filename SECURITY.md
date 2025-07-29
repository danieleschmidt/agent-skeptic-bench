# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### Private Disclosure Process

1. **Do NOT create a public GitHub issue**
2. Email us at **skeptic-bench@yourdomain.com** with:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 5 business days
- **Status Updates**: Weekly until resolution
- **Resolution Timeline**: Critical issues within 30 days, others within 90 days

### Security Best Practices

When using this benchmark:

- **API Keys**: Never commit API keys to version control
- **Data Privacy**: Be mindful of sensitive data in evaluation scenarios
- **Environment Isolation**: Run evaluations in sandboxed environments
- **Input Validation**: Always validate custom scenarios and adversarial inputs

### Scope

This security policy covers:
- The agent-skeptic-bench package
- Official documentation and examples
- CI/CD infrastructure

### Recognition

We appreciate responsible disclosure and will:
- Credit reporters in our changelog (unless they prefer anonymity)
- Provide swag for significant findings
- Consider monetary rewards for critical vulnerabilities

## Security Features

- **Input Sanitization**: All user inputs are validated
- **Sandboxed Execution**: Agent evaluations run in isolated environments  
- **Dependency Scanning**: Regular automated dependency vulnerability checks
- **Code Analysis**: Static analysis for security issues

For questions about this policy, contact: skeptic-bench@yourdomain.com