# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please follow these guidelines for responsible disclosure:

### How to Report

**DO NOT** create public GitHub issues for security vulnerabilities.

Instead, please report security issues by emailing: **security@agent-skeptic-bench.org**

### What to Include

Please include the following information in your report:

- **Vulnerability Description**: Clear description of the security issue
- **Steps to Reproduce**: Detailed steps to reproduce the vulnerability
- **Impact Assessment**: Your assessment of the potential impact
- **Proof of Concept**: If applicable, a minimal proof of concept
- **Suggested Fix**: If you have ideas for how to fix the issue
- **Contact Information**: How we can reach you for follow-up questions

### Response Timeline

- **Initial Response**: Within 24 hours
- **Assessment**: Within 7 days
- **Fix Development**: 14-30 days (depending on complexity)
- **Public Disclosure**: After fix is deployed and users have had time to update

### Responsible Disclosure

We follow a coordinated disclosure process:

1. **Private Disclosure**: You report the issue privately
2. **Assessment**: We assess and confirm the vulnerability
3. **Fix Development**: We develop and test a fix
4. **Pre-disclosure**: We may share details with key maintainers
5. **Fix Release**: We release the security fix
6. **Public Disclosure**: We publicly disclose the issue after users can update

### Bug Bounty

While we don't currently offer a formal bug bounty program, we recognize security researchers' contributions through:

- Public acknowledgment (if desired)
- Priority support for future issues
- Consideration for future bug bounty programs

## Security Best Practices

### For Users

- Always use the latest version
- Follow our deployment security guidelines
- Use proper API key management
- Enable audit logging in production
- Regularly update dependencies

### For Contributors

- Follow secure coding practices
- Use pre-commit hooks for security scanning
- Never commit secrets or API keys
- Review security implications of changes
- Follow the principle of least privilege

## Security Features

### Current Security Measures

- **Dependency Scanning**: Automated vulnerability detection
- **Static Analysis**: Code security scanning with Bandit and Semgrep
- **Container Security**: Docker image vulnerability scanning
- **Secret Management**: Secure handling of API keys and credentials
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting and abuse prevention

### Planned Security Enhancements

- **SBOM Generation**: Software Bill of Materials for transparency
- **SLSA Compliance**: Supply-chain security framework
- **Runtime Security**: Application security monitoring
- **Compliance Framework**: SOC2/ISO27001 alignment

## Known Security Considerations

### AI Model Interactions

- **API Key Security**: Secure storage and transmission of AI provider API keys
- **Rate Limiting**: Protection against API abuse
- **Input Sanitization**: Validation of user-provided prompts and scenarios
- **Response Filtering**: Sanitization of AI model responses

### Data Handling

- **Evaluation Data**: Secure handling of benchmark scenarios and results
- **User Data**: Minimal data collection with explicit consent
- **Audit Logs**: Comprehensive logging for security monitoring
- **Data Retention**: Clear policies for data lifecycle management

## Security Contact

- **Primary Contact**: security@agent-skeptic-bench.org
- **Maintainer**: @danieleschmidt
- **Security Team**: security-team@agent-skeptic-bench.org

## Acknowledgments

We thank the following security researchers for their responsible disclosure:

- [List will be updated as vulnerabilities are reported and fixed]

## Security Advisories

For the latest security advisories, please check:

- [GitHub Security Advisories](https://github.com/yourusername/agent-skeptic-bench/security/advisories)
- [Project Security Page](https://agent-skeptic-bench.org/security)

---

*This security policy is regularly updated. Last updated: July 31, 2025*