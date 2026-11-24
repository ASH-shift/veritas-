# NimbusWorks Information Security Policy

**Version 4.1 | Effective: January 1, 2024 | Classification: Internal**

## 1. Purpose

This policy establishes minimum security standards for protecting NimbusWorks information assets, customer data, and infrastructure.

## 2. Password Requirements

All NimbusWorks accounts must use passwords that meet the following criteria:
- Minimum **12 characters** in length
- Must include at least one uppercase letter, one lowercase letter, one digit, and one special character
- Must not contain the employee's name, username, or common dictionary words
- Must not reuse the last **10 passwords**

Passwords must be rotated every **90 days**. Use of a company-approved password manager (1Password) is strongly encouraged. Sharing passwords is strictly prohibited.

## 3. Multi-Factor Authentication (MFA)

MFA is **mandatory for all employees** since January 1, 2023. MFA is required for:
- Company email (Google Workspace)
- Source code repositories (GitHub)
- Cloud infrastructure console (AWS)
- Internal admin panels
- VPN access

TOTP (time-based one-time passwords) is the preferred MFA method. SMS-based OTP is permitted as a fallback but discouraged due to SIM-swap risks.

## 4. Remote Access and VPN

All remote access to NimbusWorks internal systems **must use the company VPN** (WireGuard-based, endpoint: vpn.nimbusworks.com). Accessing internal tools on public Wi-Fi without VPN is prohibited.

Employees working remotely for more than 3 consecutive days in an unfamiliar location should notify IT security at security@nimbusworks.com.

## 5. Data Classification

NimbusWorks uses four data classification levels:

| Level | Description | Examples |
|-------|-------------|---------|
| **Public** | Approved for public release | Marketing materials, blog posts |
| **Internal** | For employee use only | This policy, internal wikis |
| **Confidential** | Restricted to specific teams | Customer PII, contracts, source code |
| **Restricted** | Requires explicit authorization | Encryption keys, credentials, board materials |

Employees must handle data according to its classification level. Confidential and Restricted data must never be stored in personal accounts, unencrypted USB drives, or personal cloud services.

## 6. Device Security

All company-issued laptops must have:
- Full-disk encryption enabled (FileVault for Mac, BitLocker for Windows)
- Automatic screen lock after **5 minutes** of inactivity
- Approved endpoint detection software (CrowdStrike Falcon)
- Operating system updated within **14 days** of a critical security patch release

Personal devices may only access company email and calendar. Access to Confidential or Restricted systems from personal devices is prohibited.

## 7. Incident Reporting

Security incidents must be reported to security@nimbusworks.com **within 2 hours of detection**. Incidents include:
- Suspected phishing or malware
- Lost or stolen devices
- Unauthorized access attempts
- Accidental data exposure

Failure to report a known security incident is a serious policy violation and may result in disciplinary action.

## 8. Third-Party Software

Installation of software on company devices requires approval from IT. Use the company's managed software catalog (Jamf) for approved applications. Unapproved browser extensions are prohibited on browsers used for work.

## 9. Annual Training

All employees must complete the **Security Awareness Training** (1 hour, mandatory) within 30 days of joining and annually thereafter. Completion is tracked and reported to department heads.

---

*Contact security@nimbusworks.com for questions or to report incidents.*
