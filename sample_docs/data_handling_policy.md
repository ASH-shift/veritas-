# NimbusWorks Data Handling and Privacy Policy

**Version 2.3 | Effective: January 1, 2024 | Classification: Internal**

## 1. Data Categories

NimbusWorks processes the following categories of data:

**Customer Data**: Task content, project data, user-generated documents, and any information customers upload or create within NimbusWorks.

**Personal Data**: Employee and customer personal identifiable information (PII) including names, email addresses, phone numbers, and payment information.

**Usage Data**: Telemetry, logs, and analytics data about how customers and employees use NimbusWorks products.

**Financial Data**: Billing records, invoices, payroll data, and expense reports.

## 2. Data Retention

| Data Type | Retention Period | Notes |
|-----------|-----------------|-------|
| Financial records | **7 years** | Required by law (US, UK, EU) |
| Customer account data | Duration of contract + 30 days | Active and grace period |
| Customer data after account closure | **30 days** after closure | Then permanently deleted |
| Employee records | 7 years after employment ends | HR and payroll records |
| Application logs | 90 days | Security logs: 1 year |
| Backup data | **90 days** | Daily backups, retained for 90 days |
| Audit logs | 2 years | Compliance requirement |

Customer data is permanently deleted within **30 days of account closure**. Customers may request earlier deletion via a formal deletion request to privacy@nimbusworks.com.

## 3. Encryption

- **At rest**: AES-256 encryption for all stored data (databases, file storage, backups)
- **In transit**: TLS 1.2 or higher for all data transmission. TLS 1.0 and 1.1 are disabled.
- **Database encryption**: Column-level encryption for PII (credit card numbers, social security numbers, passport numbers) using application-level encryption in addition to disk encryption
- **Backups**: Encrypted before transmission to backup storage (AWS S3 with SSE-KMS)

## 4. Access Controls

Access to customer data follows the principle of **least privilege**:
- Engineers have no direct access to production databases without a formal access request
- All production access is logged and reviewed monthly
- Privileged access requires a manager and security team approval
- Access is revoked within **24 hours** of an employee's departure

## 5. GDPR Compliance

NimbusWorks is GDPR-compliant for customers in the European Economic Area (EEA).

Data Subject Rights:
- **Right of access**: Customers can request a data export within **30 days**
- **Right to erasure**: Deletion requests processed within **30 days**
- **Right to portability**: Data export in JSON or CSV format available
- **Right to rectification**: Customers can correct their data via account settings

Data Processing Agreements (DPAs) are available for Enterprise customers. Contact legal@nimbusworks.com.

NimbusWorks is registered as a Data Processor under GDPR. Our Data Protection Officer can be reached at dpo@nimbusworks.com.

## 6. Sub-Processors

NimbusWorks uses the following categories of sub-processors for customer data:
- Cloud infrastructure: AWS (US-East-1, EU-West-1, AP-Southeast-1)
- Email delivery: SendGrid
- Error monitoring: Sentry
- Analytics (anonymized only): Amplitude

The full list of sub-processors is published at nimbusworks.com/subprocessors and is updated with 30 days notice before adding new sub-processors.

## 7. Data Breach Response

In the event of a personal data breach affecting EU/EEA residents, NimbusWorks will:
1. Assess the breach within **24 hours** of detection
2. Notify the relevant supervisory authority within **72 hours** if required
3. Notify affected individuals without undue delay if there is high risk to their rights

All suspected breaches must be reported immediately to security@nimbusworks.com and the DPO.

## 8. Employee Data Handling

Employees handling customer data must:
- Complete GDPR Awareness Training upon joining (30 minutes, mandatory)
- Access only data required for their specific role
- Never transfer customer data to personal devices or accounts
- Report unauthorized access or suspected breaches immediately

---

*For privacy questions: privacy@nimbusworks.com | DPO: dpo@nimbusworks.com*
