# NimbusWorks Engineering SOP

**Audience: Engineering Team | Version: 3.0 | Last Updated: January 2024**

## 1. Deployment Process

### Deployment Windows

Production deployments are permitted **only on Tuesdays and Thursdays** between 2:00 PM and 5:00 PM Pacific Time. Deployments outside this window require VP of Engineering approval.

**Exception — Hotfixes**: P0 and P1 incidents may trigger hotfixes at any time. The on-call engineer has authority to deploy hotfixes without the standard window restriction.

### Pre-Deployment Checklist
- [ ] All CI/CD checks pass (GitHub Actions green)
- [ ] Code review approved (see review requirements below)
- [ ] QA sign-off on staging environment
- [ ] Feature flags configured if rolling out gradually
- [ ] Rollback plan documented in deployment ticket
- [ ] On-call engineer notified

### Rollback Procedure
If a deployment causes elevated error rates or alerts within **15 minutes**, the on-call engineer must initiate rollback immediately without waiting for root-cause analysis. Rollback is initiated via: `make rollback ENV=prod TAG=<previous-tag>`.

## 2. Code Review

### Requirements
- **Feature branches** (merged to main): minimum **1 approval** required
- **Direct merges to main** (hotfixes, releases): minimum **2 approvals** required, at least one from a senior engineer
- Author cannot approve their own PR

### Code Review Standards
Reviewers should check:
- Correctness and test coverage (unit + integration tests required for all non-trivial changes)
- Security implications (no hardcoded secrets, input validation, auth checks)
- Performance (no N+1 queries, no unbounded loops on large data sets)
- Observability (logging, metrics, error handling)
- Documentation (docstrings for public APIs, README updates if needed)

Target PR size: under **400 lines changed** (excluding tests and generated code). Large PRs should be broken up.

## 3. Incident Response

### Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|-----------|---------------|---------|
| **P0** | Service completely down; data loss risk | **Immediate** (page on-call) | All users unable to log in; database outage |
| **P1** | Major feature broken; >20% of users affected | **Within 1 hour** | API returning 500 errors; webhook failures |
| **P2** | Feature degraded; workaround exists | **Within 4 hours** | Slow queries; minor UI bugs in production |
| **P3** | Low impact; cosmetic issues | **Next business day** | Typos in UI; non-critical documentation |

### Incident Response Steps
1. Acknowledge the alert in PagerDuty (auto-assigned to on-call)
2. Create an incident channel in Slack (#incident-YYYYMMDD-brief-description)
3. Identify and communicate impact to affected customers via status.nimbusworks.com
4. Resolve the root cause or apply a temporary mitigation
5. Write a post-mortem (required for all P0 and P1 incidents, within 48 hours)

## 4. On-Call Rotation

The on-call rotation consists of **2 engineers** per week rotating on a weekly basis (Sunday midnight handoff). Engineers joining the on-call rotation must have at least 6 months of tenure.

On-call engineers are expected to respond to P0 pages within **5 minutes** and P1 pages within **15 minutes**, at any hour.

On-call compensation: **$200 flat per week** for carrying the pager, plus $50 per incident response outside business hours.

## 5. Branching Strategy

- **main**: production-ready code; protected branch
- **develop**: integration branch for features
- **feature/\<ticket-id\>-description**: feature branches off develop
- **hotfix/\<ticket-id\>-description**: hotfix branches off main

Feature branches are deleted after merge. We use squash-and-merge for feature branches to keep main history clean.

## 6. Testing Standards

- Minimum **80% code coverage** for new modules (checked in CI)
- All database queries must have corresponding integration tests
- End-to-end tests run on every PR against a staging environment
- Load tests are run quarterly and before major releases

## 7. Secrets Management

All secrets are stored in AWS Secrets Manager. Never commit secrets to version control. Use `make secrets-pull` to sync secrets to your local `.env` file. Rotate secrets annually or immediately upon suspected compromise.

---

*Questions? Ask in #engineering-ops or email engineering-leadership@nimbusworks.com*
