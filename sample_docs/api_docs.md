# NimbusWorks API Documentation

**API Version: v2 | Base URL: https://api.nimbusworks.com/v2**

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <your_api_token>
```

Tokens are generated from **Settings → API Keys** in the NimbusWorks dashboard. Tokens do not expire by default but can be revoked at any time. For OAuth flows, access tokens expire after **24 hours** and can be refreshed using the refresh token endpoint.

## Rate Limits

| Plan | Requests per Hour | Burst |
|------|------------------|-------|
| Free | 100 | 10/second |
| Starter | 500 | 20/second |
| Business | 1,000 | 50/second |
| Enterprise | 5,000 | 200/second |

When rate limited, the API returns HTTP **429 Too Many Requests** with a `Retry-After` header indicating seconds until the limit resets. Rate limit headers are returned on every response:
- `X-RateLimit-Limit`: requests allowed per hour
- `X-RateLimit-Remaining`: requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when window resets

## Core Endpoints

### Projects

**List Projects**
```
GET /v2/projects
```
Query parameters: `workspace_id` (required), `status` (active|archived|all), `page`, `per_page` (max 100)

**Create Project**
```
POST /v2/projects
```
Body: `{ "name": string, "workspace_id": string, "description": string, "color": string }`

**Get Project**
```
GET /v2/projects/{project_id}
```

**Update Project**
```
PATCH /v2/projects/{project_id}
```

**Delete Project**
```
DELETE /v2/projects/{project_id}
```
Soft-deletes the project. Data is retained for **30 days** before permanent deletion.

### Tasks

**List Tasks**
```
GET /v2/tasks
```
Query parameters: `project_id`, `assignee_id`, `status`, `due_before`, `due_after`, `page`, `per_page` (max 200)

**Create Task**
```
POST /v2/tasks
```
Body: `{ "project_id": string, "title": string, "description": string, "assignee_id": string, "due_date": ISO8601, "priority": "urgent"|"high"|"normal"|"low" }`

**Update Task**
```
PATCH /v2/tasks/{task_id}
```

**Delete Task**
```
DELETE /v2/tasks/{task_id}
```

### Users

**Get Current User**
```
GET /v2/users/me
```

**List Workspace Members**
```
GET /v2/workspaces/{workspace_id}/members
```

**Invite Member**
```
POST /v2/workspaces/{workspace_id}/members
```
Body: `{ "email": string, "role": "admin"|"member"|"viewer" }`

### Webhooks

**Create Webhook**
```
POST /v2/webhooks
```
Body: `{ "workspace_id": string, "url": string, "events": string[], "secret": string }`

Supported events: `task.created`, `task.updated`, `task.deleted`, `task.completed`, `project.created`, `project.archived`, `member.invited`, `member.removed`

NimbusWorks signs webhook payloads with HMAC-SHA256 using the secret you provide. Verify the `X-NimbusWorks-Signature` header on all incoming webhooks.

## Pagination

All list endpoints use cursor-based pagination. The response includes:
```json
{
  "data": [...],
  "meta": {
    "next_cursor": "eyJ...",
    "has_more": true,
    "total_count": 1482
  }
}
```
Pass `cursor=<next_cursor>` to fetch the next page.

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request — invalid parameters |
| 401 | Unauthorized — missing or invalid token |
| 403 | Forbidden — insufficient permissions |
| 404 | Not Found |
| 409 | Conflict — duplicate resource |
| 422 | Unprocessable Entity — validation error |
| 429 | Rate Limited |
| 500 | Internal Server Error |

## SDKs

Official SDKs are available for:
- **Python**: `pip install nimbusworks-sdk` (supports async)
- **Node.js**: `npm install @nimbusworks/sdk`
- **Go**: `go get github.com/nimbusworks/sdk-go`

Community SDKs are available for Ruby, PHP, and Java.

---

*API changelog: developer.nimbusworks.com/changelog | Support: api-support@nimbusworks.com*
