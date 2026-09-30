# LLM Usage & Cost Metering Subsystem
## Coding Agent Implementation Specification

## Objective

Build a **standalone LLM Usage & Cost Metering subsystem** for the application.

This subsystem must be architecturally independent from the Route Optimizer and any other business logic.

The Route Optimizer should only:

1. Make an LLM/provider request.
2. Receive the provider response.
3. Normalize provider usage information through a provider adapter.
4. Emit a `UsageEvent`.

The Usage & Cost Metering subsystem owns:

- Token accounting
- Usage persistence
- Cost calculation
- Pricing catalogs
- Historical pricing
- Aggregation
- Budgets
- Alerts
- Projections
- Provider reconciliation
- Usage reporting
- Cost reporting
- Optional customer billing calculations

Do not mix these responsibilities into the Route Optimizer.

---

# 1. Architecture

Use this logical architecture:

```text
                    ┌──────────────────────┐
                    │    Route Optimizer   │
                    │                      │
                    │    Business Logic    │
                    └──────────┬───────────┘
                               │
                               │ LLM Request
                               ▼
                    ┌──────────────────────┐
                    │   Provider Adapter   │
                    │                      │
                    │ OpenAI / Anthropic / │
                    │ Gemini / Compatible  │
                    └──────────┬───────────┘
                               │
                               │ Normalized UsageEvent
                               ▼
              ┌─────────────────────────────────┐
              │     USAGE & COST METERING       │
              │                                 │
              │ • Usage ingestion               │
              │ • Token accounting              │
              │ • Pricing catalog               │
              │ • Cost calculation              │
              │ • Aggregation                   │
              │ • Budgets                       │
              │ • Alerts                        │
              │ • Reconciliation                │
              │ • Reporting                     │
              └─────────────────────────────────┘
```

The Route Optimizer must NOT contain:

- Pricing logic
- Provider-specific pricing
- Monthly cost calculations
- Budget calculations
- Cost aggregation
- Billing-period logic
- Provider balance logic
- Usage dashboards
- Pricing tables
- Cost alerts

---

# 2. Provider-Agnostic Design

Do not design the metering subsystem specifically around OpenAI.

OpenAI is the initial provider, but the architecture must support:

- OpenAI
- Anthropic
- Google
- Azure OpenAI
- OpenAI-compatible APIs
- Ollama
- Local models
- Other providers added later

Every provider adapter must normalize its provider-specific usage response into the application's canonical `UsageEvent`.

The rest of the application must never need to know whether the provider called a field:

```text
prompt_tokens
input_tokens
completion_tokens
output_tokens
cached_tokens
reasoning_tokens
```

The adapter translates provider-specific terminology into the canonical model.

---

# 3. Canonical UsageEvent

Create a canonical internal event named:

```text
UsageEvent
```

Minimum logical structure:

```json
{
  "event_id": "uuid",
  "timestamp": "2026-09-26T15:31:22.123Z",

  "tenant_id": "tenant-123",
  "account_id": "customer-123",
  "user_id": "optional-user-id",

  "provider": "openai",
  "model": "gpt-5.6",
  "service_tier": "standard",

  "request_id": "internal-request-id",
  "provider_request_id": "provider-request-id",

  "input_tokens": 12500,
  "output_tokens": 2300,
  "total_tokens": 14800,

  "cached_input_tokens": 4200,
  "reasoning_tokens": 700,

  "application": "route-optimizer",
  "operation": "optimize-route",
  "profile_id": "supplychain.route_optimizer",

  "latency_ms": 2380,

  "status": "success",

  "usage_status": "received"
}
```

Additional metadata may be included.

Do not require the core schema to change whenever a provider adds another usage field.

---

# 4. Raw Provider Usage

Always retain:

1. Normalized usage
2. Raw provider usage metadata

Example:

```json
{
  "input_tokens": 12500,
  "output_tokens": 2300,
  "total_tokens": 14800,

  "provider_usage_raw": {
    "...": "provider-specific data"
  }
}
```

Raw usage should be retained for debugging and reconciliation.

Do not store entire prompts or completions merely for usage accounting unless explicitly required by another subsystem.

---

# 5. Token Categories

Support at minimum:

```text
input_tokens
cached_input_tokens
output_tokens
reasoning_tokens
total_tokens
```

Do not double-count token categories.

For example:

```text
Input tokens          10,000
Cached input           4,000
Output                 2,000
Reasoning                500
-----------------------------
Total tokens           12,000
```

Do NOT calculate:

```text
10,000 + 4,000 + 2,000 + 500
```

The provider's definition of `total_tokens` should be preserved as the authoritative total when available.

---

# 6. Internal Input Attribution

The system should optionally support an internal breakdown of input tokens.

Example:

```json
{
  "input_breakdown": {
    "system_tokens": 1200,
    "profile_tokens": 3400,
    "user_tokens": 850,
    "conversation_tokens": 2100,
    "tool_tokens": 1700,
    "other_tokens": 400
  }
}
```

This is an internal attribution estimate.

The provider-reported total remains authoritative.

Do not force the internal breakdown to equal the provider total.

Example:

```text
Internal estimate:       8,000
Provider total:          8,247
```

Store both.

This is particularly important if the system injects profile/context information into LLM calls.

---

# 7. Pricing Catalog

Create a separate:

```text
PricingCatalog
```

Pricing must be data-driven and must NOT be embedded in Route Optimizer code.

Example:

```json
{
  "pricing_version": "openai-gpt-5.6-2026-09-01",
  "provider": "openai",
  "model": "gpt-5.6",

  "effective_from": "2026-09-01T00:00:00Z",
  "effective_to": null,

  "currency": "USD",

  "input_price_per_1m": 2.50,
  "cached_input_price_per_1m": 0.25,
  "output_price_per_1m": 10.00,

  "service_tier": "standard"
}
```

The example prices above are illustrative only. Do not use them as real pricing without verifying the provider's current pricing.

---

# 8. Effective-Dated Pricing

Pricing must be versioned and effective-dated.

Example:

```text
Model X

2026-01-01 → 2026-06-30
Input:  price A
Output: price B

2026-07-01 → present
Input:  price C
Output: price D
```

When calculating historical cost:

```text
usage timestamp
       ↓
find pricing record effective at that timestamp
       ↓
calculate cost
```

Never recalculate historical costs using today's pricing.

---

# 9. Pricing Loader

Create a separate pricing-loader mechanism.

Responsibilities:

1. Retrieve or receive current provider pricing.
2. Validate the pricing structure.
3. Create a new pricing version.
4. Set effective dates.
5. Preserve previous versions.
6. Never mutate historical pricing.

The pricing loader must be independently executable.

Example:

```text
pricing/
    openai/
    anthropic/
    google/
```

The loader must not modify existing usage events.

---

# 10. Cost Calculation

Create a standalone `CostCalculator`.

Conceptually:

```text
input_cost =
    input_tokens / 1,000,000
    × input_price_per_1m

cached_input_cost =
    cached_input_tokens / 1,000,000
    × cached_input_price_per_1m

output_cost =
    output_tokens / 1,000,000
    × output_price_per_1m

total_cost =
    input_cost
    + cached_input_cost
    + output_cost
```

If a provider uses another billing unit, support provider-specific pricing dimensions.

Do not assume every provider is token-priced forever.

---

# 11. Immutable Cost Snapshot

When a usage event is priced, store the pricing information used.

Example:

```json
{
  "cost": {
    "currency": "USD",

    "input_cost": 0.03125,
    "cached_input_cost": 0.00105,
    "output_cost": 0.02300,

    "total_cost": 0.05530,

    "pricing_version": "openai-gpt-5.6-2026-09-01"
  }
}
```

Historical cost must remain tied to its pricing version.

---

# 12. Estimated vs Provider-Reported Cost

Keep these as separate concepts:

```text
estimated_cost
provider_reported_cost
```

Example:

```json
{
  "estimated_cost_usd": 0.0553,
  "provider_reported_cost_usd": null
}
```

Do not assume internally calculated cost equals the provider's eventual invoice.

Never overwrite the original calculated cost.

---

# 13. Time and Time Zones

Store the canonical timestamp in UTC:

```text
timestamp_utc
```

Also support reporting dimensions:

```text
date_utc
hour_utc
week_utc
month_utc
billing_period
```

The UI may display local time.

Do not use local server time as the canonical accounting time.

---

# 14. Usage Does Not Mean Pricing Changes by Time of Day

The system must support hourly analytics:

```text
Hour      Requests    Tokens      Cost
00:00       12         82K       $0.44
01:00        4         21K       $0.11
09:00      812        8.2M      $41.20
10:00      921        9.7M      $52.18
```

However, do not assume LLM token prices vary by time of day.

Usage varies naturally with activity.

Pricing changes should come from:

- Model
- Provider
- Token category
- Service tier
- Context/pricing rules
- Effective-dated pricing changes
- Provider-specific billing rules

Time-of-day is primarily an analytics dimension.

---

# 15. Aggregation

Build aggregation independently from raw event ingestion.

Support:

```text
minute
hour
day
week
month
billing period
```

Support dimensions:

```text
provider
model
tenant
account
user
profile
application
operation
service tier
status
```

Example:

```text
2026-09-26
    provider = OpenAI
    model = GPT-X
    application = route-optimizer

    requests       1,245
    input tokens   8,421,332
    output tokens  1,821,932
    cached tokens  4,201,112

    estimated cost $73.42
```

---

# 16. Route Optimizer Attribution

The Route Optimizer should attach metadata to every LLM request.

Example:

```json
{
  "tenant_id": "customer-123",
  "account_id": "customer-123",
  "application": "route-optimizer",
  "operation": "optimize-route",
  "profile_id": "route.default"
}
```

The Route Optimizer must not calculate cost.

It simply provides attribution metadata.

---

# 17. Profile Attribution

Support the profile architecture.

Examples:

```text
supplychain.inventory.replenishment
supplychain.allocation.business.rules
finance.sales.audit
finance.sales.audit.walmart
finance.sales.audit.kroger
supplychain.route_optimizer
```

Each usage event should optionally contain:

```text
profile_id
```

This enables reporting such as:

```text
Profile                          Cost
--------------------------------------
route.optimizer.default          $12.43
route.optimizer.enterprise       $42.18
supplychain.replenishment        $87.21
finance.sales.audit              $31.82
```

---

# 18. Usage Event Ingestion

Prefer asynchronous ingestion.

Recommended flow:

```text
LLM request
    ↓
Provider response
    ↓
Normalize usage
    ↓
Emit UsageEvent
    ↓
Queue / event bus
    ↓
Metering service
    ↓
Database
    ↓
Aggregation
    ↓
Reporting
```

The Route Optimizer request path should not depend on the reporting database being available.

---

# 19. Failure Handling

Usage tracking must not cause the Route Optimizer to fail.

Bad:

```text
LLM request succeeds
      ↓
usage database unavailable
      ↓
route optimization fails
```

Good:

```text
LLM request succeeds
      ↓
route optimization succeeds
      ↓
usage event queued
      ↓
metering temporarily unavailable
      ↓
event retried later
```

Use durable retries.

---

# 20. Idempotency

Every usage event must have:

```text
event_id
```

Also store, when available:

```text
provider
provider_request_id
```

Ingestion must be idempotent.

If the same event arrives twice:

```text
DO NOT count it twice.
```

Use a unique constraint or equivalent idempotency mechanism.

---

# 21. Streaming Requests

Support streaming providers.

Usage may arrive only in a final stream chunk.

Design for:

```text
usage_status:
    pending
    received
    estimated
    unavailable
    corrected
```

If the final usage chunk is missing, do NOT interpret that as zero tokens.

Record usage as unavailable or estimated.

---

# 22. Late-Arriving Usage

Usage may arrive after the initial request.

Allow usage events to transition:

```text
pending
    ↓
received
```

or:

```text
received
    ↓
corrected
```

Maintain an audit trail of corrections.

---

# 23. Provider Balance

Do not invent a generic provider balance.

Provider billing APIs differ.

Support optional provider-specific values such as:

```text
provider_balance
provider_credit
provider_usage
provider_spend
```

If unavailable:

```json
{
  "provider_balance": null
}
```

Do not infer provider account balance from internally calculated usage.

---

# 24. Budgets

Implement budgets as a separate subsystem.

Support scopes:

```text
global
provider
model
tenant
account
profile
application
operation
```

Example:

```json
{
  "scope": "account",
  "scope_id": "customer-123",

  "period": "monthly",

  "limit_usd": 500.00,

  "warning_percent": 75,
  "critical_percent": 90
}
```

---

# 25. Budget Metrics

For every budget calculate:

```text
current_spend
budget_limit
remaining_budget
percentage_used
projected_spend
```

Example:

```text
Monthly budget       $500
Current spend        $312
Remaining            $188
Used                  62.4%

Projected spend      $487
```

Projection must always be labeled as an estimate.

---

# 26. Cost Projection

Support projection based on:

```text
current daily burn rate
7-day average
30-day average
month-to-date average
```

Calculate:

```text
projected_monthly_cost
```

Do not present projected provider cost as an actual provider balance.

---

# 27. Reconciliation

Create a reconciliation process comparing:

```text
our recorded usage
        vs
provider reported usage
```

and:

```text
our calculated cost
        vs
provider reported cost
```

Track:

```text
variance_tokens
variance_cost
reconciliation_status
```

Example:

```text
Our calculated cost:       $341.27
Provider reported cost:    $344.18

Variance:                    $2.91
Variance percentage:         0.85%
```

Do not silently alter historical events to eliminate discrepancies.

---

# 28. Database Model

Create separate logical entities/tables for:

```text
usage_events
pricing_versions
cost_records
budget_definitions
usage_aggregates
provider_accounts
reconciliation_records
```

Do not create one giant table containing all concerns.

---

# 29. Usage Event Suggested Schema

Use a schema approximately like:

```text
usage_events
-------------
event_id
timestamp_utc

tenant_id
account_id
user_id

provider
model
service_tier

request_id
provider_request_id

input_tokens
cached_input_tokens
output_tokens
reasoning_tokens
total_tokens

application
operation
profile_id

latency_ms

status
usage_status

input_breakdown_json
provider_usage_raw_json

created_at
updated_at
```

Add indexes for common queries:

```text
timestamp_utc
provider
model
tenant_id
account_id
profile_id
application
operation
provider_request_id
```

Use composite indexes based on actual query patterns.

---

# 30. Cost Record Suggested Schema

```text
cost_records
------------
cost_id
usage_event_id

currency

input_cost
cached_input_cost
output_cost
total_cost

pricing_version

estimated_cost
provider_reported_cost

calculated_at
created_at
updated_at
```

Keep cost records logically separate from raw usage.

---

# 31. Pricing Version Suggested Schema

```text
pricing_versions
----------------
pricing_version_id

provider
model
service_tier

effective_from
effective_to

currency

pricing_json

source
source_reference

created_at
updated_at
```

Do not mutate an effective historical pricing version.

Create a new version instead.

---

# 32. Customer Billing

Keep provider cost and customer billing separate.

Provider cost:

```text
provider_cost
```

Customer charge:

```text
customer_charge
```

Example:

```text
Provider cost:       $0.042
Customer charge:     $0.075
```

Do not derive customer billing directly from provider pricing.

---

# 33. Customer Pricing

Eventually support pricing models such as:

```text
cost_plus
fixed_per_request
token_markup
profile_fee
subscription_allowance
hybrid
```

Example:

```json
{
  "customer": "ACME",
  "pricing_model": "cost_plus",
  "markup_percent": 35
}
```

Customer pricing must be independent from provider pricing.

---

# 34. Dashboard

Create a Usage & Cost dashboard independent of the Route Optimizer UI.

Top-level cards:

```text
Today's Requests
Today's Tokens
Today's Cost

Month-to-Date Requests
Month-to-Date Tokens
Month-to-Date Cost

Projected Monthly Cost
Budget Remaining
```

Charts:

```text
Cost over time
Tokens over time
Requests over time
Cost by model
Cost by provider
Cost by profile
Cost by account
```

Filters:

```text
Date range
Provider
Model
Tenant
Account
Profile
Application
Operation
```

---

# 35. Drill-Down

Support:

```text
Month
  ↓
Day
  ↓
Hour
  ↓
Request
  ↓
Provider
  ↓
Model
  ↓
Profile
```

Example:

```text
September 26
    $84.22
       ↓
10:00 AM
    $12.41
       ↓
route.optimizer.enterprise
    $7.31
       ↓
GPT-X
    $5.82
       ↓
Request abc123
    $0.042
```

---

# 36. API

Provide APIs approximately equivalent to:

```text
POST /usage/events

GET /usage
GET /usage/summary
GET /usage/timeseries

GET /usage/by-provider
GET /usage/by-model
GET /usage/by-tenant
GET /usage/by-account
GET /usage/by-profile
GET /usage/by-operation

GET /costs
GET /costs/timeseries

GET /budgets
POST /budgets

GET /pricing
POST /pricing

GET /provider-balances
```

Exact routes can follow the existing application conventions.

---

# 37. Provider Adapter Contract

Every provider adapter should expose functionality conceptually equivalent to:

```text
send(request)
normalizeUsage(response)
getProviderRequestId(response)
getModel(response)
```

Return a normalized provider response:

```text
ProviderResponse
    response
    usage
    provider_request_id
    model
    metadata
```

The metering subsystem consumes the normalized usage.

---

# 38. OpenAI Adapter

For OpenAI, support at minimum:

```text
Responses API
Chat Completions
streaming
cached input
reasoning tokens
service tier
```

OpenAI and OpenAI-compatible providers may expose usage fields differently.

Normalize those differences at the adapter boundary.

Do not spread provider-specific field names throughout the application.

Bad:

```text
routeOptimizer.prompt_tokens
```

Good:

```text
UsageEvent.input_tokens
```

---

# 39. No Pricing Logic in the Route Optimizer

Never write logic like:

```text
if model == "xyz":
    cost = tokens * 0.000002
```

inside Route Optimizer code.

Use:

```text
UsageEvent
      ↓
PricingResolver
      ↓
PricingVersion
      ↓
CostCalculator
      ↓
CostRecord
```

---

# 40. Observability

Every LLM request should have a correlation chain:

```text
trace_id
request_id
provider_request_id
usage_event_id
```

Example:

```text
trace:
    91b7...

route request:
    r-123

LLM request:
    llm-456

provider request:
    openai-789

usage event:
    usage-abc
```

This makes cost anomalies and failed accounting easy to investigate.

---

# 41. Metrics

Expose application metrics such as:

```text
llm_requests_total
llm_requests_failed

llm_input_tokens_total
llm_output_tokens_total
llm_cached_tokens_total
llm_reasoning_tokens_total

llm_cost_usd_total

llm_request_latency

llm_usage_events_pending

llm_usage_reconciliation_variance
```

Add provider/model/profile/account dimensions carefully to avoid excessive metric-cardinality problems.

For high-cardinality dimensions, prefer database analytics rather than metric labels.

---

# 42. Alerts

Support alerts for:

```text
Budget > 75%
Budget > 90%
Budget > 100%

Unexpected token spike
Unexpected cost spike
Unexpected model usage

Provider usage unavailable
Pricing configuration missing

Usage reconciliation variance
```

Example:

```text
ALERT

Account: ACME
Monthly budget: $500

Current spend: $462
Budget consumed: 92.4%

Projected month-end spend: $681
```

---

# 43. Security

Usage data can contain commercially sensitive information.

Protect:

```text
tenant_id
account_id
user_id
profile_id
provider account information
cost information
usage information
```

Use appropriate authentication and authorization.

Customers should only see their own usage.

Administrators may see aggregate usage according to authorization rules.

Do not store prompt/completion content merely to perform usage accounting.

---

# 44. Multi-Tenant Design

Design for:

```text
tenant_id
account_id
user_id
profile_id
application_id
```

Not every event must have every identifier.

Example:

```text
tenant_id  = ACME
account_id = ACME-US-SUPPLY
profile_id = supplychain.inventory.replenishment
```

This supports future SaaS and billing use cases.

---

# 45. Retention

Define configurable retention policies.

At minimum distinguish:

```text
raw usage events
aggregated usage
pricing history
reconciliation records
audit records
```

Do not delete historical pricing versions that are needed to explain historical costs.

If raw usage events are eventually archived, aggregated historical usage must remain reproducible or auditable.

---

# 46. Data Quality

Validate usage events.

Reject or quarantine events when:

```text
event_id is missing
timestamp is invalid
provider is missing
model is missing
token counts are negative
total token count is invalid
pricing is unavailable
```

Do not silently convert malformed values to zero.

Use explicit states such as:

```text
invalid
pending
unpriced
priced
reconciled
```

---

# 47. Missing Pricing

If a usage event arrives but there is no matching pricing version:

```text
DO NOT fail the LLM request.
DO NOT invent a price.
DO NOT calculate $0.
```

Store:

```text
usage_status = received
cost_status = unpriced
```

Then allow the pricing system to resolve the cost later.

---

# 48. Offline / Local Models

Support providers where token pricing is effectively zero or where the cost model is different.

Example:

```text
provider = ollama
model = local-model
pricing_type = local
```

The system may track:

```text
tokens
requests
latency
GPU time
CPU time
energy estimates
infrastructure cost
```

without forcing everything into a token price.

---

# 49. Provider Compatibility

An OpenAI-compatible API should be treated as a provider adapter.

Do not assume that:

```text
OpenAI-compatible
```

means:

```text
identical billing
identical usage fields
identical pricing
identical balance APIs
```

Each provider configuration should define its own:

```text
provider_id
base_url
authentication method
model mapping
usage mapping
pricing
billing capabilities
```

---

# 50. Configuration

Separate configuration into:

### Application configuration

```text
provider endpoints
API credentials
database
queue
logging
```

### Metering configuration

```text
pricing sources
aggregation intervals
retention
budgets
alert thresholds
currency
```

### Tenant/customer configuration

```text
tenant
account
profile
billing model
budget
limits
```

Do not mix these configuration layers.

---

# 51. No Provider Secrets in Usage Events

Never store:

```text
API keys
access tokens
provider secrets
```

inside usage events.

Usage events may contain provider IDs and request IDs but never credentials.

---

# 52. Testing Requirements

Create automated tests for:

## Token accounting

```text
input
output
cached
reasoning
total
```

## Pricing

```text
current price
historical price
price change
missing price
```

## Providers

```text
OpenAI
OpenAI-compatible provider
unknown provider
```

## Streaming

```text
normal completion
missing final usage chunk
interrupted stream
```

## Idempotency

Submit the same event twice.

Expected result:

```text
one usage record
```

## Attribution

Verify preservation of:

```text
tenant
account
profile
application
operation
```

## Budgets

Test:

```text
75%
90%
100%
over-budget
```

## Pricing changes

Verify that an old usage event continues to use its historical pricing version.

## Missing pricing

Verify that usage is retained as `unpriced` rather than recorded as zero cost.

## Late usage

Verify that pending usage can later become authoritative.

---

# 53. Acceptance Criteria

The implementation is complete when:

- [ ] Route Optimizer contains no pricing calculations.
- [ ] Route Optimizer contains no budget logic.
- [ ] Provider-specific usage formats are normalized.
- [ ] Usage events are persisted independently.
- [ ] Usage events are idempotent.
- [ ] Raw provider usage can be retained.
- [ ] Input tokens are tracked.
- [ ] Output tokens are tracked.
- [ ] Cached tokens are tracked.
- [ ] Reasoning tokens are tracked where available.
- [ ] Total tokens are tracked.
- [ ] Pricing is versioned.
- [ ] Historical pricing is preserved.
- [ ] Costs can be calculated independently.
- [ ] Estimated cost is distinguishable from provider-reported cost.
- [ ] Usage can be aggregated by minute/hour/day/week/month.
- [ ] Usage can be filtered by provider/model/tenant/account/profile/application/operation.
- [ ] Budgets are independent of usage collection.
- [ ] Monthly projections are supported.
- [ ] Provider balances are optional and provider-specific.
- [ ] Streaming usage is supported.
- [ ] Missing usage is not interpreted as zero.
- [ ] Late-arriving usage can be reconciled.
- [ ] Cost reconciliation is supported.
- [ ] Usage dashboard is independent of Route Optimizer UI.
- [ ] Pricing can be updated without modifying Route Optimizer code.
- [ ] Customer billing is separated from provider cost.
- [ ] Architecture can support additional LLM providers.
- [ ] Local/self-hosted providers can be represented.
- [ ] Usage accounting failure cannot break route optimization.

---

# 54. Recommended Implementation Sequence

Implement in this order:

### Phase 1 — Canonical model

Create:

```text
UsageEvent
ProviderResponse
CostRecord
PricingVersion
```

### Phase 2 — Provider adapter

Implement OpenAI first.

Normalize:

```text
input tokens
output tokens
cached tokens
reasoning tokens
total tokens
model
provider request ID
```

### Phase 3 — Usage ingestion

Implement:

```text
POST /usage/events
```

with idempotency.

### Phase 4 — Persistence

Create:

```text
usage_events
pricing_versions
cost_records
```

### Phase 5 — Pricing engine

Implement:

```text
PricingResolver
CostCalculator
```

with effective-dated pricing.

### Phase 6 — Aggregation

Implement:

```text
hourly
daily
weekly
monthly
```

aggregations.

### Phase 7 — Reporting

Implement:

```text
usage summary
cost summary
time series
provider breakdown
model breakdown
profile breakdown
account breakdown
```

### Phase 8 — Budgets

Implement:

```text
budget definitions
budget utilization
remaining budget
alerts
projection
```

### Phase 9 — Reconciliation

Implement provider-vs-internal reconciliation.

### Phase 10 — Additional providers

Add:

```text
Anthropic
Google
OpenAI-compatible providers
Ollama
```

without changing the core metering model.

---

# 55. Critical Architectural Rule

Treat these as three separate systems:

```text
             ┌──────────────────────┐
             │   ROUTE OPTIMIZER    │
             │                      │
             │ "What should I do?"  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │    LLM PROVIDER      │
             │                      │
             │ "Generate response"  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ USAGE / COST SYSTEM  │
             │                      │
             │ "What did it cost?"  │
             └──────────────────────┘
```

The Route Optimizer must be completely unaware of how the Usage & Cost system calculates money.

The Usage & Cost system must be completely unaware of how the Route Optimizer makes business decisions.

The provider adapter is the boundary between them.

This separation must remain even if everything is initially deployed in a single application or container.

---

# 56. Future Extension: Context/Profile Cost Accounting

The architecture should make it possible to answer a particularly important future question:

> How much does our injected expert/business-rule context cost?

For example:

```text
Request
├── Base system prompt          1,200 tokens
├── Profile context             3,400 tokens
├── User request                  850 tokens
├── Conversation history        2,100 tokens
└── Tool context                1,700 tokens
                                -------
                                9,250 tokens
```

The system should eventually be able to report:

```text
Total provider cost:             $0.055
Estimated profile-context cost:  $0.020
User-request cost:               $0.005
Conversation-context cost:       $0.012
Tool-context cost:               $0.010
Other/system cost:               $0.008
```

These component values are internal attribution estimates and must remain distinguishable from the provider's authoritative token usage.

This capability is important for measuring the economics of profile-based MCP/context enrichment.

---

# 57. Future Extension: Cross-Application Metering

The Usage & Cost subsystem should eventually support multiple applications:

```text
route-optimizer
mcp-context-server
chat-interface
batch-processor
analytics-agent
customer-support-agent
```

All applications should emit the same canonical `UsageEvent`.

Example:

```text
                     Usage & Cost
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   Route Optimizer   MCP Context      Support Agent
          │           Server              │
          └───────────────┼───────────────┘
                          ▼
                     UsageEvent
```

This makes the metering system a reusable platform service rather than a Route Optimizer feature.

---

# 58. Final Instruction to Coding Agent

Before implementing anything:

1. Inspect the existing application architecture.
2. Identify the current LLM/provider abstraction.
3. Identify where Route Optimizer requests are made.
4. Do not duplicate the existing provider abstraction.
5. Insert a clean metering boundary around the existing provider call.
6. Keep business logic and accounting logic separate.
7. Reuse existing infrastructure where appropriate.
8. Do not introduce unnecessary infrastructure if the current application can support the subsystem cleanly.
9. Implement the canonical data model first.
10. Implement OpenAI usage normalization second.
11. Implement pricing as independently versioned configuration.
12. Implement asynchronous/idempotent usage ingestion.
13. Add aggregation and reporting only after raw usage accounting is reliable.
14. Add budgets and projections after accounting is reliable.
15. Add additional providers without modifying the core metering model.

The final architecture must allow the Route Optimizer to continue functioning even if the Usage & Cost subsystem is temporarily unavailable.

**Usage accounting is observability and financial metering. It is not Route Optimizer business logic.**
