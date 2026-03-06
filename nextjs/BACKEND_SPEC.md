# ☀️ SolarWatch — Solar Plant Inverter Monitoring System

## Complete System Architecture & Specification Document

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow Architecture](#2-data-flow-architecture)
3. [Database Schema](#3-database-schema)
4. [User Roles & Permissions](#4-user-roles--permissions)
5. [Operator — Complete Feature Spec](#5-operator--complete-feature-spec)
6. [Admin — Complete Feature Spec](#6-admin--complete-feature-spec)
7. [Inverter Health Classification](#7-inverter-health-classification)
8. [API Endpoints](#8-api-endpoints)
9. [Real-Time Data Pipeline](#9-real-time-data-pipeline)
10. [RAG Chatbot Integration](#10-rag-chatbot-integration)
11. [UI/UX Specification](#11-uiux-specification)
12. [Page-by-Page Breakdown](#12-page-by-page-breakdown)
13. [Edge Cases & Error Handling](#13-edge-cases--error-handling)
14. [Security Considerations](#14-security-considerations)
15. [Tech Stack](#15-tech-stack)
16. [Folder Structure](#16-folder-structure)

---

## 1. System Overview

SolarWatch is a real-time solar plant inverter monitoring platform that:

- Receives sensor data every **5 minutes** per inverter
- Classifies inverter health into **5 categories (A–E)** using an AI model
- Provides **SHAP-based explainability** for faulty inverters
- Offers a **RAG chatbot** for operator assistance
- Supports **multi-plant, multi-block, multi-inverter** hierarchy
- Separates concerns via **Operator** and **Admin** dashboards

### System Hierarchy

```
Platform (SolarWatch)
│
├── 🏭 Plant A (e.g., "Rajasthan Solar Park")
│   ├── 📦 Block 1 (e.g., "Block-A1")
│   │   ├── ⚡ Inverter INV-001  ← sensor data every 5 min
│   │   ├── ⚡ Inverter INV-002  ← sensor data every 5 min
│   │   ├── ⚡ Inverter INV-003
│   │   └── ⚡ ... (N inverters)
│   ├── 📦 Block 2
│   │   └── ⚡ ...
│   └── 📦 Block N
│
├── 🏭 Plant B (e.g., "Gujarat Solar Farm")
│   └── ...
│
└── 🏭 Plant N
```

---

## 2. Data Flow Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Physical   │    │  Sensor          │    │   AI Model       │
│   Sensors    │───►│  Simulator       │───►│  (Classification │
│   (Field)    │    │  (Data Source)    │    │   + SHAP)        │
└─────────────┘    └──────────────────┘    └────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND (Next.js API Routes)              │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │ Auth       │  │ Plant/Block│  │ Real-time Data         │  │
│  │ Service    │  │ /Inverter  │  │ Ingestion Service      │  │
│  │            │  │ CRUD       │  │ (WebSocket / SSE)      │  │
│  └────────────┘  └────────────┘  └────────────────────────┘  │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │ Operator   │  │ Admin      │  │ Chatbot                │  │
│  │ APIs       │  │ APIs       │  │ RAG Proxy              │  │
│  └────────────┘  └────────────┘  └────────────────────────┘  │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Database │ │ Operator │ │  Admin   │
        │ (Store)  │ │ Dashboard│ │ Dashboard│
        └──────────┘ └──────────┘ └──────────┘
```

### Data Ingestion Flow (per data point)

```
1. Sensor Simulator sends data point for Inverter X
        │
        ▼
2. AI Model receives raw sensor values
        │
        ▼
3. AI Model returns:
   {
     "inverter_id": "INV-001",
     "timestamp": "2025-01-18T10:05:00Z",
     "category": "D",           // A | B | C | D | E
     "confidence": 0.87,
     "is_faulty": true,
     "fault_type": "String Voltage Mismatch",
     "shap_values": {
       "dc_voltage": 0.42,
       "dc_current": -0.15,
       "ac_power": 0.31,
       "module_temp": 0.08,
       "ambient_temp": -0.03,
       "irradiation": 0.22
     },
     "raw_readings": {
       "dc_voltage": 320.5,
       "dc_current": 8.2,
       "ac_power": 2450,
       "module_temp": 45.3,
       "ambient_temp": 32.1,
       "irradiation": 0.85
     }
   }
        │
        ▼
4. Backend stores in DB + pushes to connected clients via WebSocket/SSE
        │
        ▼
5. Operator dashboard updates inverter card color + data in real-time
```

---

## 3. Database Schema

### Entity Relationship

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│    Admin     │     │  Operator   │     │     Plant       │
├─────────────┤     ├─────────────┤     ├─────────────────┤
│ id (PK)     │     │ id (PK)     │     │ id (PK)         │
│ name        │     │ name        │     │ name            │
│ email       │     │ email       │     │ location        │
│ password    │     │ password    │     │ status          │
│ created_at  │     │ created_by  │──┐  │ total_capacity  │
│ updated_at  │     │ is_active   │  │  │ created_at      │
└─────────────┘     │ created_at  │  │  │ created_by (FK) │
                    │ last_login  │  │  └────────┬────────┘
                    └─────────────┘  │           │
                                     │           │ 1:N
                    ┌────────────────┘           │
                    │                    ┌───────┴────────┐
                    │                    │     Block      │
        ┌───────────┴──────────┐        ├────────────────┤
        │ OperatorPlantAccess  │        │ id (PK)        │
        ├──────────────────────┤        │ plant_id (FK)  │
        │ operator_id (FK)     │        │ name           │
        │ plant_id (FK)        │        │ inverter_count │
        └──────────────────────┘        │ created_at     │
                                        └───────┬────────┘
                                                │ 1:N
                                        ┌───────┴────────────┐
                                        │     Inverter       │
                                        ├────────────────────┤
                                        │ id (PK)            │
                                        │ block_id (FK)      │
                                        │ name / serial      │
                                        │ capacity_kw        │
                                        │ current_category   │
                                        │ is_online          │
                                        │ last_data_at       │
                                        │ installed_at       │
                                        └───────┬────────────┘
                                                │ 1:N
                                   ┌────────────┴──────────────┐
                                   │    InverterReading        │
                                   ├───────────────────────────┤
                                   │ id (PK)                   │
                                   │ inverter_id (FK)          │
                                   │ timestamp                 │
                                   │ category (A|B|C|D|E)      │
                                   │ confidence                │
                                   │ is_faulty                 │
                                   │ fault_type                │
                                   │ dc_voltage                │
                                   │ dc_current                │
                                   │ ac_power                  │
                                   │ module_temp               │
                                   │ ambient_temp              │
                                   │ irradiation               │
                                   │ shap_values (JSON)        │
                                   └───────────────────────────┘

                                   ┌───────────────────────────┐
                                   │       Alert               │
                                   ├───────────────────────────┤
                                   │ id (PK)                   │
                                   │ inverter_id (FK)          │
                                   │ type (warning|critical)   │
                                   │ message                   │
                                   │ category_from             │
                                   │ category_to               │
                                   │ acknowledged              │
                                   │ acknowledged_by (FK)      │
                                   │ acknowledged_at           │
                                   │ created_at                │
                                   └───────────────────────────┘

                                   ┌───────────────────────────┐
                                   │      AuditLog             │
                                   ├───────────────────────────┤
                                   │ id (PK)                   │
                                   │ user_id                   │
                                   │ user_role                 │
                                   │ action                    │
                                   │ details (JSON)            │
                                   │ ip_address                │
                                   │ created_at                │
                                   └───────────────────────────┘
```

---

## 4. User Roles & Permissions

| Action                          | Operator | Admin |
|---------------------------------|----------|-------|
| View assigned plants            | ✅       | ✅    |
| View all plants                 | ❌       | ✅    |
| View inverter real-time data    | ✅       | ✅    |
| View SHAP explanations          | ✅       | ✅    |
| Use RAG chatbot                 | ✅       | ✅    |
| Acknowledge alerts              | ✅       | ✅    |
| Add/Edit/Delete plants          | ❌       | ✅    |
| Add/Edit/Delete blocks          | ❌       | ✅    |
| Add/Edit/Delete inverters       | ❌       | ✅    |
| Add/Remove operators            | ❌       | ✅    |
| Assign operators to plants      | ❌       | ✅    |
| View audit logs                 | ❌       | ✅    |
| Configure alert thresholds      | ❌       | ✅    |
| Reset own password              | ✅       | ✅    |
| Reset other's password          | ❌       | ✅    |
| View system-wide analytics      | ❌       | ✅    |
| Export data                     | ✅ (own) | ✅    |

---

## 5. Operator — Complete Feature Spec

### 5.1 Sidebar Navigation

```
┌──────────────────────────┐
│  ☀️ SolarWatch            │
│  Operator Panel           │
├──────────────────────────┤
│                          │
│  📊 Dashboard (Summary)  │
│                          │
│  ─── MY PLANTS ───       │
│                          │
│  🏭 Plant A              │
│   ├── 📦 Block 1         │
│   ├── 📦 Block 2         │
│   └── 📦 Block 3         │
│                          │
│  🏭 Plant B              │
│   ├── 📦 Block 1         │
│   └── 📦 Block 2         │
│                          │
│  ─── TOOLS ───           │
│                          │
│  🔔 Alerts & Notifications│
│  💬 AI Chatbot            │
│  📈 Reports               │
│                          │
│  ─── ACCOUNT ───         │
│                          │
│  👤 Profile               │
│  🔑 Reset Password        │
│  🚪 Logout                │
│                          │
└──────────────────────────┘
```

### 5.2 Feature List (Point-wise)

#### Dashboard (Summary View)

- **F-OP-01**: High-level summary cards showing:
  - Total inverters monitored
  - Healthy count (A+B)
  - Warning count (C)
  - Critical count (D+E)
  - Offline count
- **F-OP-02**: Plant-wise health summary bar (stacked bar per plant)
- **F-OP-03**: Recent alerts list (last 10, with timestamp)
- **F-OP-04**: "Needs Attention" section — auto-sorted list of all D and E inverters across all assigned plants

#### Block View (Inverter Grid)

- **F-OP-05**: When operator clicks a block in sidebar → show **inverter grid**
  - Each inverter = a card/cell in a responsive grid
  - Card shows: Inverter ID, current category letter, color fill, last reading time
- **F-OP-06**: Color coding applied per category:
  - A → Green (#22c55e)
  - B → Yellow-Green (#84cc16)
  - C → Amber (#f59e0b)
  - D → Orange-Red (#ef4444 at 70% opacity)
  - E → Red (#ef4444)
  - Offline → Grey (#94a3b8) with dashed border
- **F-OP-07**: Hover on inverter card → tooltip showing last reading values
- **F-OP-08**: Pulsing animation on category D and E cards to draw attention
- **F-OP-09**: "Last updated X min ago" timestamp on each card
- **F-OP-10**: Auto-refresh every 5 minutes (match sensor interval) + manual refresh button
- **F-OP-11**: Filter bar above grid: "Show All | Healthy | Warning | Critical | Offline"
- **F-OP-12**: Sort options: "By Status (worst first) | By ID | By Last Updated"

#### Inverter Detail View (Click on any inverter card)

- **F-OP-13**: **Modal or slide-out panel** with:
  - Inverter ID, Block, Plant (breadcrumb)
  - Current category with large color indicator
  - Current sensor readings (table)
  - Confidence score from AI model
- **F-OP-14**: **SHAP Explanation Panel** (only visible for C/D/E categories):
  - Horizontal bar chart of SHAP values
  - Sorted by absolute impact (highest first)
  - Color: positive impact = pushing toward fault (red bars)
  - Color: negative impact = pushing toward healthy (green bars)
  - Plain English summary: "The primary contributor to this fault is **DC Voltage** being 42% higher than normal, combined with **AC Power** drop of 31%"
- **F-OP-15**: **Trend Chart** (line chart):
  - Last 24 hours of readings (288 data points at 5-min intervals)
  - Toggle between: DC Voltage, DC Current, AC Power, Temps, Irradiation
  - Category color band overlay on chart background
- **F-OP-16**: **Fault History** — past instances when this inverter was flagged D or E
- **F-OP-17**: **"Ask AI about this inverter"** button → opens chatbot with pre-filled context

#### Alert System

- **F-OP-18**: Browser notification (with permission) when any inverter transitions to D or E
- **F-OP-19**: Sound alert (configurable on/off) for critical (E) faults
- **F-OP-20**: Alert banner at top of page for unacknowledged critical alerts
- **F-OP-21**: Alert acknowledgment — click "Acknowledge" to mark as seen
  - Acknowledged alerts move from "Active" to "History"
  - Records: who acknowledged, when
- **F-OP-22**: Alert notification center (bell icon in topbar):
  - Badge count of unacknowledged alerts
  - Dropdown list of recent alerts
  - Click → navigates to the relevant inverter

#### RAG Chatbot

- **F-OP-23**: Floating chat button (bottom-right corner)
- **F-OP-24**: Chat window with:
  - Message input
  - Chat history (scrollable)
  - Context indicator: "Discussing: INV-001 in Block-A1, Plant A"
- **F-OP-25**: Context-aware: if opened from an inverter detail view, automatically includes inverter context in the query
- **F-OP-26**: Example queries the chatbot can handle:
  - "What is causing the fault in INV-001?"
  - "What does high DC voltage mean?"
  - "How do I fix a string voltage mismatch?"
  - "Show me the maintenance procedure for this inverter type"
  - "Is this a common issue in Block A1?"
- **F-OP-27**: Chatbot shows "typing..." indicator while waiting for response
- **F-OP-28**: Chatbot responses can include formatted text, bullet points, and links

#### Account Management

- **F-OP-29**: Profile page — view own details (name, email, assigned plants) — read-only except password
- **F-OP-30**: Reset password — requires current password + new password + confirm
- **F-OP-31**: Logout — clears session, redirects to login
- **F-OP-32**: Session timeout after 30 minutes of inactivity (configurable)

---

## 6. Admin — Complete Feature Spec

### 6.1 Sidebar Navigation

```
┌──────────────────────────┐
│  ☀️ SolarWatch            │
│  Admin Panel              │
├──────────────────────────┤
│                          │
│  📊 Dashboard (Overview)  │
│                          │
│  ─── MANAGEMENT ───      │
│                          │
│  🏭 Plant Management      │
│  👥 Operator Management   │
│                          │
│  ─── MONITORING ───      │
│                          │
│  ⚡ Live Plant View       │
│  🔔 Alerts Overview       │
│                          │
│  ─── SYSTEM ───          │
│                          │
│  📋 Audit Logs            │
│  ⚙️ Settings              │
│  👤 My Profile            │
│  🚪 Logout                │
│                          │
└──────────────────────────┘
```

### 6.2 Feature List (Point-wise)

#### Admin Dashboard

- **F-AD-01**: Summary cards:
  - Total plants
  - Total blocks (across all plants)
  - Total inverters
  - Total operators
  - Inverters currently in fault (D+E)
  - Inverters offline
- **F-AD-02**: Plant health overview — table or card view showing each plant's health distribution
- **F-AD-03**: Recent admin activity (audit log preview — last 5 actions)
- **F-AD-04**: System status indicators:
  - Data pipeline: Connected / Disconnected
  - AI Model: Responsive / Down
  - Last data ingestion timestamp

#### Plant Management

- **F-AD-05**: **View all plants** — table with: Name, Location, Blocks count, Inverters count, Status, Created date, Actions
- **F-AD-06**: **Add new plant**:
  - Form fields: Plant Name, Location (text), Status (Active/Maintenance), Capacity (kW)
  - On creation, plant starts with 0 blocks
- **F-AD-07**: **Edit plant** — update name, location, status, capacity
- **F-AD-08**: **Decommission / Soft-delete plant**:
  - Confirmation dialog: "This will mark Plant X as decommissioned. All blocks and inverters will be archived. Operators assigned to only this plant will lose access. Continue?"
  - Does NOT hard-delete — historical data preserved
  - Decommissioned plants hidden from operator view but visible in admin archive
- **F-AD-09**: **Click into a plant → Block Management**:
  - View all blocks in this plant
  - Add block: Block Name, expected inverter count
  - Edit block name
  - Delete block (only if no inverters or all inverters removed first)
- **F-AD-10**: **Click into a block → Inverter Management**:
  - View all inverters in this block
  - Add inverter: Inverter ID/Serial, Capacity (kW)
  - Edit inverter details
  - Remove/Decommission inverter
  - Bulk add: upload CSV with inverter details
- **F-AD-11**: **Plant hierarchy breadcrumb**: Platform > Plant A > Block 1 > (Inverter list)

#### Operator Management

- **F-AD-12**: **View all operators** — table with: Name, Email, Assigned Plants, Status (Active/Inactive), Last Login, Created date, Actions
- **F-AD-13**: **Add new operator**:
  - Form: Full Name, Email Address
  - Admin sets a temporary initial password (system can also auto-generate)
  - On submit:
    - Operator account created with `is_active = true`
    - Email sent to operator with login credentials (or displayed on screen for hackathon)
    - Operator is forced to reset password on first login
  - Validations:
    - Email must be unique (no duplicate)
    - Email format validation
    - Name cannot be empty
    - Password minimum 8 characters
- **F-AD-14**: **Edit operator** — update name, email
- **F-AD-15**: **Assign plants to operator**:
  - Multi-select dropdown or checkbox list of all active plants
  - Operator can be assigned to 1 or more plants
  - Changing assignments takes effect immediately
- **F-AD-16**: **Deactivate operator**:
  - Sets `is_active = false`
  - If operator is currently logged in, their session is invalidated on next API call
  - Deactivated operators cannot log in
  - Deactivated operators still visible in admin panel (greyed out)
  - Can be reactivated
- **F-AD-17**: **Delete operator**:
  - Hard delete — confirmation required: "This will permanently remove Operator X and all their alert acknowledgment history. This cannot be undone."
  - Alternative: only allow deactivation, not deletion (recommended)
- **F-AD-18**: **Reset operator password**:
  - Admin can force-reset an operator's password
  - Generates new temporary password
  - Operator must change on next login

#### Live Plant View (Admin monitoring)

- **F-AD-19**: Admin can view the same inverter grid view as operators but for ALL plants (not just assigned ones)
- **F-AD-20**: Read-only — admin can view but alert acknowledgment is done by operators

#### Alert Overview

- **F-AD-21**: View all alerts across all plants
- **F-AD-22**: Filter by: Plant, Block, Severity, Acknowledged/Unacknowledged, Date range
- **F-AD-23**: See which operator acknowledged which alert and when

#### Audit Logs

- **F-AD-24**: Searchable, filterable table of all admin and system actions:
  - "Admin 'Alice' created Plant 'Solar Park X' at 2025-01-18 10:30"
  - "Admin 'Alice' added Operator 'Bob' (bob@email.com) at 2025-01-18 11:00"
  - "Admin 'Alice' deactivated Operator 'Charlie' at 2025-01-18 14:00"
  - "Operator 'Bob' acknowledged Alert #45 at 2025-01-18 15:30"
- **F-AD-25**: Filter by: User, Action type, Date range
- **F-AD-26**: Export audit logs as CSV

#### Settings

- **F-AD-27**: Alert threshold configuration:
  - Define what confidence score maps to each category (A/B/C/D/E)
  - Configure stale data timeout (default: 10 minutes — if no data for 10 min, show as offline)
- **F-AD-28**: Notification settings:
  - Enable/disable email notifications for critical alerts
  - Configure escalation timeout

#### Admin Account

- **F-AD-29**: View own profile
- **F-AD-30**: Reset own password
- **F-AD-31**: Logout

---

## 7. Inverter Health Classification

### Category Definitions

| Category | Label        | Color                     | Hex Code  | Meaning                                    |
|----------|-------------|---------------------------|-----------|---------------------------------------------|
| A        | Healthy     | 🟢 Green                 | `#22c55e` | All parameters normal, no risk              |
| B        | Low Risk    | 🟡 Yellow-Green           | `#84cc16` | Minor deviation, monitor                    |
| C        | Moderate    | 🟠 Amber                 | `#f59e0b` | Notable deviation, attention needed         |
| D        | High Risk   | 🔴 Orange-Red            | `#f97316` | Significant anomaly, likely pre-fault       |
| E        | Critical    | 🔴 Red                   | `#ef4444` | Confirmed fault, immediate action needed    |
| —        | Offline     | ⚪ Grey (dashed border)  | `#94a3b8` | No data received beyond threshold timeout   |

### Visual Behavior per Category

| Category | Card Background | Text Color | Animation      | Alert Generated |
|----------|----------------|------------|----------------|-----------------|
| A        | Light green bg  | Dark green | None           | No              |
| B        | Light yellow bg | Dark yellow| None           | No              |
| C        | Light amber bg  | Dark amber | None           | Warning (silent)|
| D        | Light red bg    | Dark red   | Slow pulse     | Warning (visual)|
| E        | Red bg          | White      | Fast pulse     | Critical (sound)|
| Offline  | Grey bg, dashed | Grey       | Fade in/out    | Info            |

---

## 8. API Endpoints

### Authentication

```
POST   /api/auth/login              → { id, password } → { token, user, role }
POST   /api/auth/logout             → invalidate session
POST   /api/auth/reset-password     → { current, new, confirm }
POST   /api/auth/force-reset        → (admin only) { operator_id, new_password }
```

### Operator APIs

```
GET    /api/operator/dashboard       → summary stats for assigned plants
GET    /api/operator/plants          → list of assigned plants with blocks
GET    /api/operator/plants/:plantId/blocks/:blockId/inverters
                                     → inverter grid data for a block
GET    /api/operator/inverters/:id   → single inverter detail + latest reading
GET    /api/operator/inverters/:id/readings?range=24h
                                     → historical readings for trend chart
GET    /api/operator/inverters/:id/faults
                                     → fault history for this inverter
GET    /api/operator/alerts          → alerts for assigned plants
POST   /api/operator/alerts/:id/acknowledge
                                     → mark alert as acknowledged
POST   /api/chatbot/query           → { message, context? } → AI response
```

### Admin APIs

```
# Dashboard
GET    /api/admin/dashboard          → system-wide stats

# Plant CRUD
GET    /api/admin/plants             → all plants
POST   /api/admin/plants             → create plant
PUT    /api/admin/plants/:id         → update plant
DELETE /api/admin/plants/:id         → decommission plant

# Block CRUD
GET    /api/admin/plants/:plantId/blocks
POST   /api/admin/plants/:plantId/blocks
PUT    /api/admin/blocks/:id
DELETE /api/admin/blocks/:id

# Inverter CRUD
GET    /api/admin/blocks/:blockId/inverters
POST   /api/admin/blocks/:blockId/inverters
POST   /api/admin/blocks/:blockId/inverters/bulk  → CSV upload
PUT    /api/admin/inverters/:id
DELETE /api/admin/inverters/:id

# Operator CRUD
GET    /api/admin/operators          → all operators
POST   /api/admin/operators          → create operator
PUT    /api/admin/operators/:id      → update operator
PUT    /api/admin/operators/:id/assign-plants → assign plants
PUT    /api/admin/operators/:id/deactivate
PUT    /api/admin/operators/:id/activate
DELETE /api/admin/operators/:id
POST   /api/admin/operators/:id/reset-password

# Alerts & Logs
GET    /api/admin/alerts             → all alerts (filterable)
GET    /api/admin/audit-logs         → all audit logs (filterable)

# Settings
GET    /api/admin/settings
PUT    /api/admin/settings
```

### Real-Time (WebSocket / SSE)

```
WS     /ws/inverter-updates          → pushes new readings as they arrive
WS     /ws/alerts                    → pushes new alerts in real-time
```

---

## 9. Real-Time Data Pipeline

### Option A: Server-Sent Events (SSE) — Simpler

```
Client                          Server
  │                               │
  │── GET /api/stream/updates ──► │
  │                               │
  │◄── event: inverter-update ──  │  (every 5 min per inverter)
  │    data: { inverter_id,       │
  │            category,          │
  │            readings,          │
  │            shap_values }      │
  │                               │
  │◄── event: alert ────────────  │  (when category transitions to D/E)
  │    data: { alert_id,          │
  │            inverter_id,       │
  │            severity }         │
  │                               │
```

### Option B: Polling (Simplest for hackathon)

```
Client polls GET /api/operator/plants/:id/blocks/:bid/inverters
every 30 seconds (or 60 seconds)

Backend returns latest state of all inverters in that block
Client diffs with previous state and updates UI accordingly
```

**Recommendation for hackathon: Start with polling, upgrade to SSE if time permits.**

---

## 10. RAG Chatbot Integration

### Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Chat UI      │     │ Next.js API      │     │ RAG Backend      │
│ (Frontend)   │────►│ /api/chatbot     │────►│ (External API    │
│              │◄────│ /query           │◄────│  or Python svc)  │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

### Request Payload

```json
{
  "message": "What is causing the fault in INV-001?",
  "context": {
    "inverter_id": "INV-001",
    "block": "Block-A1",
    "plant": "Plant A",
    "current_category": "E",
    "fault_type": "String Voltage Mismatch",
    "shap_values": {
      "dc_voltage": 0.42,
      "dc_current": -0.15,
      "ac_power": 0.31
    },
    "latest_readings": {
      "dc_voltage": 320.5,
      "dc_current": 8.2,
      "ac_power": 2450
    }
  },
  "conversation_history": [
    { "role": "user", "content": "previous message" },
    { "role": "assistant", "content": "previous response" }
  ]
}
```

---

## 11. UI/UX Specification

### Design Principles

1. **Minimal & Clean**: No clutter — operators are non-tech users in control rooms
2. **Color-driven**: Status communicated primarily through color, not text
3. **Glanceable**: Operator should understand plant health in < 3 seconds
4. **Accessible**: Works on 1080p monitors (control room), tablets, and mobile
5. **Dark/Light mode**: Control rooms are often dim — dark mode is essential

### Theme Tokens

```
Light Mode:
  --bg-primary:    #ffffff
  --bg-secondary:  #f8fafc
  --bg-tertiary:   #f1f5f9
  --text-primary:  #0f172a
  --text-secondary:#64748b
  --border:        #e2e8f0
  --sidebar-bg:    #1e293b
  --sidebar-text:  #f8fafc

Dark Mode:
  --bg-primary:    #0f172a
  --bg-secondary:  #1e293b
  --bg-tertiary:   #334155
  --text-primary:  #f8fafc
  --text-secondary:#94a3b8
  --border:        #334155
  --sidebar-bg:    #020617
  --sidebar-text:  #f8fafc
```

### Typography

```
Font: Inter (Google Fonts) — clean, modern, highly legible
Headings: 600-700 weight
Body: 400-500 weight
Monospace (IDs, data): JetBrains Mono
```

### Component Library (TailwindCSS)

All components built with Tailwind utility classes:
- Cards with `rounded-xl shadow-sm border`
- Buttons with `rounded-lg font-medium transition-colors`
- Tables with `divide-y` pattern
- Modals with backdrop blur
- Toast notifications for alerts

---

## 12. Page-by-Page Breakdown

### Page Map

```
/                           → Redirect to /login
/login                      → Login page

/operator                   → Operator dashboard (summary)
/operator/plant/[plantId]/block/[blockId]
                            → Inverter grid view
/operator/inverter/[inverterId]
                            → Inverter detail (can also be modal)
/operator/alerts            → Alerts list
/operator/chatbot           → Full-page chatbot (also available as floating widget)
/operator/profile           → Profile & password reset

/admin                      → Admin dashboard (overview)
/admin/plants               → Plant management list
/admin/plants/[plantId]     → Plant detail → block management
/admin/plants/[plantId]/blocks/[blockId]
                            → Block detail → inverter management
/admin/operators            → Operator management list
/admin/operators/new        → Add new operator form
/admin/alerts               → All alerts overview
/admin/audit-logs           → Audit log viewer
/admin/settings             → System settings
/admin/profile              → Profile & password reset
```

---

## 13. Edge Cases & Error Handling

### Data Edge Cases

| Scenario | Handling |
|----------|---------|
| No data received for >10 min | Mark inverter as "Offline" (grey) |
| AI model returns error | Show last known category with "⚠️ Stale" indicator |
| Duplicate data point (same timestamp) | Deduplicate by inverter_id + timestamp |
| Data arrives out of order | Sort by timestamp, display latest |
| All inverters in a block go offline simultaneously | Show block-level warning banner |
| Sensor sends impossible values (e.g., negative irradiation) | Backend validates; flag as "Data Error" |

### User Edge Cases

| Scenario | Handling |
|----------|---------|
| Operator tries to access unassigned plant | 403 Forbidden + redirect |
| Admin deletes plant with active operators | Remove plant from operator assignments, notify |
| Admin deactivates currently logged-in operator | Next API call returns 401, force redirect to login |
| Two admins edit same plant simultaneously | Last-write-wins (acceptable for hackathon) |
| Operator's session expires mid-action | Show "Session expired" modal with re-login option |
| Login with wrong credentials 5+ times | Show "Too many attempts, wait 5 minutes" |
| Browser tab inactive for hours | On tab focus, re-fetch all data immediately |
| Password reset with mismatched confirm | Client-side validation + server-side check |

### Network Edge Cases

| Scenario | Handling |
|----------|---------|
| Backend server down | Show "Unable to connect to server" full-page error |
| Slow network | Loading skeletons for all data-dependent components |
| WebSocket disconnects | Auto-reconnect with exponential backoff; show "Reconnecting..." banner |
| Chatbot API timeout | "The AI assistant is currently unavailable. Please try again." |

---

## 14. Security Considerations

| Area | Implementation |
|------|---------------|
| **Authentication** | JWT tokens stored in httpOnly cookies (not localStorage) |
| **Password Storage** | bcrypt hashing with salt rounds = 12 |
| **Route Protection** | Middleware checks JWT + role on every API call |
| **CSRF** | SameSite cookie attribute + CSRF tokens for mutations |
| **Rate Limiting** | 5 login attempts per minute per IP |
| **Input Validation** | Zod schemas on all API inputs |
| **XSS** | React auto-escapes; sanitize chatbot responses |
| **SQL Injection** | Parameterized queries (Prisma/Drizzle ORM) |
| **Session Timeout** | 30 min inactivity → auto-logout |
| **Audit Trail** | All admin actions logged with timestamp, IP, user |

---

## 15. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | Next.js 15 (App Router) with Turbopack |
| **Styling** | TailwindCSS v4 |
| **UI Components** | shadcn/ui (Radix-based, accessible) |
| **Charts** | Recharts (for trend charts, SHAP bar charts) |
| **State Management** | React Context + TanStack Query (server state) |
| **Real-time** | SSE (Server-Sent Events) or polling |
| **Database** | PostgreSQL (via Supabase) or SQLite (for hackathon) |
| **ORM** | Prisma |
| **Auth** | NextAuth.js or custom JWT |
| **Chatbot UI** | Custom component with streaming support |
| **Dark Mode** | next-themes + Tailwind dark: variant |
| **Icons** | Lucide React |
| **Forms** | React Hook Form + Zod validation |
| **Notifications** | Sonner (toast) + Browser Notification API |

---

## 16. Folder Structure

```
solarwatch/
├── app/
│   ├── (auth)/
│   │   └── login/
│   │       └── page.tsx
│   ├── (operator)/
│   │   ├── layout.tsx              ← operator sidebar + topbar
│   │   ├── operator/
│   │   │   ├── page.tsx            ← operator dashboard
│   │   │   ├── plant/
│   │   │   │   └── [plantId]/
│   │   │   │       └── block/
│   │   │   │           └── [blockId]/
│   │   │   │               └── page.tsx  ← inverter grid
│   │   │   ├── alerts/
│   │   │   │   └── page.tsx
│   │   │   ├── chatbot/
│   │   │   │   └── page.tsx
│   │   │   └── profile/
│   │   │       └── page.tsx
│   ├── (admin)/
│   │   ├── layout.tsx              ← admin sidebar + topbar
│   │   ├── admin/
│   │   │   ├── page.tsx            ← admin dashboard
│   │   │   ├── plants/
│   │   │   │   ├── page.tsx        ← plant list
│   │   │   │   └── [plantId]/
│   │   │   │       └── page.tsx    ← plant detail / blocks
│   │   │   ├── operators/
│   │   │   │   ├── page.tsx        ← operator list
│   │   │   │   └── new/
│   │   │   │       └── page.tsx    ← add operator form
│   │   │   ├── alerts/
│   │   │   │   └── page.tsx
│   │   │   ├── audit-logs/
│   │   │   │   └── page.tsx
│   │   │   ├── settings/
│   │   │   │   └── page.tsx
│   │   │   └── profile/
│   │   │       └── page.tsx
│   ├── api/
│   │   ├── auth/
│   │   │   ├── login/route.ts
│   │   │   ├── logout/route.ts
│   │   │   └── reset-password/route.ts
│   │   ├── operator/
│   │   │   ├── dashboard/route.ts
│   │   │   ├── plants/route.ts
│   │   │   ├── inverters/
│   │   │   │   ├── [id]/route.ts
│   │   │   │   └── [id]/readings/route.ts
│   │   │   └── alerts/route.ts
│   │   ├── admin/
│   │   │   ├── dashboard/route.ts
│   │   │   ├── plants/route.ts
│   │   │   ├── operators/route.ts
│   │   │   ├── audit-logs/route.ts
│   │   │   └── settings/route.ts
│   │   ├── chatbot/
│   │   │   └── query/route.ts
│   │   └── stream/
│   │       └── updates/route.ts     ← SSE endpoint
│   ├── layout.tsx                   ← root layout (theme provider)
│   └── globals.css
├── components/
│   ├── ui/                          ← shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── dialog.tsx
│   │   ├── input.tsx
│   │   ├── table.tsx
│   │   ├── badge.tsx
│   │   ├── tooltip.tsx
│   │   ├── dropdown-menu.tsx
│   │   ├── sheet.tsx               ← slide-out panel
│   │   └── ...
│   ├── layout/
│   │   ├── operator-sidebar.tsx
│   │   ├── admin-sidebar.tsx
│   │   ├── topbar.tsx
│   │   └── theme-toggle.tsx
│   ├── operator/
│   │   ├── dashboard-stats.tsx
│   │   ├── inverter-grid.tsx
│   │   ├── inverter-card.tsx
│   │   ├── inverter-detail-modal.tsx
│   │   ├── shap-chart.tsx
│   │   ├── trend-chart.tsx
│   │   ├── alert-banner.tsx
│   │   ├── alert-list.tsx
│   │   └── chatbot-widget.tsx
│   ├── admin/
│   │   ├── dashboard-stats.tsx
│   │   ├── plant-table.tsx
│   │   ├── plant-form.tsx
│   │   ├── operator-table.tsx
│   │   ├── operator-form.tsx
│   │   ├── block-manager.tsx
│   │   ├── inverter-manager.tsx
│   │   └── audit-log-table.tsx
│   └── shared/
│       ├── loading-skeleton.tsx
│       ├── error-boundary.tsx
│       ├── empty-state.tsx
│       ├── confirm-dialog.tsx
│       └── status-badge.tsx
├── lib/
│   ├── auth.ts                      ← JWT utilities
│   ├── db.ts                        ← Prisma client
│   ├── utils.ts                     ← helper functions
│   ├── constants.ts                 ← category colors, thresholds
│   └── validators.ts                ← Zod schemas
├── hooks/
│   ├── use-auth.ts
│   ├── use-realtime.ts              ← SSE/polling hook
│   ├── use-inverter-data.ts
│   └── use-alerts.ts
├── providers/
│   ├── theme-provider.tsx
│   ├── auth-provider.tsx
│   └── query-provider.tsx
├── prisma/
│   ├── schema.prisma
│   └── seed.ts                      ← seed demo data
├── public/
│   └── ...
├── tailwind.config.ts
├── next.config.ts
├── package.json
└── tsconfig.json
```

---

## Summary: What Was Missing (Corrections Applied)

### Operator Side — You Missed:
1. Offline/Stale data state (6th state beyond A-E)
2. Alert notification & acknowledgment system
3. Historical trend charts per inverter
4. Filter & search across inverters
5. Dashboard summary before drilling into blocks
6. Session timeout handling
7. Network disconnection handling
8. Loading states during data fetch
9. Plant assignment (which plants the operator can see)
10. SHAP explanation as visual chart (not just raw data)

### Admin Side — You Missed:
1. Block CRUD (add/edit/delete blocks within a plant)
2. Inverter CRUD (add/edit/delete inverters within a block)
3. Edit operator (not just add/remove)
4. Operator-to-plant assignment
5. Deactivate vs Delete distinction
6. Audit logging of admin actions
7. Admin dashboard with system-wide overview
8. Soft-delete / decommission vs hard-delete for plants
9. Handling of currently-logged-in operator being deactivated
10. Duplicate email prevention
11. System settings (threshold configuration)
12. What the admin sees for live monitoring (read-only plant view) 