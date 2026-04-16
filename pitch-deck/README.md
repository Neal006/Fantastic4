# LUMIN.AI — Pitch Deck

> **Sales-oriented presentation for companies interested in deploying LUMIN.AI for utility-scale solar plant monitoring.**

---

## Overview

This pitch deck is a self-contained, interactive HTML presentation designed for pitching LUMIN.AI to potential enterprise customers — solar farm operators, utility companies, energy asset managers, and O&M service providers.

It covers the **complete product story**: from the industry problem and market opportunity through the full technical architecture, GenAI capabilities, security posture, competitive differentiation, pricing model, ROI projections, and product roadmap.

---

## How to Use

### Open Locally

Simply open `index.html` in any modern web browser:

```bash
# From the project root
start pitch-deck/index.html        # Windows
open pitch-deck/index.html          # macOS
xdg-open pitch-deck/index.html     # Linux
```

No build step, no dependencies, no internet required — the presentation is fully self-contained.

### Navigation

| Control | Action |
|---------|--------|
| **Arrow Keys** (`←` / `→`) | Previous / Next slide |
| **Spacebar** | Next slide |
| **Home / End** | Jump to first / last slide |
| **Click dots** | Jump to specific slide |
| **Nav buttons** | Previous / Next slide |
| **Swipe** (touch) | Previous / Next slide |

---

## Slide Outline (20 Slides)

| # | Slide | Content |
|:-:|-------|---------|
| 01 | **Title** | LUMIN.AI branding, tagline, high-level data flow |
| 02 | **The Problem** | 4 key pain points in solar operations (alarm fatigue, black-box AI, manual tickets, siloed data) |
| 03 | **Market Opportunity** | $11B+ solar O&M market, 1.6 TW installed capacity, $3.2M avg loss per 100MW plant |
| 04 | **Our Solution** | 4 microservices overview (ML Pipeline, Inference, GenAI, Web App) |
| 05 | **How It Works** | Architecture flow: Sensors → ML → GenAI → Dashboard |
| 06 | **Operator Experience** | What operators actually see: AI explanations, health grid, chatbot, PDF tickets |
| 07 | **ML Engine Deep Dive** | 7-stage pipeline, 183 features, Optuna tuning, SHAP explainability |
| 08 | **GenAI Layer** | RAG pipeline, 4-layer hallucination guardrails, Groq Llama 3.3 70B, LangSmith |
| 09 | **LLM Ablation Study** | 3 models × 27 test cases, comparative results, Groq selection rationale |
| 10 | **Explainability** | 3-layer XAI (SHAP → Plain English → Visual), A–E category mapping |
| 11 | **Performance** | Latency benchmarks (<50ms ML, ~1s LLM), scalability metrics |
| 12 | **Technology Stack** | Full 9-layer stack table (ML, GenAI, RAG, Frontend, Backend, DB, etc.) |
| 13 | **Security** | JWT auth, RBAC, input validation, SQL injection protection, audit trails |
| 14 | **Digital Twin** | Future 3D visualization architecture (Three.js, WebSocket, GenAI Copilot) |
| 15 | **Competitive Advantage** | Feature comparison: Traditional SCADA vs Basic ML vs LUMIN.AI |
| 16 | **Business Model** | 3-tier SaaS pricing (Starter, Professional, Enterprise) |
| 17 | **ROI** | 30% downtime reduction, 60% faster tickets, 80% faster diagnostics, 5x operator efficiency |
| 18 | **Product Roadmap** | Now → Q3-Q4 2026 → 2027 → 2028+ vision |
| 19 | **Team** | Team Fantastic4 profiles, skills, and credentials |
| 20 | **Call to Action** | Live demo, pilot program, contact information |

---

## Target Audience

This deck is designed for:

- **Solar Farm Operators** — Companies running utility-scale PV plants (10MW+)
- **Utility Companies** — Organizations managing distributed solar assets
- **Energy Asset Managers** — Firms overseeing portfolios of renewable energy installations
- **O&M Service Providers** — Third-party maintenance companies servicing solar fleets
- **EPC Contractors** — Engineering firms seeking post-installation monitoring solutions
- **Investors / VCs** — For fundraising and partnership discussions

---

## Technical Details Covered

The pitch deck comprehensively covers:

### Machine Learning
- 7-stage pipeline (ingestion → cleaning → 183 features → auto-labeling → anomaly enrichment → SMOTE + split → XGBoost + SHAP)
- Walk-forward cross validation (no future data leakage)
- Optuna hyperparameter optimization (40 Bayesian trials)
- SHAP TreeExplainer for exact per-prediction feature attribution
- 3-class risk classification → 5-category operational mapping (A–E)

### Generative AI
- Groq Llama 3.3 70B (91.7% accuracy, ~1.0s latency)
- RAG pipeline (PyMuPDF → SentenceTransformers → FAISS)
- 4-layer hallucination prevention (0% hallucination rate)
- Agentic PDF ticket generation
- Multi-turn conversational Q&A with session memory
- LangSmith observability and tracing

### Web Application
- Next.js 15 with TailwindCSS v4 and shadcn/ui
- Express.js REST API with JWT authentication
- AWS RDS MySQL cloud database
- Real-time simulator (15s cycle, CSV-derived sensor data)
- Operator and Admin dashboards with role-based access

### Security & Infrastructure
- JWT httpOnly cookies, bcrypt, rate limiting
- Parameterized SQL, Zod validation, XSS protection
- 4 independently deployable microservices
- Graceful fallback when ML servers are unavailable

---

## Customization

The pitch deck is a single HTML file with inline CSS and JavaScript. To customize:

- **Colors**: Edit CSS variables in the `:root` block at the top of the `<style>` section
- **Content**: Edit slide HTML directly — each slide is a `<div class="slide">` block
- **Add slides**: Copy an existing slide block, update `data-slide` index, and increment the total
- **Branding**: Update the `.logo` element and title tag
- **Fonts**: Change the Google Fonts import URL

---

## Design Features

- **Dark theme** optimized for projection/screen sharing
- **Smooth transitions** with CSS cubic-bezier animations
- **Progress bar** showing deck completion
- **Responsive** — works on desktop, tablet, and mobile
- **Touch support** — swipe navigation on mobile/tablet
- **Keyboard navigation** — arrow keys, spacebar, home/end
- **No dependencies** — zero external JS/CSS libraries required
- **Ambient glow orbs** — subtle background visual effects per slide

---

## File Structure

```
pitch-deck/
├── index.html    # Complete self-contained pitch deck
└── README.md     # This file
```

---

## Related Project Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **Project README** | `../README.md` | Full technical documentation (1200+ lines) |
| **System Architecture Report** | `../SYSTEM_ARCHITECTURE_REPORT.md` | Academic-style architecture paper |
| **Backend Spec** | `../nextjs/BACKEND_SPEC.md` | Complete backend system specification |

---

**Built by Team Fantastic4 for HACKaMINeD 2026**
