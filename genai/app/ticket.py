"""
PDF maintenance-ticket generator using ReportLab.
Produces a professional, print-ready A4 document.
"""

import os
from datetime import datetime
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from app.config import TICKET_DIR


# ---------------------------------------------------------------------------
# Custom styles
# ---------------------------------------------------------------------------
def _styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            "TicketHeader",
            parent=base["Heading1"],
            fontSize=20,
            textColor=colors.HexColor("#1a237e"),
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
        )
    )
    base.add(
        ParagraphStyle(
            "TicketSubHeader",
            parent=base["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#37474f"),
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        )
    )
    base.add(
        ParagraphStyle(
            "SectionTitle",
            parent=base["Heading3"],
            fontSize=12,
            textColor=colors.HexColor("#1a237e"),
            spaceBefore=6 * mm,
            spaceAfter=2 * mm,
        )
    )
    base.add(
        ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
        )
    )
    base.add(
        ParagraphStyle(
            "Disclaimer",
            parent=base["BodyText"],
            fontSize=7,
            textColor=colors.grey,
            spaceBefore=8 * mm,
        )
    )
    return base


# ---------------------------------------------------------------------------
# Priority colour mapping
# ---------------------------------------------------------------------------
_PRIORITY_COLORS = {
    "P1-Critical": colors.HexColor("#c62828"),
    "P2-High": colors.HexColor("#e65100"),
    "P3-Medium": colors.HexColor("#f9a825"),
    "P4-Low": colors.HexColor("#2e7d32"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_ticket_pdf(
    ticket_id: str,
    inverter_id: str,
    plant_id: str,
    plant_name: str,
    block: str,
    risk_score: float,
    risk_class: str,
    ticket_data: Dict,
    timestamp: datetime,
) -> str:
    """Generate a PDF and return its file path."""
    os.makedirs(TICKET_DIR, exist_ok=True)
    pdf_path = os.path.join(TICKET_DIR, f"{ticket_id}.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = _styles()
    elems: list = []

    # ---- Header ----
    elems.append(Paragraph("SOLARGUARD AI", styles["TicketHeader"]))
    elems.append(
        Paragraph("AUTOMATED MAINTENANCE TICKET", styles["TicketSubHeader"])
    )
    elems.append(
        HRFlowable(
            width="100%", thickness=1.5, color=colors.HexColor("#1a237e")
        )
    )
    elems.append(Spacer(1, 4 * mm))

    # ---- Meta table ----
    priority = ticket_data.get("priority", "P3-Medium")
    pri_color = _PRIORITY_COLORS.get(priority, colors.black)
    meta = [
        ["Ticket ID", ticket_id, "Priority", priority],
        ["Inverter", inverter_id, "Risk Score", f"{risk_score:.4f}"],
        ["Plant", f"{plant_name} ({plant_id})", "Risk Class", risk_class],
        ["Block", block, "Created", timestamp.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    meta_table = Table(meta, colWidths=[28 * mm, 55 * mm, 28 * mm, 55 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e8eaf6")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (3, 0), (3, 0), pri_color),
                ("FONTNAME", (3, 0), (3, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdbdbd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elems.append(meta_table)

    # ---- Title ----
    title = ticket_data.get("title", f"Risk Alert – {inverter_id}")
    elems.append(Paragraph("Issue Title", styles["SectionTitle"]))
    elems.append(Paragraph(title, styles["BodyText"]))

    # ---- Description ----
    elems.append(Paragraph("Description", styles["SectionTitle"]))
    elems.append(
        Paragraph(
            ticket_data.get("description", "N/A"), styles["BodySmall"]
        )
    )

    # ---- Root Cause ----
    elems.append(Paragraph("Root Cause Analysis", styles["SectionTitle"]))
    elems.append(
        Paragraph(
            ticket_data.get("root_cause_analysis", "N/A"),
            styles["BodySmall"],
        )
    )

    # ---- Recommended Actions ----
    actions: List[str] = ticket_data.get("recommended_actions", [])
    if actions:
        elems.append(
            Paragraph("Recommended Actions", styles["SectionTitle"])
        )
        for i, action in enumerate(actions, 1):
            elems.append(
                Paragraph(f"{i}. {action}", styles["BodySmall"])
            )

    # ---- Estimated Downtime ----
    elems.append(Paragraph("Estimated Downtime", styles["SectionTitle"]))
    elems.append(
        Paragraph(
            ticket_data.get("estimated_downtime", "TBD"),
            styles["BodySmall"],
        )
    )

    # ---- Parts Needed ----
    parts: List[str] = ticket_data.get("parts_needed", [])
    if parts:
        elems.append(Paragraph("Parts / Materials Needed", styles["SectionTitle"]))
        for part in parts:
            elems.append(Paragraph(f"• {part}", styles["BodySmall"]))

    # ---- Safety Notes ----
    notes: List[str] = ticket_data.get("safety_notes", [])
    if notes:
        elems.append(Paragraph("Safety Notes", styles["SectionTitle"]))
        for note in notes:
            elems.append(
                Paragraph(f"⚠ {note}", styles["BodySmall"])
            )

    # ---- Escalation ----
    if ticket_data.get("escalation_needed"):
        elems.append(Spacer(1, 3 * mm))
        esc_style = ParagraphStyle(
            "Escalation",
            parent=styles["BodyText"],
            fontSize=10,
            textColor=colors.white,
            backColor=colors.HexColor("#c62828"),
            alignment=TA_CENTER,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            leftIndent=10 * mm,
            rightIndent=10 * mm,
            leading=16,
        )
        elems.append(
            Paragraph(
                "ESCALATION REQUIRED – Notify plant supervisor immediately",
                esc_style,
            )
        )

    # ---- Disclaimer ----
    elems.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.grey)
    )
    elems.append(
        Paragraph(
            "This ticket was auto-generated by SolarGuard AI based on ML "
            "model predictions and SHAP feature analysis. All referenced "
            "sensor values are sourced directly from telemetry. Verify "
            "critical readings on-site before executing maintenance.",
            styles["Disclaimer"],
        )
    )

    doc.build(elems)
    return pdf_path
