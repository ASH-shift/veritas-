"""
Generate PDF versions of two NimbusWorks sample documents using fpdf2.
Run once: python scripts/create_sample_pdfs.py
"""

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).parent.parent / "sample_docs"


def make_pdf(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 10, title, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 5, "NimbusWorks Internal Document", align="C")
    pdf.ln(8)

    for heading, body in sections:
        # Section heading
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(1)

        # Body text
        pdf.set_font("Helvetica", size=10)
        for para in body.strip().split("\n\n"):
            para = para.strip()
            if para:
                pdf.multi_cell(0, 6, para)
                pdf.ln(3)
        pdf.ln(2)

    out_path = OUT_DIR / filename
    pdf.output(str(out_path))
    print(f"Created {out_path}")


def main() -> None:
    # ── Document 1: Equipment & Asset Policy ──────────────────────────────────
    make_pdf(
        "equipment_asset_policy.pdf",
        "NimbusWorks Equipment and Asset Policy\nVersion 1.1 | Effective: January 1, 2024",
        [
            (
                "1. Company-Issued Equipment",
                "NimbusWorks provides all full-time employees with a MacBook Pro (14-inch) as their primary work device. "
                "Engineers and designers receive 32 GB RAM configurations; all other roles receive the 16 GB standard configuration. "
                "Equipment is company property and must be returned within 5 business days of the employee's last working day.\n\n"
                "Employees who lose or damage company equipment through negligence may be held liable for repair or replacement costs "
                "up to a maximum of $500. Accidental damage should be reported to IT within 24 hours.",
            ),
            (
                "2. Equipment Refresh Cycle",
                "Company laptops are refreshed every 3 years. Employees approaching the 3-year mark will be notified by IT 60 days "
                "before their refresh date. Devices older than 4 years must be replaced immediately due to security support constraints.\n\n"
                "Peripherals (monitors, keyboards, mice) are refreshed every 5 years or replaced when defective.",
            ),
            (
                "3. Personal Device Policy",
                "Employees may use personal devices only for company email and calendar access. Access to codebases, customer data, "
                "or Confidential/Restricted information from personal devices is prohibited.\n\n"
                "Personal devices used for email must have a screen lock enabled and must be enrolled in the MDM (Mobile Device Management) "
                "profile available from IT.",
            ),
            (
                "4. Software Licenses",
                "Software licenses are managed centrally by IT. Employees must not purchase software subscriptions using personal "
                "accounts on behalf of NimbusWorks. All software procurement must go through the IT catalog or a formal purchase request.\n\n"
                "The company holds enterprise licenses for: JetBrains IDEs, Figma, Notion, Slack, Zoom, 1Password, and GitHub Teams. "
                "Additional license requests should be submitted to it@nimbusworks.com.",
            ),
            (
                "5. Asset Inventory",
                "All company assets are tracked in the IT Asset Register. Equipment is assigned to individual employees and "
                "must not be transferred to another employee without IT approval.\n\n"
                "Annual asset audits are conducted in October. Employees are required to confirm the status of their assigned equipment "
                "via the IT portal within 5 business days of receiving an audit request.",
            ),
        ],
    )

    # ── Document 2: Workplace Health and Safety Policy ─────────────────────────
    make_pdf(
        "health_safety_policy.pdf",
        "NimbusWorks Workplace Health and Safety Policy\nVersion 1.0 | Effective: January 1, 2024",
        [
            (
                "1. Our Commitment",
                "NimbusWorks is committed to providing a safe and healthy workplace for all employees, contractors, and visitors. "
                "The company complies with all applicable health and safety legislation in each jurisdiction where it operates.\n\n"
                "The VP of People Operations holds primary responsibility for health and safety compliance, supported by a Health "
                "and Safety Committee that meets quarterly.",
            ),
            (
                "2. Ergonomics",
                "All employees working at a desk for more than 4 hours per day are entitled to an ergonomic assessment. Contact "
                "facilities@nimbusworks.com to request an assessment.\n\n"
                "The company will provide ergonomic equipment (adjustable desk, ergonomic chair, monitor stand) as recommended by "
                "the assessment. For remote employees, ergonomic equipment can be purchased using the $1,500 home office stipend.",
            ),
            (
                "3. Mental Health",
                "NimbusWorks takes mental health seriously. All employees have access to the Employee Assistance Program (EAP), "
                "which provides 6 free confidential counseling sessions per issue per year.\n\n"
                "Managers are trained to recognize signs of burnout and distress. Employees are encouraged to raise mental health "
                "concerns with HR confidentially at hr@nimbusworks.com.",
            ),
            (
                "4. Incident Reporting",
                "Workplace injuries or accidents must be reported to HR within 24 hours. Near-misses should also be reported "
                "to help prevent future incidents. Reports are filed at safety.nimbusworks.com.\n\n"
                "For medical emergencies, call emergency services (911 in the US) first, then notify HR.",
            ),
            (
                "5. Emergency Procedures",
                "Each office has a designated Fire Warden and First Aid Officer. Emergency exit routes are posted on each floor. "
                "Employees should familiarise themselves with the nearest emergency exit on their first day.\n\n"
                "Fire drills are conducted twice per year (March and September). Participation is mandatory for all office-based staff.",
            ),
        ],
    )


if __name__ == "__main__":
    main()
