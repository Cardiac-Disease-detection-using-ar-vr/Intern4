"""
src/intern4_fusion/report_pdf.py
===================================
Generates fusion_comparison_report.pdf — Task 5.3 deliverable.
"""

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)


def generate_pdf_report(results_table: list,
                        best_strategy: str,
                        best_metrics: dict,
                        calibration_plot_path: Path,
                        reliability_plot_path: Path,
                        out_path: Path,
                        n_vasanth: int, n_tushar: int):

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]

    story = []

    story.append(Paragraph("Intern 4 — Unified Fusion Comparison Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                           body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Dataset Summary", h2))
    story.append(Paragraph(
        f"Base Module (Intern 2 / Vasanth): {n_vasanth} patients, EchoNet-Dynamic, "
        f"3-class labels derived from EF thresholds.", body))
    story.append(Paragraph(
        f"Extended Module (Intern 3 / Tushar): {n_tushar} patients (test split), "
        f"binary ground-truth labels (0=healthy, 1=disease).", body))
    story.append(Paragraph(
        "<b>Note:</b> The two modules were trained on disjoint patient populations "
        "with no overlapping patient IDs. Fusion is therefore performed at score "
        "level on Tushar's test cohort, using Tushar's predicted EF as a proxy "
        "input to simulate what the Base module would output for the same patient.",
        body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Fusion Strategy Comparison", h2))
    table_data = [["Model", "Bal. Acc", "Macro F1", "AUROC", "ECE", "MCE"]]
    for row in results_table:
        table_data.append([
            row["model"], str(row["balanced_acc"]), str(row["macro_f1"]),
            str(row["auroc"]), str(row["ece"]), str(row.get("mce", "—")),
        ])

    t = Table(table_data, colWidths=[2.3 * inch, 0.85 * inch, 0.85 * inch,
                                     0.8 * inch, 0.7 * inch, 0.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("3. Best Strategy", h2))
    story.append(Paragraph(f"<b>{best_strategy}</b>", body))
    story.append(Paragraph(
        f"Balanced Accuracy: {best_metrics['balanced_acc']} &nbsp;&nbsp; "
        f"Macro F1: {best_metrics['macro_f1']} &nbsp;&nbsp; "
        f"AUROC: {best_metrics['auroc']} &nbsp;&nbsp; "
        f"ECE: {best_metrics['ece']} (target &lt; 0.05) &nbsp;&nbsp; "
        f"MCE: {best_metrics.get('mce', '—')}", body))
    story.append(Spacer(1, 14))

    if reliability_plot_path and Path(reliability_plot_path).exists():
        story.append(Paragraph("4. Reliability Diagram (Best Strategy)", h2))
        story.append(Image(str(reliability_plot_path), width=6.5 * inch,
                           height=6.5 * inch * 0.35))
        story.append(Spacer(1, 10))

    if calibration_plot_path and Path(calibration_plot_path).exists():
        story.append(Paragraph("5. Calibration: Before vs After", h2))
        story.append(Image(str(calibration_plot_path), width=6.5 * inch,
                           height=6.5 * inch * 0.4))

    story.append(PageBreak())
    story.append(Paragraph("6. Known Limitations", h2))
    story.append(Paragraph(
        "1. No patient-level pairing exists between the Base and Extended modules; "
        "fusion is score-level only, validated on Tushar's cohort.", body))
    story.append(Paragraph(
        "2. Tushar's raw 'mild' probability output was found to be non-functional "
        "(capped near 0.15, never the argmax across all 1,026 patients). This was "
        "corrected by reconstructing 3-class probabilities from his Dempster-Shafer "
        "common_alpha score combined with a sigmoid split on predicted EF.", body))
    story.append(Paragraph(
        "3. Early/Intermediate/Cross-Attention fusion models are trained on "
        "Tushar's 512-d embeddings paired with simulated Base features; true "
        "intermediate fusion requires Vasanth's ecg_embeddings.pt, not yet received.",
        body))

    doc.build(story)
    print(f"PDF report saved to {out_path}")
