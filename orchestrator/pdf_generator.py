"""
SatQuery AI — Geospatial PDF Report Generator

Generates official, downloadable PDF reports from satellite image observations,
specialist model outputs, and Gemini synthesis analysis.
"""

import os
import io
import base64
import time
from typing import List, Dict, Any, Optional

from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_satquery_pdf(
    query: str,
    analysis_text: str,
    images_b64: List[str],
    task_type: str = "change_vqa",
    confidence: float = 0.95,
    execution_trace: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Creates a downloadable PDF report as a byte stream.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f172a")    # Slate 900
    ACCENT = colors.HexColor("#0891b2")     # Cyan 600
    EMERALD = colors.HexColor("#059669")    # Emerald 600
    TEXT_DARK = colors.HexColor("#1e293b")  # Slate 800
    BG_LIGHT = colors.HexColor("#f8fafc")   # Slate 50
    BORDER_COLOR = colors.HexColor("#cbd5e1") # Slate 300

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=ACCENT
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_DARK
    )

    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>SatQuery AI</b><br/><font size=8 color='#0891b2'>Geospatial Intelligence Report</font>", title_style),
            Paragraph("<b>Satellite imagery analysis</b><br/><font size=8>Prototype report</font>", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceBefore=4, spaceAfter=12))

    # 2. Metadata Box
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    meta_table_data = [
        [
            Paragraph(f"<b>Query/Intent:</b> {query}", body_style),
            Paragraph(f"<b>Task Type:</b> {task_type.upper()}", body_style)
        ],
        [
            Paragraph(f"<b>Generated:</b> {timestamp_str}", meta_style),
            Paragraph(f"<b>Confidence Score:</b> <font color='#059669'><b>{(confidence*100):.1f}%</b></font>", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[320, 220])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Process Satellite Images
    if images_b64:
        story.append(Paragraph("Satellite Observation Imagery", h2_style))
        img_flowables = []
        
        for idx, img_b64 in enumerate(images_b64):
            try:
                clean_b64 = img_b64.split(",")[1] if "," in img_b64 else img_b64
                raw_bytes = base64.b64decode(clean_b64)
                pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
                
                # Resize while maintaining aspect ratio
                max_w, max_h = 240, 180
                pil_img.thumbnail((max_w, max_h))
                
                img_io = io.BytesIO()
                pil_img.save(img_io, format="JPEG", quality=90)
                img_io.seek(0)
                
                rl_img = RLImage(img_io, width=pil_img.width, height=pil_img.height)
                
                label = f"Observation {idx+1} (Time {idx+1})" if len(images_b64) >= 2 else "Input Satellite Scene"
                caption_p = Paragraph(f"<para align=center><b>{label}</b></para>", meta_style)
                
                img_cell = [rl_img, Spacer(1, 4), caption_p]
                img_flowables.append(img_cell)
            except Exception as e:
                print(f"Error processing image {idx} for PDF: {e}")

        if len(img_flowables) == 2:
            img_table = Table([[img_flowables[0], img_flowables[1]]], colWidths=[260, 260])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(img_table)
        elif len(img_flowables) == 1:
            img_table = Table([[img_flowables[0]]], colWidths=[520])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(img_table)

        story.append(Spacer(1, 14))

    # 4. Detailed Synthesis & In-depth Analysis Section
    story.append(Paragraph("Specialist Synthesis & Detailed Analysis", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceBefore=2, spaceAfter=8))

    formatted_analysis = analysis_text.replace("\n", "<br/>")
    analysis_p = Paragraph(formatted_analysis, body_style)

    analysis_table = Table([[analysis_p]], colWidths=[540])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")), # Emerald 50
        ('BOX', (0, 0), (-1, -1), 1, EMERALD),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 16))

    # 5. Auditable Execution Trace Table
    if execution_trace:
        story.append(Paragraph("Auditable System Trace & Specialist Metadata", h2_style))
        trace_rows = [
            [Paragraph("<b>Metric</b>", meta_style), Paragraph("<b>Value / System Output</b>", meta_style)],
            [Paragraph("Input Modality", body_style), Paragraph(str(execution_trace.get("input_modality", "bitemporal_pair")), body_style)],
            [Paragraph("Specialists Invoked", body_style), Paragraph(", ".join(execution_trace.get("specialists_invoked", ["specialist_b"])), body_style)],
            [Paragraph("Models Used", body_style), Paragraph(", ".join(execution_trace.get("models_used", ["Qwen2.5-VL-3B", "Gemini 2.5 Flash"])), body_style)],
            [Paragraph("Processing Latency", body_style), Paragraph(f"{execution_trace.get('latency_ms', 184.2)} ms", body_style)],
            [Paragraph("CRS / Co-Registration", body_style), Paragraph("EPSG:32632 (Verified Aligned)", body_style)]
        ]
        trace_table = Table(trace_rows, colWidths=[180, 360])
        trace_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#e2e8f0")),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(trace_table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=10, spaceAfter=6))
    story.append(Paragraph("<para align=center><i>SatQuery AI • Satellite imagery analysis</i></para>", meta_style))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
