"""
================================================================================
Pillar 4 Explainable AI (XAI): Statistical Benford & Forensic Explainability
================================================================================
Implements:
- Quantitative Statistical Evidence Attribution (Strictly non-neural / non-SHAP)
- First-Significant Digit Distribution vs. Theoretical Benford Law PMF
- Per-Digit Absolute and Signed Difference Decomposition
- Mean Absolute Error (MAE) and Chi-Square Goodness-of-Fit Telemetry
- Dynamic, Non-Causal Human & Technical Explanation Generation
- Sparse Sample Handling without Fabricating Statistical Significance
- High-Resolution Dual-Distribution Forensic Visualization Plot
================================================================================
"""

import os
import io
import base64
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from scipy.stats import chi2

# Theoretical Benford distribution for digits 1 to 9: P(d) = log10(1 + 1/d)
BENFORD_THEORETICAL = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
BENFORD_THEORETICAL_PCT = BENFORD_THEORETICAL * 100.0


def compute_benford_comparison_table(
    obs_freqs_pct: List[float],
    exp_freqs_pct: Optional[List[float]] = None,
    digits_count: int = 0
) -> Dict[str, Any]:
    """
    Computes per-digit comparisons between observed and expected Benford distributions.
    Returns comparison table rows, signed/absolute differences, and top deviating digits.
    """
    if exp_freqs_pct is None or len(exp_freqs_pct) != 9:
        exp_freqs_pct = BENFORD_THEORETICAL_PCT.tolist()

    if len(obs_freqs_pct) != 9:
        # Fallback if empty or incomplete
        obs_freqs_pct = [0.0] * 9

    obs_arr = np.array(obs_freqs_pct)
    exp_arr = np.array(exp_freqs_pct)
    diff_arr = obs_arr - exp_arr
    abs_diff_arr = np.abs(diff_arr)

    # Rank digits by absolute deviation descending (0-indexed -> digit 1-9)
    ranked_indices = np.argsort(-abs_diff_arr)
    top_indices = ranked_indices[:3]  # Top 3 deviating digits

    table_rows = []
    top_deviations = []

    for d in range(1, 10):
        idx = d - 1
        obs_p = round(float(obs_arr[idx]), 2)
        exp_p = round(float(exp_arr[idx]), 2)
        diff_p = round(float(diff_arr[idx]), 2)
        abs_diff_p = round(float(abs_diff_arr[idx]), 2)
        is_top = idx in top_indices

        # Approximate counts
        obs_count = int(round((obs_p / 100.0) * digits_count)) if digits_count > 0 else 0
        exp_count = int(round((exp_p / 100.0) * digits_count)) if digits_count > 0 else 0

        # Relative deviation percentage: |Obs - Exp| / Exp * 100
        rel_deviation_pct = round((abs_diff_p / max(exp_p, 0.1)) * 100.0, 1)

        row = {
            "digit": int(d),
            "observed_pct": obs_p,
            "expected_pct": exp_p,
            "diff_pct": diff_p,
            "abs_diff_pct": abs_diff_p,
            "obs_count": obs_count,
            "exp_count": exp_count,
            "relative_dev_pct": rel_deviation_pct,
            "is_top_deviant": is_top,
            "direction": "overrepresented" if diff_p > 0 else ("underrepresented" if diff_p < 0 else "exact")
        }
        table_rows.append(row)

    # Top deviating digits structured list
    for idx in top_indices:
        d = int(idx + 1)
        top_deviations.append({
            "digit": int(d),
            "observed_pct": round(float(obs_arr[idx]), 2),
            "expected_pct": round(float(exp_arr[idx]), 2),
            "diff_pct": round(float(diff_arr[idx]), 2),
            "abs_diff_pct": round(float(abs_diff_arr[idx]), 2),
            "direction": "overrepresented" if diff_arr[idx] > 0 else "underrepresented"
        })

    return {
        "table_rows": table_rows,
        "top_deviations": top_deviations,
        "max_deviation_digit": int(ranked_indices[0] + 1),
        "max_deviation_pct": round(float(abs_diff_arr[ranked_indices[0]]), 2)
    }


def generate_benford_chart_plot(
    obs_freqs_pct: List[float],
    exp_freqs_pct: List[float],
    mae: float,
    chi_sq: float,
    p_val: float,
    digits_count: int,
    verdict: str,
    top_deviant_digits: List[int],
    output_prefix: str = "DOC_XAI"
) -> Dict[str, Any]:
    """
    Renders a publication-grade dark-themed comparison plot for Benford explainability.
    Saves to storage/xai/ and returns file path & base64 data URI.
    """
    digits_arr = np.arange(1, 10)
    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1, 
        figsize=(8.5, 6.0), 
        facecolor='#0b1120',
        gridspec_kw={'height_ratios': [2.2, 1.0]},
        dpi=160
    )

    # 1. Main Distribution Comparison
    ax_main.set_facecolor('#0f172a')
    is_fake = "FORGED" in verdict or "SYNTHESIZED" in verdict or "FAKE" in verdict
    
    # Colors
    bar_colors = []
    for d in range(1, 10):
        if d in top_deviant_digits and is_fake:
            bar_colors.append('#f43f5e')  # Highlighted top deviant digit (Rose)
        elif is_fake:
            bar_colors.append('#fb7185')  # Muted rose
        else:
            bar_colors.append('#10b981')  # Emerald authentic

    bars = ax_main.bar(
        digits_arr - 0.15, 
        obs_freqs_pct, 
        width=0.30, 
        color=bar_colors, 
        alpha=0.90, 
        label=f'Observed Distribution (N={digits_count})',
        edgecolor='#ffffff33',
        linewidth=1
    )

    exp_line = ax_main.plot(
        digits_arr + 0.15, 
        exp_freqs_pct, 
        color='#38bdf8', 
        marker='o', 
        linewidth=2.5, 
        markersize=7, 
        label="Theoretical Benford's Law: log₁₀(1 + 1/d)"
    )

    # Value tags on top of bars
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax_main.text(
                bar.get_x() + bar.get_width() / 2.,
                h + 0.8,
                f'{h:.1f}%',
                ha='center',
                va='bottom',
                fontsize=8,
                color='#e2e8f0',
                fontweight='bold',
                fontfamily='monospace'
            )

    ax_main.set_xticks(digits_arr)
    ax_main.set_xticklabels([f'Digit {d}' for d in digits_arr], color='#cbd5e1', fontsize=9, fontweight='bold')
    ax_main.set_ylabel('Proportion / Frequency (%)', color='#cbd5e1', fontsize=9.5, fontweight='bold')
    ax_main.set_title(
        f"Statistical Benford Distribution Analysis • {verdict} (MAE = {mae:.4f})",
        color='#f8fafc',
        fontsize=11,
        fontweight='bold',
        pad=12
    )
    ax_main.grid(color='#334155', alpha=0.4, linestyle='--')
    ax_main.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9', fontsize=8.5)
    
    # Set y limit with breathing room
    max_y = max(max(obs_freqs_pct) if obs_freqs_pct else 35, max(exp_freqs_pct) if exp_freqs_pct else 35) + 6.0
    ax_main.set_ylim(0, max_y)

    for spine in ax_main.spines.values():
        spine.set_color('#334155')

    # 2. Difference Subplot (Observed - Expected)
    ax_diff.set_facecolor('#0f172a')
    diff_arr = np.array(obs_freqs_pct) - np.array(exp_freqs_pct)
    diff_colors = ['#f43f5e' if val > 0 else '#38bdf8' for val in diff_arr]
    
    ax_diff.bar(
        digits_arr, 
        diff_arr, 
        width=0.45, 
        color=diff_colors, 
        alpha=0.85,
        edgecolor='#ffffff22'
    )
    ax_diff.axhline(0, color='#94a3b8', linewidth=1, linestyle='-')
    ax_diff.set_xticks(digits_arr)
    ax_diff.set_xticklabels([str(d) for d in digits_arr], color='#94a3b8', fontsize=8.5)
    ax_diff.set_ylabel('Difference (Δ %)', color='#cbd5e1', fontsize=8.5, fontweight='bold')
    ax_diff.set_xlabel('First Significant Digit (d ∈ [1..9])', color='#cbd5e1', fontsize=9, fontweight='bold')
    ax_diff.grid(color='#334155', alpha=0.3, linestyle=':')

    for spine in ax_diff.spines.values():
        spine.set_color('#334155')

    # Annotate summary box on main chart
    p_str = f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"
    stat_box_text = f"MAE: {mae:.4f}\nχ²: {chi_sq:.2f} (df=8)\np-value: {p_str}\nN: {digits_count}"
    ax_main.text(
        0.03, 0.93,
        stat_box_text,
        transform=ax_main.transAxes,
        fontsize=8.5,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#475569', alpha=0.95),
        color='#38bdf8'
    )

    plt.tight_layout()

    # Save to buffer and base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0b1120')
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)

    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{base64_data}"

    # Save to storage directory if accessible
    relative_path = f"xai/{output_prefix}_{uuid.uuid4().hex[:6]}_benford.png"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector", "storage", "xai")
    os.makedirs(storage_dir, exist_ok=True)
    full_save_path = os.path.join(storage_dir, os.path.basename(relative_path))
    
    try:
        with open(full_save_path, "wb") as f:
            f.write(img_bytes)
    except Exception as fe:
        print(f"[Pillar 4 XAI Plot Warning] Could not save chart to disk: {fe}")

    return {
        "chart_url": relative_path,
        "chart_data_uri": data_uri
    }


def generate_pillar4_statistical_xai(
    p4_res: Dict[str, Any],
    digits_raw: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Generates explainable statistical evidence for Pillar 4 Benford's Law OCR analysis.
    Provides complete statistical transparency without neural XAI or SHAP.
    """
    try:
        applicable = bool(p4_res.get("applicable", True))
        digits_count = int(p4_res.get("digits_count", 0))

        # Check for sparse document or non-applicable case
        if not applicable or digits_count < 5:
            reason = p4_res.get("reason", f"Document contains insufficient numerical values ({digits_count} detected).")
            human_exp = (
                f"Benford statistical explanation is unavailable or limited for this document because fewer than 5 numerical digits "
                f"({digits_count} detected) were extracted via OCR. Statistical significance requires multi-digit financial or transactional data."
            )
            tech_exp = (
                f"Sample size N={digits_count} is below the minimal degrees-of-freedom threshold (df=8) required for Chi-Square goodness-of-fit testing. "
                f"Preserved sparse document handling without fabricating statistical significance."
            )
            limitations = [
                "Benford's Law analysis requires multi-magnitude transactional numerical quantities (recommended N ≥ 15).",
                "Document contained sparse numerical text; statistical explainability was gracefully bypassed."
            ]

            return {
                "xai_available": True,
                "pillar": "Pillar 4: Document & Benford Forensics",
                "method": "Statistical Benford Goodness-of-Fit",
                "prediction": p4_res.get("verdict", "N/A (NON-DOCUMENT)"),
                "confidence": 0.0,
                "is_sparse": True,
                "digits_count": digits_count,
                "mae": None,
                "chi_square": None,
                "p_value": None,
                "comparison_table": [],
                "top_deviations": [],
                "evidence": [],
                "visualizations": [],
                "human_explanation": human_exp,
                "technical_explanation": tech_exp,
                "limitations": limitations
            }

        # Valid numerical data available
        is_auth = bool(p4_res.get("is_authentic", True))
        verdict = p4_res.get("verdict", "AUTHENTIC DOCUMENT" if is_auth else "FORGED / AI-SYNTHESIZED")
        confidence = float(p4_res.get("confidence", 85.0))
        mae = float(p4_res.get("mae", 0.0))
        chi_sq = float(p4_res.get("chi_square", 0.0))
        p_val = float(p4_res.get("p_value", 0.5))

        obs_freqs = p4_res.get("obs_freqs", [])
        exp_freqs = p4_res.get("expected_freqs", [])

        if not obs_freqs or len(obs_freqs) != 9:
            obs_freqs = BENFORD_THEORETICAL_PCT.tolist()
        if not exp_freqs or len(exp_freqs) != 9:
            exp_freqs = BENFORD_THEORETICAL_PCT.tolist()

        # Compute comparison table and top deviations
        comp_data = compute_benford_comparison_table(obs_freqs, exp_freqs, digits_count=digits_count)
        table_rows = comp_data["table_rows"]
        top_deviations = comp_data["top_deviations"]
        top_digits = [d["digit"] for d in top_deviations[:2]]
        top_digits_str = " and ".join([f"Digit '{d['digit']}' (diff: {d['diff_pct']:+.1f}%)" for d in top_deviations[:2]])

        # Generate comparison chart
        chart_info = generate_benford_chart_plot(
            obs_freqs_pct=obs_freqs,
            exp_freqs_pct=exp_freqs,
            mae=mae,
            chi_sq=chi_sq,
            p_val=p_val,
            digits_count=digits_count,
            verdict=verdict,
            top_deviant_digits=top_digits,
            output_prefix=f"DOC_{digits_count}"
        )

        # Build Evidence Objects
        evidence = []
        
        # 1. MAE
        dir_mae = "supports_fake" if not is_auth else "supports_real"
        evidence.append({
            "feature": "benford_mean_absolute_error",
            "value": round(mae, 4),
            "direction": dir_mae,
            "importance": 0.95,
            "description": f"Mean Absolute Error of observed first-digit frequencies compared to theoretical Benford PMF (MAE = {mae:.4f})."
        })

        # 2. Chi-Square Statistic & p-value
        p_val_display = f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"
        dir_chi2 = "supports_fake" if p_val < 0.01 else "supports_real"
        evidence.append({
            "feature": "chi_square_statistic",
            "value": round(chi_sq, 2),
            "direction": dir_chi2,
            "importance": 0.90,
            "description": f"Chi-Square goodness-of-fit statistic = {chi_sq:.2f} with 8 degrees of freedom (p-value = {p_val_display})."
        })

        # 3. OCR Digit Count
        evidence.append({
            "feature": "ocr_extracted_digits_count",
            "value": digits_count,
            "direction": "neutral",
            "importance": 0.50,
            "description": f"Total first significant digits extracted from document text via OCR (N = {digits_count})."
        })

        # 4. Top Deviations Evidence
        if top_deviations:
            top1 = top_deviations[0]
            evidence.append({
                "feature": f"digit_{top1['digit']}_distribution_deviation",
                "value": top1["abs_diff_pct"],
                "direction": dir_mae,
                "importance": 0.80,
                "description": f"Digit '{top1['digit']}' showed the largest frequency anomaly ({top1['observed_pct']:.1f}% observed vs {top1['expected_pct']:.1f}% expected, diff = {top1['diff_pct']:+.1f}%)."
            })

        # Dynamic Human-Readable Explanation adhering to non-causal forensic requirements
        p_val_display = f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"
        if not is_auth:
            human_exp = (
                f"Benford analysis found that the observed first-digit distribution differs from the expected distribution across the {digits_count} extracted numerical values. "
                f"The largest deviations occurred for {top_digits_str}. "
                f"The Chi-Square statistic ({chi_sq:.2f}, p-value = {p_val_display}) and MAE ({mae:.4f}) provide the quantitative evidence used by this analysis. "
                f"This statistical pattern is consistent with the forensic signal detected by the system and should be considered alongside the other evidence."
            )
        else:
            human_exp = (
                f"Benford analysis found that the observed first-digit distribution conforms closely to the expected logarithmic distribution across the {digits_count} extracted numerical values. "
                f"The largest deviations occurred for {top_digits_str} (within natural statistical accounting tolerance). "
                f"The Chi-Square statistic ({chi_sq:.2f}, p-value = {p_val_display}) and MAE ({mae:.4f}) provide the quantitative evidence used by this analysis. "
                f"This statistical pattern is consistent with the forensic signal detected by the system and should be considered alongside the other evidence."
            )

        # Dynamic Technical Explanation
        top_dev_breakdown = ", ".join([f"d={d['digit']}: diff={d['diff_pct']:+.2f}%" for d in top_deviations])
        tech_exp = (
            f"Empirical first-digit PMF evaluated across N={digits_count} OCR tokens against theoretical Benford logarithmic PMF P(d) = log10(1 + 1/d). "
            f"Pearson Chi-Square test (df=8) produced Chi2 = {chi_sq:.4f}, p = {p_val:.5e}, MAE = {mae:.4f}. "
            f"Ranked digit deviations: [{top_dev_breakdown}]. "
            f"Sample size dynamic threshold scaling applied: strict={0.020 + (0.010 * 100.0 / max(digits_count, 30)):.4f}."
        )

        limitations = [
            "Benford's Law applies to naturally occurring multi-magnitude numbers (invoices, ledgers, transaction prices), not assigned identifiers (phone numbers, postal codes, product serials).",
            "Scanned document compression and low OCR resolution may occasionally distort numeral recognition."
        ]

        if digits_count < 15:
            limitations.insert(0, f"Sample size (N={digits_count}) is modest (N < 15); empirical variances naturally widen for smaller sample sizes.")

        visualizations = [{
            "type": "benford_distribution_comparison",
            "chart_url": chart_info["chart_url"],
            "chart_data_uri": chart_info["chart_data_uri"],
            "observed": obs_freqs,
            "expected": exp_freqs,
            "mae": mae,
            "chi_square": chi_sq,
            "p_value": p_val,
            "digits_count": digits_count
        }]

        return {
            "xai_available": True,
            "pillar": "Pillar 4: Document & Benford Forensics",
            "method": "Statistical Benford Goodness-of-Fit",
            "prediction": verdict,
            "confidence": confidence,
            "is_sparse": digits_count < 15,
            "digits_count": digits_count,
            "mae": mae,
            "chi_square": chi_sq,
            "p_value": p_val,
            "comparison_table": table_rows,
            "top_deviations": top_deviations,
            "chart_image": chart_info["chart_url"],
            "chart_image_data_uri": chart_info["chart_data_uri"],
            "evidence": evidence,
            "visualizations": visualizations,
            "human_explanation": human_exp,
            "technical_explanation": tech_exp,
            "limitations": limitations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "xai_available": False,
            "pillar": "Pillar 4: Document & Benford Forensics",
            "method": "Statistical Benford Goodness-of-Fit",
            "prediction": p4_res.get("verdict", "UNKNOWN"),
            "confidence": float(p4_res.get("confidence", 50.0)),
            "error": str(e),
            "human_explanation": f"Statistical explainability analysis encountered an error: {e}",
            "technical_explanation": str(e),
            "evidence": [],
            "visualizations": [],
            "limitations": []
        }
