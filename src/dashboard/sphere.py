"""
dashboard/sphere.py
────────────────────
Reactive Listening Sphere — real-time audio-reactive animation.

Renders a CSS/JS animated sphere inside Streamlit that reacts to
acoustic energy from the ConvinceSense pipeline.

Usage:
    from src.dashboard.sphere import render_reactive_sphere
    render_reactive_sphere(energy=0.6, is_listening=True)
"""

import streamlit.components.v1 as components


def render_reactive_sphere(
    energy: float = 0.0,
    is_listening: bool = False,
    height: int = 280,
) -> None:
    """Render the reactive listening sphere.

    Parameters
    ----------
    energy : float
        Normalized acoustic energy in [0, 1].  Drives sphere scale,
        glow intensity, and ripple spawning.
    is_listening : bool
        Whether the pipeline is actively capturing audio.
    height : int 
        Height of the HTML component in pixels.
    """
    # Clamp and normalize
    e = max(0.0, min(float(energy) * 10, 1.0))   # raw RMS → 0-1

    # Animation parameters driven by energy
    scale      = 1.0 + e * 0.35
    glow_size  = 20 + e * 60
    glow_alpha = 0.3 + e * 0.5
    pulse_dur  = max(1.8 - e * 1.0, 0.6)
    ripple_count = min(int(e * 5) + 1, 4) if e > 0.05 else 0

    # State-dependent colors
    if not is_listening:
        core_c1, core_c2 = "#4b5563", "#1f2937"   # gray idle
        glow_color = "107, 114, 128"                # gray glow
        status_text = "Ready"
        status_dot  = "#6b7280"
    elif e < 0.1:
        core_c1, core_c2 = "#7c3aed", "#3b82f6"   # calm purple-blue
        glow_color = "124, 58, 237"
        status_text = "Listening…"
        status_dot  = "#7c3aed"
    elif e < 0.4:
        core_c1, core_c2 = "#a855f7", "#3b82f6"   # active purple-blue
        glow_color = "168, 85, 247"
        status_text = "Speech detected"
        status_dot  = "#a855f7"
    elif e < 0.7:
        core_c1, core_c2 = "#c084fc", "#60a5fa"   # bright purple-blue
        glow_color = "192, 132, 252"
        status_text = "Active speech"
        status_dot  = "#c084fc"
    else:
        core_c1, core_c2 = "#e879f9", "#818cf8"   # vibrant pink-indigo
        glow_color = "232, 121, 249"
        status_text = "High energy"
        status_dot  = "#e879f9"

    # Build ripple rings HTML
    ripple_rings = ""
    for i in range(ripple_count):
        delay = i * 0.4
        ripple_rings += f"""
        <div class="ripple" style="
            animation-delay: {delay}s;
            animation-duration: {pulse_dur + 0.5}s;
        "></div>"""

    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

        .sphere-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: {height - 30}px;
            font-family: 'Inter', sans-serif;
            user-select: none;
        }}

        .sphere-container {{
            position: relative;
            width: 200px;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .sphere {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(
                circle at 35% 35%,
                {core_c1},
                {core_c2} 70%,
                #1e1b4b
            );
            box-shadow:
                0 0 {glow_size}px rgba({glow_color}, {glow_alpha}),
                0 0 {glow_size * 2}px rgba({glow_color}, {glow_alpha * 0.4}),
                inset 0 -4px 12px rgba(0, 0, 0, 0.3),
                inset 0 4px 12px rgba(255, 255, 255, 0.08);
            transform: scale({scale});
            animation: pulse {pulse_dur}s ease-in-out infinite;
            transition: transform 0.5s ease, box-shadow 0.5s ease;
            z-index: 2;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale({scale}); }}
            50% {{ transform: scale({scale * 1.06}); }}
        }}

        .ripple {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 120px;
            height: 120px;
            margin-top: -60px;
            margin-left: -60px;
            border-radius: 50%;
            border: 2px solid rgba({glow_color}, 0.4);
            animation: ripple-expand {pulse_dur + 0.5}s ease-out infinite;
            z-index: 1;
        }}

        @keyframes ripple-expand {{
            0% {{
                transform: scale(1);
                opacity: 0.6;
            }}
            100% {{
                transform: scale(2.5);
                opacity: 0;
            }}
        }}

        /* Ambient particle ring */
        .particle-ring {{
            position: absolute;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            border: 1px solid rgba({glow_color}, 0.12);
            animation: ring-rotate {max(8 - e * 5, 3)}s linear infinite;
            z-index: 0;
        }}
        .particle-ring::before,
        .particle-ring::after {{
            content: '';
            position: absolute;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: rgba({glow_color}, {0.3 + e * 0.5});
            box-shadow: 0 0 6px rgba({glow_color}, 0.4);
        }}
        .particle-ring::before {{
            top: -2px;
            left: 50%;
            margin-left: -2px;
        }}
        .particle-ring::after {{
            bottom: -2px;
            left: 50%;
            margin-left: -2px;
        }}

        @keyframes ring-rotate {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
        }}

        .status-label {{
            margin-top: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 500;
            color: #94a3b8;
            letter-spacing: 0.5px;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {status_dot};
            box-shadow: 0 0 8px {status_dot};
            animation: dot-pulse 1.5s ease-in-out infinite;
        }}
        @keyframes dot-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
    </style>

    <div class="sphere-wrapper">
        <div class="sphere-container">
            <div class="sphere"></div>
            {ripple_rings}
            <div class="particle-ring"></div>
        </div>
        <div class="status-label">
            <div class="status-dot"></div>
            {status_text}
        </div>
    </div>
    """

    components.html(html, height=height, scrolling=False)
