"""Animated SVG fruit fly avatar — anatomically-detailed Drosophila.

CSS @keyframes on transform properties only (compositor-thread friendly).
Wing-beat speed driven by --wing-beat-duration CSS custom property
computed from FCI: high FCI → fast/energetic, low FCI → sluggish.
"""

from __future__ import annotations


def _wing_beat_duration(fci: float) -> float:
    """Map FCI ∈ [0,1] to wing-beat CSS duration in seconds.

    High FCI → fast wings (0.06s ≈ 200Hz realistic Drosophila wing-beat).
    Low FCI  → sluggish wings (0.8s).
    """
    # Linear interpolation: fci=1→0.06s, fci=0→0.8s
    return 0.8 - fci * 0.74


def _resolve_body_color(fci: float) -> str:
    """Abdomen stripe highlight: brighter amber when confident."""
    if fci > 0.7:
        return "#B8860B"  # dark goldenrod — vigorous
    if fci > 0.4:
        return "#8B6914"  # default amber
    return "#6B4226"  # dull brown — distressed


def _glow_params(neural_activity: float | None) -> tuple[float, float, str]:
    """Compute glow intensity, pulse duration, and glow color from activity.

    Returns (intensity 0-1, pulse_duration_s, css_color).
    """
    if neural_activity is None or neural_activity <= 0:
        return 0.0, 3.0, "rgba(0,255,136,0.0)"
    # Clamp to [0, 1] range for parameterization
    a = min(max(neural_activity, 0.0), 1.0)
    intensity = 0.2 + a * 0.7  # 0.2 → 0.9
    pulse_dur = 3.0 - a * 2.6   # 3.0s → 0.4s
    # Color: blend from green to amber at high activity
    if a > 0.7:
        color = f"rgba(255,204,0,{intensity:.2f})"
    else:
        color = f"rgba(0,255,136,{intensity:.2f})"
    return intensity, pulse_dur, color


def to_html(
    fci: float,
    event: str,
    neural_activity: float | None = None,
) -> str:
    """Return a self-contained HTML snippet with an animated SVG fruit fly.

    Parameters
    ----------
    fci : float
        Fly Confidence Index ∈ [0, 1].
    event : str
        Current economy event name (unused visually but reserved).
    neural_activity : float | None
        Mean output neuron rate. Controls glow intensity and pulse speed.
        ``None`` disables glow.

    Returns
    -------
    str
        HTML string containing ``<style>`` + ``<svg>`` + ``<script>``.
    """
    duration = _wing_beat_duration(fci)
    stripe_color = _resolve_body_color(fci)
    glow_intensity, glow_pulse_dur, glow_color = _glow_params(neural_activity)
    glow_visible = "visible" if glow_intensity > 0 else "hidden"

    # Scale factor ~1.33x for 400x520 viewBox (original 300x420)
    # All coordinates scaled inline below.
    return f"""\
<style>
  :root {{
    --wing-beat-duration: {duration:.3f}s;
    --glow-intensity: {glow_intensity:.2f};
    --glow-pulse-duration: {glow_pulse_dur:.2f}s;
  }}
  .fly-svg {{
    display: block;
    margin: 0 auto;
  }}
  /* ── Wing beat ───────────────────────────── */
  @keyframes wingBeatLeft {{
    0%   {{ transform: rotate(-5deg) scaleY(1.0); }}
    50%  {{ transform: rotate(-55deg) scaleY(0.6); }}
    100% {{ transform: rotate(-5deg) scaleY(1.0); }}
  }}
  @keyframes wingBeatRight {{
    0%   {{ transform: rotate(5deg) scaleY(1.0); }}
    50%  {{ transform: rotate(55deg) scaleY(0.6); }}
    100% {{ transform: rotate(5deg) scaleY(1.0); }}
  }}
  .wing-left {{
    transform-origin: 200px 213px;
    animation: wingBeatLeft var(--wing-beat-duration) ease-in-out infinite;
  }}
  .wing-right {{
    transform-origin: 200px 213px;
    animation: wingBeatRight var(--wing-beat-duration) ease-in-out infinite;
  }}
  /* ── Body bob ────────────────────────────── */
  @keyframes bodyBob {{
    0%   {{ transform: translateY(0px); }}
    50%  {{ transform: translateY(-5px); }}
    100% {{ transform: translateY(0px); }}
  }}
  .fly-body-group {{
    animation: bodyBob 1.2s ease-in-out infinite;
  }}
  /* ── Leg movement ────────────────────────── */
  @keyframes legWiggleLeft {{
    0%   {{ transform: rotate(0deg); }}
    50%  {{ transform: rotate(-4deg); }}
    100% {{ transform: rotate(0deg); }}
  }}
  @keyframes legWiggleRight {{
    0%   {{ transform: rotate(0deg); }}
    50%  {{ transform: rotate(4deg); }}
    100% {{ transform: rotate(0deg); }}
  }}
  .legs-left {{
    transform-origin: 187px 260px;
    animation: legWiggleLeft 0.8s ease-in-out infinite;
  }}
  .legs-right {{
    transform-origin: 213px 260px;
    animation: legWiggleRight 0.8s ease-in-out infinite 0.1s;
  }}
  /* ── Antenna sway ────────────────────────── */
  @keyframes antennaSway {{
    0%   {{ transform: rotate(0deg); }}
    50%  {{ transform: rotate(3deg); }}
    100% {{ transform: rotate(0deg); }}
  }}
  .antenna-left {{
    transform-origin: 187px 133px;
    animation: antennaSway 2.0s ease-in-out infinite;
  }}
  .antenna-right {{
    transform-origin: 213px 133px;
    animation: antennaSway 2.0s ease-in-out infinite 1.0s;
  }}
  /* ── Neural glow pulse ──────────────────── */
  @keyframes pulseGlow {{
    0%   {{ opacity: 0.3; transform: scale(1.0); }}
    50%  {{ opacity: 0.85; transform: scale(1.08); }}
    100% {{ opacity: 0.3; transform: scale(1.0); }}
  }}
  .neural-halo {{
    transform-origin: 200px 224px;
    animation: pulseGlow var(--glow-pulse-duration) ease-in-out infinite;
    visibility: {glow_visible};
  }}
</style>

<svg class="fly-svg" xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 400 520" width="400" height="520"
     role="img" aria-label="Animated fruit fly avatar">

  <defs>
    <!-- Wing vein gradient -->
    <linearGradient id="wingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="rgba(180,220,255,0.35)"/>
      <stop offset="100%" stop-color="rgba(200,235,255,0.15)"/>
    </linearGradient>
    <!-- Eye facet pattern -->
    <pattern id="facets" width="6" height="6" patternUnits="userSpaceOnUse">
      <circle cx="3" cy="3" r="2.2" fill="#CC0000" stroke="#990000"
              stroke-width="0.4"/>
    </pattern>
    <!-- Neural glow filter -->
    <filter id="neuralGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur"/>
      <feColorMatrix type="matrix"
        values="0 0 0 0 0
                0 1 0 0 0.5
                0 0 0 0 0.3
                0 0 0 1 0" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <g class="fly-body-group">

    <!-- ════════ NEURAL GLOW HALO (behind body) ════════ -->
    <ellipse class="neural-halo" cx="200" cy="224" rx="55" ry="45"
             fill="{glow_color}" filter="url(#neuralGlow)"
             opacity="{glow_intensity:.2f}"/>

    <!-- ════════ ANTENNAE ════════ -->
    <g class="antenna-left">
      <line x1="187" y1="133" x2="167" y2="96" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="167" y1="96" x2="153" y2="67" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="160" y1="80" x2="140" y2="60" stroke="#5C4033"
            stroke-width="1.2" stroke-linecap="round" opacity="0.7"/>
      <line x1="157" y1="73" x2="133" y2="64" stroke="#5C4033"
            stroke-width="1" stroke-linecap="round" opacity="0.5"/>
    </g>
    <g class="antenna-right">
      <line x1="213" y1="133" x2="233" y2="96" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="233" y1="96" x2="247" y2="67" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="240" y1="80" x2="260" y2="60" stroke="#5C4033"
            stroke-width="1.2" stroke-linecap="round" opacity="0.7"/>
      <line x1="243" y1="73" x2="267" y2="64" stroke="#5C4033"
            stroke-width="1" stroke-linecap="round" opacity="0.5"/>
    </g>

    <!-- ════════ HEAD ════════ -->
    <ellipse cx="200" cy="140" rx="43" ry="37" fill="#5C4033"/>
    <!-- Compound eyes -->
    <ellipse cx="171" cy="133" rx="19" ry="21" fill="url(#facets)"
             stroke="#990000" stroke-width="1"/>
    <ellipse cx="229" cy="133" rx="19" ry="21" fill="url(#facets)"
             stroke="#990000" stroke-width="1"/>
    <!-- Eye shine highlights -->
    <ellipse cx="165" cy="127" rx="5" ry="7" fill="rgba(255,255,255,0.15)"/>
    <ellipse cx="224" cy="127" rx="5" ry="7" fill="rgba(255,255,255,0.15)"/>
    <!-- Proboscis -->
    <line x1="200" y1="167" x2="200" y2="187" stroke="#5C4033"
          stroke-width="2.5" stroke-linecap="round"/>

    <!-- ════════ THORAX ════════ -->
    <ellipse cx="200" cy="224" rx="47" ry="40" fill="#5C4033"/>
    <ellipse cx="200" cy="224" rx="37" ry="32" fill="none"
             stroke="#4A3328" stroke-width="1" opacity="0.5"/>
    <!-- Scutellum -->
    <ellipse cx="200" cy="197" rx="24" ry="11" fill="#6B4226" opacity="0.6"/>

    <!-- ════════ WINGS ════════ -->
    <g class="wing-left">
      <ellipse cx="120" cy="187" rx="73" ry="27" fill="url(#wingGrad)"
               stroke="rgba(180,220,255,0.5)" stroke-width="1"/>
      <line x1="193" y1="207" x2="67" y2="180" stroke="rgba(150,190,230,0.4)"
            stroke-width="0.9"/>
      <line x1="187" y1="211" x2="80" y2="197" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.7"/>
      <line x1="180" y1="213" x2="93" y2="207" stroke="rgba(150,190,230,0.25)"
            stroke-width="0.7"/>
      <line x1="127" y1="177" x2="120" y2="197" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.7"/>
    </g>
    <g class="wing-right">
      <ellipse cx="280" cy="187" rx="73" ry="27" fill="url(#wingGrad)"
               stroke="rgba(180,220,255,0.5)" stroke-width="1"/>
      <line x1="207" y1="207" x2="333" y2="180" stroke="rgba(150,190,230,0.4)"
            stroke-width="0.9"/>
      <line x1="213" y1="211" x2="320" y2="197" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.7"/>
      <line x1="220" y1="213" x2="307" y2="207" stroke="rgba(150,190,230,0.25)"
            stroke-width="0.7"/>
      <line x1="273" y1="177" x2="280" y2="197" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.7"/>
    </g>

    <!-- ════════ ABDOMEN ════════ -->
    <ellipse cx="200" cy="280" rx="40" ry="24" fill="{stripe_color}"/>
    <ellipse cx="200" cy="280" rx="40" ry="24" fill="none"
             stroke="#4A3328" stroke-width="1"/>
    <ellipse cx="200" cy="320" rx="35" ry="21" fill="#5C4033"/>
    <ellipse cx="200" cy="320" rx="35" ry="21" fill="none"
             stroke="#4A3328" stroke-width="1"/>
    <ellipse cx="200" cy="357" rx="29" ry="19" fill="{stripe_color}"/>
    <ellipse cx="200" cy="357" rx="29" ry="19" fill="none"
             stroke="#4A3328" stroke-width="1"/>
    <ellipse cx="200" cy="389" rx="21" ry="16" fill="#5C4033"/>
    <ellipse cx="200" cy="389" rx="21" ry="16" fill="none"
             stroke="#4A3328" stroke-width="1"/>
    <ellipse cx="200" cy="413" rx="11" ry="11" fill="#4A3328"/>

    <!-- ════════ LEGS (3 pairs) ════════ -->
    <g class="legs-left">
      <line x1="173" y1="207" x2="140" y2="233" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="140" y1="233" x2="113" y2="267" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="113" y1="267" x2="100" y2="287" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="167" y1="233" x2="127" y2="273" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="127" y1="273" x2="100" y2="313" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="100" y1="313" x2="87" y2="336" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="171" y1="260" x2="133" y2="320" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="133" y1="320" x2="109" y2="373" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="109" y1="373" x2="96" y2="400" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
    </g>
    <g class="legs-right">
      <line x1="227" y1="207" x2="260" y2="233" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="260" y1="233" x2="287" y2="267" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="287" y1="267" x2="300" y2="287" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="233" y1="233" x2="273" y2="273" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="273" y1="273" x2="300" y2="313" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="300" y1="313" x2="313" y2="336" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="229" y1="260" x2="267" y2="320" stroke="#5C4033"
            stroke-width="3" stroke-linecap="round"/>
      <line x1="267" y1="320" x2="291" y2="373" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="291" y1="373" x2="304" y2="400" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
    </g>

  </g><!-- /fly-body-group -->
</svg>

<script>
function updateAvatar(fci) {{
  var duration = 0.8 - fci * 0.74;
  document.documentElement.style.setProperty('--wing-beat-duration', duration.toFixed(3) + 's');
}}

var _smoothedActivity = 0.0;
var _lastActivityTime = Date.now();

function updateAvatarActivity(activity) {{
  // Exponential smoothing: smoothed += (target - smoothed) * (1 - exp(-dt/tau))
  var now = Date.now();
  var dt = (now - _lastActivityTime) / 1000.0;
  _lastActivityTime = now;
  var tau = 0.5;
  var alpha = 1.0 - Math.exp(-dt / tau);
  _smoothedActivity += (activity - _smoothedActivity) * alpha;
  var a = Math.min(Math.max(_smoothedActivity, 0.0), 1.0);

  // Map to glow intensity and pulse duration
  var intensity = 0.2 + a * 0.7;
  var pulseDur = 3.0 - a * 2.6;
  document.documentElement.style.setProperty('--glow-intensity', intensity.toFixed(2));
  document.documentElement.style.setProperty('--glow-pulse-duration', pulseDur.toFixed(2) + 's');

  // Update halo visibility and opacity
  var halo = document.querySelector('.neural-halo');
  if (halo) {{
    halo.style.visibility = a > 0.01 ? 'visible' : 'hidden';
    halo.setAttribute('opacity', intensity.toFixed(2));
    // Update color: green → amber at high activity
    if (a > 0.7) {{
      halo.setAttribute('fill', 'rgba(255,204,0,' + intensity.toFixed(2) + ')');
    }} else {{
      halo.setAttribute('fill', 'rgba(0,255,136,' + intensity.toFixed(2) + ')');
    }}
  }}
}}
</script>
"""
