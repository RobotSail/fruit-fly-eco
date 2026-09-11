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


def to_html(fci: float, event: str) -> str:
    """Return a self-contained HTML snippet with an animated SVG fruit fly.

    Parameters
    ----------
    fci : float
        Fly Confidence Index ∈ [0, 1].
    event : str
        Current economy event name (unused visually but reserved).

    Returns
    -------
    str
        HTML string containing ``<style>`` + ``<svg>`` + ``<script>``.
    """
    duration = _wing_beat_duration(fci)
    stripe_color = _resolve_body_color(fci)

    return f"""\
<style>
  :root {{
    --wing-beat-duration: {duration:.3f}s;
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
    transform-origin: 150px 160px;
    animation: wingBeatLeft var(--wing-beat-duration) ease-in-out infinite;
  }}
  .wing-right {{
    transform-origin: 150px 160px;
    animation: wingBeatRight var(--wing-beat-duration) ease-in-out infinite;
  }}
  /* ── Body bob ────────────────────────────── */
  @keyframes bodyBob {{
    0%   {{ transform: translateY(0px); }}
    50%  {{ transform: translateY(-4px); }}
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
    transform-origin: 140px 195px;
    animation: legWiggleLeft 0.8s ease-in-out infinite;
  }}
  .legs-right {{
    transform-origin: 160px 195px;
    animation: legWiggleRight 0.8s ease-in-out infinite 0.1s;
  }}
  /* ── Antenna sway ────────────────────────── */
  @keyframes antennaSway {{
    0%   {{ transform: rotate(0deg); }}
    50%  {{ transform: rotate(3deg); }}
    100% {{ transform: rotate(0deg); }}
  }}
  .antenna-left {{
    transform-origin: 140px 100px;
    animation: antennaSway 2.0s ease-in-out infinite;
  }}
  .antenna-right {{
    transform-origin: 160px 100px;
    animation: antennaSway 2.0s ease-in-out infinite 1.0s;
  }}
</style>

<svg class="fly-svg" xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 300 420" width="300" height="420"
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
  </defs>

  <g class="fly-body-group">

    <!-- ════════ ANTENNAE ════════ -->
    <g class="antenna-left">
      <!-- Left antenna: 2 segments + arista -->
      <line x1="140" y1="100" x2="125" y2="72" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="125" y1="72" x2="115" y2="50" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <!-- Arista (feathery branch) -->
      <line x1="120" y1="60" x2="105" y2="45" stroke="#5C4033"
            stroke-width="1" stroke-linecap="round" opacity="0.7"/>
      <line x1="118" y1="55" x2="100" y2="48" stroke="#5C4033"
            stroke-width="0.8" stroke-linecap="round" opacity="0.5"/>
    </g>
    <g class="antenna-right">
      <!-- Right antenna: 2 segments + arista -->
      <line x1="160" y1="100" x2="175" y2="72" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="175" y1="72" x2="185" y2="50" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="180" y1="60" x2="195" y2="45" stroke="#5C4033"
            stroke-width="1" stroke-linecap="round" opacity="0.7"/>
      <line x1="182" y1="55" x2="200" y2="48" stroke="#5C4033"
            stroke-width="0.8" stroke-linecap="round" opacity="0.5"/>
    </g>

    <!-- ════════ HEAD ════════ -->
    <ellipse cx="150" cy="105" rx="32" ry="28" fill="#5C4033"/>
    <!-- Compound eyes -->
    <ellipse cx="128" cy="100" rx="14" ry="16" fill="url(#facets)"
             stroke="#990000" stroke-width="1"/>
    <ellipse cx="172" cy="100" rx="14" ry="16" fill="url(#facets)"
             stroke="#990000" stroke-width="1"/>
    <!-- Eye shine highlights -->
    <ellipse cx="124" cy="95" rx="4" ry="5" fill="rgba(255,255,255,0.15)"/>
    <ellipse cx="168" cy="95" rx="4" ry="5" fill="rgba(255,255,255,0.15)"/>
    <!-- Proboscis -->
    <line x1="150" y1="125" x2="150" y2="140" stroke="#5C4033"
          stroke-width="2" stroke-linecap="round"/>

    <!-- ════════ THORAX ════════ -->
    <ellipse cx="150" cy="168" rx="35" ry="30" fill="#5C4033"/>
    <!-- Thorax texture lines -->
    <ellipse cx="150" cy="168" rx="28" ry="24" fill="none"
             stroke="#4A3328" stroke-width="0.8" opacity="0.5"/>
    <!-- Scutellum -->
    <ellipse cx="150" cy="148" rx="18" ry="8" fill="#6B4226" opacity="0.6"/>

    <!-- ════════ WINGS ════════ -->
    <g class="wing-left">
      <ellipse cx="90" cy="140" rx="55" ry="20" fill="url(#wingGrad)"
               stroke="rgba(180,220,255,0.5)" stroke-width="0.8"/>
      <!-- Wing veins -->
      <line x1="145" y1="155" x2="50" y2="135" stroke="rgba(150,190,230,0.4)"
            stroke-width="0.7"/>
      <line x1="140" y1="158" x2="60" y2="148" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.5"/>
      <line x1="135" y1="160" x2="70" y2="155" stroke="rgba(150,190,230,0.25)"
            stroke-width="0.5"/>
      <!-- Cross vein -->
      <line x1="95" y1="133" x2="90" y2="148" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.5"/>
    </g>
    <g class="wing-right">
      <ellipse cx="210" cy="140" rx="55" ry="20" fill="url(#wingGrad)"
               stroke="rgba(180,220,255,0.5)" stroke-width="0.8"/>
      <line x1="155" y1="155" x2="250" y2="135" stroke="rgba(150,190,230,0.4)"
            stroke-width="0.7"/>
      <line x1="160" y1="158" x2="240" y2="148" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.5"/>
      <line x1="165" y1="160" x2="230" y2="155" stroke="rgba(150,190,230,0.25)"
            stroke-width="0.5"/>
      <line x1="205" y1="133" x2="210" y2="148" stroke="rgba(150,190,230,0.3)"
            stroke-width="0.5"/>
    </g>

    <!-- ════════ ABDOMEN ════════ -->
    <!-- Segment 1 -->
    <ellipse cx="150" cy="210" rx="30" ry="18" fill="{stripe_color}"/>
    <ellipse cx="150" cy="210" rx="30" ry="18" fill="none"
             stroke="#4A3328" stroke-width="0.8"/>
    <!-- Segment 2 -->
    <ellipse cx="150" cy="240" rx="26" ry="16" fill="#5C4033"/>
    <ellipse cx="150" cy="240" rx="26" ry="16" fill="none"
             stroke="#4A3328" stroke-width="0.8"/>
    <!-- Segment 3 -->
    <ellipse cx="150" cy="268" rx="22" ry="14" fill="{stripe_color}"/>
    <ellipse cx="150" cy="268" rx="22" ry="14" fill="none"
             stroke="#4A3328" stroke-width="0.8"/>
    <!-- Segment 4 (tapered tip) -->
    <ellipse cx="150" cy="292" rx="16" ry="12" fill="#5C4033"/>
    <ellipse cx="150" cy="292" rx="16" ry="12" fill="none"
             stroke="#4A3328" stroke-width="0.8"/>
    <!-- Abdomen tip -->
    <ellipse cx="150" cy="310" rx="8" ry="8" fill="#4A3328"/>

    <!-- ════════ LEGS (3 pairs) ════════ -->
    <!-- Fore legs (attached near head-thorax junction) -->
    <g class="legs-left">
      <!-- Left fore leg: femur → tibia → tarsus -->
      <line x1="130" y1="155" x2="105" y2="175" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="105" y1="175" x2="85" y2="200" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="85" y1="200" x2="75" y2="215" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
      <!-- Left mid leg -->
      <line x1="125" y1="175" x2="95" y2="205" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="95" y1="205" x2="75" y2="235" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="75" y1="235" x2="65" y2="252" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
      <!-- Left hind leg -->
      <line x1="128" y1="195" x2="100" y2="240" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="100" y1="240" x2="82" y2="280" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="82" y1="280" x2="72" y2="300" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
    </g>
    <g class="legs-right">
      <!-- Right fore leg -->
      <line x1="170" y1="155" x2="195" y2="175" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="195" y1="175" x2="215" y2="200" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="215" y1="200" x2="225" y2="215" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
      <!-- Right mid leg -->
      <line x1="175" y1="175" x2="205" y2="205" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="205" y1="205" x2="225" y2="235" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="225" y1="235" x2="235" y2="252" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
      <!-- Right hind leg -->
      <line x1="172" y1="195" x2="200" y2="240" stroke="#5C4033"
            stroke-width="2.5" stroke-linecap="round"/>
      <line x1="200" y1="240" x2="218" y2="280" stroke="#5C4033"
            stroke-width="2" stroke-linecap="round"/>
      <line x1="218" y1="280" x2="228" y2="300" stroke="#5C4033"
            stroke-width="1.5" stroke-linecap="round"/>
    </g>

  </g><!-- /fly-body-group -->
</svg>

<script>
function updateAvatar(fci) {{
  // Map FCI to wing-beat duration: high FCI → fast (0.06s), low → sluggish (0.8s)
  var duration = 0.8 - fci * 0.74;
  document.documentElement.style.setProperty('--wing-beat-duration', duration.toFixed(3) + 's');
}}
</script>
"""
