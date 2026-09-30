---
name: Forensic Precision Minimal
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#45464d'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e74'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#131b2e'
  on-primary-container: '#7c839b'
  inverse-primary: '#bec6e0'
  secondary: '#006d30'
  on-secondary: '#ffffff'
  secondary-container: '#92f5a4'
  on-secondary-container: '#007233'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#410002'
  on-tertiary-container: '#ef453c'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fd'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3f465c'
  secondary-fixed: '#95f8a7'
  secondary-fixed-dim: '#79db8d'
  on-secondary-fixed: '#00210a'
  on-secondary-fixed-variant: '#005323'
  tertiary-fixed: '#ffdad6'
  tertiary-fixed-dim: '#ffb4ab'
  on-tertiary-fixed: '#410002'
  on-tertiary-fixed-variant: '#93000b'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.03em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  margin: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system serves technical and forensic analysts conducting exhaustive digital video investigations. The interface must communicate clinical objectivity, incontrovertible certainty, and non-distracting visual clarity. 

The aesthetic is purely minimal and analytical. By stripping away extraneous decorative gradients, aggressive shadows, and high-saturation interface accents, the interface steps back to elevate evidentiary video frames, hex/metadata logs, and algorithmic confidence metrics. Every pixel, separator line, and typographic weight serves an informative function.

Visual hierarchy relies on crisp borders, clinical contrast ratios, and deliberate binary cues: an authentic state versus an altered state. The atmosphere is that of a modern scientific laboratory console—rigorous, silent, and indisputable.

## Colors

The system uses a strictly controlled, high-legibility light palette designed to prevent eye fatigue during prolonged analysis sessions.

- **Canvas & Base**: The root viewport background is strictly `#FFFFFF`. Structural cards, inspector panels, and grouped metadata containers use `#F9FAFB`. 
- **Dividers & Outlines**: Boundaries, table cells, and panel seams are rendered in crisp, subtle `#E5E7EB`.
- **Text & Hierarchy**: Primary typography, analytical values, and cryptographic hashes leverage deep slate `#0F172A`. Secondary labels, structural headings, and metadata keys use muted slate `#64748B`. Tertiary informational text and disabled states sit at `#94A3B8`.
- **Forensic Status - Authentic**: A muted, clinical forest green (`#15803D`) represents verified authentic media, valid cryptographic signatures, and unaltered compression streams. Paired with a delicate `#F0FDF4` surface tint and `#BBF7D0` outline for status pills.
- **Forensic Status - Altered / Synthetic**: A controlled, non-vibrant crimson (`#B91C1C`) denotes face reenactment, frame-splicing, deepfake artifacts, and anomalous audio-visual alignment. Paired with `#FEF2F2` surface tint and `#FECACA` border.
- **Forensic Status - Inconclusive**: A balanced neutral amber (`#B45309`) with `#FFFBEB` background for low-confidence margins requiring manual inspector override.

## Typography

Typography focuses on immediate, error-free legibility. `Inter` provides high legibility at dense scales across headings, descriptions, and summary findings. Tabular figures (`tnum`) must be forced across all analytical and timecode instances to maintain spatial stability during scrub operations.

`JetBrains Mono` is reserved for forensic data artifacts: timestamps (`HH:MM:SS:FF`), SHA-256 signatures, codec profiles, frame indexes, and pill badge labels. 

Headings remain compact and restrained; forensic interfaces prioritize dense horizontal data grids over oversized editorial typography. Line heights are tuned tightly to ensure rapid scanning without visual drift.

## Layout & Spacing

The layout is built for high-density, multi-pane desktop environments (1440px minimum target breakpoint). It operates on a fixed-gutter, fluid-panel architecture:

- **Panels & Canvases**: Standard layouts comprise a persistent 3-column split view: Left Inspector / Artifact Index (280px–320px fixed), Center Diagnostic Canvas (flexible viewport with synchronized frame-scrubber), and Right Evidence Findings Panel (360px–420px fixed).
- **Rhythm & Compaction**: Spacing tokens prioritize compact visual economy. Base increment is 4px. Structural cards and table cells use `space-sm` (8px) and `space-md` (12px) to maximize on-screen analytical data points without vertical scrolling.
- **Margins & Safe Zones**: Canvas perimeter margin is maintained at `margin` (24px) to frame the investigative workstation cleanly against hardware edges.

## Elevation & Depth

This system avoids decorative drop shadows and ambient blur elevations. Visual depth is established entirely through structural borders and tonal layering.

- **Level 0 (Canvas)**: Background workspace sits at `#FFFFFF`.
- **Level 1 (Card / Container Panels)**: Raised modules, sidebars, and analytical cards sit on `#F9FAFB`, framed with a 1px solid border of `#E5E7EB`.
- **Level 2 (Popovers, Tooltips & Overlays)**: Frame inspect overlays, context menus, and frame zoom loupes sit on `#FFFFFF` with a 1px `#CBD5E1` border and a single, clinical micro-shadow: `0 2px 4px 0 rgba(15, 23, 42, 0.04)`.
- **Interactive State Depth**: Hovering over rows or items does not translate or elevate the component; instead, it triggers a background shift from `#F9FAFB` to `#F3F4F6` or outlines with `#CBD5E1`.

## Shapes

The interface adopts a disciplined shape language. Standard structural modules—containers, video frames, tabular groupings, and input fields—use subtle 4px corner rounding (`roundedness: 1`), conveying scientific precision and structural rigidity. 

The sole exception to this rule is the forensic status badge, which utilizes a fully rounded pill form (`9999px`) to create a clear morphological contrast against rectangular data fields and metadata tables.

## Components

### Buttons
- **Primary**: Background `#0F172A`, text `#FFFFFF`, 4px border radius. Height 32px. Padding 0 12px. Font `Inter` 13px weight 500. No shadow. Hover: `#1E293B`.
- **Secondary / Ghost**: Background `#FFFFFF`, border 1px solid `#E5E7EB`, text `#0F172A`. Hover: `#F9FAFB` with border `#CBD5E1`.
- **Forensic Action**: Export Report / Flag Tampering uses contextual semantic outlines (e.g., `#B91C1C` border with `#FEF2F2` background).

### Status Badges (Pills)
- Fully rounded (`border-radius: 9999px`), padding `2px 8px`, font `JetBrains Mono` 11px uppercase weight 500.
- **Authentic**: `#F0FDF4` background, `#BBF7D0` border (1px solid), `#15803D` text. Preceded by a 5px solid dot in `#15803D`.
- **Altered / Deepfake**: `#FEF2F2` background, `#FECACA` border (1px solid), `#B91C1C` text. Preceded by a 5px solid dot in `#B91C1C`.
- **Unverified**: `#F8FAFC` background, `#E2E8F0` border (1px solid), `#64748B` text.

### Score & Probability Progress Bars
- Linear, non-rounded internal progress indicators with an overall height of 4px or 6px.
- Track container: `#F1F5F9`, 2px radius.
- Fill state: Pure `#15803D` for authenticity confidence; pure `#B91C1C` for synthetic likelihood. Transitions must be instant or capped at 150ms linear to prevent decorative sluggishness.

### Data Tables & Evidence Lists
- Headers: `#F9FAFB`, 11px `JetBrains Mono` uppercase text `#64748B`, 1px solid bottom border `#E5E7EB`. Height 28px.
- Rows: `#FFFFFF` alternate or single color with 1px horizontal bottom border `#F3F4F6`. Row height 36px. Compact text `#0F172A` in 12px or 13px. Hover state: `#F8FAFC`.

### Form Fields & Scrub Filters
- Inputs: `#FFFFFF` background, 1px solid `#E5E7EB`, 4px radius, 32px height, 13px font. Focus: 1px outline `#0F172A` with zero ring/glow.

### Video Frame Preview Cards
- Base container: `#F9FAFB` with 1px `#E5E7EB` border.
- Metadata footer: Split 2-column key/value display with monospaced timestamps and forensic verdicts docked to the card bottom.