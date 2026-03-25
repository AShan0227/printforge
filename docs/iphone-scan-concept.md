# iPhone LiDAR Scan → PrintForge Concept

## Overview

Leverage iPhone Pro's LiDAR scanner to capture real-world objects and feed them directly into PrintForge for print-ready mesh generation. This creates a seamless "scan → print" workflow within the Apple ecosystem.

## Technical Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  iPhone Pro  │────▶│  Point Cloud │────▶│  PrintForge  │────▶│  3D Printer  │
│  LiDAR Scan  │     │  + RGB Mesh  │     │  Optimization│     │  (Bambu/etc) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     ARKit               USDZ/OBJ            Repair +              Direct
    RealityKit           Export              Watertight             Send
```

### Step-by-step

1. **Capture** — User scans object with iPhone Pro using native ARKit/RealityKit
2. **Export** — iOS app exports point cloud or mesh as USDZ/OBJ/PLY
3. **Transfer** — AirDrop, iCloud, or direct API upload to PrintForge server
4. **Process** — PrintForge repairs mesh, ensures watertight, optimizes for printing
5. **Print** — Send directly to Bambu Lab printer or export 3MF/STL

## Required APIs

### ARKit (Capture)
- `ARWorldTrackingConfiguration` with `.sceneReconstruction = .meshWithClassification`
- `ARMeshAnchor` for real-time mesh geometry access
- Scene depth via `ARFrame.sceneDepth` (LiDAR depth map)

### RealityKit (Processing)
- `ModelEntity` for mesh manipulation
- `PhotogrammetrySession` (iOS 17+) for high-quality reconstruction
- `MeshResource` for programmatic mesh access

### Object Capture API (iOS 17+)
- `ObjectCaptureSession` — guided capture with feedback
- `PhotogrammetrySession` — async reconstruction from captured images
- Outputs USDZ with PBR materials

## User Flow Mockup

```
┌─────────────────────────────┐
│     PrintForge Scanner      │
│                             │
│  ┌───────────────────────┐  │
│  │                       │  │
│  │   [Camera viewfinder] │  │
│  │   LiDAR depth overlay │  │
│  │   Progress: 67%       │  │
│  │                       │  │
│  └───────────────────────┘  │
│                             │
│  Object: Coffee Mug        │
│  Points captured: 45,231   │
│  Mesh quality: Good        │
│                             │
│  [ Scan More ]  [ Done ✓ ] │
│                             │
├─────────────────────────────┤
│                             │
│  Processing...              │
│  ████████████░░░░ 75%       │
│                             │
│  → Mesh repair              │
│  → Watertight check         │
│  → Print optimization       │
│                             │
│  Estimated print time: 2h   │
│  Material: 23g PLA          │
│                             │
│  [ Send to Bambu A1 ]      │
│  [ Export 3MF ]             │
│                             │
└─────────────────────────────┘
```

## Competitive Advantage: Apple Ecosystem Lock-in

### Why This Wins

1. **Hardware moat** — LiDAR is only on iPhone Pro/iPad Pro. No Android equivalent with this quality.
2. **Seamless UX** — Scan → process → print in one app, no file juggling.
3. **Apple ecosystem** — AirDrop to Mac for slicing, iCloud sync, Shortcuts automation.
4. **Developer tools** — ARKit/RealityKit are mature, well-documented, GPU-optimized.

### Competitive Landscape

| Feature | PrintForge + iPhone | Polycam | Luma AI | 3D Scanner App |
|---------|-------------------|---------|---------|----------------|
| LiDAR scan | Yes | Yes | No (photogrammetry) | Yes |
| Print optimization | Yes | No | No | No |
| Watertight guarantee | Yes | No | No | No |
| Direct printer send | Yes | No | No | No |
| Cost estimation | Yes | No | No | No |
| Failure prediction | Yes | No | No | No |

### Key Differentiator
No competitor offers scan-to-print in one pipeline. Polycam and Luma generate meshes for visualization, not printing. PrintForge would be the first to guarantee print-ready output from an iPhone scan.

## Implementation Phases

### Phase 1: Import (2 weeks)
- Accept USDZ/OBJ/PLY uploads via API
- Auto-detect scan artifacts (noise, holes, disconnected components)
- Repair and optimize for printing

### Phase 2: iOS App (4 weeks)
- Swift/SwiftUI app with ARKit scanning
- Real-time mesh preview
- Direct upload to local PrintForge server

### Phase 3: On-Device (8 weeks)
- CoreML model for on-device mesh optimization
- Offline scan-to-print capability
- Apple Watch companion for print monitoring

## Technical Risks

- **LiDAR resolution** — ~1cm accuracy, insufficient for sub-mm mechanical parts
- **Reflective surfaces** — LiDAR struggles with glass, mirrors, shiny metal
- **Large objects** — Scanning area limited to ~5m range
- **Battery drain** — Continuous LiDAR + processing drains battery quickly
- **App Store review** — Apple may reject if app interferes with AR frameworks
