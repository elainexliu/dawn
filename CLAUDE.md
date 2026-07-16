# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Dawn is a closed-loop wrist-worn neuromodulation device that predicts BFRB (body-focused
repetitive behavior) and tic urge states from physiological signals and
(eventually) delivers median nerve stimulation (MNS) before the behavior
begins. Solo-built, MIT EECS background, Neo Residency project.

## Commands

**Firmware (PlatformIO — run from `firmware/`):**
```
pio run                    # build
pio run -t upload          # flash to device
pio device monitor         # open serial monitor
pio run -t clean           # clean build artifacts
```

**Host Python (run from repo root):**
```
python -m host.acquisition.record    # collect a data session
python -m host.labeling.label        # annotate session
python -m host.training.train        # train model
```
*(Paths subject to change as host/ is built out.)*

## Current phase

**Sensing side only.** The TENS/MNS hardware has not arrived yet, so all
actuation logic is stubbed as a no-op. Do not build out TENS control,
stimulation parameters, or closed-loop actuation code unless explicitly asked
— that work is gated behind a sham-controlled self-experiment that hasn't run
yet.

Everything right now is: firmware bring-up -> data collection -> labeling ->
feature pipeline -> classical ML training -> (later) on-device inference.

## Architecture

Two halves, one repo. The ESP32 is intentionally "dumb" — it only talks to
sensors, timestamps, and streams raw packets over USB serial. All modeling,
labeling, and training logic lives on the laptop in Python.

```
firmware/     ESP32 (PlatformIO + Arduino framework). Sensor drivers, packet
              formatting, serial streaming. No inference, no ML, no MNS logic.
host/         Python. acquisition/, labeling/, pipeline/ (segmentation,
              features, normalization), training/, inference/ (phase 2).
data/         gitignored. raw/, labeled/, processed/.
models/       gitignored (or git-lfs later). Trained model artifacts.
notebooks/    exploratory only, not production code.
docs/         packet_format.md, session_log.md.
```

Communication: wired USB serial for now, not BLE. Tethered-to-laptop data
collection is deliberate at this phase, not a limitation to fix.

## Hardware in use

- ESP32 WROOM-32 / XIAO ESP32-S3 (dev)
- MPU6050 (IMU, prototyping) — LSM6DSV16X is the production target later
- MAX30102 (PPG/HR)
- MyoWare 2.0 + Kendall electrodes, via ADS1115, thenar placement — this is a
  **labeling/ground-truth instrument**, not a product sensor. Don't treat it
  as something the shipped device will include.
- TENS 7000 + MCP4131 digital potentiometer (on order, not yet integrated)

## Key technical decisions — don't relitigate these without new evidence

- **Classical ML first (LR / RF / GBT), not CNN-LSTM.** The Cambridge/Nokia
  paper (Searle et al.) hit 0.89–0.94 AUC on N=10 with hand-crafted features
  and simple models. A CNN-LSTM architecture referenced in other literature
  needed ~575K labeled examples to reach only 65.8% F1. Solo data collection
  will never approach that scale — don't reach for deep learning here.
- **Personalized models, not generic/cross-subject.** The goal is a model
  that works for one person (the user), not one that generalizes across
  strangers. Personalized cross-validation consistently outperformed generic
  in the reference paper, and it's the easier target for a solo N=1 build.
- **Store raw signals, not pre-extracted features**, so windows can be
  re-sliced later without redoing data collection.
- **Feature extraction/normalization is a single shared module** used
  identically at training time and inference time. Never duplicate this logic
  between an offline training script and an online inference script.
- **Day-based train/test splits, not random splits**, when evaluating on
  solo-collected data — random splits leak across overlapping windows from
  the same session and inflate scores.
- **Raw signals from all sensors are combined into a single packet per
  timestep** on the firmware side, so host and device never need to
  reconcile two independent clocks after the fact.

## Things intentionally deferred (don't build unless asked)

- BLE streaming (deliberately wired serial for now)
- On-device inference (deliberately laptop-side for now)
- TENS/MNS actuation and control loop (hardware not arrived, mechanism not
  yet validated)
- CI/CD, automated test suites, OTA firmware updates, protocol backward-compat
  handling, multi-repo split — all premature for a one-person, one-device,
  fast-iterating project. Revisit if the project scales beyond a solo build.
- One cheap exception: the packet format should include a version byte from
  the start (see `docs/packet_format.md`) since it costs nothing now and
  avoids ambiguity in old session logs later.

## Working style

- Be direct and concise. Push back on premature complexity (production
  infra, deep learning, generalization) rather than defaulting to it.
- The user is newer to the software/ML side than the hardware/research side
  — prefer clear module boundaries and explicit walkthroughs over dense,
  clever code.
- When touching `pipeline/` (segmentation, features, normalization), remember
  it's shared between training and inference — a change here affects both.
