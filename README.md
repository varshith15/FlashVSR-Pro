# flashvsr-scope-plugin

A Daydream Scope plugin for real-time video super resolution using FlashVSR.

This plugin is a fork of [FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro) — see the original repository for detailed documentation on FlashVSR capabilities, Docker setup, troubleshooting, and advanced configuration options.

## Features

- **Real-Time Video Upscaling** — 2x video super resolution using the FlashVSR tiny pipeline with TCDecoder for fast, high-quality results
- **Post-Processor Integration** — Designed to run as a post-processor in your Scope generation pipeline, keeping tensors on GPU for maximum efficiency

## Demo

Check out the [FlashVSR Scope Demo](https://app.daydream.live/creators/mammoth-peach-salmon-34/realtime-upscaler-flashvsr-demo) to see the plugin in action, installation walkthrough, and performance comparisons.

## Install

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to install this plugin using the URL:

```
https://github.com/varshith15/FlashVSR-Pro.git
```

Or install directly with:

```bash
uv run daydream-scope install git+https://github.com/varshith15/FlashVSR-Pro.git
```

## Upgrade

Follow the [Scope plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md) to upgrade this plugin to the latest version.

## Architecture

This plugin registers a single pipeline via the `register_pipelines` hook:

### FlashVSR (`flashvsr`)

A **video-mode** post-processor pipeline that performs 2x super resolution on input video frames. It uses the FlashVSR tiny pipeline with the TCDecoder (Tiny Conditional Decoder) for balanced quality and speed.

- **Scale Factor**: 2x upscaling
- **Decoder**: TCDecoder (TCD)
- **Pipeline**: Tiny mode for real-time performance
- **Input**: 8 frames at a time
- **Output**: Upscaled video frames aligned to 128px multiples

## Requirements

- NVIDIA GPU with 15GB+ VRAM
- CUDA support

## Performance

- ~30 FPS on RTX 5090 when used as a post-processor 
- ~22 FPS on RTX 5090 (end-to-end with WebRTC decode overhead)

## License

See the [original FlashVSR-Pro repository](https://github.com/LujiaJin/FlashVSR-Pro) for full license details and citations.
