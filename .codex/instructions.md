# Codex Authoring Guidelines

## Model
- model: gpt-5.2-codex
- model_reasoning_effort: high
- approval_policy: never

## Output Expectations
- Produce compact, well-structured code and text with comments and docs while preserving clarity through naming and decomposition.
- Prefer reliable, modern language and framework features  with an emphasis on speed and running performance
- Apply clean code and best design patterns and best practices; keep methods small and cohesive.
- Avoid any tests and test frameworks unless explicitly required.
- Avoid design antipatterns and documentation antipatterns

## Project Context
- description: "Deep learning training framework for image super resolution and restoration."
- Target Python >= 12; torch>=2.7
- Optimize for maintainability: clear documenation and configuration points
- keywords: super-resolution, machine-learning, image-restoration, pytorch, computer vision (cv), safetensors, cuda, esrgan

## Style Preferences
- Keep formatting compact, but stylish, avoiding superfluous blank lines and inline comments.
