# NAFNet module overview

This repository currently keeps the baseline NAFNet integration plus the
optional GDPM conditioner.

## Active modules

- `basicsr/models/GDPM.py`: Global Directional Prior Modulation.
- `basicsr/models/archs/NAFNet_arch.py`: NAFNet and NAFNetLocal.

## Integration

`GDPM` is applied after the intro convolution and before the encoder stack:

1. `x = intro(inp)`
2. `x = gdpm(inp, x)`
3. `x` enters the encoder path

Removed experimental modules no longer have model files, configuration entries,
imports, constructor arguments, or forward-pass hooks.
