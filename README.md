# Triforce - a beamformer for Apple Silicon laptops
Triforce implements a Minimum Variance Distortionless Response adaptive beamformer
for the microphone array found in the following Apple Silicon laptops:
* MacBook Pro 13" (M1/M2)
* MacBook Air 13" (M1/M2)
* MacBook Pro 14" (M1 Pro/Max, M2 Pro/Max)
* MacBook Pro 16" (M1 Pro/Max, M2 Pro/Max)
* MacBook Air 15" (M2)

## Dependencies
Triforce tries to keep dependencies to a minimum. On top of the
crates specified in Cargo.lock, you will only require:
* LV2

## Why?
The Apple Silicon laptops mentioned above have an array of microphones arranged
either in a triangle or in a straight line. Much like with the speakers, Apple are trying way too hard to be
fancy here, and implement an adaptive beamformer in userspace to try and isolate
the desired signal from background noise. Without beamforming, the array is far
too sensitive and omnidirectional to be at all useful. Thus, to make it useful outside
of macOS, we need a beamformer.

## Expectation management
Finding accessible literature on any DSP-related topics, let alone something like
wideband adaptive beamforming, is challenging to say the least. This is an attempt
at a beamformer armed only with first year undergrad level engineering maths and some vague
idea of the principles gleaned from various webpages and PDFs. Do not expect it to
outperform Apple's implementation at this time. Patches to improve it are always welcome!

## Known limitations
* nalgebra does not do any explicit SIMD optimisation, relying only on LLVM auto-vectorisation.
  Performance and efficiency of matrix math routines are not very good.
* Following from that, we are not doing wideband decomposition due to the added computational
  burden. Without SIMD/NEON support, this is simply too slow for a realtime audio plugin.
* Output is mono only. Much like with wideband decomposition, adding additional matrix processing
  to fake stereo output would be too computationally intensive
