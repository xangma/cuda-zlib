# Provenance

The decoder, wrapper and original byte-level tests were written by coding agents
for xangma in an unmerged development branch of xangma's PyCBC fork. They were
introduced together in development commit
`1b4563168adc3a5862621bd068b9a22d19e58443`; extraction used snapshot
`637ad7dfd0a8ec8541ea4edab9dce063ef89a2ce`.

Those development files carried PyCBC Collaboration / GPL boilerplate. xangma
confirmed their original development and authorized this independent MIT
release. The copyright and license headers now reflect that ownership.
This release does not change PyCBC's license. Earlier development copies retain
their existing license grants.

The standalone compressor, owned-output bridge, packaging, adapter and
fixed-block decoder repairs were subsequently developed for xangma. Version
`0.1.0a1` retains the executable codec logic and CUDA source strings of the
validated standalone candidate commit
`98974923a97c16c73a938e324d0a605834ee3eeb`. Release changes concern license
headers, package version, documentation and CI.

The implementation follows RFC 1950 and RFC 1951. It contains no vendored zlib
or nvCOMP implementation. Python's stdlib zlib is used only as an independent
test and benchmark oracle. NumPy, CuPy and optional JAX are separate dependencies
with their own licenses. The optional adapter does not make PyCBC or LAL core
package dependencies.
