# Provenance

The implementation was written for xangma, who owns the code and authorized
its release under the [MIT license](LICENSE).

The implementation follows RFC 1950 and RFC 1951. It contains no vendored zlib
or nvCOMP implementation. Python's stdlib zlib is used only as an independent
test and benchmark oracle. NumPy, JAX and the CUDA toolkit are separate dependencies
with their own licenses.
