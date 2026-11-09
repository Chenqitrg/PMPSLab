# Third-party notices

## TNRKit.jl — `pseudopow`

`src/gauge/pseudopow.jl` contains a modified version of the `pseudopow` helper
in [TNRKit.jl](https://github.com/QuantumKitHub/TNRKit.jl),
[`src/schemes/btrg.jl`](https://github.com/QuantumKitHub/TNRKit.jl/blob/e2907a3878f3e49f22d981be76575712763840a1/src/schemes/btrg.jl#L50-L58).
The local version accesses diagonal storage directly, uses an absolute-value
threshold comparison, and omits the unused sector variable.

The following upstream notice applies to this helper:

```text
MIT License

Copyright (c) 2025 Victor Vanthilt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
