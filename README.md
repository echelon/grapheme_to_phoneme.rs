grapheme\_to\_phoneme
=====================
**grapheme\_to\_phoneme** is a prediction tool to turn graphemes (words) into Arpabet phonemes.

It is based on [Kyubyong Park's and Jongseok Kim's g2p.py](https://github.com/Kyubyong/g2p),
but only focuses on the prediction model (OOV prediction). CMUDict lookup and hetronym handling
are best handled by other libraries, such as my
[Arpabet crate](https://crates.io/crates/arpabet).

Usage
-----

```rust
extern crate grapheme_to_phoneme;
use grapheme_to_phoneme::Model;

let model = Model::load_in_memory()
  .expect("should load");

assert_eq!(model.predict("test").expect("should encode"),
  vec!["T", "EH1", "S", "T"].iter()
    .map(|s| s.to_string())
    .collect::<Vec<String>>());
```

License
-------
**BSD 4-clause**

Copyright (c) 2020, Brandon Thomas. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:

   This product includes software developed by Brandon Thomas
   (bt@brand.io, echelon@gmail.com).

4. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

