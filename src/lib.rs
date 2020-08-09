// Copyright (c) 2020 Brandon Thomas <bt@brand.io>, <echelon@gmail.com>
// Based on g2p.py (https://github.com/Kyubyong/g2p), by Kyubyong Park & Jongseok Kim

#![deny(dead_code)]
//#![deny(missing_docs)]
#![deny(unreachable_patterns)]
//#![deny(unused_extern_crates)]
//#![deny(unused_imports)]
#![deny(unused_qualifications)]

//! **Phoneme** is a prediction tool to turn graphemes into phonemes.
//! It is based on g2p.py

use ndarray::Array2;
use ndarray::Slice;
use ndarray::{Array1, Axis, Array3, ArrayView2};
use ndarray_npy::{read_npy, ReadNpyError, NpzReader, ReadNpzError};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{Cursor, Seek, Read};
use std::{io, fmt};
use std::path::Path;

const MODEL_NPZ : &'static [u8; 3342208] = include_bytes!("../model/model.npz");

pub struct Model {
  enc_emb : Array2<f32>,
  enc_w_ih : Array2<f32>,
  enc_w_hh : Array2<f32>,
  enc_b_ih : Array1<f32>,
  enc_b_hh : Array1<f32>,

  dec_emb : Array2<f32>,
  dec_w_ih : Array2<f32>,
  dec_w_hh : Array2<f32>,
  dec_b_ih : Array1<f32>,
  dec_b_hh : Array1<f32>,

  fc_w : Array2<f32>,
  fc_b : Array1<f32>,

  g2idx: HashMap<String, usize>,
  idx2p: HashMap<usize, String>,

  unknown_grapheme_idx: usize,
}

impl Model {
  /// Load the model that comes bundled in-memory with the library.
  /// This is baked into the library at compile time, and it cannot be updated
  /// without re-releasing the library.
  pub fn load_in_memory() -> Result<Self, PhonemeError> {
    // NB: Rust doesn't support arrays larger than 32 for read due to
    // `std::array::LengthAtMost32`, so we convert it to a slice.
    let cursor = Cursor::new(&MODEL_NPZ[..]);
    let reader = NpzReader::new(cursor)?;
    Self::from_npz_reader(reader)
  }

  /// Read a model from a NumPy '.npz' file.
  pub fn read_npz_file(filepath: &Path) -> Result<Self, PhonemeError> {
    let file = File::open(filepath)?;
    let reader = NpzReader::new(file)?;
    Self::from_npz_reader(reader)
  }

  /// Read a model from a directory containing '.npy' files of the expected names.
  ///
  /// This is provided in the event native NumPy '.npz' serialization fails with this
  /// library. (I was unable to get 'checkpoint20.npz to work without unzipping and rebundling it.)
  ///
  /// This looks for 'enc_emb.npy', 'enc_w_ih.npy', 'enc_w_hh.npy', 'enc_b_ih.npy',
  /// 'enc_b_hh.npy', 'dec_emb.npy', 'dec_w_ih.npy', 'dec_w_hh.npy', 'dec_b_ih.npy',
  /// 'dec_b_hh.npy', 'fc_w.npy', 'fc_b.npy'.  It's a bit heavy-handed, but may help if it's
  /// difficult to rebundle as 'npz'.
  pub fn from_npy_directory(directory: &Path) -> Result<Self, PhonemeError> {
    let enc_emb : Array2<f32> = read_npy(directory.join("enc_emb.npy"))?; // (29, 64). (len(graphemes), emb)
    let enc_w_ih : Array2<f32> = read_npy(directory.join("enc_w_ih.npy"))?; // (3*128, 64)
    let enc_w_hh : Array2<f32> = read_npy(directory.join("enc_w_hh.npy"))?; // (3*128, 128)
    let enc_b_ih : Array1<f32> = read_npy(directory.join("enc_b_ih.npy"))?; // (3*128,)
    let enc_b_hh : Array1<f32> = read_npy(directory.join("enc_b_hh.npy"))?; // (3*128,)
    let dec_emb : Array2<f32> = read_npy(directory.join("dec_emb.npy"))?; // (74, 64). (len(phonemes), emb)
    let dec_w_ih : Array2<f32> = read_npy(directory.join("dec_w_ih.npy"))?; // (3*128, 64)
    let dec_w_hh : Array2<f32> = read_npy(directory.join("dec_w_hh.npy"))?; // (3*128, 128)
    let dec_b_ih : Array1<f32> = read_npy(directory.join("dec_b_ih.npy"))?; // (3*128,)
    let dec_b_hh : Array1<f32> = read_npy(directory.join("dec_b_hh.npy"))?; // (3*128,)
    let fc_w : Array2<f32> = read_npy(directory.join("fc_w.npy"))?; // (74, 128)
    let fc_b : Array1<f32> = read_npy(directory.join("fc_b.npy"))?; // (74,)

    Self::from_arrays(
      enc_emb,
      enc_w_ih,
      enc_w_hh,
      enc_b_ih,
      enc_b_hh,
      dec_emb,
      dec_w_ih,
      dec_w_hh,
      dec_b_ih,
      dec_b_hh,
      fc_w,
      fc_b,
    )
  }

  fn from_npz_reader<T: Read + Seek>(mut npz_reader: NpzReader<T>) -> Result<Self, PhonemeError> {
    let enc_emb : Array2<f32> = npz_reader.by_name("enc_emb")?; // (29, 64). (len(graphemes), emb)
    let enc_w_ih : Array2<f32> = npz_reader.by_name("enc_w_ih")?; // (3*128, 64)
    let enc_w_hh : Array2<f32> = npz_reader.by_name("enc_w_hh")?; // (3*128, 128)
    let enc_b_ih : Array1<f32> = npz_reader.by_name("enc_b_ih")?; // (3*128,)
    let enc_b_hh : Array1<f32> = npz_reader.by_name("enc_b_hh")?; // (3*128,)
    let dec_emb : Array2<f32> = npz_reader.by_name("dec_emb")?; // (74, 64). (len(phonemes), emb)
    let dec_w_ih : Array2<f32> = npz_reader.by_name("dec_w_ih")?; // (3*128, 64)
    let dec_w_hh : Array2<f32> = npz_reader.by_name("dec_w_hh")?; // (3*128, 128)
    let dec_b_ih : Array1<f32> = npz_reader.by_name("dec_b_ih")?; // (3*128,)
    let dec_b_hh : Array1<f32> = npz_reader.by_name("dec_b_hh")?; // (3*128,)
    let fc_w : Array2<f32> = npz_reader.by_name("fc_w")?; // (74, 128)
    let fc_b : Array1<f32> = npz_reader.by_name("fc_b")?; // (74,)

    Self::from_arrays(
      enc_emb,
      enc_w_ih,
      enc_w_hh,
      enc_b_ih,
      enc_b_hh,
      dec_emb,
      dec_w_ih,
      dec_w_hh,
      dec_b_ih,
      dec_b_hh,
      fc_w,
      fc_b,
    )
  }

  fn from_arrays(
    enc_emb : Array2<f32>,
    enc_w_ih : Array2<f32>,
    enc_w_hh : Array2<f32>,
    enc_b_ih : Array1<f32>,
    enc_b_hh : Array1<f32>,
    dec_emb : Array2<f32>,
    dec_w_ih : Array2<f32>,
    dec_w_hh : Array2<f32>,
    dec_b_ih : Array1<f32>,
    dec_b_hh : Array1<f32>,
    fc_w : Array2<f32>,
    fc_b : Array1<f32>,
  ) -> Result<Self, PhonemeError> {
    let mut graphemes = Vec::new();
    graphemes.push("<pad>".to_string());
    graphemes.push("<unk>".to_string());
    graphemes.push("</s>".to_string());

    let characters : Vec<String> = "abcdefghijklmnopqrstuvwxyz".chars()
      .map(|c| c.to_string())
      .collect();

    graphemes.extend(characters);

    let phones = vec![
      "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
      "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D",
      "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F",
      "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
      "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
      "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
      "Z", "ZH"
    ];

    let mut phonemes = Vec::new();
    phonemes.push("<pad>".to_string());
    phonemes.push("<unk>".to_string());
    phonemes.push("<s>".to_string());
    phonemes.push("</s>".to_string());
    phonemes.extend(phones.iter().map(|p| p.to_string()).collect::<Vec<String>>());

    let mut g2idx : HashMap<String, usize> = HashMap::new();
    for (i, val) in graphemes.iter().enumerate() {
      g2idx.insert(val.to_string(), i);
    }

    let unknown_grapheme_idx = g2idx.get("<unk>")
      .map(|u| u.clone())
      .expect("<unk> should be a grapheme"); // TODO error handling

    let mut idx2p : HashMap<usize, String> = HashMap::new();
    for (i, val) in phonemes.iter().enumerate() {
      idx2p.insert(i, val.to_string());
    }

    Ok(Self {
      enc_emb,
      enc_w_ih,
      enc_w_hh,
      enc_b_ih,
      enc_b_hh,
      dec_emb,
      dec_w_ih,
      dec_w_hh,
      dec_b_ih,
      dec_b_hh,
      fc_w,
      fc_b,
      g2idx,
      idx2p,
      unknown_grapheme_idx,
    })
  }

  /// Predict phonemes from single-word grapheme input.
  pub fn predict(&self, grapheme: &str) -> Vec<String> {
    let enc = self.encode(grapheme);
    let enc = self.gru(&enc, grapheme.len() + 1);

    let last_hidden = enc.index_axis(Axis(1), grapheme.len()); // TODO: correct?

    let mut dec = self.dec_emb.index_axis(Axis(0), 2)// 2 = <s>
      .insert_axis(Axis(0)); // (1, 256)

    let mut h = last_hidden.to_owned();

    let mut preds = Vec::new();

    for _i in 0..20 {
      h = self.grucell(&dec, &h, &self.dec_w_ih, &self.dec_w_hh, &self.dec_b_ih, &self.dec_b_hh);

      // For 2d arrays, `dot(&Rhs)` computes the matrix multiplication
      let logits = h.dot(&self.fc_w.t()) + &self.fc_b;

      let pred = self.argmax(&logits);

      if pred == 3 {
        break; // 3 = </s>
      }

      preds.push(pred);

      dec = self.dec_emb.index_axis(Axis(0), pred)
        .insert_axis(Axis(0)); // (1, 256)
    }

    let preds = preds.iter()
      .map(|idx| self.idx2p.get(&idx)
        .map(|x| x.clone())
        .unwrap_or("<unk>".to_string()))
      .collect();

    preds
  }

  pub (crate) fn encode(&self, grapheme: &str) -> Array3<f32> {
    let mut chars : Vec<String> = grapheme.chars()
      .map(|c| c.to_string())
      .collect();

    chars.push("</s>".to_string());

    // Incredibly useful guide for porting NumPy to Rust ndarray:
    // https://docs.rs/ndarray/0.13.1/ndarray/doc/ndarray_for_numpy_users/index.html

    let encoded : Vec<usize> = chars.iter()
      .map(|c| c.to_string())
      .map(|c| self.g2idx.get(&c)
        .map(|u| u.clone())
        .unwrap_or(self.unknown_grapheme_idx))
      .collect();

    let shape = (encoded.len(), 256);
    let mut embeddings : Array2<f32> = Array2::zeros(shape);

    for (i, mut row) in embeddings.axis_iter_mut(Axis(0)).enumerate() {
      let embedding_index = *encoded.get(i)
        .expect("error handling"); // TODO ERROR HANDLING
      let embedding = self.enc_emb.index_axis(Axis(0), embedding_index);
      row.assign(&embedding);
    }

    embeddings.insert_axis(Axis(0)) // (1, N, 256)
  }

  pub (crate) fn gru(&self, x: &Array3<f32>, steps: usize) -> Array3<f32> {
    // Initial hidden state
    let mut h : Array2<f32> = Array2::zeros((1, 256));
    let mut outputs : Array3<f32> = Array3::zeros((1, steps, 256));

    for (i, mut row) in outputs.axis_iter_mut(Axis(1)).enumerate() {
      let sub_x = x.index_axis(Axis(1), i);
      h = self.grucell(&sub_x, &h, &self.enc_w_ih, &self.enc_w_hh, &self.enc_b_ih, &self.enc_b_hh);
      row.assign(&h);
    }

    outputs // (1, N, 256)
  }

  /// x: (1, 256)
  /// h: (1, 256)
  pub (crate) fn grucell(&self,
                 x: &ArrayView2<f32>,
                 h: &Array2<f32>,
                 w_ih: &Array2<f32>,
                 w_hh: &Array2<f32>,
                 b_ih: &Array1<f32>,
                 b_hh: &Array1<f32>,

  ) -> Array2<f32> {
    let rzn_ih  = x.dot(&w_ih.view().t()) + &b_ih.view();
    let rzn_hh  = h.dot(&w_hh.view().t()) + &b_hh.view();

    let t_ih = rzn_ih.shape()[1] * 2 / 3;
    let rz_ih = rzn_ih.slice_axis(Axis(1), Slice::from(0..t_ih));
    let n_ih = rzn_ih.slice_axis(Axis(1), Slice::from(t_ih..));

    let t_hh = rzn_hh.shape()[1] * 2 / 3;
    let rz_hh = rzn_hh.slice_axis(Axis(1), Slice::from(0..t_hh));
    let n_hh = rzn_hh.slice_axis(Axis(1), Slice::from(t_hh..));

    let rz_ih : Array2<f32> = rz_ih.to_owned();
    let rz_hh : Array2<f32> = rz_hh.to_owned();

    let result = rz_ih + rz_hh;
    let rz = self.sigmoid(&result);

    let (r, z) = rz.view().split_at(Axis(1), 256);

    let n_ih : Array2<f32> = n_ih.to_owned();
    let n_hh : Array2<f32> = n_hh.to_owned();
    let r = r.to_owned();

    let inner = n_ih + r * n_hh;

    let n = inner.map(|x: &f32| x.tanh());
    let z = z.into_owned();

    // output is (1, 256)
    (z.map(|x: &f32| 1.0 - x)) * n + z * h
  }

  /// x: (1, 512)
  /// output: (1, 512)
  pub (crate) fn sigmoid(&self, x: &Array2<f32>) -> Array2<f32> {
    let x : Array2<f32> = x.map(|x: &f32| 1.0 / (1.0 + (-x).exp()));
    x
  }

  pub (crate) fn argmax(&self, x: &Array2<f32>) -> usize {
    let mut max = f32::MIN;
    let mut argmax = 0;

    let mut i = 0;
    for y in x.slice_axis(Axis(1), Slice::from(0..)) {
      let y = *y;
      if y > max {
        max = y;
        argmax = i;
      }
      i += 1;
    }
    argmax
  }
}

#[derive(Debug)]
pub enum PhonemeError {
  IoError(std::io::Error),
  ReadNpyError(ReadNpyError),
  ReadNpzError(ReadNpzError),
}

impl From<io::Error> for PhonemeError {
  fn from(e: io::Error) -> Self {
    PhonemeError::IoError(e)
  }
}

impl From<ReadNpyError> for PhonemeError {
  fn from(e: ReadNpyError) -> Self {
    PhonemeError::ReadNpyError(e)
  }
}

impl From<ReadNpzError> for PhonemeError {
  fn from(e: ReadNpzError) -> Self {
    PhonemeError::ReadNpzError(e)
  }
}


impl fmt::Display for PhonemeError {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "PhonemeError")
  }
}

impl Error for PhonemeError {
  fn source(&self) -> Option<&(dyn Error + 'static)> {
    match &self {
      PhonemeError::IoError(e) => Some(e),
      PhonemeError::ReadNpyError(e) => Some(e),
      PhonemeError::ReadNpzError(e) => Some(e),
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::Model;
  use std::path::Path;

  #[test]
  fn load_in_memory() {
    let model = Model::load_in_memory();
    assert_eq!(true, model.is_ok());
    let _ = model.expect("should have loaded");
  }

  #[test]
  fn read_npz_file() {
    let model = Model::read_npz_file(Path::new("model/model.npz"));
    assert_eq!(true, model.is_ok());
    let _ = model.expect("should have loaded");
  }

  #[test]
  fn from_npy_directory() {
    let model = Model::from_npy_directory(Path::new("model/"));
    assert_eq!(true, model.is_ok());
    let _ = model.expect("should have loaded");
  }

  #[test]
  fn predict() {
    let model = Model::load_in_memory()
      .expect("Should be able to read");

    assert_eq!(model.predict("test"),
               vec!["T", "EH1", "S", "T"].iter()
                 .map(|s| s.to_string())
                 .collect::<Vec<String>>());

    assert_eq!(model.predict("zelda"),
               vec!["Z", "EH1", "L", "D", "AH0"].iter()
                 .map(|s| s.to_string())
                 .collect::<Vec<String>>());

    assert_eq!(model.predict("symphonia"),
               vec!["S", "IH0", "M", "F", "OW1", "N", "IY0", "AH0"].iter()
                 .map(|s| s.to_string())
                 .collect::<Vec<String>>());
  }

  #[test]
  fn predict_lots() {
    let model = Model::load_in_memory()
      .expect("Should be able to read");

    for _i in 0..100 {
      assert_eq!(model.predict("test"),
                 vec!["T", "EH1", "S", "T"].iter()
                   .map(|s| s.to_string())
                   .collect::<Vec<String>>());
    }
  }
}
