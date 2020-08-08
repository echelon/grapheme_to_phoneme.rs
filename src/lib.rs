// Copyright (c) 2020 Brandon Thomas <bt@brand.io>, <echelon@gmail.com>

//#![deny(dead_code)]
//#![deny(missing_docs)]
//#![deny(unreachable_patterns)]
//#![deny(unused_extern_crates)]
//#![deny(unused_imports)]
//#![deny(unused_qualifications)]

//! **Phoneme** is a prediction tool to turn graphemes into phonemes.
//! It is based on g2p.py

use std::fs::File;
use std::io::Read;
use std::{io, fmt};
use std::error::Error;
use std::fmt::Formatter;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_npy::{NpzReader, ReadNpzError, read_npy, ReadNpyError};
use std::collections::HashMap;

pub struct Phoneme {
}


pub struct Model {
  graphemes: Vec<String>,
  phonemes: Vec<String>,
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
  pub fn read_npy() -> Result<Self, PhonemeError> {
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
      .expect("<unk> should be a grapheme");

    let mut idx2p : HashMap<usize, String> = HashMap::new();
    for (i, val) in phonemes.iter().enumerate() {
      idx2p.insert(i, val.to_string());
    }

    let enc_emb : Array2<f32> = read_npy("data/enc_emb.npy")?; // (29, 64). (len(graphemes), emb)
    let enc_w_ih : Array2<f32> = read_npy("data/enc_w_ih.npy")?; // (3*128, 64)
    let enc_w_hh : Array2<f32> = read_npy("data/enc_w_hh.npy")?; // (3*128, 128)
    let enc_b_ih : Array1<f32> = read_npy("data/enc_b_ih.npy")?; // (3*128,)
    let enc_b_hh : Array1<f32> = read_npy("data/enc_b_hh.npy")?; // (3*128,)

    let dec_emb : Array2<f32> = read_npy("data/dec_emb.npy")?; // (74, 64). (len(phonemes), emb)
    let dec_w_ih : Array2<f32> = read_npy("data/dec_w_ih.npy")?; // (3*128, 64)
    let dec_w_hh : Array2<f32> = read_npy("data/dec_w_hh.npy")?; // (3*128, 128)
    let dec_b_ih : Array1<f32> = read_npy("data/dec_b_ih.npy")?; // (3*128,)
    let dec_b_hh : Array1<f32> = read_npy("data/dec_b_hh.npy")?; // (3*128,)

    let fc_w : Array2<f32> = read_npy("data/fc_w.npy")?; // (74, 128)
    let fc_b : Array1<f32> = read_npy("data/fc_b.npy")?; // (74,)

    Ok(Self {
      graphemes,
      phonemes,
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

  pub fn encode(&self, grapheme: &str) {
    let mut chars : Vec<String> = grapheme.chars()
      .map(|c| c.to_string())
      .collect();

    chars.push("</s>".to_string());

    let encoded : Vec<usize> = chars.iter()
      .map(|c| c.to_string())
      .map(|c| self.g2idx.get(&c)
        .map(|u| u.clone())
        .unwrap_or(self.unknown_grapheme_idx))
      .collect();

    //x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

    //grapheme.chars()
    //  .map()

    println!("Chars: {:?}", chars);
    println!("Encoded: {:?}", encoded);
  }
}

impl Phoneme {
  pub fn new() -> Self {
    Self {}
  }

  /*pub fn grapheme_to_phoneme(&self, polyphone: &str) -> String {
    "".to_string()
  }*/
}

#[derive(Debug)]
pub enum PhonemeError {
  IoError(std::io::Error),
  ReadNpyError(ndarray_npy::ReadNpyError),
  ReadNpzError(ndarray_npy::ReadNpzError),
}

impl From<io::Error> for PhonemeError {
  fn from(e: io::Error) -> Self {
    PhonemeError::IoError(e)
  }
}

impl From<ReadNpzError> for PhonemeError {
  fn from(e: ReadNpzError) -> Self {
    PhonemeError::ReadNpzError(e)
  }
}

impl From<ReadNpyError> for PhonemeError {
  fn from(e: ReadNpyError) -> Self {
    PhonemeError::ReadNpyError(e)
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
      PhonemeError::ReadNpzError(e) => Some(e),
      PhonemeError::ReadNpyError(e) => Some(e),
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::Model;
  use anyhow::Result as AnyhowResult;

  #[test]
  fn test_read_npz() -> AnyhowResult<()> {
    let model = Model::read_npy()?;
    Ok(())
  }

  #[test]
  fn test_encode() -> AnyhowResult<()> {
    let model = Model::read_npy()?;
    model.encode("test");

    assert_eq!(1,2);

    Ok(())
  }
}
