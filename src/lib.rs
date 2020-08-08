// Copyright (c) 2015, 2018, 2020 Brandon Thomas <bt@brand.io>

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

pub struct Phoneme {
}


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
}

impl Model {
pub fn read_npy() -> Result<Self, PhonemeError> {
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
    })
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
}
